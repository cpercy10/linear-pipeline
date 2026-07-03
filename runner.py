"""Batch directory runner (ADAPTED from Module 3 ``runner/directory_runner.py``).

scan input dir → load shared background once (if provided) → START the warm Blender
worker → spin up the resource tiers → process every image with bounded concurrency →
write outputs (same filenames, INPUT dims) → SHUT DOWN the warm worker → print a
per-stage timing + lane + error summary.

The tiers:
  * preprocess pool — a ThreadPoolExecutor (YOLO / orientation / DINOv2 / GeoCalib /
    occupancy-resize / disc maths / the manual composite paste).
  * blender pool    — a single-slot async gate to ONE warm Blender process. Its
    lifecycle is owned here: started ONCE before the batch (loads scene + materials +
    car-free setup), torn down ONCE after.
  * remove.bg       — network-bound background removal (interior / exterior-partial and
    now the exterior-full cutout), gated by BackgroundRemover's own semaphore.

FLUX / diffusion has been removed: exterior-full composites the remove.bg cutout
directly onto the rendered plate, so there is no diffusion pool and no VRAM
co-residency planning. The warm Blender worker is the only heavy GPU user.

The end-of-run summary surfaces the fused-path stages
(``retrieval`` / ``geocalib`` / ``plate_render`` / ``resize`` / ``anchor`` /
``remove_bg`` / ``composite``) alongside the Module-3 ones, so the slowest stage is
visible per batch.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from PIL import Image

from config.pipeline_config import PipelineSettings, VisualOccupancyRegistry
from models.model_manager import ModelManager
from pipeline import Pipeline, PipelineResult
from processing.background_remover import BackgroundRemover
from processing.exceptions import ConfigError
from render.blender_worker import BlenderWorker
from runtime.blender_pool import BlenderPool
from utils.logging import get_logger
from utils.metrics import get_metrics

_log = get_logger("runner")

# Stages worth surfacing first in the summary (fused-path + headline Module-3 ones).
# Everything else still prints, sorted, after these.
_PRIORITY_STAGES = (
    "classify", "yolo", "orientation", "retrieval", "geocalib",
    "resize", "occupancy_resize", "plate_render", "anchor",
    "remove_bg", "composite",
)


class DirectoryRunner:
    def __init__(
        self,
        settings: PipelineSettings,
        manager: ModelManager,
        registry: VisualOccupancyRegistry,
    ) -> None:
        self.s = settings
        self.manager = manager
        self.registry = registry

    def _discover(self) -> List[Path]:
        exts = set(self.s.image_extensions)
        return sorted(
            p for p in self.s.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

    async def run(self) -> None:
        s = self.s
        # Batch mode requires the I/O dirs (config keeps them Optional for non-batch use).
        if s.input_dir is None or s.output_dir is None:
            raise ConfigError(
                "MOTOCUT_INPUT_DIR and MOTOCUT_OUTPUT_DIR are required for the batch runner"
            )
        paths = self._discover()
        if not paths:
            _log.warning("no_images", input_dir=str(s.input_dir))
            return
        s.output_dir.mkdir(parents=True, exist_ok=True)

        _log.info("batch.start", images=len(paths))

        # Shared background for the exterior-partial lane only (optional). exterior-full
        # uses the rendered plate, not this; interior forces a solid background.
        bg_rgba: Optional[Image.Image] = None
        if s.background_image and Path(s.background_image).exists():
            bg_rgba = Image.open(s.background_image).convert("RGBA")
        elif s.background_image:
            _log.warning("background_image.missing", path=str(s.background_image))

        remover = BackgroundRemover(s)

        # Warm Blender worker — started ONCE before the batch (loads scene + materials
        # + car-free setup), torn down ONCE after. The single-slot pool serializes
        # renders against each other.
        worker = BlenderWorker(s)
        blender_pool = BlenderPool(worker)

        pre_exec = ThreadPoolExecutor(
            max_workers=s.preprocess_workers, thread_name_prefix="pre"
        )
        pipeline = Pipeline(
            manager=self.manager, remover=remover,
            blender_pool=blender_pool,
            preprocess_executor=pre_exec, settings=s, registry=self.registry,
            background_rgba=bg_rgba,
        )

        results: List[PipelineResult] = []
        t0 = perf_counter()
        try:
            _log.info("blender.starting")
            await blender_pool.start()

            # Bounded concurrency: a fixed set of consumers drains a work queue.
            # Enough consumers to keep the network + Blender + preprocess tiers busy.
            work: asyncio.Queue[Path] = asyncio.Queue()
            for p in paths:
                work.put_nowait(p)
            n_consumers = (
                s.preprocess_workers + s.removebg_concurrency + blender_pool.size + 4
            )

            async def consumer() -> None:
                while True:
                    try:
                        p = work.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    results.append(await pipeline.process_path(p))

            await asyncio.gather(*[consumer() for _ in range(n_consumers)])
        finally:
            # Tear the tiers + warm worker down in reverse order of need.
            await blender_pool.shutdown()
            await remover.aclose()
            pre_exec.shutdown(wait=True)
        elapsed = perf_counter() - t0

        self._print_summary(results, elapsed)

    def _print_summary(self, results: List[PipelineResult], elapsed: float) -> None:
        done = sum(1 for r in results if r.status == "done")
        skipped = sum(1 for r in results if r.status == "skipped")
        errored = sum(1 for r in results if r.status == "error")
        total = len(results)
        ips = (done / elapsed) if elapsed > 0 else 0.0

        metrics = get_metrics()
        stages = metrics.stage_summary()
        lanes = metrics.lane_counts()

        _log.info(
            "batch.complete",
            total=total, done=done, skipped=skipped, errored=errored,
            elapsed_s=round(elapsed, 2), images_per_s=round(ips, 3),
            lanes=lanes,
        )

        # Order: the priority (fused-path) stages first in their canonical order, then
        # any remaining stages sorted alphabetically.
        ordered = [st for st in _PRIORITY_STAGES if st in stages]
        ordered += sorted(st for st in stages if st not in _PRIORITY_STAGES)

        # Human-readable table (stdout) alongside the structured log line above.
        lines = [
            "",
            "═══════════════════════ BATCH SUMMARY ═══════════════════════",
            f" images: {total}   done: {done}   skipped: {skipped}   errored: {errored}",
            f" elapsed: {elapsed:.2f}s   throughput: {ips:.3f} img/s (completed)",
            f" lanes: {lanes}",
            "",
            f" {'stage':<22}{'count':>7}{'p50_ms':>10}{'p95_ms':>10}{'mean_ms':>10}{'max_ms':>10}",
            " " + "-" * 67,
        ]
        for stage in ordered:
            st = stages[stage]
            lines.append(
                f" {stage:<22}{st['count']:>7}{st['p50_ms']:>10.1f}"
                f"{st['p95_ms']:>10.1f}{st['mean_ms']:>10.1f}{st['max_ms']:>10.1f}"
            )
        errors = metrics.errors()
        if errors:
            lines.append("")
            lines.append(f" errors ({len(errors)}):")
            for name, msg in errors[:20]:
                lines.append(f"   - {name}: {msg[:120]}")
            if len(errors) > 20:
                lines.append(f"   … and {len(errors) - 20} more")
        lines.append("══════════════════════════════════════════════════════════════")
        print("\n".join(lines))
