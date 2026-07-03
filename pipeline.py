"""End-to-end orchestration for a single image (ADAPTED from Module 3 ``pipeline.py``).

Routing → lane. The expensive concerns are dispatched off the event loop so many
images progress concurrently across the THREE resource tiers (BLUEPRINT §5):

  * GPU-light preprocess (classify / YOLO / orientation / DINOv2 / GeoCalib /
    occupancy-resize)                                         → preprocess pool
  * the warm Blender plate render                             → blender pool (1 slot)
  * remove.bg cutout + manual composite (exterior-full)       → network + CPU
  * network (remove.bg, interior/partial lanes)               → async + semaphore

Lanes (router output → :class:`config.pipeline_config.ImageClass`):
  * ``interior``         → ``processing.interior`` UNCHANGED (remove.bg + flatten).
  * ``exterior-partial`` → ``processing.partial`` UNCHANGED (remove.bg + composite
                            onto the supplied background). Terminal — no plate, no
                            diffusion.
  * ``exterior-full``    → the fused path: ``stages.exterior_full`` split across the
                            tiers: preprocess (tier 1) → render the per-car studio plate
                            (tier 2) → remove.bg cutout + manual composite onto the
                            plate (tier 3). The car is pasted at the SAME placement the
                            old FLUX path used; there is no diffusion model anymore.

The Blender worker's ``render`` surface is ``render(camera, disc_diam, out_jpg,
out_json, *, photo_w, photo_h, ...)`` returning a META dict and writing the plate
JPG to ``out_jpg``. The fused ``exterior_full`` stage exposes three plain callables
(``preprocess`` / ``composite_on_plate``) plus a ``_coerce_plate_result`` helper that normalises
a ``(plate, meta)`` pair into a :class:`stages.exterior_full.PlateRenderResult`. This
module bridges the two: it converts the fuser's flat camera dict into Module-2's
``*_deg/*_m`` export keys, drives the Blender pool, loads the rendered JPG, and feeds
the ``(plate, meta)`` pair to the coercer — so the heavy stage stays decoupled from
the async worker plumbing.
"""

from __future__ import annotations

import io
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import asyncio
from PIL import Image

from config.pipeline_config import ImageClass, PipelineSettings, VisualOccupancyRegistry
from models.model_manager import ModelManager
from models.timm_classifier import Prediction, TimmClassifier
from processing import interior, partial as partial_lane
from processing.background_remover import BackgroundRemover
from processing.exceptions import BlenderRenderError, DetectionError, PipelineError
from processing.validators import load_image
from runtime.blender_pool import BlenderPool
from stages import exterior_full
from utils.logging import get_logger, stage_timer
from utils.metrics import get_metrics


@dataclass
class PipelineResult:
    filename: str
    lane: str
    status: str            # "done" | "skipped" | "error"
    output: Optional[Path]
    error: Optional[str]


# Map the fuser's flat camera dict ({azimuth, elevation, distance, cam_height,
# focal, roll}) → Module-2 render_server.export_plate's payload keys
# ({azimuth_deg, elevation_deg, distance_m, cam_height_m, focal_mm, roll_deg}).
# This is the ONE naming bridge between the perspective fuser and the Blender worker.
_CAMERA_TO_EXPORT_KEYS = {
    "azimuth":    "azimuth_deg",
    "elevation":  "elevation_deg",
    "distance":   "distance_m",
    "cam_height": "cam_height_m",
    "focal":      "focal_mm",
    "roll":       "roll_deg",
}


def _camera_export_payload(camera: Dict[str, float]) -> Dict[str, float]:
    """Translate the fuser's flat camera dict into the worker/export key naming."""
    out: Dict[str, float] = {}
    for flat_key, export_key in _CAMERA_TO_EXPORT_KEYS.items():
        if flat_key in camera and camera[flat_key] is not None:
            out[export_key] = camera[flat_key]
    return out


def _classify(router: TimmClassifier, image: Image.Image, log) -> Prediction:
    with stage_timer("classify", device=router.device, gpu=True, log=log):
        return router.predict_one(image)


def _save(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = {}
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        image = image.convert("RGB")
        save_kwargs["quality"] = 95
    image.save(out_path, **save_kwargs)


def _encode_png(image: Image.Image) -> bytes:
    """Encode a crop to PNG bytes for the remove.bg upload (lossless — no artifacts)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


class Pipeline:
    def __init__(
        self,
        *,
        manager: ModelManager,
        remover: BackgroundRemover,
        blender_pool: BlenderPool,
        preprocess_executor: ThreadPoolExecutor,
        settings: PipelineSettings,
        registry: VisualOccupancyRegistry,
        background_rgba: Image.Image,
    ) -> None:
        self.m = manager
        self.remover = remover
        self.blender = blender_pool
        self.pre_exec = preprocess_executor
        self.s = settings
        self.reg = registry
        self.bg_rgba = background_rgba

    async def process_path(self, path: Path) -> PipelineResult:
        log = get_logger("pipeline").bind(request_id=path.name)
        loop = asyncio.get_running_loop()
        lane_value = "?"
        try:
            image_bytes = path.read_bytes()
            image = load_image(path, self.s.image_extensions)

            # ── route ────────────────────────────────────────────────────────
            pred = await loop.run_in_executor(
                self.pre_exec, partial(_classify, self.m.router, image, log)
            )
            lane = ImageClass(pred.label)
            lane_value = lane.value
            get_metrics().record_lane(lane_value)
            log.info("route", lane=lane_value, confidence=round(pred.confidence, 4))

            # ── lanes ────────────────────────────────────────────────────────
            if lane is ImageClass.INTERIOR:
                out = await interior.process_interior(
                    image_bytes, path.name,
                    remover=self.remover, settings=self.s, log=log,
                )
            elif lane is ImageClass.EXTERIOR_PARTIAL:
                out = await partial_lane.process_partial(
                    image_bytes, path.name,
                    remover=self.remover, background_rgba=self.bg_rgba,
                    settings=self.s, log=log,
                )
            else:  # EXTERIOR_FULL — the fused 3-module path across the three tiers.
                out = await self._exterior_full(image, path, log)

            out_path = self.s.output_dir / path.name
            _save(out, out_path)
            log.info("image.done", lane=lane_value, output=str(out_path))
            return PipelineResult(path.name, lane_value, "done", out_path, None)

        except DetectionError as exc:
            get_metrics().record_error(path.name, f"no_vehicle: {exc}")
            log.warning("image.skipped", reason="no_vehicle")
            return PipelineResult(path.name, lane_value, "skipped", None, str(exc))
        except PipelineError as exc:
            get_metrics().record_error(path.name, str(exc))
            log.error("image.error", error=str(exc))
            return PipelineResult(path.name, lane_value, "error", None, str(exc))
        except Exception as exc:  # noqa: BLE001 — one bad image must not kill the batch
            get_metrics().record_error(path.name, repr(exc))
            log.error("image.error", error=repr(exc), exc_info=True)
            return PipelineResult(path.name, lane_value, "error", None, repr(exc))

    # ── exterior-full: preprocess (tier 1) → plate (tier 2) → composite (tier 3) ──
    async def _exterior_full(self, image: Image.Image, path: Path, log) -> Image.Image:
        loop = asyncio.get_running_loop()

        # Tier 1 — GPU-light preprocess (preprocess pool). GeoCalib (inside the
        # fuser) loads the full frame itself, so it needs the on-disk path.
        pre = await loop.run_in_executor(
            self.pre_exec,
            partial(
                exterior_full.preprocess, image, path,
                yolo=self.m.yolo,
                orientation_model=self.m.orientation,
                retriever=self.m.retriever,
                settings=self.s,
                registry=self.reg,
                log=log,
            ),
        )

        # Tier 2 — warm Blender plate render (single-slot blender pool). Renders at
        # INPUT dims so cx,cy align 1:1 with the composited canvas.
        plate = await self._render_plate(pre, path, log)

        # Tier 3 — remove.bg cutout + manual composite (no diffusion). remove.bg is a
        # network call (async, semaphore-gated inside BackgroundRemover); the composite
        # is CPU image work, run in the preprocess pool so it never blocks the loop.
        crop_bytes = await loop.run_in_executor(
            self.pre_exec, partial(_encode_png, pre.raw_crop)
        )
        cutout = await self.remover.remove(
            crop_bytes, path.name, add_shadow=self.s.removebg.exterior_add_shadow
        )
        return await loop.run_in_executor(
            self.pre_exec,
            partial(exterior_full.composite_on_plate, pre, plate, cutout,
                    settings=self.s, log=log),
        )

    async def _render_plate(
        self, pre: "exterior_full.PreprocessResult", path: Path, log
    ) -> "exterior_full.PlateRenderResult":
        """Drive the warm Blender worker for one plate and normalise its output.

        The worker writes ``plate.jpg`` to ``out_jpg`` and returns the Module-2 meta
        dict (``car_spot_px`` + ``pixels_per_metre`` + ``camera.resolution``). We load
        the JPG back and hand the ``(plate, meta)`` pair to the fused stage's coercer,
        which recovers ``cx,cy`` + the lateral ppm into a ``PlateRenderResult``.
        """
        w, h = int(pre.orig_size[0]), int(pre.orig_size[1])
        long_edge = max(w, h)
        export_camera = _camera_export_payload(pre.camera)

        out_jpg = Path(tempfile.gettempdir()) / "motuva_plates" / f"{path.stem}_plate.jpg"
        out_json = out_jpg.with_suffix(".json")
        out_jpg.parent.mkdir(parents=True, exist_ok=True)

        meta = await self.blender.render(
            export_camera,
            pre.disc_m,
            str(out_jpg),
            str(out_json),
            photo_w=w,
            photo_h=h,
            studio=self.s.blender.studio,
            long_edge=long_edge,
            samples=self.s.blender.render_samples,
        )

        try:
            plate_img = Image.open(out_jpg)
            plate_img.load()
        except Exception as exc:  # noqa: BLE001 — surface a clean pipeline error
            raise BlenderRenderError(
                f"could not read rendered plate at {out_jpg}: {exc}"
            ) from exc

        # The fused stage already knows how to recover cx,cy + ppm from a (plate, meta)
        # pair in the Module-2 schema; reuse it so the contract lives in one place.
        return exterior_full._coerce_plate_result((plate_img, meta), log)
