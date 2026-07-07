"""Lazy FLUX.2 Klein final image-edit refinement.

This is intentionally not an inpaint/mask path. The model sees either the manual rembg
composite alone or the composite plus references, then returns a full-frame polish.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple

import torch
from PIL import Image

from config.pipeline_config import PipelineSettings
from processing.exceptions import ConfigError, PipelineError
from utils.logging import get_logger, stage_timer

_log = get_logger("flux_refine")


@dataclass
class FluxRefineRequest:
    enabled: bool
    prompt: str
    num_steps: int
    seed: Optional[int]
    max_long_edge: int
    guidance_scale: float
    strength: Optional[float]
    reference_mode: str


def default_refine_request(settings: PipelineSettings) -> FluxRefineRequest:
    cfg = settings.flux_refine
    mode = cfg.reference_mode
    prompt = cfg.prompt
    if not prompt:
        prompt = (
            cfg.prompt_composite_only
            if mode == "composite_only"
            else cfg.prompt_with_reference
        )
    return FluxRefineRequest(
        enabled=bool(cfg.enabled),
        prompt=prompt,
        num_steps=int(cfg.num_steps),
        seed=cfg.seed,
        max_long_edge=int(cfg.max_long_edge),
        guidance_scale=float(cfg.guidance_scale),
        strength=cfg.strength,
        reference_mode=cfg.reference_mode,
    )


def _snap(n: int, multiple: int = 32) -> int:
    return max(multiple, (int(n) // multiple) * multiple)


def _fit_size(size: Tuple[int, int], max_long_edge: int) -> Tuple[int, int]:
    w, h = int(size[0]), int(size[1])
    max_long_edge = max(64, int(max_long_edge))
    scale = min(1.0, max_long_edge / max(1, max(w, h)))
    if w >= h:
        new_w = _snap(int(w * scale))
        new_h = _snap(round(new_w * h / max(1, w)))
    else:
        new_h = _snap(int(h * scale))
        new_w = _snap(round(new_h * w / max(1, h)))
    return max(64, new_w), max(64, new_h)


class FluxKleinRefiner:
    """Loads FLUX.2 Klein only when a request actually enables refinement."""

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._pipe = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def model_id(self) -> str:
        return self._settings.flux_refine.model_id

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            if not torch.cuda.is_available():
                raise ConfigError("FLUX refine requires a CUDA GPU")
            try:
                from diffusers import Flux2KleinPipeline
                pipe_cls = Flux2KleinPipeline
            except ImportError:
                try:
                    from diffusers import DiffusionPipeline
                    pipe_cls = DiffusionPipeline
                except ImportError as exc:
                    raise ConfigError(
                        "diffusers is not installed; install the FLUX.2 Klein dependencies"
                    ) from exc

            _log.info("flux_refine.loading", model=self.model_id)
            try:
                pipe = pipe_cls.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                )
            except TypeError:
                pipe = pipe_cls.from_pretrained(
                    self.model_id,
                    dtype=torch.bfloat16,
                )

            if self._settings.flux_refine.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            self._pipe = pipe
            _log.info("flux_refine.ready", model=self.model_id)
            return self._pipe

    @staticmethod
    def _prepare_inputs(
        composite: Image.Image,
        guide: Image.Image,
        car_reference: Optional[Image.Image],
        max_long_edge: int,
    ) -> Tuple[Image.Image, Image.Image, Optional[Image.Image], Tuple[int, int]]:
        original_size = composite.size
        work_size = _fit_size(original_size, max_long_edge)
        comp = composite.convert("RGB")
        ref = guide.convert("RGB")
        if comp.size != work_size:
            comp = comp.resize(work_size, Image.LANCZOS)
        if ref.size != work_size:
            ref = ref.resize(work_size, Image.LANCZOS)
        car_ref = None
        if car_reference is not None:
            car_ref = car_reference.convert("RGB")
            ref_long = max(64, min(max_long_edge, max(work_size)))
            ref_size = _fit_size(car_ref.size, ref_long)
            if car_ref.size != ref_size:
                car_ref = car_ref.resize(ref_size, Image.LANCZOS)
        return comp, ref, car_ref, original_size

    def refine(
        self,
        composite: Image.Image,
        gray_guide: Image.Image,
        car_reference: Optional[Image.Image],
        req: FluxRefineRequest,
    ) -> Image.Image:
        if not req.enabled:
            return composite.convert("RGB")

        pipe = self._ensure_pipe()
        comp_in, guide_in, car_ref_in, original_size = self._prepare_inputs(
            composite, gray_guide, car_reference, req.max_long_edge
        )

        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

        references = comp_in
        if req.reference_mode in {"with_reference", "multi_reference"}:
            references = [comp_in, guide_in]
            if car_ref_in is not None:
                references.append(car_ref_in)

        kwargs = {
            "prompt": req.prompt,
            "image": references,
            "height": comp_in.height,
            "width": comp_in.width,
            "guidance_scale": float(req.guidance_scale),
            "num_inference_steps": int(req.num_steps),
        }
        if generator is not None:
            kwargs["generator"] = generator
        if req.strength is not None:
            kwargs["strength"] = float(req.strength)

        try:
            with stage_timer("flux_refine", device="cuda", gpu=True, log=_log) as extra:
                result = pipe(**kwargs).images[0]
                extra["steps"] = int(req.num_steps)
                extra["seed"] = req.seed
                extra["work_size"] = [comp_in.width, comp_in.height]
                extra["reference_mode"] = req.reference_mode
                extra["reference_count"] = len(references) if isinstance(references, list) else 1
        except TypeError as exc:
            raise PipelineError(
                "FLUX refine call failed. If your installed diffusers build does not "
                "accept multiple input images, set "
                "MOTOCUT_FLUX_REFINE__REFERENCE_MODE=composite_only or upgrade diffusers."
            ) from exc
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise PipelineError(f"FLUX refine ran out of memory: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"FLUX refine failed: {exc}") from exc

        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        return result.convert("RGB")
