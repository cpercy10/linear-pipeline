"""Lazy FLUX Fill inpaint wrapper for the server experiment."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

import torch
from PIL import Image

from config.pipeline_config import PipelineSettings
from processing.exceptions import ConfigError, PipelineError
from processing import image_processor as ip
from utils.logging import get_logger, stage_timer

_log = get_logger("flux_inpaint")


@dataclass
class InpaintRequest:
    enabled: bool
    mode: str
    prompt: str
    num_steps: int
    seed: Optional[int]
    max_long_edge: int
    body_opacity: float
    guidance_scale: float


def default_request(settings: PipelineSettings) -> InpaintRequest:
    cfg = settings.inpaint
    return InpaintRequest(
        enabled=bool(cfg.enabled),
        mode=cfg.mode,
        prompt=cfg.prompt,
        num_steps=int(cfg.num_steps),
        seed=cfg.seed,
        max_long_edge=int(cfg.max_long_edge),
        body_opacity=float(cfg.body_opacity),
        guidance_scale=float(cfg.guidance_scale),
    )


class FluxFillInpainter:
    """Loads FluxFillPipeline only when an enabled request first needs it."""

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._pipe = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def model_id(self) -> str:
        return self._settings.inpaint.model_id

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            if not torch.cuda.is_available():
                raise ConfigError("FLUX inpaint requires a CUDA GPU")
            try:
                from diffusers import FluxFillPipeline
            except ImportError as exc:
                raise ConfigError(
                    "diffusers is not installed; install the FLUX inpaint dependencies"
                ) from exc

            _log.info("flux.loading", model=self.model_id)
            pipe = FluxFillPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            )
            pipe.to("cuda")
            self._pipe = pipe
            _log.info("flux.ready", model=self.model_id)
            return self._pipe

    @staticmethod
    def _fit_pair(image: Image.Image, mask: Image.Image, max_long_edge: int):
        fitted = ip.fit_longest_edge(image.convert("RGB"), max_long_edge)
        if fitted.size == image.size:
            return fitted, mask.convert("L"), image.size
        return (
            fitted,
            mask.convert("L").resize(fitted.size, Image.LANCZOS),
            image.size,
        )

    def inpaint(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        if not req.enabled:
            return image.convert("RGB")
        pipe = self._ensure_pipe()
        img_in, mask_in, original_size = self._fit_pair(image, mask, req.max_long_edge)
        generator = None
        if req.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

        try:
            with stage_timer("flux_inpaint", device="cuda", gpu=True, log=_log) as extra:
                result = pipe(
                    prompt=req.prompt,
                    image=img_in,
                    mask_image=mask_in,
                    height=img_in.height,
                    width=img_in.width,
                    num_inference_steps=int(req.num_steps),
                    guidance_scale=float(req.guidance_scale),
                    generator=generator,
                ).images[0]
                extra["mode"] = req.mode
                extra["steps"] = int(req.num_steps)
                extra["seed"] = req.seed
                extra["work_size"] = [img_in.width, img_in.height]
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise PipelineError(f"FLUX inpaint ran out of memory: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"FLUX inpaint failed: {exc}") from exc

        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        return result.convert("RGB")

