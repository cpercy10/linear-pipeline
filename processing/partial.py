"""exterior_partial lane — remove.bg → composite onto the shared background.

Center-crops the shared background to the segmented image's dimensions (scaling the
background up first if it's smaller), then alpha-composites the car on top. Output
preserves input dimensions.
"""

from __future__ import annotations

from typing import Optional

import structlog
from PIL import Image

from config.pipeline_config import PipelineSettings
from processing.background_remover import BackgroundRemover
from utils.logging import get_logger, stage_timer


def composite_partial(segmented_rgba: Image.Image, background_rgba: Image.Image) -> Image.Image:
    """Composite the segmented car (RGBA) onto a center-cropped background, at the
    segmented image's exact dimensions. Returns RGB."""
    input_w, input_h = segmented_rgba.size
    bg = background_rgba
    bg_w, bg_h = bg.size

    # If the input is larger than the background, scale the background up to cover.
    if input_w > bg_w or input_h > bg_h:
        scale = max(input_w / bg_w, input_h / bg_h)
        bg = bg.resize((int(bg_w * scale), int(bg_h * scale)), Image.LANCZOS)
        bg_w, bg_h = bg.size

    left = (bg_w - input_w) // 2
    top = (bg_h - input_h) // 2
    cropped_bg = bg.crop((left, top, left + input_w, top + input_h))

    final = cropped_bg.copy()
    final.alpha_composite(segmented_rgba, (0, 0))
    return final.convert("RGB")


async def process_partial(
    image_bytes: bytes,
    filename: str,
    *,
    remover: BackgroundRemover,
    background_rgba: Image.Image,
    settings: PipelineSettings,
    log: Optional[structlog.stdlib.BoundLogger] = None,
) -> Image.Image:
    log = log or get_logger("partial")
    with stage_timer("remove_bg", log=log):
        segmented = await remover.remove(image_bytes, filename)
    with stage_timer("partial_composite", log=log):
        return composite_partial(segmented, background_rgba)
