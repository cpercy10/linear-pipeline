"""interior lane — remove.bg → force solid background (default white).

Output preserves input dimensions (remove.bg returns same-size RGBA).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import structlog
from PIL import Image

from config.pipeline_config import PipelineSettings
from processing.background_remover import BackgroundRemover
from utils.logging import get_logger, stage_timer


def replace_background_solid(
    rgba_image: Image.Image,
    bg_color: Tuple[int, int, int],
    alpha_threshold: int,
) -> Image.Image:
    """Force pixels with alpha < threshold to the solid background color, flatten to RGB."""
    arr = np.array(rgba_image)
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3]
    rgb[alpha < alpha_threshold] = bg_color
    return Image.fromarray(rgb, mode="RGB")


async def process_interior(
    image_bytes: bytes,
    filename: str,
    *,
    remover: BackgroundRemover,
    settings: PipelineSettings,
    log: Optional[structlog.stdlib.BoundLogger] = None,
) -> Image.Image:
    log = log or get_logger("interior")
    with stage_timer("remove_bg", log=log):
        rgba = await remover.remove(image_bytes, filename)
    with stage_timer("interior_flatten", log=log):
        return replace_background_solid(
            rgba,
            bg_color=settings.interior.bg_color,
            alpha_threshold=settings.removebg.interior_alpha_threshold,
        )
