"""Mask construction for the server-only FLUX inpaint experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING, Tuple

import cv2
import numpy as np
import structlog
from PIL import Image, ImageChops, ImageFilter

from config.pipeline_config import PipelineSettings
from stages import anchor as anchor_stage
from utils.logging import get_logger, stage_timer

if TYPE_CHECKING:
    from stages import exterior_full


@dataclass
class InpaintInputs:
    manual_composite: Image.Image
    cutout_resized: Image.Image
    car_mask: Image.Image
    preserve_mask: Image.Image
    edge_band_mask: Image.Image
    shadow_mask: Image.Image
    body_harmonize_mask: Image.Image
    inpaint_mask: Image.Image
    paste_top_left: Tuple[int, int]
    placed_size: Tuple[int, int]
    clamped: bool
    mask_stats: Dict[str, int] = field(default_factory=dict)


def _as_l_mask(mask: Image.Image) -> Image.Image:
    return mask.convert("L")


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def clean_alpha(alpha: Image.Image, settings: PipelineSettings) -> Image.Image:
    """Remove islands, fill holes, smooth jagged edges, and return a soft L mask."""
    cfg = settings.rembg
    mask = np.array(alpha.convert("L"))
    binary = (mask >= int(cfg.alpha_threshold)).astype(np.uint8) * 255

    if cfg.clean_mask:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(np.argmax(areas) + 1)
            binary = np.where(labels == largest, 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

        k = _odd_kernel(cfg.morph_kernel)
        kernel = np.ones((k, k), np.uint8)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        binary = filled

    soft = Image.fromarray(binary, mode="L")
    if cfg.feather_px > 0:
        soft = soft.filter(ImageFilter.GaussianBlur(float(cfg.feather_px)))
    return soft


def apply_alpha(cutout_rgba: Image.Image, alpha: Image.Image) -> Image.Image:
    out = cutout_rgba.convert("RGBA").copy()
    out.putalpha(alpha.resize(out.size, Image.LANCZOS).convert("L"))
    return out


def _dilate(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return _as_l_mask(mask)
    size = _odd_kernel(px * 2 + 1)
    return _as_l_mask(mask).filter(ImageFilter.MaxFilter(size))


def _erode(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return _as_l_mask(mask)
    size = _odd_kernel(px * 2 + 1)
    return _as_l_mask(mask).filter(ImageFilter.MinFilter(size))


def _mask_union(*masks: Image.Image) -> Image.Image:
    out = Image.new("L", masks[0].size, 0)
    for mask in masks:
        out = ImageChops.lighter(out, _as_l_mask(mask))
    return out


def _mask_subtract(a: Image.Image, b: Image.Image) -> Image.Image:
    return ImageChops.subtract(_as_l_mask(a), _as_l_mask(b))


def _paste_mask(mask: Image.Image, canvas_size: Tuple[int, int], xy: Tuple[int, int]) -> Image.Image:
    full = Image.new("L", canvas_size, 0)
    full.paste(_as_l_mask(mask), xy)
    return full


def _composite_target(pre, plate, settings: PipelineSettings) -> Tuple[float, float]:
    acfg = settings.anchor
    bias_frac = {
        "frontrear": acfg.forward_bias_frontrear,
        "threeq": acfg.forward_bias_threeq,
        "side": acfg.forward_bias_side,
    }.get(pre.pose, acfg.forward_bias_threeq)
    radius_m = max(0.0, float(pre.disc_m) / 2.0)
    ppm_depth = plate.plate_ppm_depth or plate.plate_ppm or 0.0
    bias_px = float(bias_frac) * radius_m * float(ppm_depth)
    return (plate.cx, plate.cy + bias_px)


def _shadow_mask(
    car_mask: Image.Image,
    xy: Tuple[int, int],
    placed_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    settings: PipelineSettings,
) -> Image.Image:
    cfg = settings.rembg
    x, y = xy
    cw, ch = placed_size
    width = max(8, int(cw * 0.92))
    height = max(6, int(ch * cfg.shadow_height_frac))
    cx = x + cw // 2
    cy = y + int(ch * (1.0 - cfg.shadow_offset_frac))

    arr = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
    axes = (max(4, width // 2), max(3, height // 2))
    cv2.ellipse(arr, (int(cx), int(cy)), axes, 0, 0, 360, 255, -1)
    shadow = Image.fromarray(arr, mode="L")
    if cfg.shadow_blur_px > 0:
        shadow = shadow.filter(ImageFilter.GaussianBlur(float(cfg.shadow_blur_px)))
    # Do not ask FLUX to repaint the confident car interior for pure shadow work.
    return _mask_subtract(shadow, _erode(car_mask, settings.rembg.preserve_erode_px))


def build_inputs(
    pre: "exterior_full.PreprocessResult",
    plate: "exterior_full.PlateRenderResult",
    cutout_rgba: Image.Image,
    *,
    settings: PipelineSettings,
    mode: str,
    log: Optional[structlog.stdlib.BoundLogger] = None,
) -> InpaintInputs:
    _log = log or get_logger("stages.inpaint_masks")

    with stage_timer("rembg_mask_build", log=_log) as extra:
        cleaned_alpha = clean_alpha(cutout_rgba.getchannel("A"), settings)
        cutout_clean = apply_alpha(cutout_rgba, cleaned_alpha)
        cutout_resized = cutout_clean.resize(pre.resized_crop.size, Image.LANCZOS)
        cutout_alpha = cutout_resized.getchannel("A")

        target = _composite_target(pre, plate, settings)
        xy = anchor_stage.compute_paste_top_left(
            target,
            cutout_resized.size,
            plate.plate.size,
            anchor_frac=(0.5, pre.anchor_frac[1]),
        )
        x, y, clamped = xy

        base = plate.plate.convert("RGBA")
        base.alpha_composite(cutout_resized, (x, y))
        manual = base.convert("RGB")

        car_mask = _paste_mask(cutout_alpha, plate.plate.size, (x, y))
        preserve_mask = _erode(car_mask, settings.rembg.preserve_erode_px)
        edge_band = _mask_subtract(_dilate(car_mask, settings.rembg.edge_band_px), preserve_mask)
        shadow = _shadow_mask(car_mask, (x, y), cutout_resized.size, plate.plate.size, settings)
        body = preserve_mask.filter(ImageFilter.GaussianBlur(max(1.0, settings.rembg.feather_px * 2.0)))

        if mode == "shadow":
            inpaint_mask = shadow
        elif mode == "shadow_edge_body":
            inpaint_mask = _mask_union(shadow, edge_band, body)
        else:
            inpaint_mask = _mask_union(shadow, edge_band)

        masks = {
            "car": car_mask,
            "preserve": preserve_mask,
            "edge_band": edge_band,
            "shadow": shadow,
            "body": body,
            "inpaint": inpaint_mask,
        }
        stats = {
            name: int(np.count_nonzero(np.array(mask.convert("L"))))
            for name, mask in masks.items()
        }

        extra["paste_top_left"] = [x, y]
        extra["placed_px"] = list(cutout_resized.size)
        extra["clamped"] = bool(clamped)
        extra["mask_pixels"] = stats

    return InpaintInputs(
        manual_composite=manual,
        cutout_resized=cutout_resized,
        car_mask=car_mask,
        preserve_mask=preserve_mask,
        edge_band_mask=edge_band,
        shadow_mask=shadow,
        body_harmonize_mask=body,
        inpaint_mask=inpaint_mask,
        paste_top_left=(x, y),
        placed_size=cutout_resized.size,
        clamped=bool(clamped),
        mask_stats=stats,
    )


def merge_inpaint_result(
    manual: Image.Image,
    inpainted: Image.Image,
    inputs: InpaintInputs,
    *,
    mode: str,
    body_opacity: float,
) -> Image.Image:
    """Merge FLUX output while preserving the original car silhouette by default."""
    base = manual.convert("RGBA")
    generated = inpainted.resize(manual.size, Image.LANCZOS).convert("RGBA")

    if mode == "shadow_edge_body":
        opacity = max(0.0, min(1.0, float(body_opacity)))
        body = inputs.body_harmonize_mask.point(lambda p: int(p * opacity))
        edit_mask = _mask_union(inputs.shadow_mask, inputs.edge_band_mask, body)
    else:
        edit_mask = inputs.inpaint_mask.convert("L")

    merged = Image.composite(generated, base, edit_mask)
    # Keep confident car pixels verbatim unless the user explicitly allowed body edits.
    if mode != "shadow_edge_body":
        car_layer = Image.new("RGBA", manual.size, (0, 0, 0, 0))
        car_layer.alpha_composite(inputs.cutout_resized, inputs.paste_top_left)
        merged.alpha_composite(car_layer, (0, 0))

    return merged.convert("RGB")
