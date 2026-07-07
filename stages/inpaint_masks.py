"""Mask construction for the server-only FLUX inpaint experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING, Tuple

import numpy as np
import structlog
from PIL import Image, ImageFilter

from config.pipeline_config import PipelineSettings
from stages import anchor as anchor_stage
from stages import composite_refine
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


def _dilate(mask: Image.Image, px: int) -> Image.Image:
    return composite_refine.dilate_mask(mask, px)


def _erode(mask: Image.Image, px: int) -> Image.Image:
    return composite_refine.erode_mask(mask, px)


def _mask_union(*masks: Image.Image) -> Image.Image:
    return composite_refine.mask_union(*masks)


def _mask_subtract(a: Image.Image, b: Image.Image) -> Image.Image:
    return composite_refine.mask_subtract(a, b)


def _paste_mask(mask: Image.Image, canvas_size: Tuple[int, int], xy: Tuple[int, int]) -> Image.Image:
    return composite_refine.paste_mask(mask, canvas_size, xy)


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
        cutout_resized = cutout_rgba.convert("RGBA").resize(pre.resized_crop.size, Image.LANCZOS)

        target = _composite_target(pre, plate, settings)
        xy = anchor_stage.compute_paste_top_left(
            target,
            cutout_resized.size,
            plate.plate.size,
            anchor_frac=(0.5, pre.anchor_frac[1]),
        )
        x, y, clamped = xy

        cutout_resized = composite_refine.refine_cutout_for_plate(
            cutout_resized,
            plate.plate,
            (x, y),
            settings,
        )
        cutout_alpha = cutout_resized.getchannel("A")
        shadow_hint = composite_refine.contact_shadow_mask(
            cutout_alpha,
            plate.plate.size,
            (x, y),
            settings,
        )
        base = composite_refine.apply_contact_shadow(
            plate.plate,
            shadow_hint,
            settings,
        ).convert("RGBA")
        base.alpha_composite(cutout_resized, (x, y))
        manual = base.convert("RGB")

        car_mask = _paste_mask(cutout_alpha, plate.plate.size, (x, y))
        preserve_mask = _erode(car_mask, settings.rembg.preserve_erode_px)
        edge_band = _mask_subtract(_dilate(car_mask, settings.rembg.edge_band_px), preserve_mask)
        shadow = _mask_subtract(shadow_hint, preserve_mask)
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
    """Merge FLUX output while preserving confident car pixels by default."""
    base = manual.convert("RGBA")
    generated = inpainted.resize(manual.size, Image.LANCZOS).convert("RGBA")

    if mode == "shadow_edge_body":
        opacity = max(0.0, min(1.0, float(body_opacity)))
        body = inputs.body_harmonize_mask.point(lambda p: int(p * opacity))
        edit_mask = _mask_union(inputs.shadow_mask, inputs.edge_band_mask, body)
    else:
        edit_mask = inputs.inpaint_mask.convert("L")

    merged = Image.composite(generated, base, edit_mask)
    # Keep the car body stable, but do not cover the repaired edge band for shadow_edge.
    if mode != "shadow_edge_body":
        restore_mask = inputs.car_mask if mode == "shadow" else inputs.preserve_mask
        car_layer = Image.new("RGBA", manual.size, (0, 0, 0, 0))
        car_layer.alpha_composite(inputs.cutout_resized, inputs.paste_top_left)
        layer_alpha = np.array(car_layer.getchannel("A"), dtype=np.uint8)
        restore_alpha = np.array(restore_mask.convert("L"), dtype=np.uint8)
        car_layer.putalpha(Image.fromarray(np.minimum(layer_alpha, restore_alpha), mode="L"))
        merged.alpha_composite(car_layer, (0, 0))

    return merged.convert("RGB")
