"""Shared cutout cleanup, color harmonization, and contact-shadow helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter

from config.pipeline_config import PipelineSettings


@dataclass
class RefinedComposite:
    image: Image.Image
    cutout: Image.Image
    alpha: Image.Image
    shadow_mask: Image.Image
    stats: Dict[str, object] = field(default_factory=dict)


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def as_l_mask(mask: Image.Image) -> Image.Image:
    return mask.convert("L")


def dilate_mask(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return as_l_mask(mask)
    return as_l_mask(mask).filter(ImageFilter.MaxFilter(_odd_kernel(px * 2 + 1)))


def erode_mask(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return as_l_mask(mask)
    return as_l_mask(mask).filter(ImageFilter.MinFilter(_odd_kernel(px * 2 + 1)))


def mask_union(*masks: Image.Image) -> Image.Image:
    out = Image.new("L", masks[0].size, 0)
    for mask in masks:
        out = ImageChops.lighter(out, as_l_mask(mask))
    return out


def mask_subtract(a: Image.Image, b: Image.Image) -> Image.Image:
    return ImageChops.subtract(as_l_mask(a), as_l_mask(b))


def solidify_mask(
    mask: Image.Image,
    *,
    threshold: int = 8,
    close_px: int = 5,
    fill_holes: bool = True,
) -> Image.Image:
    """Return a hard object support mask, including transparent glass holes."""
    arr = (np.array(mask.convert("L"), dtype=np.uint8) > int(threshold)).astype(np.uint8) * 255

    if close_px > 0 and int(np.count_nonzero(arr)) > 0:
        kernel = np.ones((_odd_kernel(close_px * 2 + 1), _odd_kernel(close_px * 2 + 1)), np.uint8)
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)

    if fill_holes and int(np.count_nonzero(arr)) > 0:
        padded = cv2.copyMakeBorder(arr, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        h, w = flood.shape[:2]
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
        arr = cv2.bitwise_or(arr, holes)

    return Image.fromarray(arr, mode="L")


def paste_mask(mask: Image.Image, canvas_size: Tuple[int, int], xy: Tuple[int, int]) -> Image.Image:
    full = Image.new("L", canvas_size, 0)
    full.paste(as_l_mask(mask), xy)
    return full


def clean_alpha(alpha: Image.Image, settings: PipelineSettings) -> Image.Image:
    """Return a cleaner car matte with a tight antialiased edge and fewer halos."""
    cfg = settings.rembg
    raw = np.array(alpha.convert("L"), dtype=np.uint8)
    threshold = max(1, int(cfg.alpha_threshold))
    binary = (raw >= threshold).astype(np.uint8) * 255

    if cfg.clean_mask:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas):
                largest = float(np.max(areas))
                min_area = max(32, int(largest * 0.015))
                keep = np.where(areas >= min_area)[0] + 1
                binary = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

        k = _odd_kernel(cfg.morph_kernel)
        kernel = np.ones((k, k), np.uint8)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        binary = filled

    halo_erode = max(0, int(getattr(cfg, "alpha_halo_erode_px", 1)))
    if halo_erode > 0 and int(np.count_nonzero(binary)) > 0:
        halo_kernel = np.ones((_odd_kernel(halo_erode * 2 + 1), _odd_kernel(halo_erode * 2 + 1)), np.uint8)
        binary = cv2.erode(binary, halo_kernel, iterations=1)

    feather = max(0.0, float(cfg.feather_px))
    if feather > 0 and int(np.count_nonzero(binary)) > 0:
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 3)
        signed = inside - outside
        width = max(0.75, feather)
        soft = np.clip((signed + width) / (2.0 * width), 0.0, 1.0) * 255.0
    else:
        soft = binary.astype(np.float32)

    support_kernel = np.ones((_odd_kernel(3), _odd_kernel(3)), np.uint8)
    support = cv2.dilate(binary, support_kernel, iterations=1)
    core_px = max(1, int(cfg.preserve_erode_px) // 2)
    core_kernel = np.ones((_odd_kernel(core_px * 2 + 1), _odd_kernel(core_px * 2 + 1)), np.uint8)
    core = cv2.erode(binary, core_kernel, iterations=1)

    cleaned = np.where(support > 0, soft, 0.0)
    cleaned = np.where(core > 0, np.maximum(cleaned, raw.astype(np.float32)), cleaned)
    cleaned = np.where(binary > 0, np.maximum(cleaned, float(threshold)), cleaned)
    cleaned = cleaned.clip(0, 255).astype(np.uint8)
    return Image.fromarray(cleaned, mode="L")


def apply_alpha(cutout_rgba: Image.Image, alpha: Image.Image) -> Image.Image:
    out = cutout_rgba.convert("RGBA").copy()
    out.putalpha(alpha.resize(out.size, Image.LANCZOS).convert("L"))
    return out


def _decontaminate_edge_rgb(cutout_rgba: Image.Image, settings: PipelineSettings) -> Image.Image:
    """Fill semi-transparent edge RGB from nearby opaque car pixels."""
    cfg = settings.rembg
    rgba = np.array(cutout_rgba.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    unknown = ((alpha >= int(cfg.alpha_threshold)) & (alpha < 245)).astype(np.uint8) * 255
    if int(np.count_nonzero(unknown)) < 8 or int(np.count_nonzero(alpha >= 245)) < 32:
        return cutout_rgba.convert("RGBA")

    radius = max(1, int(getattr(cfg, "edge_inpaint_radius_px", max(2, cfg.feather_px + 1))))
    rgb = rgba[:, :, :3]
    repaired = np.empty_like(rgb)
    for channel in range(3):
        repaired[:, :, channel] = cv2.inpaint(
            rgb[:, :, channel], unknown, radius, cv2.INPAINT_TELEA
        )

    blend = cv2.GaussianBlur(unknown, (0, 0), sigmaX=1.2, sigmaY=1.2).astype(np.float32) / 255.0
    rgb_out = rgb.astype(np.float32) * (1.0 - blend[:, :, None])
    rgb_out += repaired.astype(np.float32) * blend[:, :, None]
    rgba[:, :, :3] = rgb_out.clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _neutralize_glass_contamination(
    cutout_rgba: Image.Image,
    plate_rgb: Image.Image,
    xy: Tuple[int, int],
    settings: PipelineSettings,
) -> Image.Image:
    """Suppress old-scene detail in likely glass while preserving dark tint and highlights."""
    cfg = settings.rembg
    if not bool(getattr(cfg, "glass_cleanup", True)):
        return cutout_rgba.convert("RGBA")

    strength = max(0.0, min(1.0, float(getattr(cfg, "glass_cleanup_strength", 0.72))))
    if strength <= 0:
        return cutout_rgba.convert("RGBA")

    cutout = cutout_rgba.convert("RGBA")
    x, y = int(xy[0]), int(xy[1])
    cw, ch = cutout.size
    plate_crop = plate_rgb.convert("RGB").crop((x, y, x + cw, y + ch))

    rgba = np.array(cutout, dtype=np.float32)
    rgb_u8 = rgba[:, :, :3].clip(0, 255).astype(np.uint8)
    plate = np.array(plate_crop, dtype=np.float32)
    alpha = rgba[:, :, 3]

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    yy = np.arange(ch, dtype=np.float32)[:, None]

    support = alpha >= max(24, int(cfg.alpha_threshold))
    inner = cv2.erode(
        support.astype(np.uint8) * 255,
        np.ones((_odd_kernel(7), _odd_kernel(7)), np.uint8),
        iterations=1,
    ) > 0
    glass_band = (yy >= ch * 0.06) & (yy <= ch * 0.62)
    dark_neutral = (sat <= int(getattr(cfg, "glass_max_saturation", 120))) & (
        val <= int(getattr(cfg, "glass_max_value", 185))
    )
    translucent_inside = (alpha < 245) & inner
    candidate = support & glass_band & inner & (dark_neutral | translucent_inside)

    candidate_u8 = candidate.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
    keep = np.zeros_like(candidate_u8)
    min_area = max(48, int(cw * ch * 0.0015))
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        by = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        if bw < max(8, int(cw * 0.035)) or bh < max(5, int(ch * 0.018)):
            continue
        if by + bh * 0.5 > ch * 0.66:
            continue
        keep[labels == idx] = 255

    if int(np.count_nonzero(keep)) < min_area:
        return cutout

    mask = cv2.GaussianBlur(keep, (0, 0), sigmaX=2.0, sigmaY=2.0).astype(np.float32) / 255.0
    mask *= strength

    smooth_sigma = max(3.0, min(12.0, cw * 0.018))
    smooth_car = cv2.GaussianBlur(rgb_u8, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma).astype(np.float32)
    neutral_dark = np.array([22.0, 25.0, 27.0], dtype=np.float32)
    scene_tint = plate * 0.24 + neutral_dark[None, None, :] * 0.76
    target = smooth_car * 0.36 + scene_tint * 0.64

    original_luma = np.dot(rgba[:, :, :3], np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    target_luma = np.dot(target, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    target += (original_luma - target_luma)[:, :, None] * 0.28

    rgba[:, :, :3] = rgba[:, :, :3] * (1.0 - mask[:, :, None]) + target * mask[:, :, None]
    return Image.fromarray(rgba.clip(0, 255).astype(np.uint8), mode="RGBA")


def _harmonize_color(
    cutout_rgba: Image.Image,
    plate_rgb: Image.Image,
    xy: Tuple[int, int],
    settings: PipelineSettings,
) -> Image.Image:
    cfg = settings.rembg
    strength = float(getattr(cfg, "color_match_strength", 0.22))
    edge_blend_strength = float(getattr(cfg, "edge_plate_blend", 0.18))
    if strength <= 0 and edge_blend_strength <= 0:
        return cutout_rgba.convert("RGBA")

    cutout = cutout_rgba.convert("RGBA")
    x, y = int(xy[0]), int(xy[1])
    cw, ch = cutout.size
    plate_crop = plate_rgb.convert("RGB").crop((x, y, x + cw, y + ch))

    rgba = np.array(cutout, dtype=np.float32)
    plate = np.array(plate_crop, dtype=np.float32)
    alpha = rgba[:, :, 3]
    support = alpha >= max(1, int(cfg.alpha_threshold))

    if strength > 0:
        support_u8 = support.astype(np.uint8) * 255
        band_px = max(6, int(getattr(cfg, "color_match_band_px", max(12, cfg.edge_band_px))))
        dilated = cv2.dilate(
            support_u8,
            np.ones((_odd_kernel(band_px * 2 + 1), _odd_kernel(band_px * 2 + 1)), np.uint8),
            iterations=1,
        )
        ring = (dilated > 0) & (~support)
        if int(np.count_nonzero(ring)) >= 64:
            bg = np.median(plate[ring], axis=0)
            weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            bg_luma = float(np.dot(bg, weights))
            if bg_luma > 1.0:
                gains = bg / bg_luma
                gains_luma = float(np.dot(gains, weights))
                if gains_luma > 1e-4:
                    gains = gains / gains_luma
                gains = 1.0 + (gains - 1.0) * strength
                gains = np.clip(gains, 0.88, 1.12)
                body_blend = ((alpha / 255.0) ** 0.65) * min(1.0, max(0.0, strength * 1.35))
                corrected = rgba[:, :, :3] * gains[None, None, :]
                rgba[:, :, :3] = rgba[:, :, :3] * (1.0 - body_blend[:, :, None])
                rgba[:, :, :3] += corrected * body_blend[:, :, None]

    if edge_blend_strength > 0:
        edge = ((alpha > 0) & (alpha < 190)).astype(np.float32)
        edge_blend = edge * (1.0 - alpha / 255.0) * edge_blend_strength
        rgba[:, :, :3] = rgba[:, :, :3] * (1.0 - edge_blend[:, :, None])
        rgba[:, :, :3] += plate * edge_blend[:, :, None]

    return Image.fromarray(rgba.clip(0, 255).astype(np.uint8), mode="RGBA")


def refine_cutout_for_plate(
    cutout_rgba: Image.Image,
    plate_rgb: Image.Image,
    xy: Tuple[int, int],
    settings: PipelineSettings,
) -> Image.Image:
    alpha = clean_alpha(cutout_rgba.getchannel("A"), settings)
    refined = apply_alpha(cutout_rgba, alpha)
    refined = _decontaminate_edge_rgb(refined, settings)
    refined = _neutralize_glass_contamination(refined, plate_rgb, xy, settings)
    refined = _harmonize_color(refined, plate_rgb, xy, settings)
    return refined


def contact_shadow_mask(
    cutout_alpha: Image.Image,
    canvas_size: Tuple[int, int],
    xy: Tuple[int, int],
    settings: PipelineSettings,
) -> Image.Image:
    """Build a soft shadow from the lower silhouette instead of a fixed ellipse."""
    cfg = settings.rembg
    cw, ch = cutout_alpha.size
    canvas_w, canvas_h = int(canvas_size[0]), int(canvas_size[1])
    x, y = int(xy[0]), int(xy[1])

    alpha = np.array(cutout_alpha.convert("L"), dtype=np.float32) / 255.0
    bottom_start = min(ch - 1, max(0, int(ch * 0.54)))
    bottom = alpha[bottom_start:, :]
    density = np.percentile(bottom, 88, axis=0) if bottom.size else np.zeros((cw,), dtype=np.float32)
    if float(np.max(density)) <= 1e-4:
        density = np.ones((cw,), dtype=np.float32)
    else:
        density = density / max(float(np.max(density)), 1e-4)

    blur_x = max(5, int(cw * 0.035))
    density = cv2.GaussianBlur(density.reshape(1, -1), (0, 0), sigmaX=blur_x).reshape(-1)
    density = density / max(float(np.max(density)), 1e-4)

    broad = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    contact = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    center_y = y + ch * (1.0 - float(cfg.shadow_offset_frac))
    sigma_y = max(2.0, ch * float(cfg.shadow_height_frac) * 0.34)
    y0 = max(0, int(center_y - sigma_y * 3.0))
    y1 = min(canvas_h, int(center_y + sigma_y * 3.0) + 1)
    x0 = max(0, x)
    x1 = min(canvas_w, x + cw)
    if y1 > y0 and x1 > x0:
        ys = np.arange(y0, y1, dtype=np.float32)
        gy = np.exp(-0.5 * ((ys - center_y) / sigma_y) ** 2)
        local_density = density[(x0 - x):(x1 - x)]
        footprint = gy[:, None] * local_density[None, :] * 118.0
        broad[y0:y1, x0:x1] = np.maximum(broad[y0:y1, x0:x1], footprint)

    axes = (
        max(4, int(cw * float(getattr(cfg, "shadow_width_frac", 0.84)) * 0.5)),
        max(3, int(ch * float(cfg.shadow_height_frac) * 0.42)),
    )
    center = (int(x + cw * 0.5), int(center_y + ch * 0.01))
    cv2.ellipse(broad, center, axes, 0, 0, 360, 110, -1)

    blur = max(0.0, float(cfg.shadow_blur_px))
    broad_img = Image.fromarray(broad.clip(0, 255).astype(np.uint8), mode="L")
    if blur > 0:
        broad_img = broad_img.filter(ImageFilter.GaussianBlur(blur))

    support = alpha > 0.18
    if int(np.count_nonzero(support)) > 0:
        ys, xs = np.nonzero(support)
        foot_y = int(np.max(ys))
        bottom_y = np.full((cw,), -1, dtype=np.int32)
        for col in np.unique(xs):
            bottom_y[int(col)] = int(np.max(ys[xs == col]))
        contact_band = max(3, int(ch * 0.055))
        contact_density = np.where(
            bottom_y >= foot_y - contact_band,
            0.45 + density * 0.55,
            0.0,
        ).astype(np.float32)
        if float(np.max(contact_density)) > 1e-4:
            contact_density = cv2.GaussianBlur(
                contact_density.reshape(1, -1),
                (0, 0),
                sigmaX=max(2.0, cw * 0.014),
            ).reshape(-1)
            contact_density = contact_density / max(float(np.max(contact_density)), 1e-4)
            contact_center_y = y + foot_y + max(1.0, ch * 0.012)
            contact_sigma_y = max(1.5, ch * 0.014)
            cy0 = max(0, int(contact_center_y - contact_sigma_y * 4.0))
            cy1 = min(canvas_h, int(contact_center_y + contact_sigma_y * 4.0) + 1)
            if cy1 > cy0 and x1 > x0:
                cys = np.arange(cy0, cy1, dtype=np.float32)
                gy = np.exp(-0.5 * ((cys - contact_center_y) / contact_sigma_y) ** 2)
                local_contact = contact_density[(x0 - x):(x1 - x)]
                contact[cy0:cy1, x0:x1] = np.maximum(
                    contact[cy0:cy1, x0:x1],
                    gy[:, None] * local_contact[None, :] * 255.0,
                )

    contact_img = Image.fromarray(contact.clip(0, 255).astype(np.uint8), mode="L")
    contact_blur = max(1.0, min(blur * 0.28, ch * 0.018))
    contact_img = contact_img.filter(ImageFilter.GaussianBlur(contact_blur))
    return ImageChops.lighter(broad_img, contact_img)


def apply_contact_shadow(
    plate_rgb: Image.Image,
    shadow_mask: Image.Image,
    settings: PipelineSettings,
) -> Image.Image:
    cfg = settings.rembg
    opacity = max(0.0, min(1.0, float(getattr(cfg, "contact_shadow_opacity", 0.34))))
    plate = np.array(plate_rgb.convert("RGB"), dtype=np.float32)
    mask = np.array(shadow_mask.convert("L"), dtype=np.float32) / 255.0
    plate *= (1.0 - mask[:, :, None] * opacity)
    return Image.fromarray(plate.clip(0, 255).astype(np.uint8), mode="RGB")


def composite_cutout_on_plate(
    cutout_rgba: Image.Image,
    plate_rgb: Image.Image,
    xy: Tuple[int, int],
    settings: PipelineSettings,
) -> RefinedComposite:
    refined = refine_cutout_for_plate(cutout_rgba, plate_rgb, xy, settings)
    alpha = refined.getchannel("A")
    shadow = contact_shadow_mask(alpha, plate_rgb.size, xy, settings)
    base = apply_contact_shadow(plate_rgb, shadow, settings).convert("RGBA")
    base.alpha_composite(refined, xy)
    stats = {
        "shadow_pixels": int(np.count_nonzero(np.array(shadow, dtype=np.uint8))),
        "alpha_pixels": int(np.count_nonzero(np.array(alpha, dtype=np.uint8))),
    }
    return RefinedComposite(
        image=base.convert("RGB"),
        cutout=refined,
        alpha=alpha,
        shadow_mask=shadow,
        stats=stats,
    )
