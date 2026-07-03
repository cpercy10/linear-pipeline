"""Composite the car cutout onto the rendered plate by its GROUND-CONTACT point (S0).

The car is an RGBA cutout (remove.bg). It is pinned by its WHEEL-FOOTPRINT CENTRE —
supplied as ``anchor_frac=(fx, fy)``, the fractional position of that point inside the
crop — onto the target ``car_spot_px`` (the plate's turntable centre, optionally slid
toward the camera by the caller's forward bias). The top-left is CLAMPED so the car
stays fully inside the plate, then the cutout is alpha-composited onto the plate.

Placement (S0 — ground-contact anchoring, see ``perspective/anchor_geometry.py``):
``cx,cy`` is a point on the FLOOR (the projection of the turntable centre), whereas the
bbox centre is ~half-way up the car. Pinning by the wheel-footprint centre makes the
wheels land consistently for every car (tall SUV vs low coupe), instead of riding the
top/bottom edge of the disc.

Because ``stages/occupancy_resize`` guarantees the resized car fits inside the canvas,
the clamp ALWAYS yields a fully in-bounds placement → no clipping, ever.

NOTE: the placement math (``compute_paste_top_left``) is the SAME computation the
pipeline has always used to anchor the car; the only change from the old FLUX path is
that we alpha-paste the cutout onto the plate here instead of onto a gray canvas fed to
FLUX.
"""

from __future__ import annotations

from typing import Optional, Tuple

import structlog
from PIL import Image

from utils.logging import get_logger, stage_timer

log = get_logger("stages.anchor")


def compute_paste_top_left(
    target: Tuple[float, float],
    crop_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    anchor_frac: Tuple[float, float] = (0.5, 0.5),
) -> Tuple[int, int, bool]:
    """Top-left ``(x, y)`` so the crop's ``anchor_frac`` point lands on ``target``,
    CLAMPED so the whole crop stays inside the canvas.

    Returns ``(x, y, clamped)`` where ``clamped`` is True if the ideal position had to
    be pulled back on-canvas.
    """
    cx, cy = float(target[0]), float(target[1])
    cw, ch = int(crop_size[0]), int(crop_size[1])
    canvas_w, canvas_h = int(canvas_size[0]), int(canvas_size[1])
    fx, fy = float(anchor_frac[0]), float(anchor_frac[1])

    x_ideal = int(round(cx - fx * cw))
    y_ideal = int(round(cy - fy * ch))
    x = max(0, min(canvas_w - cw, x_ideal))
    y = max(0, min(canvas_h - ch, y_ideal))
    clamped = (x != x_ideal) or (y != y_ideal)
    return x, y, clamped


def composite_car_on_plate(
    cutout_rgba: Image.Image,
    plate_rgb: Image.Image,
    car_spot_px: Tuple[float, float],
    *,
    anchor_frac: Tuple[float, float] = (0.5, 0.5),
    log: Optional[structlog.stdlib.BoundLogger] = None,
) -> Image.Image:
    """Alpha-composite the RGBA car cutout onto the plate so the car's ground-contact
    point (``anchor_frac`` within the cutout) lands on ``car_spot_px`` — CLAMPED so the
    car stays fully on-plate.

    Parameters
    ----------
    cutout_rgba:
        The remove.bg car cutout (RGBA), already occupancy-resized to its placed size.
    plate_rgb:
        The rendered studio plate (RGB) at INPUT dimensions — the final background.
    car_spot_px:
        ``(cx, cy)`` target in plate/canvas pixels — the plate's turntable centre,
        optionally shifted toward the camera by the caller's forward bias.
    anchor_frac:
        ``(fx, fy)`` in ``[0, 1]`` — where the car's ground-contact centre sits inside
        the cutout (from ``perspective.anchor_geometry.ground_contact_frac``).

    Returns an RGB image at the plate's dimensions (== input dimensions).
    """
    _log = log if log is not None else get_logger("stages.anchor")
    cutout = cutout_rgba.convert("RGBA")
    cw, ch = cutout.size
    canvas_w, canvas_h = plate_rgb.size

    with stage_timer("composite", log=_log) as extra:
        x, y, clamped = compute_paste_top_left(
            car_spot_px, (cw, ch), (canvas_w, canvas_h), anchor_frac
        )
        base = plate_rgb.convert("RGBA")
        base.alpha_composite(cutout, (x, y))
        result = base.convert("RGB")

        extra["target_px"] = [round(float(car_spot_px[0]), 1), round(float(car_spot_px[1]), 1)]
        extra["anchor_frac"] = [round(float(anchor_frac[0]), 4), round(float(anchor_frac[1]), 4)]
        extra["paste_top_left"] = [x, y]
        extra["placed_px"] = [cw, ch]
        extra["canvas_px"] = [canvas_w, canvas_h]
        extra["clamped"] = clamped

    return result
