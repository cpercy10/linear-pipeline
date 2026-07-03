"""Area-based visual-occupancy resize of the RGB bbox crop (UNIFORM, no distortion).

The car's bbox crop is scaled so it covers a target FRACTION of the canvas AREA
(``TARGET_VISUAL_OCCUPANCY`` per class — see ``config/orientation.yaml`` +
``VisualOccupancyRegistry``). FLUX cuts the car out implicitly downstream, so this
operates on the plain RGB bbox crop (NO segmentation).

Locked rules (agreed):
  * **Uniform scale only** — width and height are scaled by the SAME factor, so the
    car's aspect ratio is NEVER changed (no stretch/squish). (This replaces the old
    per-axis "aspect nudge", which distorted the car.)
  * **Target-sized, fit-capped:** scale toward the per-class occupancy target (UP or
    DOWN), but never beyond the largest uniform scale that still fits the car inside the
    canvas when CENTERED (``min(W/car_w, H/car_h)``). So the resized car ALWAYS fits the
    canvas — the anchor can then place its centre on ``cx,cy`` and clamp it fully
    in-bounds without ever clipping.
  * **Bidirectional to target** — scale UP toward the target for small cars and DOWN
    for cars that over-fill the frame, so every car lands at the per-class occupancy.
    (Was previously upscale-only, which left frame-filling cars too big to place.)
  * **No positioning here** — placement (centre on the plate's ``cx,cy``) happens in
    ``stages/anchor`` after the render. This step only decides the car's SIZE.

The function returns the resized RGB crop plus a record of the realised occupancy +
scale, for logging / A-B checks. (NOTE: the disc/footprint math is fed the RAW bbox,
not this resized size — they are intentionally decoupled.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

from PIL import Image

from config.pipeline_config import (
    Orientation,
    PipelineSettings,
    VisualOccupancyRegistry,
    get_orientation_registry,
    get_settings,
)
from utils.logging import get_logger, stage_timer

log = get_logger("stages.occupancy_resize")


@dataclass
class OccupancyResizeResult:
    """The occupancy-resized crop + the record needed downstream.

    ``crop`` is the resized RGB image (uniform scale, aspect preserved, guaranteed to
    fit the canvas). ``width``/``height`` are its new pixel dims. ``scale_w``/``scale_h``
    are reported for verification — they are equal (modulo integer rounding), which is
    the proof that the car was NOT distorted.
    """
    crop: Image.Image
    width: int
    height: int
    occupancy_target: float
    occupancy_realised: float
    scale_w: float
    scale_h: float
    info: Dict[str, object] = field(default_factory=dict)


def resize_car_by_footprint(
    rgb_crop: Image.Image,
    orientation: Union[Orientation, str, None],
    canvas_size: Tuple[int, int],
    *,
    settings: Optional[PipelineSettings] = None,
    registry: Optional[VisualOccupancyRegistry] = None,
) -> OccupancyResizeResult:
    """Uniformly resize ``rgb_crop`` toward its per-class visual-occupancy target.

    Parameters
    ----------
    rgb_crop:
        The RAW YOLO bbox crop (RGB). Resized in place of any segmentation — FLUX does
        the implicit cutout downstream.
    orientation:
        The 8-way orientation label (``Orientation`` / its string value / ``"partial"``
        / ``None``). Selects the ``TARGET_VISUAL_OCCUPANCY`` entry; unknown → ``default``.
    canvas_size:
        ``(width, height)`` of the gray canvas the car will be anchored on — the INPUT
        image dimensions.

    Returns
    -------
    OccupancyResizeResult
        The resized crop (uniform scale, guaranteed to fit the canvas) + new dims +
        realised occupancy + scale record.
    """
    settings = settings or get_settings()
    registry = registry or get_orientation_registry()

    canvas_w, canvas_h = int(canvas_size[0]), int(canvas_size[1])
    car_w, car_h = rgb_crop.size

    target = float(registry.get(orientation))
    canvas_area = max(1, canvas_w * canvas_h)
    current_occ = (car_w * car_h) / canvas_area

    with stage_timer("resize", log=log) as extra:
        # UNIFORM scale toward the target area — UP for small cars, DOWN for cars that
        # over-fill the frame — capped by the largest uniform scale that still fits the
        # centered car inside the canvas.
        s_target = (target / current_occ) ** 0.5         # >1 upscales, <1 downscales to target
        s_fit = min(canvas_w / max(1, car_w),            # max uniform scale that fits (centered)
                    canvas_h / max(1, car_h))
        scale = min(s_target, s_fit)
        fit_capped = s_fit < s_target

        new_w = min(canvas_w, max(1, int(round(car_w * scale))))
        new_h = min(canvas_h, max(1, int(round(car_h * scale))))

        if (new_w, new_h) == (car_w, car_h):
            resized = rgb_crop
        else:
            resized = rgb_crop.resize((new_w, new_h), Image.LANCZOS)

        realised = (new_w * new_h) / canvas_area

        extra["orientation"] = (
            orientation.value if isinstance(orientation, Orientation) else orientation
        )
        extra["occupancy_target"] = round(target, 4)
        extra["occupancy_realised"] = round(realised, 4)
        extra["scale_w"] = round(new_w / max(1, car_w), 4)   # == scale_h (uniform) → no distortion
        extra["scale_h"] = round(new_h / max(1, car_h), 4)
        extra["fit_capped"] = fit_capped
        extra["native_px"] = [car_w, car_h]
        extra["resized_px"] = [new_w, new_h]

    return OccupancyResizeResult(
        crop=resized,
        width=new_w,
        height=new_h,
        occupancy_target=target,
        occupancy_realised=realised,
        scale_w=new_w / max(1, car_w),
        scale_h=new_h / max(1, car_h),
        info={
            "fit_capped": fit_capped,
            "native_px": [car_w, car_h],
            "resized_px": [new_w, new_h],
            "canvas_px": [canvas_w, canvas_h],
        },
    )
