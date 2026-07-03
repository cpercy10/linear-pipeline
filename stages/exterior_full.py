"""exterior_full lane — the fused 3-module heavy path (ADAPTED from Module 3).

Module 3's exterior_full was YOLO → neutral canvas → orientation → fill-ratio canvas
→ FLUX(background). The unified pipeline FUSES in Modules 1 (perspective) and 2 (plate
render): the car is sized by area-based visual occupancy, a per-car studio plate is
rendered at the photo's perspective, and the car is anchored onto a gray canvas at the
plate's turntable centre before FLUX composites it against the rendered plate.

Per-image flow (BLUEPRINT §3):

  preprocess()    GPU-light (preprocess pool):
    1  YOLO largest bbox (shared crop)              → models.yolo_detector
    2a orientation class on the neutral canvas      → models.timm_classifier
    2b+2c+3 perspective fuser (retrieval+geocalib+gate+azimuth)
                                                     → perspective.estimate.estimate
    4  occupancy resize of the RGB crop             → stages.occupancy_resize
    5  disc diameter on the RESIZED car             → perspective.footprint

  (plate render)  Blender pool (single warm worker):
    6  render the per-car studio plate (camera + disc) at INPUT dims.
       This is driven by ``pipeline.Pipeline._render_plate`` (NOT a function here):
       it maps the fuser's flat camera dict to Module-2 export keys, calls
       ``render.blender_worker.BlenderWorker.render(...)``, reads the written JPG,
       and normalises it via ``_coerce_plate_result`` below.

  composite_on_plate()   remove.bg cutout + manual composite (no diffusion):
    7  remove.bg the YOLO crop → RGBA cutout, occupancy-resized (same sizing)
    8  alpha-paste the cutout onto the plate at cx,cy → RGB at INPUT dims
                                                     → stages.anchor

The stage exposes two plain callables — ``preprocess`` (tier 1) and
``composite_on_plate`` (tier 3) — plus the ``_coerce_plate_result`` helper. Tier 2
(the plate render) lives
in ``pipeline.Pipeline._render_plate``, which owns the SINGLE coupling point to the
warm worker so the key-translation and the worker's real signature live in one place.

>>> Blender-worker contract (host-side ``render.blender_worker.BlenderWorker``). The
actual call lives in ``pipeline.Pipeline._render_plate``; its real surface is::

    worker.render(
        camera: dict,          # Module-2 export keys {azimuth_deg, elevation_deg,
                               #   distance_m, cam_height_m, focal_mm, roll_deg} —
                               #   translated from the fuser's flat dict by
                               #   pipeline._camera_export_payload
        disc_diam: float,      # footprint.disc_diameter_m on the RESIZED car
        out_jpg: str,          # the worker writes plate.jpg here
        out_json: str,         # …and meta.json here
        *, photo_w: int, photo_h: int,   # INPUT dims (plate renders at input long edge)
        long_edge: int = ...,  # max(photo_w, photo_h) → plate at INPUT dimensions
        samples: int = ...,
    ) -> dict                  # the Module-2 meta (car_spot_px, pixels_per_metre, …)

``pipeline._render_plate`` loads the written JPG and hands the ``(plate, meta)`` pair
to ``_coerce_plate_result`` (below), which recovers ``cx,cy`` + the lateral ppm into a
:class:`PlateRenderResult`. The coercer is written defensively (accepts a
PlateRender-like object OR a ``(plate, meta)`` tuple OR a mapping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import structlog
from PIL import Image

from config.pipeline_config import (
    Orientation,
    PipelineSettings,
    VisualOccupancyRegistry,
)
from models.dino_retriever import Retriever
from models.timm_classifier import TimmClassifier
from models.yolo_detector import YoloDetector
from perspective import anchor_geometry
from perspective import estimate as perspective_estimate
from perspective import footprint
from processing import image_processor as ip
from processing.exceptions import BlenderRenderError, DetectionError
from stages import anchor as anchor_stage
from stages import occupancy_resize as resize_stage
from utils.logging import get_logger, stage_timer


# ─────────────────────────────────────────────────────────────────────────────
# Phase result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """GPU-light preprocess output — everything the render + composite tail needs."""
    resized_crop: Image.Image                       # occupancy-resized RGB bbox crop
    orientation: Orientation
    orientation_confidence: float
    camera: Dict[str, float]                         # fuser flat dict (drives plate + footprint)
    disc_m: float                                    # disc sized to the RESIZED car
    orig_size: Tuple[int, int]                       # INPUT (w, h)
    estimate: "perspective_estimate.CameraEstimate"  # full fused estimate (confidence, azimuth info)
    pose: str                                        # footprint pose_class of the corrected azimuth
    recovered_length_m: float
    input_ppm: float                                 # input image pixels/metre (for plate↔input scale)
    anchor_frac: Tuple[float, float] = (0.5, 0.5)    # (fx,fy) of the car's ground-contact centre in the resized crop
    resize_info: Dict[str, object] = field(default_factory=dict)
    raw_crop: Optional[Image.Image] = None           # RGB bbox crop BEFORE occupancy resize (debug)


@dataclass
class PlateRenderResult:
    """Blender-pool output — the rendered per-car studio plate + its anchor point."""
    plate: Image.Image                               # rendered plate at INPUT dims
    cx: float
    cy: float
    plate_ppm: float                                 # plate pixels/metre (lateral)
    plate_ppm_depth: float = 0.0                     # plate pixels/metre (toward camera) — drives the forward bias
    meta: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — GPU-light preprocess (preprocess pool)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    image: Image.Image,
    full_image_path: Union[str, Path],
    *,
    yolo: YoloDetector,
    orientation_model: TimmClassifier,
    retriever: Retriever,
    settings: PipelineSettings,
    registry: VisualOccupancyRegistry,
    log: Optional[structlog.stdlib.BoundLogger] = None,
) -> PreprocessResult:
    """YOLO → orientation → perspective fuse → occupancy resize → disc size.

    ``image`` is the FULL input (RGB). ``full_image_path`` is its path on disk —
    GeoCalib (inside the fuser) loads the full frame itself for the horizon. The YOLO
    crop is computed ONCE and shared by orientation, retrieval, occupancy resize, and
    the disc/footprint maths.
    """
    log = log or get_logger("stages.exterior_full")
    orig_size = image.size                          # (w, h) — INPUT dimensions
    fill = settings.canvas.fill_color

    # 1 — YOLO largest bbox (shared crop).
    with stage_timer("yolo", device=yolo.device, gpu=True, log=log):
        bbox = yolo.detect_largest(image)
    if bbox is None:
        raise DetectionError("no vehicle detected in exterior_full image")
    bbox_w = bbox[2] - bbox[0]

    # The RGB bbox crop — bbox-only, no segmentation (FLUX cuts the car out implicitly).
    car_crop = ip.crop_with_padding(image, bbox, settings.yolo.crop_padding)

    # 2a — orientation class on the NEUTRAL training canvas (M3 transform, unchanged).
    with stage_timer("orientation_canvas", log=log):
        neutral_canvas = ip.build_orientation_canvas(car_crop, orig_size, fill)
    with stage_timer("orientation", device=orientation_model.device, gpu=True, log=log) as extra:
        pred = orientation_model.predict_one(neutral_canvas)
        extra["orientation"] = pred.label
        extra["confidence"] = round(pred.confidence, 4)
    orientation = Orientation(pred.label)

    # 2b/2c/3 — perspective fuser: retrieval + GeoCalib + gate + azimuth correction.
    #   Runs on the SHARED crop (retrieval do_crop=False). Each sub-stage is timed
    #   inside the fuser (retrieval / geocalib). Advisory confidence is logged, not
    #   branched (locked invariant).
    cam_est = perspective_estimate.estimate(
        crop=car_crop,
        full_image_path=full_image_path,
        orientation=orientation,
        retriever=retriever,
    )
    camera = cam_est.camera                          # flat dict {azimuth, elevation, ...}

    # 4 — occupancy resize of the RGB crop onto the INPUT-sized canvas.
    rz = resize_stage.resize_car_by_footprint(
        car_crop, orientation, orig_size, settings=settings, registry=registry
    )

    # 5 — disc diameter, sized to the RAW YOLO bbox (the car's true projected footprint —
    #     the same detection DINOv2 keys on), NOT the occupancy-resized car (which is only
    #     a framing size for the FLUX canvas). footprint expects azimuth in its front=180
    #     convention — the corrected azimuth from the fuser is already in that convention.
    img_w_px = orig_size[0]
    disc_m, recovered_L, input_ppm, pose = footprint.disc_diameter_m(
        bbox_w,                                      # RAW tight YOLO bbox width
        camera["azimuth"],
        camera["focal"],
        camera["distance"],
        img_w_px,
    )

    # Ground-contact anchor (S0): where the car's WHEEL-FOOTPRINT CENTRE sits inside
    # the crop, so the composite tail can pin THAT (not the bbox centre) onto the plate's
    # turntable centre. Re-projects a stand-in car box through the recovered camera
    # (perspective/anchor_geometry) — pose-only, independent of the plate render.
    acfg = settings.anchor
    fx, fy = anchor_geometry.ground_contact_frac(
        azimuth_deg=camera["azimuth"],
        elevation_deg=camera.get("elevation"),
        distance_m=camera["distance"],
        cam_height_m=camera.get("cam_height", 1.35),
        focal_mm=camera["focal"],
        roll_deg=camera.get("roll", 0.0),
        res_x=orig_size[0],
        res_y=orig_size[1],
        box_lwh=tuple(acfg.car_box_lwh),
        sensor_mm=acfg.sensor_mm,
    )
    # fx,fy are fractions of the TIGHT bbox; remap into the PADDED crop the car is
    # placed from (occupancy resize preserves the fraction under uniform scale).
    cl, ct, cr, cb = ip.crop_box_with_padding(orig_size, bbox, settings.yolo.crop_padding)
    fp_x = bbox[0] + fx * (bbox[2] - bbox[0])
    fp_y = bbox[1] + fy * (bbox[3] - bbox[1])
    anchor_frac = (
        min(1.0, max(0.0, (fp_x - cl) / max(1, cr - cl))),
        min(1.0, max(0.0, (fp_y - ct) / max(1, cb - ct))),
    )

    # ── Step-1 fit DIAGNOSTIC (no behavior change) — is the car bigger than its disc? ──
    # disc depth: project the disc rims at the plate camera. The worker zooms 3/4 plates
    # out ×1.08 (worker_entry.THREEQ_ZOOMOUT), so match that here for an honest disc size.
    disc_dist = camera["distance"] * (1.08 if pose == "threeq" else 1.0)
    disc_depth_px = anchor_geometry.project_disc_depth_px(
        elevation_deg=camera.get("elevation"), distance_m=disc_dist,
        cam_height_m=camera.get("cam_height", 1.35), focal_mm=camera["focal"],
        roll_deg=camera.get("roll", 0.0), res_x=orig_size[0], res_y=orig_size[1],
        disc_m=disc_m, sensor_mm=acfg.sensor_mm,
    )
    # footprint depth in resized-crop px = depth-fraction × tight-bbox height × occupancy scale
    foot_frac_depth = anchor_geometry.footprint_depth_frac(
        azimuth_deg=camera["azimuth"], elevation_deg=camera.get("elevation"),
        distance_m=camera["distance"], cam_height_m=camera.get("cam_height", 1.35),
        focal_mm=camera["focal"], roll_deg=camera.get("roll", 0.0),
        res_x=orig_size[0], res_y=orig_size[1],
        box_lwh=tuple(acfg.car_box_lwh), sensor_mm=acfg.sensor_mm,
    )
    occ_scale = rz.height / max(1, (cb - ct))
    footprint_depth_px = foot_frac_depth * (bbox[3] - bbox[1]) * occ_scale
    slack_px = disc_depth_px - footprint_depth_px

    log.info(
        "exterior_full.preprocess",
        orientation=orientation.value,
        orientation_confidence=round(pred.confidence, 4),
        azimuth=camera["azimuth"],
        azimuth_snapped=cam_est.azimuth_snapped,
        confidence=cam_est.confidence,
        disc_m=round(disc_m, 3),
        pose=pose,
        recovered_length_m=round(recovered_L, 2),
        occupancy_realised=round(rz.occupancy_realised, 4),
        resized_px=[rz.width, rz.height],
        anchor_frac=[round(anchor_frac[0], 4), round(anchor_frac[1], 4)],
        disc_depth_px=round(disc_depth_px, 1),
        footprint_depth_px=round(footprint_depth_px, 1),
        slack_px=round(slack_px, 1),
        fits=bool(slack_px > 0),
    )

    return PreprocessResult(
        resized_crop=rz.crop,
        orientation=orientation,
        orientation_confidence=pred.confidence,
        camera=camera,
        disc_m=disc_m,
        orig_size=orig_size,
        estimate=cam_est,
        pose=pose,
        recovered_length_m=recovered_L,
        input_ppm=input_ppm,
        anchor_frac=anchor_frac,
        resize_info=rz.info,
        raw_crop=car_crop,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — plate render (Blender pool / warm worker)
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_plate_result(raw: Any, log: structlog.stdlib.BoundLogger) -> PlateRenderResult:
    """Normalise whatever the warm worker returns into a PlateRenderResult.

    Accepts (in priority order):
      * an object exposing ``.plate`` + ``.cx`` + ``.cy`` (+ optional ``.pixels_per_metre``
        / ``.plate_ppm`` / ``.meta``);
      * a ``(plate, meta)`` tuple where ``meta`` carries ``car_spot_px`` +
        ``pixels_per_metre.lateral`` (the Module-2 ``render_master.export_metadata``
        schema);
      * a mapping with ``plate`` + ``meta`` / ``cx`` / ``cy`` keys.
    """
    plate: Optional[Image.Image] = None
    meta: Dict[str, Any] = {}
    cx = cy = None
    plate_ppm = 0.0

    if hasattr(raw, "plate"):
        plate = getattr(raw, "plate")
        meta = dict(getattr(raw, "meta", {}) or {})
        cx = getattr(raw, "cx", None)
        cy = getattr(raw, "cy", None)
        plate_ppm = float(
            getattr(raw, "plate_ppm", None)
            or getattr(raw, "pixels_per_metre", 0.0)
            or 0.0
        )
    elif isinstance(raw, (tuple, list)) and len(raw) == 2:
        plate, meta = raw[0], dict(raw[1] or {})
    elif isinstance(raw, dict):
        plate = raw.get("plate")
        meta = dict(raw.get("meta", raw) or {})
        cx, cy = raw.get("cx"), raw.get("cy")

    if plate is None:
        raise BlenderRenderError("blender worker returned no plate image")

    # Recover cx/cy + plate ppm from the Module-2 meta schema when not given directly.
    if (cx is None or cy is None) and isinstance(meta, dict):
        spot = meta.get("car_spot_px")
        if spot and len(spot) >= 2:
            cx, cy = float(spot[0]), float(spot[1])
    plate_ppm_depth = 0.0
    if isinstance(meta, dict):
        ppm = meta.get("pixels_per_metre")
        if isinstance(ppm, dict):
            if not plate_ppm:
                plate_ppm = float(ppm.get("lateral", 0.0) or 0.0)
            plate_ppm_depth = float(ppm.get("toward_camera", 0.0) or 0.0)
        elif ppm and not plate_ppm:
            plate_ppm = float(ppm)

    if cx is None or cy is None:
        raise BlenderRenderError(
            "blender plate meta is missing car_spot_px (cx,cy) — cannot anchor the car"
        )

    return PlateRenderResult(
        plate=plate.convert("RGB") if isinstance(plate, Image.Image) else plate,
        cx=float(cx),
        cy=float(cy),
        plate_ppm=float(plate_ppm),
        plate_ppm_depth=float(plate_ppm_depth),
        meta=meta,
    )


# NOTE: the plate-render invocation is NOT a stage function here. ``pipeline.py``
# owns the ONE render path (``Pipeline._render_plate``): it translates the fuser's
# flat camera dict into Module-2's ``*_deg/*_m`` export keys via
# ``_camera_export_payload``, calls the warm worker's real surface
# ``BlenderWorker.render(camera, disc_diam, out_jpg, out_json, *, photo_w, photo_h, …)``,
# loads the written ``plate.jpg`` back, and feeds the ``(plate, meta)`` pair to
# ``_coerce_plate_result`` above. An earlier ``render_plate`` / ``_render_with_worker``
# pair here drove a DIVERGENT, non-existent worker surface (a ``render_plate`` method,
# a ``disc`` kwarg, no ``out_jpg``/``out_json``) that never matched
# ``BlenderWorker.render`` — it was dead and a trap for future callers, so it was
# removed in favour of the single wired path in ``pipeline.py``.




# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — remove.bg cutout + manual composite (no diffusion)
# ─────────────────────────────────────────────────────────────────────────────

def composite_on_plate(
    pre: PreprocessResult,
    plate: PlateRenderResult,
    cutout_rgba: Image.Image,
    *,
    settings: PipelineSettings,
    log: Optional[structlog.stdlib.BoundLogger] = None,
) -> Image.Image:
    """Manual composite: alpha-paste the remove.bg car cutout onto the rendered plate.

    Reproduces the EXACT placement the old FLUX path used, then pastes the cutout
    straight onto the plate instead of feeding a gray canvas + plate to FLUX:

      * SIZE  — resize the RGBA cutout to the occupancy-resized dims (occupancy resize
                is still the ONLY sizing step; ``pre.resized_crop.size`` is its output).
      * WHERE — pin the car's ground-contact centre (``pre.anchor_frac``) onto the plate
                turntable centre ``cx,cy``, slid TOWARD THE CAMERA by the per-pose
                forward bias (``anchor.forward_bias_*``, a fraction of the disc radius),
                then clamp fully on-plate (see ``stages.anchor``).

    Returns an RGB image at INPUT dimensions (the plate already renders at input dims).
    """
    log = log or get_logger("stages.exterior_full")
    acfg = settings.anchor

    # Forward bias: shift the target toward the camera (DOWN in the image = +y) by a
    # fraction of the disc RADIUS, via the plate's toward-camera pixels/metre (fall back
    # to lateral, then no shift). Per pose group (pre.pose, the azimuth-derived bucket).
    bias_frac = {
        "frontrear": acfg.forward_bias_frontrear,
        "threeq":    acfg.forward_bias_threeq,
        "side":      acfg.forward_bias_side,
    }.get(pre.pose, acfg.forward_bias_threeq)
    radius_m = max(0.0, float(pre.disc_m) / 2.0)
    ppm_depth = plate.plate_ppm_depth or plate.plate_ppm or 0.0
    bias_px = float(bias_frac) * radius_m * float(ppm_depth)
    target = (plate.cx, plate.cy + bias_px)

    # Resize the cutout to the SAME dims occupancy resize produced for the RGB crop, so
    # the placement fractions (pre.anchor_frac) stay valid and sizing is unchanged.
    cutout_resized = cutout_rgba.convert("RGBA").resize(pre.resized_crop.size, Image.LANCZOS)

    log.info(
        "exterior_full.composite",
        pose=pre.pose,
        forward_bias=round(float(bias_frac), 4),
        bias_px=round(bias_px, 1),
        target_px=[round(float(target[0]), 1), round(float(target[1]), 1)],
        placed_px=list(cutout_resized.size),
    )

    return anchor_stage.composite_car_on_plate(
        cutout_resized,
        plate.plate,
        target,
        anchor_frac=(0.5, pre.anchor_frac[1]),   # center X; wheel offset on Y only
        log=log,
    )
