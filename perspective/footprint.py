"""Tier-1 footprint -> turntable-disc size.

Recovers the car's REAL length from the YOLO bounding box, the predicted azimuth, and the camera
(focal + distance), then sizes the disc a touch bigger than that footprint. Pose-invariant: the
azimuth term un-rotates the projected width, so the same car yields the same disc at any angle.

    w   = bbox_width_px / pixels_per_metre              # un-zoom: projected footprint in metres
    L   = w / (|sinθ| + |cosθ| / CAR_LW_RATIO)          # un-rotate: real car length  (θ = azimuth-180)
    disc = clamp(L * DISC_MARGIN, DISC_MIN, DISC_MAX)
    pixels_per_metre = focal_mm * img_w_px / (distance_m * SENSOR_MM)

Copied VERBATIM from Module 2 ``plate-rendering/footprint.py``. The disc math is
unchanged; in the unified pipeline it is simply fed the RESIZED car's dimensions
(see ``stages/occupancy_resize`` + ``perspective/estimate``) so the ring fits the
car as it will actually be composited.
"""
import math

SENSOR_MM    = 36.0    # full-frame sensor width assumed by the pipeline
CAR_LW_RATIO = 2.4     # typical car length : width
DISC_MARGIN  = 1.25    # disc a bit bigger than the footprint (sample-like ring around the car)
DISC_MIN     = 3.0
DISC_MAX     = 8.0

# ── pose-aware, overflow-safe calibration ────────────────────────────────────────────────────
# The single-image size recovery is noisy: 3/4 shots UNDER-read the footprint (the rotated car
# projects narrower than it is) → disc too small → the real car overflows after compositing. That is
# the unacceptable direction, so 3/4 gets a bigger ring AND a hard floor; front/rear can't observe
# length at all (it foreshortens into depth) so they use a safe default. Erring large is fine — a
# generous ring looks like the samples; a small disc that the car spills off does not.
MARGIN_3Q   = 1.50     # 3/4 ring multiplier (raised) — bigger ring so the car reads "placed", not edge-to-edge
MARGIN_SIDE = 1.25     # side measures length directly — modest ring
# A round disc must contain the footprint DIAGONAL (~L*1.084), not just L. The biggest common dealer
# vehicle (~5.3 m SUV / small van) has a ~5.7-6.0 m diagonal, so the floor is set above that with a
# ring to match the samples. 3/4 shots under-read the most (and read as overflowed) so they get their
# OWN, higher floor; big cars scale up; the noisy recovery can NEVER shrink it into overflow.
FLOOR_M     = 6.5      # side / front-rear floor
FLOOR_3Q    = 6.5      # 3/4-ONLY floor — bigger than 6.5 (looked small) but below 7.5 (felt too big); tunable
CEIL_M      = 9.0      # raised so genuinely large 3/4 cars can scale past the floor
DEFAULT_L_M = 5.0      # assumed length when it isn't observable (front / rear)
BAND_DEG    = 25.0     # +/- band around each pose centre


def pose_class(azimuth_deg):
    az = float(azimuth_deg) % 360.0
    near = lambda c: abs((az - c + 180.0) % 360.0 - 180.0)
    if min(near(45), near(135), near(225), near(315)) <= BAND_DEG:
        return "threeq"
    if min(near(90), near(270)) <= BAND_DEG:
        return "side"
    return "frontrear"


def pixels_per_metre(focal_mm, distance_m, img_w_px, sensor_mm=SENSOR_MM):
    return float(focal_mm) * float(img_w_px) / (max(float(distance_m), 0.1) * sensor_mm)


def car_length_m(bbox_w_px, azimuth_deg, focal_mm, distance_m, img_w_px):
    """Recover real car length (m) + the input image's pixels_per_metre."""
    ppm = pixels_per_metre(focal_mm, distance_m, img_w_px)
    w_proj = float(bbox_w_px) / ppm                         # projected footprint width in metres
    theta = math.radians(float(azimuth_deg) - 180.0)        # convention: front=180 -> 0
    factor = abs(math.sin(theta)) + abs(math.cos(theta)) / CAR_LW_RATIO
    factor = max(factor, 0.30)                              # guard the near-front degenerate case
    return w_proj / factor, ppm


def disc_diameter_m(bbox_w_px, azimuth_deg, focal_mm, distance_m, img_w_px):
    """Pose-aware, overflow-safe disc diameter. Returns (disc_m, recovered_L_m, pixels_per_metre, pose)."""
    L, ppm = car_length_m(bbox_w_px, azimuth_deg, focal_mm, distance_m, img_w_px)
    w_proj = float(bbox_w_px) / ppm                       # projected footprint width (directly measured)
    pose = pose_class(azimuth_deg)
    if pose == "threeq":
        disc = max(w_proj * MARGIN_3Q, FLOOR_3Q)          # 3/4: bigger ring + higher floor (don't overflow / look placed)
    elif pose == "side":
        disc = max(L * MARGIN_SIDE, FLOOR_M)              # side length is reliable
    else:
        disc = max(DEFAULT_L_M * DISC_MARGIN, FLOOR_M)    # front/rear: length unobservable → safe default
    disc = max(FLOOR_M, min(CEIL_M, disc))
    return disc, L, ppm, pose
