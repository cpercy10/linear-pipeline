"""Where do the car's wheels sit inside its crop? (S0 — ground-contact anchoring).

The unified pipeline used to anchor the car by its bbox GEOMETRIC CENTRE on the
plate's turntable centre ``cx,cy``. But ``cx,cy`` is a point on the FLOOR (the
projection of the turntable centre at floor height — see ``render_master.export_
metadata``), whereas the bbox centre is a point ~half-way up the car. Gluing a
mid-height point to a floor point makes the wheels land in a different place for
every car (tall SUV vs low coupe), which is why some cars rode the top/bottom edge
of the disc.

This module finds, per image, WHERE the car's ground-contact centre sits inside its
bounding box — as a fraction ``(fx, fy)`` — so the anchor can pin THAT point to
``cx,cy`` instead of the bbox centre. It needs no new model: it re-projects a plain
stand-in car box through the SAME camera the perspective fuser recovered, and reads
off where the footprint centre lands relative to the box's projected bounding box.

It deliberately mirrors the plate camera build in
``render/studio_engine/configurator/render_server.py:662-677`` (and the studio
invariants in ``render/studio_engine/scripts/render_master.py:40-43``) so the
geometry matches the rendered plate. The ONE difference: the plate fixes the camera
azimuth at 180° (turntable principle — the empty disc is azimuth-invariant), but the
real PHOTO was taken at the recovered ``azimuth``, and that is what shaped the car's
silhouette — so ``ground_contact_frac`` orbits the virtual camera to the real
azimuth. This is a throwaway projection: it renders nothing and never touches Blender.

Self-check: :func:`project_turntable_centre` re-projects the floor centre at the
PLATE azimuth (180°); it MUST reproduce a rendered plate's ``meta['car_spot_px']``
(within rounding). That is the proof the camera replica's conventions (sensor fit,
Y-flip, roll, look-at basis) match Blender. See the ``__main__`` demo below.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# ── Studio-engine invariants — MUST match render_master.py:40-43 ──────────────
# The turntable centre on the floor, the floor height, and the standard car box
# (L×W×H) the engine itself uses to frame plates. car_spot_px is the projection of
# (CAR_SPOT_XY, FLOOR_Z). If the .blend changes these, update here (the self-check
# against meta['car_spot_px'] will flag a mismatch).
CAR_SPOT_XY: Tuple[float, float] = (-0.08, -1.29)
FLOOR_Z: float = 0.16
STD_CAR_LWH: Tuple[float, float, float] = (4.60, 1.90, 1.45)
DEFAULT_SENSOR_MM: float = 36.0

Vec3 = Tuple[float, float, float]


# ── tiny vector helpers (no numpy dependency, so this stays standalone) ───────
def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec3) -> Vec3:
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


# ── camera replica (mirrors render_server.export_plate's pose build) ──────────
def _camera_pose(
    azimuth_deg: float,
    elevation_deg: Optional[float],
    distance_m: float,
    cam_height_m: float,
    roll_deg: float,
):
    """Camera location + (right, up, forward) basis, replicating export_plate.

    forward = the view direction (Blender's local −Z), up = local +Y, right = +X.
    ``azimuth_deg`` is the REAL photo azimuth (the plate fixes 180, but the photo's
    car silhouette was produced at the real angle). Everything else is verbatim from
    render_server.py:662-677.
    """
    cx, cy = CAR_SPOT_XY
    aim_z_default = FLOOR_Z + STD_CAR_LWH[2] / 2.0          # standard car volume centre
    dz_guess = cam_height_m + FLOOR_Z - aim_z_default
    horiz = math.sqrt(max(distance_m * distance_m - dz_guess * dz_guess, 0.25))

    a = math.radians(azimuth_deg)
    cam = (cx - math.cos(a) * horiz, cy - math.sin(a) * horiz, FLOOR_Z + cam_height_m)

    if elevation_deg is None or elevation_deg == "":
        aim_z = aim_z_default
    else:
        aim_z = cam[2] - math.tan(math.radians(float(elevation_deg))) * horiz
    aim = (cx, cy, aim_z)

    # Look-at basis: camera looks along −Z (forward), +Y up, +X right (Blender).
    fwd = _norm(_sub(aim, cam))
    world_up = (0.0, 0.0, 1.0)
    right = _norm(_cross(fwd, world_up))
    up = _cross(right, fwd)

    # Roll about the view axis (Blender applies Euler((0,0,roll)) on local Z = −fwd).
    r = math.radians(float(roll_deg or 0.0))
    cr, sr = math.cos(r), math.sin(r)
    right_r = (right[0] * cr + up[0] * sr,
               right[1] * cr + up[1] * sr,
               right[2] * cr + up[2] * sr)
    up_r = (-right[0] * sr + up[0] * cr,
            -right[1] * sr + up[1] * cr,
            -right[2] * sr + up[2] * cr)
    return cam, fwd, right_r, up_r


def _project(
    P: Vec3, cam: Vec3, fwd: Vec3, right: Vec3, up: Vec3,
    focal_mm: float, sensor_mm: float, res_x: int, res_y: int,
) -> Optional[Tuple[float, float]]:
    """Project a world point to plate pixels (top-left origin), replicating
    ``world_to_camera_view`` + the Y-flip in ``export_metadata.to_px``.

    Sensor fit is Blender's AUTO: the sensor maps to the LARGER image dimension.
    Returns ``None`` for points at/behind the camera.
    """
    v = _sub(P, cam)
    depth = _dot(v, fwd)
    if depth <= 1e-6:
        return None
    xc = _dot(v, right)
    yc = _dot(v, up)
    if res_x >= res_y:                       # landscape → sensor fits width
        sens_x = sensor_mm
        sens_y = sensor_mm * res_y / res_x
    else:                                    # portrait → sensor fits height
        sens_y = sensor_mm
        sens_x = sensor_mm * res_x / res_y
    ndc_x = 0.5 + (xc / depth) * (focal_mm / sens_x)
    ndc_y = 0.5 + (yc / depth) * (focal_mm / sens_y)
    return (ndc_x * res_x, (1.0 - ndc_y) * res_y)


def ground_contact_frac(
    *,
    azimuth_deg: float,
    elevation_deg: Optional[float],
    distance_m: float,
    cam_height_m: float,
    focal_mm: float,
    roll_deg: float = 0.0,
    res_x: int,
    res_y: int,
    box_lwh: Tuple[float, float, float] = STD_CAR_LWH,
    sensor_mm: float = DEFAULT_SENSOR_MM,
) -> Tuple[float, float]:
    """Where the car's ground-contact CENTRE sits inside its bounding box.

    Returns ``(fx, fy)`` in ``[0, 1]``: ``fx`` across the bbox (0=left, 1=right),
    ``fy`` down the bbox (0=top, 1=bottom). For a symmetric view (front / rear /
    side) ``fx≈0.5``; for a 3/4 view it shifts to the leaning side. ``fy`` is well
    below 0.5 from the bottom (the wheels live in the lower part of the box) and
    moves with elevation. These are the fractions the anchor pins onto ``cx,cy``.

    The car box is the engine's standard ``STD_CAR_LWH`` by default (robust — the
    fraction depends on box PROPORTIONS, not the noisy recovered length). Height is
    the dominant driver of ``fy``; override ``box_lwh`` per body-type later to refine.
    """
    cam, fwd, right, up = _camera_pose(
        azimuth_deg, elevation_deg, distance_m, cam_height_m, roll_deg
    )
    cx, cy = CAR_SPOT_XY
    L, W, H = box_lwh

    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for z in (FLOOR_Z, FLOOR_Z + H):
                p = _project((cx + sx * L / 2.0, cy + sy * W / 2.0, z),
                             cam, fwd, right, up, focal_mm, sensor_mm, res_x, res_y)
                if p is not None:
                    pts.append(p)

    fp = _project((cx, cy, FLOOR_Z), cam, fwd, right, up,
                  focal_mm, sensor_mm, res_x, res_y)

    if len(pts) < 2 or fp is None:
        return (0.5, 0.85)                   # degenerate fallback: centre-x, low-y
    minx = min(p[0] for p in pts); maxx = max(p[0] for p in pts)
    miny = min(p[1] for p in pts); maxy = max(p[1] for p in pts)
    if maxx <= minx or maxy <= miny:
        return (0.5, 0.85)

    fx = (fp[0] - minx) / (maxx - minx)
    fy = (fp[1] - miny) / (maxy - miny)
    return (min(1.0, max(0.0, fx)), min(1.0, max(0.0, fy)))


def project_turntable_centre(
    *,
    elevation_deg: Optional[float],
    distance_m: float,
    cam_height_m: float,
    focal_mm: float,
    roll_deg: float = 0.0,
    res_x: int,
    res_y: int,
    sensor_mm: float = DEFAULT_SENSOR_MM,
) -> Optional[Tuple[float, float]]:
    """SELF-CHECK: project the floor centre at the PLATE azimuth (180°).

    This MUST match a rendered plate's ``meta['car_spot_px']`` (within rounding) for
    the same elevation/distance/cam_height/focal/roll/resolution — that proves this
    module's camera conventions match Blender's. The turntable centre always projects
    to the horizontal image centre (``≈ res_x/2``) regardless of elevation/distance,
    which is a cheap invariant to assert even without a rendered plate.
    """
    cam, fwd, right, up = _camera_pose(
        180.0, elevation_deg, distance_m, cam_height_m, roll_deg
    )
    return _project((CAR_SPOT_XY[0], CAR_SPOT_XY[1], FLOOR_Z),
                    cam, fwd, right, up, focal_mm, sensor_mm, res_x, res_y)


def project_disc_depth_px(
    *,
    elevation_deg: Optional[float],
    distance_m: float,
    cam_height_m: float,
    focal_mm: float,
    roll_deg: float = 0.0,
    res_x: int,
    res_y: int,
    disc_m: float,
    sensor_mm: float = DEFAULT_SENSOR_MM,
) -> float:
    """DIAGNOSTIC: vertical pixel span of the turntable disc (near rim → far rim) at the
    plate camera (azimuth 180). The disc is a circle of radius ``disc_m/2`` on the floor;
    we project its near (+X) and far (−X) rim points and return ``|Δy|``."""
    cam, fwd, right, up = _camera_pose(180.0, elevation_deg, distance_m, cam_height_m, roll_deg)
    R = disc_m / 2.0
    cx, cy = CAR_SPOT_XY
    near = _project((cx + R, cy, FLOOR_Z), cam, fwd, right, up, focal_mm, sensor_mm, res_x, res_y)
    far_ = _project((cx - R, cy, FLOOR_Z), cam, fwd, right, up, focal_mm, sensor_mm, res_x, res_y)
    if near is None or far_ is None:
        return 0.0
    return abs(near[1] - far_[1])


def footprint_depth_frac(
    *,
    azimuth_deg: float,
    elevation_deg: Optional[float],
    distance_m: float,
    cam_height_m: float,
    focal_mm: float,
    roll_deg: float = 0.0,
    res_x: int,
    res_y: int,
    box_lwh: Tuple[float, float, float] = STD_CAR_LWH,
    sensor_mm: float = DEFAULT_SENSOR_MM,
) -> float:
    """DIAGNOSTIC: the car's wheel-contact face vertical span as a fraction of its bbox
    height (real azimuth). Multiply by the tight-bbox height (in resized-crop px) to get
    the footprint depth in canvas pixels."""
    cam, fwd, right, up = _camera_pose(azimuth_deg, elevation_deg, distance_m, cam_height_m, roll_deg)
    cx, cy = CAR_SPOT_XY
    L, W, H = box_lwh
    all_pts, foot_pts = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for z in (FLOOR_Z, FLOOR_Z + H):
                p = _project((cx + sx * L / 2.0, cy + sy * W / 2.0, z),
                             cam, fwd, right, up, focal_mm, sensor_mm, res_x, res_y)
                if p is None:
                    continue
                all_pts.append(p)
                if z == FLOOR_Z:
                    foot_pts.append(p)
    if len(all_pts) < 2 or len(foot_pts) < 2:
        return 0.0
    bbox_h = max(p[1] for p in all_pts) - min(p[1] for p in all_pts)
    if bbox_h <= 0:
        return 0.0
    foot_span = max(p[1] for p in foot_pts) - min(p[1] for p in foot_pts)
    return foot_span / bbox_h


if __name__ == "__main__":
    # Standalone sanity demo (no Blender / torch needed): python perspective/anchor_geometry.py
    RES = (1600, 1200)
    print(f"resolution {RES}, sensor {DEFAULT_SENSOR_MM}mm\n")

    # Invariant: the turntable centre projects to the horizontal image centre.
    for elev in (0.0, 10.0, 25.0):
        px = project_turntable_centre(elevation_deg=elev, distance_m=7.0,
                                      cam_height_m=1.35, focal_mm=50.0,
                                      res_x=RES[0], res_y=RES[1])
        print(f"car_spot_px @elev={elev:>4}: {tuple(round(v,1) for v in px)} "
              f"(x should be ~{RES[0]/2:.0f})")

    print("\nground-contact (fx, fy) by pose [dist 7, h 1.35, focal 50]:")
    for label, az, elev in [
        ("front          ", 180, 5), ("side            ", 90, 5),
        ("front-3/4 low   ", 135, 5), ("front-3/4 high  ", 135, 22),
        ("rear-3/4 low    ", 315, 5), ("rear-3/4 high   ", 315, 22),
    ]:
        fx, fy = ground_contact_frac(azimuth_deg=az, elevation_deg=elev,
                                     distance_m=7.0, cam_height_m=1.35,
                                     focal_mm=50.0, res_x=RES[0], res_y=RES[1])
        print(f"  az={az:>3} elev={elev:>2}  {label}  fx={fx:.3f}  fy={fy:.3f}")
