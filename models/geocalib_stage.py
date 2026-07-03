"""Stage 2 — GeoCalib: measure the camera's downward tilt and map it to the
studio's `elevation` parameter. We use GeoCalib's pitch only (its strong axis);
its focal is unreliable and azimuth/height it cannot give.

Reused from Module 1 ``perspective-estimation/geocalib_stage.py``. The pitch→
elevation calibration and the elevation clamp are UNCHANGED — only adapted to read
the slope / intercept / clamp from the unified ``PipelineSettings`` (``geocalib``
sub-config). The heavy ``geocalib`` package is LAZY-imported inside ``_model`` so
merely importing this module (or the package) never drags in that runtime-only dep.
"""
import math
import threading
from functools import lru_cache
from dataclasses import dataclass

import torch

from config.pipeline_config import get_settings


@dataclass
class GeoCalibResult:
    elevation_deg: float
    vfov_deg: float
    gc_pitch: float
    gc_roll: float
    pitch_uncertainty: float
    roll_uncertainty: float


# GeoCalib is a single shared model. Serialize calibrate() across the preprocess
# threads: calling one torch model concurrently is not thread-safe, and the lock
# bounds peak VRAM while GeoCalib is co-resident with FLUX on the one GPU.
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    # Lazy import: the geocalib package is a runtime-only GPU dependency.
    from geocalib import GeoCalib
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return GeoCalib().to(dev), dev


def refine(image_path) -> GeoCalibResult:
    cfg = get_settings().geocalib
    model, dev = _model()
    img = model.load_image(str(image_path)).to(dev)   # ON GPU (was CPU → 10–70 s/image)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    with _LOCK, torch.inference_mode():
        res = model.calibrate(img)
    gc_roll, gc_pitch = [math.degrees(float(x)) for x in res["gravity"].rp[0]]
    true_pitch = cfg.pitch_slope * gc_pitch + cfg.pitch_intercept
    lo, hi = cfg.elevation_clamp
    elevation = max(lo, min(hi, -true_pitch))
    return GeoCalibResult(
        elevation_deg=round(elevation, 2),
        vfov_deg=round(math.degrees(float(res["camera"].vfov[0])), 2),
        gc_pitch=round(gc_pitch, 2),
        gc_roll=round(gc_roll, 2),
        pitch_uncertainty=round(math.degrees(float(res["pitch_uncertainty"][0])), 2),
        roll_uncertainty=round(math.degrees(float(res["roll_uncertainty"][0])), 2),
    )
