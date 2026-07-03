"""Perspective fuser (NEW) — orientation + retrieval + GeoCalib → camera values.

This is the thin Module-1 fuser for the unified pipeline. It does NOT re-implement
any of the three estimators; it orchestrates them and applies the ONE genuinely new
rule the unified design adds: the **azimuth correction**.

Per-image flow (exterior-full lane):

  1. Retrieval (``models.dino_retriever.Retriever.estimate``) on the SHARED YOLO
     crop (``do_crop=False``) → a continuous azimuth + the continuous camera values
     (elevation / distance / cam_height / focal / roll) + similarities.
  2. GeoCalib (``models.geocalib_stage.refine``) on the FULL image → an elevation
     estimate + pitch uncertainty.
  3. Gate (``perspective.gate.decide``) → the final elevation + an advisory
     HIGH/MEDIUM/LOW confidence label. Elevation choice + label are UNCHANGED from
     Module 1 (gate is advisory/logged only in the unified pipeline — no branching).
  4. **Azimuth correction (NEW):** retrieval gives the azimuth; the 8-way
     orientation class CORRECTS it ONLY when the circular |Δ| to the orientation's
     canonical degrees exceeds ``AZIMUTH_CORRECTION_DELTA_DEG`` (45°) — in which case
     we snap to the canonical degrees. Otherwise we keep retrieval's continuous value.
     The corrected azimuth drives the plate render + ``footprint.pose_class``.

Output: a flat ``camera`` dict {azimuth, elevation, distance, cam_height, focal,
roll} (the keys the Blender plate render + footprint math consume) plus the gate's
confidence label and a record of whether the azimuth was snapped.

>>> TODO(on-pod) <<< The 8-label→degrees table (``ORIENTATION_AZIMUTH_DEG`` in
config) is an ASSUMPTION about the index's azimuth convention (handedness, and
whether 0° is front or back). Run ``validate_azimuth_map()`` (delegates to the
config validator that buckets the index's continuous azimuths against the 8
canonical centres) and have a human eyeball a few sample crops per bucket BEFORE
trusting the snap. A mirror flip or a front/back swap would silently corrupt every
plate render. Until validated, treat the snap as advisory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from PIL import Image

from config.pipeline_config import (
    AZIMUTH_CORRECTION_DELTA_DEG,
    ORIENTATION_AZIMUTH_DEG,
    Orientation,
    circular_delta_deg,
    get_settings,
    validate_orientation_azimuth_map,
)
from models import dino_retriever, geocalib_stage
from perspective import gate
from utils.logging import get_logger, stage_timer

log = get_logger("perspective.estimate")


# ─────────────────────────────────────────────────────────────────────────────
# Azimuth correction (the one NEW rule)
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_azimuth(orientation: Union[Orientation, str, None]) -> Optional[float]:
    """Canonical azimuth (deg) for an orientation label, or None if unknown."""
    if orientation is None:
        return None
    if not isinstance(orientation, Orientation):
        try:
            orientation = Orientation(str(orientation))
        except ValueError:
            return None
    return ORIENTATION_AZIMUTH_DEG.get(orientation)


def correct_azimuth(
    retrieval_azimuth_deg: float,
    orientation: Union[Orientation, str, None],
    threshold_deg: float = AZIMUTH_CORRECTION_DELTA_DEG,
) -> Dict[str, object]:
    """Apply the azimuth-correction rule.

    Retrieval gives a continuous azimuth. The 8-way orientation class corrects it
    ONLY when the circular |Δ| to the orientation's canonical degrees is strictly
    greater than ``threshold_deg`` (default 45°); then we SNAP to the canonical
    degrees. Otherwise we keep retrieval's continuous value.

    Returns {azimuth_deg, snapped, retrieval_azimuth_deg, canonical_azimuth_deg,
    delta_deg, reason}.
    """
    retr_az = float(retrieval_azimuth_deg) % 360.0
    canonical = _canonical_azimuth(orientation)

    if canonical is None:
        return {
            "azimuth_deg": round(retr_az, 2),
            "snapped": False,
            "retrieval_azimuth_deg": round(retr_az, 2),
            "canonical_azimuth_deg": None,
            "delta_deg": None,
            "reason": "no orientation class → keep retrieval azimuth",
        }

    delta = circular_delta_deg(retr_az, canonical)
    if delta > threshold_deg:
        return {
            "azimuth_deg": round(canonical % 360.0, 2),
            "snapped": True,
            "retrieval_azimuth_deg": round(retr_az, 2),
            "canonical_azimuth_deg": round(canonical % 360.0, 2),
            "delta_deg": round(delta, 2),
            "reason": (
                f"orientation disagrees with retrieval (delta {delta:.1f} > "
                f"{threshold_deg:.0f}) → snap to canonical {canonical:.0f}"
            ),
        }
    return {
        "azimuth_deg": round(retr_az, 2),
        "snapped": False,
        "retrieval_azimuth_deg": round(retr_az, 2),
        "canonical_azimuth_deg": round(canonical % 360.0, 2),
        "delta_deg": round(delta, 2),
        "reason": (
            f"orientation agrees with retrieval (delta {delta:.1f} <= "
            f"{threshold_deg:.0f}) → keep retrieval azimuth"
        ),
    }


def validate_azimuth_map(index_dir=None, tolerance_deg: float = 22.5) -> Dict[str, object]:
    """Validate the 8-label→degrees map against the retrieval index.

    >>> TODO(on-pod) <<< Thin wrapper over the config validator
    (``validate_orientation_azimuth_map``): it buckets the index's continuous
    camera azimuths against the 8 canonical centres and reports per-bucket
    population + circular spread. A healthy convention gives 8 well-populated,
    tightly-clustered buckets. A near-empty bucket — or a bucket whose mass sits
    at the ANTIPODE of where it should — is the fingerprint of a front/back swap
    or a mirror flip. ``ok`` proves "every bucket is populated and tight"; it does
    NOT prove the front/back assignment is right — only a human spot-check can, so
    eyeball a few sample crops per bucket before trusting the snap.
    """
    if index_dir is None:
        index_dir = get_settings().index_dir
    return validate_orientation_azimuth_map(index_dir=Path(index_dir), tolerance_deg=tolerance_deg)


# ─────────────────────────────────────────────────────────────────────────────
# The fuser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraEstimate:
    """The fused per-image perspective estimate.

    ``camera`` is the flat dict consumed by the Blender plate render + footprint
    math. ``confidence`` is the gate's advisory HIGH/MEDIUM/LOW label.
    """
    camera: Dict[str, float]
    confidence: str
    azimuth_snapped: bool
    azimuth_info: Dict[str, object]
    gate_decision: gate.Decision
    retrieval: "dino_retriever.RetrievalResult"
    geocalib: Optional["geocalib_stage.GeoCalibResult"] = None
    notes: Dict[str, object] = field(default_factory=dict)


def estimate(
    crop: Image.Image,
    full_image_path: Union[str, Path],
    orientation: Union[Orientation, str, None],
    retriever: "dino_retriever.Retriever",
) -> CameraEstimate:
    """Fuse retrieval + GeoCalib + gate (+ azimuth correction) into camera values.

    Parameters
    ----------
    crop:
        The SHARED YOLO crop (RGB) — retrieval runs on it with ``do_crop=False`` so
        it keys on the same crop the rest of the pipeline uses.
    full_image_path:
        Path to the FULL input image — GeoCalib loads it itself (it needs the full
        frame's horizon, not the crop).
    orientation:
        The 8-way orientation class (``Orientation`` / its string value / None).
        Drives the azimuth-correction snap.
    retriever:
        A loaded ``dino_retriever.Retriever`` (index loaded once, shared).
    """
    # 1. Retrieval on the shared crop (no second detection).
    with stage_timer("retrieval", gpu=True, log=log):
        retr = retriever.estimate(crop, do_crop=False)

    # 2. GeoCalib on the full image.
    geo = None
    geo_failed_reason = None
    try:
        with stage_timer("geocalib", gpu=True, log=log):
            geo = geocalib_stage.refine(str(full_image_path))
    except Exception as exc:  # noqa: BLE001 — GeoCalib is a runtime-only dep; degrade gracefully
        geo_failed_reason = f"{type(exc).__name__}: {exc}"
        log.warning("geocalib.failed", reason=geo_failed_reason)

    # 3. Gate → final elevation + advisory confidence label (UNCHANGED from M1).
    #    Gate compares retrieval vs GeoCalib elevation; with no GeoCalib we keep
    #    retrieval's elevation and a MEDIUM advisory label.
    if geo is not None:
        decision = gate.decide(retr, geo)
        elevation = decision.elevation_deg
    else:
        decision = gate.Decision(
            elevation_deg=round(retr.elevation_deg, 2),
            source="retrieval (GeoCalib unavailable)",
            label="MEDIUM",
            reason=f"GeoCalib unavailable ({geo_failed_reason}); using retrieval elevation",
            delta=0.0,
            sim=round(retr.top1_sim, 3),
            unc=0.0,
        )
        elevation = decision.elevation_deg

    # 4. Azimuth correction (NEW): snap to orientation canonical only when Δ > 45°.
    azimuth_info = correct_azimuth(retr.azimuth_deg, orientation)
    azimuth = float(azimuth_info["azimuth_deg"])

    # 5. Assemble the flat camera-values dict the plate render + footprint consume.
    #    Azimuth = corrected; elevation = gate; the rest come from retrieval.
    camera: Dict[str, float] = {
        "azimuth": round(azimuth, 2),
        "elevation": round(float(elevation), 2),
        "distance": round(float(retr.distance_m), 2),
        "cam_height": round(float(retr.cam_height_m), 2),
        "focal": round(float(retr.focal_mm), 2),
        "roll": round(float(retr.roll_deg), 2),
    }

    log.info(
        "perspective.estimate",
        azimuth=camera["azimuth"],
        elevation=camera["elevation"],
        distance=camera["distance"],
        focal=camera["focal"],
        confidence=decision.label,
        azimuth_snapped=bool(azimuth_info["snapped"]),
        azimuth_reason=azimuth_info["reason"],
        gate_source=decision.source,
        gate_reason=decision.reason,
        top1_sim=retr.top1_sim,
        geocalib_available=geo is not None,
    )

    return CameraEstimate(
        camera=camera,
        confidence=decision.label,
        azimuth_snapped=bool(azimuth_info["snapped"]),
        azimuth_info=azimuth_info,
        gate_decision=decision,
        retrieval=retr,
        geocalib=geo,
        notes={
            "orientation": orientation.value if isinstance(orientation, Orientation) else orientation,
            "geocalib_failed_reason": geo_failed_reason,
        },
    )
