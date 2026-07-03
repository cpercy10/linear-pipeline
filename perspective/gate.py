"""Stage 3 — confidence gate (Module 1, reused verbatim).

Decide the final elevation and a HIGH/MEDIUM/LOW trust label from retrieval
similarity (out-of-distribution flag), GeoCalib uncertainty, and method
agreement. Trusts GeoCalib's elevation by default.

This is copied from Module 1's ``perspective-estimation/gate.py`` UNCHANGED in
logic — the only adaptation is reading the thresholds from the unified
``PipelineSettings`` (the ``gate`` sub-config) instead of Module 1's flat
``config`` module. The elevation choice and the HIGH/MEDIUM/LOW labelling are
byte-for-byte the same decision.

LOCKED INVARIANT: in the unified pipeline this verdict is ADVISORY / LOGGED only
— it does NOT branch the flow.
"""
from dataclasses import dataclass

from config.pipeline_config import get_settings


@dataclass
class Decision:
    elevation_deg: float
    source: str
    label: str
    reason: str
    delta: float
    sim: float
    unc: float


def decide(retr, geo) -> Decision:
    """retr: RetrievalResult, geo: GeoCalibResult."""
    cfg = get_settings().gate

    delta = abs(geo.elevation_deg - retr.elevation_deg)
    agree = delta <= cfg.agree_deg
    geo_unreliable = (geo.pitch_uncertainty > cfg.geo_unc_veryhigh and delta > cfg.disagree_deg)
    # GeoCalib "collapse": near-floor reading while retrieval's prior says clearly higher
    # (its failure mode on plain studio floors; self-uncertainty does NOT flag it).
    geo_collapsed = (geo.elevation_deg <= cfg.geo_flat_max
                     and (retr.elevation_deg - geo.elevation_deg) > cfg.geo_drop_min)

    if geo_unreliable:
        elevation, source = retr.elevation_deg, "retrieval (GeoCalib unreliable)"
    elif geo_collapsed:
        elevation, source = retr.elevation_deg, "retrieval (GeoCalib collapsed flat)"
    else:
        elevation, source = geo.elevation_deg, "geocalib"

    sim = retr.top1_sim
    if sim < cfg.sim_low:
        label = "LOW"
        reason = f"out-of-distribution (retrieval sim {sim:.2f}) -> review"
    elif geo_unreliable:
        label = "LOW"
        reason = f"GeoCalib unsure (unc {geo.pitch_uncertainty:.1f}) & disagrees (delta {delta:.1f}) -> review"
    elif geo_collapsed:
        label = "HIGH" if sim >= cfg.sim_high else "MEDIUM"
        reason = (f"GeoCalib collapsed flat (geo {geo.elevation_deg:.1f} vs retr "
                  f"{retr.elevation_deg:.1f}) -> used retrieval (sim {sim:.2f})")
    elif sim >= cfg.sim_high and (agree or geo.pitch_uncertainty <= cfg.geo_confident_unc):
        label = "HIGH"
        why = f"methods agree (delta {delta:.1f})" if agree else f"GeoCalib confident (unc {geo.pitch_uncertainty:.1f})"
        reason = f"{why}, strong retrieval (sim {sim:.2f})"
    else:
        label = "MEDIUM"
        reason = f"usable (sim {sim:.2f}, unc {geo.pitch_uncertainty:.1f}, delta {delta:.1f})"

    return Decision(round(elevation, 2), source, label, reason,
                    round(delta, 2), round(sim, 3), round(geo.pitch_uncertainty, 2))
