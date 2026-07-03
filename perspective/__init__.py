"""Perspective layer — camera-value estimation for the exterior-full lane.

Reuses Module 1's estimators (retrieval + GeoCalib + confidence gate) and Module 2's
footprint→disc math UNCHANGED, plus the one NEW rule the unified design adds: the
azimuth correction (snap retrieval's continuous azimuth to the orientation class's
canonical degrees only when the circular |Δ| > 45°).

Submodules import heavy / runtime-only deps (timm, torch, geocalib) lazily or at
module import, so they are imported directly by call sites
(``from perspective.estimate import estimate``) rather than eagerly here.
"""
