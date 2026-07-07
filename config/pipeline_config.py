"""
Unified configuration for the Motuva pipeline (Modules 1 + 2 + 3 fused).

Everything tunable lives here or in ``orientation.yaml``. Logic code reads from
``PipelineSettings`` — no hardcoded paths, sizes, or thresholds anywhere else.

Values are populated from environment variables (prefix ``MOTOCUT_``) and an
``a.env`` file at the package root (copy ``.env.example`` → ``a.env``).

This is the single merged config the unified package uses; it replaces the three
modules' separate configs:

  * Module 3 (diffusion / spine): model + diffusion + remove.bg + canvas blocks,
    OrientationRegistry, the singleton accessors. KEPT verbatim where possible.
  * Module 1 (perspective):  retrieval / GeoCalib / confidence-gate constants +
    the retrieval index dir. MERGED IN under the ``retrieval`` / ``geocalib`` /
    ``gate`` sub-configs.
  * Module 2 (plate render): Blender exe path, the studio-engine repo/dir, the
    materials dir, the master .blend, the STUDIO look + render long-edge/samples.
    MERGED IN under the ``blender`` sub-config.

All in-package asset paths (weights, retrieval index, Blender assets) default
INTO this package (``assets/weights``, ``assets/index``, ``assets/blender``) and
stay env-overridable.

NOTE on car sizing: the per-orientation fill/anchor registry from Module 3 is
REPLACED by an area-based ``TARGET_VISUAL_OCCUPANCY`` map (see ``orientation.yaml``
and ``VisualOccupancyRegistry`` below). Placement is by GROUND-CONTACT anchoring
(S0): the car is pinned by its WHEEL-FOOTPRINT centre onto the plate's ``cx,cy``,
then slid toward the camera by a per-pose forward bias (``AnchorConfig``). See
``stages/anchor.py`` + ``perspective/anchor_geometry.py``.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Package-root anchored asset locations
# ─────────────────────────────────────────────────────────────────────────────
# config/pipeline_config.py -> config/ -> motuva_pipeline/  (the package root)
_PKG_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _PKG_ROOT / "assets"


# ─────────────────────────────────────────────────────────────────────────────
# Enums — the fixed vocabularies of the pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ImageClass(str, Enum):
    """Output of the 3-way router classifier → selects the lane."""
    EXTERIOR_FULL    = "exterior-full"
    EXTERIOR_PARTIAL = "exterior-partial"
    INTERIOR         = "interior"


class Orientation(str, Enum):
    """Output of the 8-way orientation model (exterior_full lane only)."""
    BACK           = "back"
    BACK_LEFT      = "back-left"
    BACK_RIGHT     = "back-right"
    FRONT          = "front"
    FRONT_LEFT     = "front-left"
    FRONT_RIGHT    = "front-right"
    SIDEWAYS_LEFT  = "sideways-left"
    SIDEWAYS_RIGHT = "sideways-right"


# ─────────────────────────────────────────────────────────────────────────────
# Orientation → canonical azimuth (degrees) map.
#
# Drives the azimuth-CORRECTION rule (Module 1 gate): retrieval gives a continuous
# azimuth; the 8-way orientation model corrects it ONLY when the circular delta
# exceeds 45° — in which case we snap to the orientation's canonical degrees here.
# Otherwise we keep retrieval's continuous value.
#
# Convention (degrees, [0, 360)):
#   front          = 180   (camera looks at the front of the car)
#   back           =   0
#   sideways-left  =  90
#   sideways-right = 270
#   the four corners halfway between their neighbours:
#   back-right     =  45   front-right    = 135
#   front-left     = 225   back-left      = 315
#
# >>> TODO(on-pod) <<< This table is an ASSUMPTION about the index's azimuth
# convention (sign/handedness, and whether 0° is front or back). It MUST be
# VALIDATED against the index before being trusted — a mirror flip or a
# front/back swap would silently corrupt every plate render. Use
# ``validate_orientation_azimuth_map()`` below (Phase 2 / on-pod checklist) to
# confirm each label's retrieved azimuths cluster around the value here before
# enabling the snap. Until validated, treat snaps as advisory.
# ─────────────────────────────────────────────────────────────────────────────

ORIENTATION_AZIMUTH_DEG: Dict[Orientation, float] = {
    Orientation.FRONT:          180.0,
    Orientation.BACK:             0.0,
    Orientation.SIDEWAYS_LEFT:   90.0,
    Orientation.SIDEWAYS_RIGHT: 270.0,
    Orientation.BACK_RIGHT:      45.0,
    Orientation.FRONT_RIGHT:    135.0,
    Orientation.FRONT_LEFT:     225.0,
    Orientation.BACK_LEFT:      315.0,
}

# Snap threshold: correct retrieval azimuth only when the circular difference to
# the orientation's canonical degrees is strictly greater than this.
AZIMUTH_CORRECTION_DELTA_DEG: float = 45.0


def circular_delta_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference between two azimuths, in [0, 180]."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs — Module 3 (diffusion spine)
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierConfig(BaseModel):
    """3-way router. timm tf_efficientnetv2_l.in21k, trained at 1024."""
    arch:       str             = "tf_efficientnetv2_l.in21k"
    img_size:   int             = 1024
    num_classes: int            = 3
    # Order MUST match the training label order.
    class_names: List[str]      = [c.value for c in
                                   (ImageClass.EXTERIOR_FULL,
                                    ImageClass.EXTERIOR_PARTIAL,
                                    ImageClass.INTERIOR)]
    norm_mean:  Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std:   Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int             = 16   # batchable — fixed-size input


class OrientationConfig(BaseModel):
    """8-way orientation model. Same arch/transform as the classifier."""
    arch:        str            = "tf_efficientnetv2_l.in21k"
    img_size:    int            = 1024
    num_classes: int            = 8
    # Order MUST match the training label order (see orientation reference script).
    class_names: List[str]      = [o.value for o in (
        Orientation.BACK, Orientation.BACK_LEFT, Orientation.BACK_RIGHT,
        Orientation.FRONT, Orientation.FRONT_LEFT, Orientation.FRONT_RIGHT,
        Orientation.SIDEWAYS_LEFT, Orientation.SIDEWAYS_RIGHT)]
    norm_mean:   Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std:    Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size:  int            = 16


class YoloConfig(BaseModel):
    """Car detection. We keep the largest detected vehicle bbox.

    ONE shared YOLO serves both the detection crop (Module 3 exterior lane) and
    the retrieval crop (Module 1) — detection only, no segmentation.
    """
    # COCO classes: 2=car, 5=bus, 7=truck
    classes:      List[int]     = [2, 5, 7]
    conf:         float         = 0.25
    iou:          float         = 0.45
    crop_padding: float         = 0.05   # fraction of bbox added on each side


class CanvasConfig(BaseModel):
    """Gray canvas the (occupancy-resized) car is anchored onto (size == input)."""
    fill_color: Tuple[int, int, int] = (200, 200, 200)
    # Final image clamps (occupancy resize): keep the car inside the canvas and
    # leave a small bottom margin so the contact shadow has room.
    bottom_margin_frac: float = 0.02
    # Upscale-only ± aspect nudge allowed during the occupancy resize.
    aspect_nudge_frac:  float = 0.05


class AnchorConfig(BaseModel):
    """Ground-contact anchoring (exterior_full) — see perspective/anchor_geometry.py.

    The car is pinned by its WHEEL-FOOTPRINT CENTRE (not its bbox centre) onto the
    plate's turntable centre ``cx,cy``, then slid toward the camera by a consistent
    forward bias so the foreground apron of the (deliberately generous) disc reads
    naturally rather than empty.
    """
    # Camera sensor width assumed by the plate (Blender cam.data.sensor_width). MUST
    # match the master .blend; the project_turntable_centre self-check validates it.
    sensor_mm:    float = 36.0
    # Stand-in car box (L×W×H, metres) used to locate the wheels in the crop. Mirrors
    # render_master STD_CAR; the footprint fraction depends on PROPORTIONS, so the
    # absolute size (and noisy recovered length) does not matter here.
    car_box_lwh:  Tuple[float, float, float] = (4.60, 1.90, 1.45)
    # Forward bias: shift the car toward the camera by this fraction of the disc
    # RADIUS (converted to pixels via the plate's toward-camera pixels/metre), to fill
    # the foreground apron. 0.0 = pure grounding (car centred on the turntable). Tune by
    # eye; raise to push the car further forward. Set PER POSE GROUP — keyed by the
    # footprint pose_class (``pre.pose``, the azimuth-derived bucket), so front/rear, the
    # four 3/4 corners, and full side can each be dialled independently.
    forward_bias_frontrear: float = 0.25   # azimuth near 0°/180°        (front / rear)
    forward_bias_threeq:    float = 0.17   # azimuth near 45/135/225/315 (the four 3/4 corners)
    forward_bias_side:      float = 0.27   # azimuth near 90°/270°       (sideways left / right)


class RemoveBgConfig(BaseModel):
    """remove.bg API (interior + exterior_partial lanes). Network-bound."""
    endpoint:       str         = "https://api.remove.bg/v1.0/removebg"
    size:           str         = "auto"
    add_shadow:     bool        = True
    exterior_add_shadow: bool   = False  # exterior-full uses the local plate-aware contact shadow
    shadow_opacity: int         = 100
    concurrency:    int         = 4      # simultaneous in-flight API calls
    max_retries:    int         = 3
    timeout_s:      float       = 60.0
    interior_alpha_threshold: int = 250  # alpha < this → forced to white


class RembgConfig(BaseModel):
    """Server exterior-full local segmentation experiment."""
    model_name: str = "birefnet-general"
    alpha_matting: bool = True
    clean_mask: bool = True
    morph_kernel: int = 7
    feather_px: int = 3
    alpha_threshold: int = 16
    preserve_erode_px: int = 5
    edge_band_px: int = 18
    edge_inpaint_radius_px: int = 3
    color_match_strength: float = 0.22
    color_match_band_px: int = 24
    edge_plate_blend: float = 0.18
    shadow_offset_frac: float = 0.08
    shadow_height_frac: float = 0.18
    shadow_width_frac: float = 0.84
    shadow_blur_px: int = 18
    contact_shadow_opacity: float = 0.34


class InpaintConfig(BaseModel):
    """Server-only FLUX Fill inpaint experiment for exterior-full images."""
    enabled: bool = False
    model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    max_long_edge: int = 1024
    num_steps: int = 28
    seed: Optional[int] = None
    mode: Literal["shadow", "shadow_edge", "shadow_edge_body"] = "shadow_edge"
    prompt: str = (
        "Create a natural soft studio contact shadow beneath the tires and repair only "
        "the cutout edge transition into the floor and background. Preserve the car "
        "identity, paint color, silhouette, wheels, lights, glass, trim, and details."
    )
    guidance_scale: float = 30.0
    body_opacity: float = 0.35


class FluxRefineConfig(BaseModel):
    """Final FLUX.2 Klein image-edit pass over the rembg composite."""
    enabled: bool = False
    model_id: str = "black-forest-labs/FLUX.2-klein-9B"
    max_long_edge: int = 768
    num_steps: int = 4
    seed: Optional[int] = None
    guidance_scale: float = 1.0
    strength: Optional[float] = None
    reference_mode: Literal["both", "with_reference", "composite_only", "multi_reference"] = "with_reference"
    cpu_offload: bool = True
    prompt: str = ""  # optional override; empty selects one of the mode prompts below
    prompt_composite_only: str = (
        "Automotive composite repair only. Use the provided image as the locked final "
        "composition and improve only the pasted car integration. Keep the background, "
        "camera viewpoint, framing, scene geometry, road, sky, buildings, signs, and "
        "all environment details unchanged. Repair jagged cutout edges, missing or "
        "thin car edge pixels, small holes, alpha fringing, background bleed-through "
        "inside the car, weak tire grounding, absent contact shadow, color mismatch, "
        "exposure mismatch, glass contamination, paint reflections, and unnatural "
        "reflections according to the visible scene lighting. Preserve the same car "
        "silhouette, dimensions, viewing angle, camera perspective, proportions, body "
        "shape, wheel size, wheel position, ride height, glass shape, grille, lights, "
        "badges, trim, license area, model text, and original paint hue. Do not "
        "repaint, recolor, redesign, resize, rotate, warp, smooth away details, add "
        "new objects, remove background objects, or hallucinate background content. "
        "Final result: photorealistic automotive integration with clean edges, better "
        "color correction, realistic reflections, natural shadow, and no pasted-on look."
    )
    prompt_with_reference: str = (
        "Automotive composite repair using references. Use the first image as the "
        "locked final composition to improve. Use the gray guide image only for the "
        "car's exact placement, silhouette, scale, tire contact points, and camera "
        "perspective. Use the cropped car reference only for car identity and missing "
        "details: same silhouette, dimensions, viewing angle, proportions, body shape, "
        "wheel size, wheel position, ride height, glass shape, grille, lights, badges, "
        "trim, license area, model text, and original paint hue. Keep the background "
        "scene, camera viewpoint, road, sky, buildings, signs, and environment details "
        "from the composition unchanged. Repair jagged edges, missing car parts, edge "
        "holes, alpha fringing, background bleed-through, weak/no contact shadow, color "
        "mismatch, exposure mismatch, glass contamination, old-environment reflections, "
        "and unnatural paint reflections. Remove only unwanted old-environment reflected "
        "objects and color contamination while preserving automotive gloss, metallic "
        "tone if present, natural panel shading, broad gradients, curved surface "
        "reflections, glass tint, chrome, trim, and specular highlights. Do not repaint, "
        "recolor, redesign, resize, rotate, warp, flatten paint, make it matte, add "
        "objects, remove background objects, or hallucinate new background details. "
        "Final result: source-accurate photorealistic car integration with clean edges, "
        "correct color, improved reflections, natural tire grounding, and realistic shadow."
    )


class InteriorConfig(BaseModel):
    bg_color: Tuple[int, int, int] = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs — Module 1 (perspective estimation)
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalConfig(BaseModel):
    """DINOv2 embedding + kNN retrieval over the prebuilt index.

    Mirrors Module 1 ``perspective-estimation/config.py`` (stage 1).
    """
    dino_model:      str   = "vit_small_patch14_dinov2.lvd142m"
    embed_size:      int   = 224
    crop_pad:        float = 0.08          # standard retrieval pad (A/B vs occupancy)
    vehicle_classes: Tuple[int, int, int] = (2, 5, 7)  # COCO car / bus / truck
    k_neighbours:    int   = 8
    softmax_temp:    float = 0.04


class GeoCalibConfig(BaseModel):
    """GeoCalib elevation mapping (Module 1 stage 2, calibrated on the pilot, r=0.92).

    true_camera_pitch = slope * gc_pitch + intercept ; elevation = -true_camera_pitch
    """
    pitch_slope:     float = 0.764
    pitch_intercept: float = -0.19
    elevation_clamp: Tuple[float, float] = (-5.0, 30.0)


class GateConfig(BaseModel):
    """Confidence gate (Module 1 stage 3).

    NOTE (locked invariant): in the unified pipeline the gate is ADVISORY /
    LOGGED only — it does NOT branch the flow. These thresholds drive the logged
    advisory verdict (similarity / agreement / GeoCalib confidence).
    """
    sim_low:           float = 0.60   # below -> out-of-distribution
    sim_high:          float = 0.70   # at/above -> retrieval pose trustworthy
    geo_unc_veryhigh:  float = 7.0    # GeoCalib essentially guessing
    agree_deg:         float = 4.0    # methods "agree" within this
    disagree_deg:      float = 6.0    # methods materially conflict
    geo_confident_unc: float = 4.0    # GeoCalib uncertainty considered "confident"
    # GeoCalib "collapse" guard: on low-texture studio inputs GeoCalib reads near-level
    # with falsely-low uncertainty (self-uncertainty does NOT catch it). When it lands
    # near-floor AND retrieval's 8-neighbour prior says clearly higher, distrust GeoCalib
    # and use retrieval. Validated on 118 Outputs-11 imgs: fixes 21/22 collapses, 0 false
    # positives; leaves GeoCalib's upward corrections (its real purpose) untouched.
    geo_flat_max:      float = 3.0    # geo elevation <= this is "suspiciously flat"
    geo_drop_min:      float = 2.0    # ...and retrieval this much higher -> collapse


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs — Module 2 (plate render / Blender studio engine)
# ─────────────────────────────────────────────────────────────────────────────

class BlenderConfig(BaseModel):
    """Blender studio-engine settings (Module 2).

    The studio engine (``scripts`` + ``configurator``) and assets (master .blend +
    Materials/_downloaded_2k) are vendored INTO this package under
    ``render/studio_engine`` and ``assets/blender`` respectively. Paths below
    default to those in-package locations and stay env-overridable via the
    top-level settings (see ``PipelineSettings`` Blender fields).

    GPU + warm-worker invariants (locked):
      * Cycles renders on GPU (the worker forces ``cycles.device='GPU'`` in place
        of Module 2 render.py's hardcoded 'CPU').
      * A persistent warm worker loads the scene / materials / _setup_carfree ONCE
        and loops over jobs (no cold subprocess-per-image).
    """
    # Studio look — CAR-FREE empty plate. Module 2 render.py applies a small zoom-out
    # for the 3/4 angles itself; this is the base look.
    studio: Dict[str, str] = {
        "room": "flatwall",
        "light": "panels",
        "turntable": "flush",
        "floor": "lib_floor-tiling-stonegranitetile-",
        "wall": "paint:54585B",
        "branding": "none",
    }
    render_long_edge: int = 1024
    render_samples:   int = 16
    device:           Literal["GPU", "CPU"] = "GPU"   # warm worker forces GPU
    release_after_render: bool = True  # stop Blender after each plate to free Cycles VRAM
    # Warm-worker IPC / supervision.
    job_timeout_s:    float = 600.0   # full-res plates on big photos can take >180s until the plate-res cap lands
    startup_timeout_s: float = 300.0   # scene + material load can be slow on cold start
    # Auto-restart: bounded retries with linear backoff so one transient crash does not
    # spin forever (and so a hung respawn cannot wedge the held job lock indefinitely —
    # _spawn() is itself startup_timeout_s-bounded). On exhaustion the worker is marked
    # permanently dead and every render() fails fast with a clear fatal.
    max_restart_attempts: int = 3
    restart_backoff_s:    float = 2.0   # sleep = attempt_index * restart_backoff_s


# ─────────────────────────────────────────────────────────────────────────────
# Top-level settings (env / .env driven)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MOTOCUT_",
        env_file=str(_PKG_ROOT / "a.env"),   # package-root a.env (works from any cwd)
        env_file_encoding="utf-8",
        env_nested_delimiter="__",   # e.g. MOTOCUT_DIFFUSION__ENABLE_CPU_OFFLOAD=true
        extra="ignore",
    )

    # ── Model weights — default INTO the package (assets/weights), env-overridable ─
    classifier_weights:  Path = _ASSETS / "weights" / "interiorvsfullvspartial.pth"
    orientation_weights: Path = _ASSETS / "weights" / "orientation-model.pth"
    yolo_weights:        str = "yolov8s.pt"   # ultralytics resolves bare names

    # ── Module 1: retrieval index dir — defaults INTO the package (assets/index) ──
    #    Holds emb.npy + meta.json (NOT the shards/).
    index_dir:           Path = _ASSETS / "index"

    # ── Module 2: Blender exe + studio engine + assets — env-overridable ─────────
    blender_exe:         Path = Path(r"D:\Blender\B\blender.exe")
    # Studio engine (scripts + configurator) vendored under render/studio_engine.
    studio_engine_dir:   Path = _PKG_ROOT / "render" / "studio_engine"
    # Master .blend + materials default into assets/blender.
    master_blend:        Path = _ASSETS / "blender" / "motuva-studio-master.blend"
    materials_2k_dir:    Path = _ASSETS / "blender" / "Materials" / "_downloaded_2k"

    # ── Batch-only paths — optional. The DirectoryRunner enforces them when used. ─
    input_dir:           Optional[Path] = None
    output_dir:          Optional[Path] = None
    background_image:    Optional[Path] = None   # supplied background for the partial lane

    # ── Secrets (read without the MOTOCUT_ prefix; validation_alias bypasses it) ─
    remove_bg_api_key:   str = Field(default="", validation_alias="REMOVE_BG_API_KEY")

    # Force a specific device for the small models; empty = same as diffusion gpu0.
    small_model_device:      str = ""
    # Inference dtype for the timm classifiers + YOLO. fp16 is faster; switch to
    # "float32" if you observe routing/orientation drift vs your fp32 training eval.
    small_model_dtype:       Literal["float16", "float32"] = "float16"

    # ── Concurrency (three async tiers) ──────────────────────────────────────
    preprocess_workers:  int = 8       # CPU/light-GPU stages (yolo/orient/dino/geocalib/resize/anchor)
    removebg_concurrency: int = 4      # mirrors RemoveBgConfig.concurrency at runtime

    # ── Observability ───────────────────────────────────────────────────────
    log_level:   str = "INFO"
    environment: Literal["dev", "prod"] = "prod"
    # On startup, validate the 8-label→degrees azimuth table against the index's
    # convention (advisory: logs azimuth_map.ok / .suspect, never blocks the run; see
    # validate_orientation_azimuth_map). Re-reads meta.json once (~1-3s) — set False to
    # skip once the convention is confirmed.
    validate_azimuth_on_startup: bool = True

    # ── Allowed input extensions ────────────────────────────────────────────
    image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    # ── Car sizing: area-based visual occupancy targets (replaces fill/anchor) ──
    #    Loaded from this YAML by VisualOccupancyRegistry.
    orientation_config_path: Path = Path(__file__).parent / "orientation.yaml"

    # ── Nested component configs ────────────────────────────────────────────
    classifier:  ClassifierConfig  = ClassifierConfig()
    orientation: OrientationConfig = OrientationConfig()
    yolo:        YoloConfig        = YoloConfig()
    canvas:      CanvasConfig      = CanvasConfig()
    anchor:      AnchorConfig      = AnchorConfig()
    removebg:    RemoveBgConfig    = RemoveBgConfig()
    rembg:       RembgConfig       = RembgConfig()
    inpaint:     InpaintConfig     = InpaintConfig()
    flux_refine: FluxRefineConfig  = FluxRefineConfig()
    interior:    InteriorConfig    = InteriorConfig()
    retrieval:   RetrievalConfig   = RetrievalConfig()
    geocalib:    GeoCalibConfig    = GeoCalibConfig()
    gate:        GateConfig        = GateConfig()
    blender:     BlenderConfig     = BlenderConfig()

    @field_validator("image_extensions", mode="after")
    @classmethod
    def _lower_exts(cls, v: List[str]) -> List[str]:
        return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in v]



# ─────────────────────────────────────────────────────────────────────────────
# Visual-occupancy registry (loads orientation.yaml once)
#
# REPLACES Module 3's per-orientation fill_ratio/vertical_anchor registry. Car
# sizing is now area-based VISUAL OCCUPANCY (the fraction of the canvas area the
# car's bbox crop should cover), upscale-only, per class.
# ─────────────────────────────────────────────────────────────────────────────

# Keys understood in orientation.yaml beyond the 8 orientations.
_OCCUPANCY_EXTRA_KEYS = ("partial", "default")


class VisualOccupancyRegistry:
    """Loads and serves area-based visual-occupancy targets from YAML.

    Each value is the target fraction of the canvas AREA the car should occupy
    (e.g. 0.70 = the car bbox covers ~70% of the canvas area). Upscale-only.

    Keys: the 8 orientation labels + ``partial`` (the exterior-partial / framed
    lane) + ``default`` (fallback when no class is resolved).
    """

    def __init__(self, mapping: Dict[str, float]):
        self._mapping = mapping

    @classmethod
    def load(cls, path: Path) -> "VisualOccupancyRegistry":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        occ = raw.get("target_visual_occupancy", raw)
        mapping: Dict[str, float] = {}
        for orient in Orientation:
            if orient.value not in occ:
                raise ValueError(
                    f"orientation.yaml is missing an entry for "
                    f"'{orient.value}' under target_visual_occupancy. "
                    f"All 8 orientations must be present."
                )
            mapping[orient.value] = float(occ[orient.value])
        for extra in _OCCUPANCY_EXTRA_KEYS:
            if extra not in occ:
                raise ValueError(
                    f"orientation.yaml is missing the required '{extra}' "
                    f"entry under target_visual_occupancy."
                )
            mapping[extra] = float(occ[extra])
        return cls(mapping)

    def get(self, key: "Orientation | str") -> float:
        """Return the occupancy target for an orientation/label, falling back to
        ``default`` when the key is unknown."""
        if isinstance(key, Orientation):
            key = key.value
        return self._mapping.get(key, self._mapping["default"])

    @property
    def mapping(self) -> Dict[str, float]:
        return dict(self._mapping)


# ─────────────────────────────────────────────────────────────────────────────
# Azimuth-map validation helper (on-pod / Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_azimuth(entry: dict) -> Optional[float]:
    """Pull an azimuth (degrees) out of one index meta entry, defensively.

    The real Module-1 index stores it at ``entry["camera"]["azimuth_deg"]``; we
    also accept a handful of flat fallbacks so this keeps working if the schema
    shifts. Returns None when no azimuth is present.
    """
    azim_keys = ("azimuth_deg", "azimuth", "yaw_deg", "yaw", "az")
    cam = entry.get("camera") if isinstance(entry, dict) else None
    for src in (cam, entry):
        if isinstance(src, dict):
            for k in azim_keys:
                v = src.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
    return None


def validate_orientation_azimuth_map(
    index_dir: Optional[Path] = None,
    tolerance_deg: float = 22.5,
) -> Dict[str, object]:
    """Validate ``ORIENTATION_AZIMUTH_DEG`` against the retrieval index.

    >>> TODO(on-pod) <<< This is the guard for the locked invariant that the
    8-label→degrees table matches the index's azimuth convention (handedness,
    and whether 0° is front or back). Get a human to eyeball a few sample crops
    against their bucket before trusting the snap.

    The real ``meta.json`` (98k entries) carries a CONTINUOUS
    ``camera.azimuth_deg`` per entry but NO discrete orientation label, so we
    cannot group by label. Instead we BUCKET each entry to its nearest canonical
    azimuth (the 8 values in ``ORIENTATION_AZIMUTH_DEG``) and report, per bucket:
    how many entries fell in it, and the circular spread (std-dev) of their
    azimuths. A healthy convention produces 8 well-populated buckets each tightly
    clustered (small ``circular_std_deg``) around its centre. A near-empty bucket,
    or a bucket whose mass sits at the ANTIPODE of where it should, is the
    fingerprint of a front/back swap or a mirror flip — the thing this guard
    exists to catch.

    Returns a report dict: per-bucket {n, circular_mean_deg, circular_std_deg,
    expected_deg, ok} + an overall ``ok``. ``ok`` here means "every bucket is
    populated and tightly clustered"; it does NOT prove the front/back assignment
    is correct (only a human spot-check can) — hence the TODO. Raises nothing.
    """
    import json

    settings_index = index_dir or (_ASSETS / "index")
    meta_path = Path(settings_index) / "meta.json"
    report: Dict[str, object] = {"meta_path": str(meta_path)}

    if not meta_path.exists():
        report["ok"] = False
        report["error"] = f"meta.json not found at {meta_path}"
        return report

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # meta.json may be a bare list or {"entries"/"items"/"meta": [...]}.
    if isinstance(meta, dict):
        entries = meta.get("entries") or meta.get("items") or meta.get("meta") or []
    else:
        entries = meta
    if not isinstance(entries, list) or not entries:
        report["ok"] = False
        report["error"] = "could not locate a list of index entries in meta.json"
        return report

    centres = list(ORIENTATION_AZIMUTH_DEG.items())   # [(Orientation, deg), ...]
    buckets: Dict[str, List[float]] = {o.value: [] for o, _ in centres}
    unverifiable = 0
    for e in entries:
        az = _extract_azimuth(e)
        if az is None:
            unverifiable += 1
            continue
        az %= 360.0
        nearest = min(centres, key=lambda kv: circular_delta_deg(az, kv[1]))
        buckets[nearest[0].value].append(az)

    if all(len(v) == 0 for v in buckets.values()):
        report["ok"] = False
        report["error"] = "no entries carried a camera azimuth"
        report["unverifiable"] = unverifiable
        return report

    per_bucket: Dict[str, object] = {}
    overall_ok = True
    total = sum(len(v) for v in buckets.values())
    # A bucket is "populated" if it holds at least 2% of the verifiable mass.
    min_n = max(1, int(0.02 * total))
    for orient, expected in centres:
        vals = buckets[orient.value]
        n = len(vals)
        if n == 0:
            per_bucket[orient.value] = {"n": 0, "expected_deg": expected, "ok": False}
            overall_ok = False
            continue
        s = sum(math.sin(math.radians(v)) for v in vals) / n
        c = sum(math.cos(math.radians(v)) for v in vals) / n
        r = math.hypot(s, c)                                   # mean resultant length
        mean_deg = math.degrees(math.atan2(s, c)) % 360.0
        # Circular standard deviation (degrees); 0 = perfectly tight.
        circ_std = math.degrees(math.sqrt(-2.0 * math.log(r))) if r > 1e-9 else 180.0
        ok = (n >= min_n) and (circ_std <= tolerance_deg)
        overall_ok = overall_ok and ok
        per_bucket[orient.value] = {
            "n": n,
            "circular_mean_deg": round(mean_deg, 2),
            "circular_std_deg": round(circ_std, 2),
            "expected_deg": expected,
            "ok": ok,
        }

    report["per_bucket"] = per_bucket
    report["unverifiable"] = unverifiable
    report["min_bucket_n"] = min_n
    report["tolerance_deg"] = tolerance_deg
    report["ok"] = overall_ok
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessors
# ─────────────────────────────────────────────────────────────────────────────

_settings: Optional[PipelineSettings] = None
_occupancy_registry: Optional[VisualOccupancyRegistry] = None


def get_settings() -> PipelineSettings:
    global _settings
    if _settings is None:
        # Populate os.environ from a.env so prefixed vars and the bare
        # REMOVE_BG_API_KEY all resolve consistently — regardless of cwd.
        try:
            from dotenv import load_dotenv
            load_dotenv(_PKG_ROOT / "a.env")
        except Exception:  # noqa: BLE001 — dotenv is optional at runtime
            pass
        _settings = PipelineSettings()  # type: ignore[call-arg]
    return _settings


def get_orientation_registry() -> VisualOccupancyRegistry:
    """Return the visual-occupancy registry (kept under the historical name so
    Module-3 call sites that import ``get_orientation_registry`` keep working)."""
    global _occupancy_registry
    if _occupancy_registry is None:
        _occupancy_registry = VisualOccupancyRegistry.load(
            get_settings().orientation_config_path
        )
    return _occupancy_registry


# Clear alias for new code that wants the explicit name.
get_visual_occupancy_registry = get_orientation_registry
