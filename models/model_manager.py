"""ModelManager — device-aware loading of the small models for the remove-bg pipeline.

Loads the models the exterior-full lane needs, on the single GPU (cuda:0):
  * router + orientation timm classifiers,
  * ONE shared YOLO vehicle detector,
  * the DINOv2 retriever (index loaded once + embedder warmed),
  * GeoCalib (best-effort warm).

FLUX / diffusion has been REMOVED: the exterior-full lane composites the remove.bg car
cutout directly onto the rendered Blender plate, so there is no resident diffusion model
and no VRAM co-residency planning. The warm Blender worker is the only heavy GPU user
and is managed by ``runtime.blender_pool``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config.pipeline_config import PipelineSettings, validate_orientation_azimuth_map
from models.dino_retriever import Retriever, _embedder
from models.timm_classifier import TimmClassifier
from models.yolo_detector import YoloDetector
from processing.exceptions import ModelLoadError
from utils.gpu_monitor import cuda_available, cuda_sync, device_count, list_devices
from utils.logging import get_logger

_log = get_logger("model_manager")

_TORCH_DTYPES = {"float16": torch.float16, "float32": torch.float32}

# The primary (and, on this deployment, only) GPU.
_PRIMARY = "cuda:0"


@dataclass
class ModelManager:
    settings: PipelineSettings
    router: TimmClassifier
    orientation: TimmClassifier
    yolo: YoloDetector
    retriever: Retriever

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def build(cls, settings: PipelineSettings) -> "ModelManager":
        if not cuda_available() or device_count() == 0:
            raise ModelLoadError("No CUDA GPU visible — the pipeline requires a GPU.")

        for d in list_devices():
            _log.info("gpu.detected", index=d.index, name=d.name,
                      total_mb=d.total_mb, free_mb=d.free_mb)

        small_dev = settings.small_model_device or _PRIMARY
        small_dtype = _TORCH_DTYPES[settings.small_model_dtype]

        # ── small models (resident on the small-model device) ────────────────
        _log.info("load.small_models", device=small_dev, dtype=settings.small_model_dtype)
        router = TimmClassifier(
            weights_path=settings.classifier_weights,
            arch=settings.classifier.arch,
            num_classes=settings.classifier.num_classes,
            class_names=settings.classifier.class_names,
            img_size=settings.classifier.img_size,
            norm_mean=settings.classifier.norm_mean,
            norm_std=settings.classifier.norm_std,
            device=small_dev, dtype=small_dtype,
            batch_size=settings.classifier.batch_size, name="router",
        )
        orientation = TimmClassifier(
            weights_path=settings.orientation_weights,
            arch=settings.orientation.arch,
            num_classes=settings.orientation.num_classes,
            class_names=settings.orientation.class_names,
            img_size=settings.orientation.img_size,
            norm_mean=settings.orientation.norm_mean,
            norm_std=settings.orientation.norm_std,
            device=small_dev, dtype=small_dtype,
            batch_size=settings.orientation.batch_size, name="orientation",
        )
        yolo = YoloDetector(
            weights=settings.yolo_weights, device=small_dev,
            classes=settings.yolo.classes, conf=settings.yolo.conf, iou=settings.yolo.iou,
        )

        # DINOv2 retriever (Module 1): load the index once + warm the embedder so the
        # first real image doesn't pay lazy-init cost.
        _log.info("load.retriever", index_dir=str(settings.index_dir))
        retriever = Retriever(index_dir=settings.index_dir)

        # Advisory: validate the 8-label→degrees azimuth table against the index's
        # convention (handedness / front-back). Logs azimuth_map.ok on success or a
        # single azimuth_map.suspect warning naming the bad buckets; it NEVER blocks the
        # run. Toggle with settings.validate_azimuth_on_startup.
        if settings.validate_azimuth_on_startup:
            try:
                rep = validate_orientation_azimuth_map(settings.index_dir)
                if rep.get("ok"):
                    _log.info("azimuth_map.ok", tolerance_deg=rep.get("tolerance_deg"))
                else:
                    per_bucket = rep.get("per_bucket") or {}
                    bad = [k for k, v in per_bucket.items() if not v.get("ok")]
                    _log.warning(
                        "azimuth_map.suspect",
                        reason=rep.get("error", "buckets empty/wide — possible mirror or front/back flip"),
                        bad_buckets=bad,
                        meta_path=rep.get("meta_path"),
                    )
            except Exception as exc:  # noqa: BLE001 — advisory only, never block startup
                _log.warning("azimuth_map.check_failed", error=repr(exc))

        try:
            _embedder()  # warms + resides the DINOv2 model (cached singleton)
            cuda_sync(_PRIMARY)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"[dino] failed to warm DINOv2 embedder: {exc}") from exc

        # GeoCalib (Module 1) is a runtime-only GPU dependency. Warm it best-effort; if
        # the package isn't installed we log and continue (the exterior-full lane will
        # surface the missing dep).
        cls._warm_geocalib()
        cuda_sync(_PRIMARY)

        return cls(
            settings=settings, router=router, orientation=orientation, yolo=yolo,
            retriever=retriever,
        )

    @staticmethod
    def _warm_geocalib() -> None:
        try:
            from models.geocalib_stage import _model
            _model()  # cached singleton; loads GeoCalib weights onto the GPU
            _log.info("load.geocalib", status="warmed")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "load.geocalib.skipped",
                reason=str(exc),
                note="GeoCalib not warmed (package missing?); exterior-full lane "
                     "will fail until it is installed.",
            )
