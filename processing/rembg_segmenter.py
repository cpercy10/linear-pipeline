"""Server-only rembg segmentation with session reuse."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional

from PIL import Image

from config.pipeline_config import PipelineSettings
from processing.exceptions import BackgroundRemovalError, ConfigError
from utils.logging import get_logger

_log = get_logger("rembg")


@dataclass
class RembgResult:
    cutout: Image.Image
    model_name: str


class RembgSegmenter:
    """Lazy rembg wrapper used by the server exterior-full experiment."""

    def __init__(self, settings: PipelineSettings) -> None:
        self._cfg = settings.rembg
        self._session = None
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._cfg.model_name

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            try:
                from rembg import new_session
            except ImportError as exc:
                raise ConfigError(
                    "rembg is not installed; install the server experiment dependencies"
                ) from exc
            except SystemExit as exc:
                raise ConfigError(
                    "rembg could not load an onnxruntime backend. Install rembg[cpu] "
                    "or a CUDA-compatible onnxruntime-gpu build."
                ) from exc
            _log.info("rembg.loading", model=self._cfg.model_name)
            self._session = new_session(self._cfg.model_name)
            _log.info("rembg.ready", model=self._cfg.model_name)
            return self._session

    def remove(self, image: Image.Image) -> RembgResult:
        """Return an RGBA cutout for a PIL image crop."""
        try:
            from rembg import remove
        except ImportError as exc:
            raise ConfigError(
                "rembg is not installed; install the server experiment dependencies"
            ) from exc
        except SystemExit as exc:
            raise ConfigError(
                "rembg could not load an onnxruntime backend. Install rembg[cpu] "
                "or a CUDA-compatible onnxruntime-gpu build."
            ) from exc

        session = self._ensure_session()
        try:
            out = remove(
                image.convert("RGB"),
                session=session,
                alpha_matting=self._cfg.alpha_matting,
            )
        except Exception as exc:  # noqa: BLE001
            raise BackgroundRemovalError(f"rembg failed: {exc}") from exc

        if not isinstance(out, Image.Image):
            raise BackgroundRemovalError("rembg returned a non-image result")
        return RembgResult(cutout=out.convert("RGBA"), model_name=self._cfg.model_name)

    async def aclose(self) -> None:
        """Symmetric with BackgroundRemover; rembg has no async resource to close."""
        self._session = None
