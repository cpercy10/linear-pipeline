"""Processing: validation, canvas/bbox ops, background removal, per-lane processors."""

from processing.exceptions import (
    BackgroundRemovalError,
    BlenderRenderError,
    ConfigError,
    DetectionError,
    InputValidationError,
    ModelLoadError,
    PipelineError,
    PipelineVRAMError,
)

__all__ = [
    "PipelineError",
    "ConfigError",
    "ModelLoadError",
    "InputValidationError",
    "DetectionError",
    "BackgroundRemovalError",
    "BlenderRenderError",
    "PipelineVRAMError",
]
