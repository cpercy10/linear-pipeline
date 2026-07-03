"""Custom exception hierarchy for the pipeline.

Every stage wraps its work and re-raises as one of these so the runner can decide
to skip, retry, or abort a single image without taking down the batch.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class ConfigError(PipelineError):
    """Misconfiguration: missing file, bad YAML, missing orientation entry, etc."""


class ModelLoadError(PipelineError):
    """A model failed to load (bad weights path, arch mismatch, OOM at load)."""


class InputValidationError(PipelineError):
    """Input image is missing, unreadable, wrong format, or out of size bounds."""


class DetectionError(PipelineError):
    """YOLO found no vehicle in an image routed to the exterior_full lane."""


class BackgroundRemovalError(PipelineError):
    """remove.bg call failed after exhausting retries."""


class PipelineVRAMError(PipelineError):
    """A GPU op ran out of memory. Raised after flushing cache + logging stats so
    the caller can retry the single image or reject it."""


class BlenderRenderError(PipelineError):
    """The warm Blender worker failed to render a plate (crash, timeout, nonzero
    exit, or a malformed/missing response). The host client raises this so the
    runner can skip/retry the single image; the worker auto-restarts on crash."""
