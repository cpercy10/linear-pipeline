"""Stage 1 — input validation. Reject bad input before touching the GPU."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError

from processing.exceptions import InputValidationError

# Decompression-bomb guard is handled by Pillow's MAX_IMAGE_PIXELS; we set a
# generous cap and surface a clean error instead of a warning/crash.
Image.MAX_IMAGE_PIXELS = 200_000_000


def validate_extension(path: Path, allowed: Sequence[str]) -> None:
    if path.suffix.lower() not in allowed:
        raise InputValidationError(
            f"{path.name}: unsupported extension '{path.suffix}' (allowed: {list(allowed)})"
        )


def load_image(path: Path, allowed_exts: Sequence[str]) -> Image.Image:
    """Validate + decode an input image to RGB. Raises InputValidationError on any
    problem (missing, unreadable, wrong format)."""
    if not path.exists():
        raise InputValidationError(f"{path}: file does not exist")
    validate_extension(path, allowed_exts)
    try:
        with Image.open(path) as im:
            im.load()  # force decode now so errors surface here, not downstream
            return im.convert("RGB")
    except UnidentifiedImageError as exc:
        raise InputValidationError(f"{path.name}: not a valid image") from exc
    except OSError as exc:
        raise InputValidationError(f"{path.name}: failed to read ({exc})") from exc


def load_image_from_bytes(image_bytes: bytes, filename: str) -> Image.Image:
    """Validate + decode in-memory image bytes (e.g. an HTTP upload) to RGB.
    Extension is not enforced here — content is what matters for an upload."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.load()
            return im.convert("RGB")
    except UnidentifiedImageError as exc:
        raise InputValidationError(f"{filename}: not a valid image") from exc
    except OSError as exc:
        raise InputValidationError(f"{filename}: failed to decode ({exc})") from exc
