"""Canvas / crop / placement ops (Stages 3 + 5) and diffusion-input helpers.

The placement math is reproduced from the user's reference `preprocess_car`,
parameterized by fill_ratio + vertical_anchor so the SAME function serves both:
  * the NEUTRAL canvas fed to the orientation model (fixed training values, the
    hardcoded constants below), and
  * the FINAL canvas fed to diffusion (per-orientation values from config YAML).

>>> The NEUTRAL constants below MUST match the canvas your orientation model was
    trained on. They are intentionally NOT in config (per project decision). If your
    orientation-training canvas generator used a different scaling rule than the
    reference `preprocess_car` (e.g. a different never-shrink clamp), adjust
    `place_car_on_canvas` accordingly. <<<
"""

from __future__ import annotations

from typing import Tuple

from PIL import Image

from models.yolo_detector import BBox

# ── Fixed NEUTRAL placement (orientation-model input). Do not move to config. ──
ORIENTATION_INPUT_FILL_RATIO: float = 0.80
ORIENTATION_INPUT_VERTICAL_ANCHOR: float = 0.50

# ── Kontext-only stitching prefix (used only by the kontext_dev backend) ──────
KONTEXT_PREFIX = (
    "This image has two halves: the LEFT half is the first image (the car), "
    "the RIGHT half is the second image (the background environment). "
)


def crop_box_with_padding(
    image_size: Tuple[int, int], bbox: BBox, padding: float
) -> Tuple[int, int, int, int]:
    """The padded crop box ``(x1, y1, x2, y2)`` in IMAGE pixels (clamped to bounds).

    The single source of truth for the padding math — both ``crop_with_padding`` (to
    cut the crop) and the ground-contact anchor (to convert a wheel point in image
    pixels into crop coordinates) read it, so the two can never drift."""
    w, h = image_size
    x1, y1, x2, y2 = bbox
    px = (x2 - x1) * padding
    py = (y2 - y1) * padding
    return (max(0, int(x1 - px)), max(0, int(y1 - py)),
            min(w, int(x2 + px)), min(h, int(y2 + py)))


def crop_with_padding(image: Image.Image, bbox: BBox, padding: float) -> Image.Image:
    """Crop `bbox` from `image` with `padding` (fraction of bbox) on each side,
    clamped to image bounds."""
    return image.crop(crop_box_with_padding(image.size, bbox, padding))


def place_car_on_canvas(
    car_crop: Image.Image,
    canvas_size: Tuple[int, int],
    fill_color: Tuple[int, int, int],
    fill_ratio: float,
    vertical_anchor: float,
) -> Image.Image:
    """Scale the car crop to `fill_ratio` of the limiting canvas edge (never shrinks
    below native size, matching the reference), center it horizontally, and anchor it
    vertically so the car's CENTER sits at `vertical_anchor` (fraction from top).
    Integer pixel arithmetic at paste time (no float coords)."""
    canvas_w, canvas_h = canvas_size
    car_w, car_h = car_crop.size

    scale = max(fill_ratio * min(canvas_w / car_w, canvas_h / car_h), 1.0)
    new_w = int(car_w * scale)
    new_h = int(car_h * scale)
    resized = car_crop.resize((new_w, new_h), Image.LANCZOS)

    x_offset = (canvas_w - new_w) // 2
    y_offset = int(canvas_h * vertical_anchor - new_h / 2)
    y_offset = max(0, min(canvas_h - new_h, y_offset))  # keep the car on-canvas

    canvas = Image.new("RGB", (canvas_w, canvas_h), fill_color)
    canvas.paste(resized, (x_offset, y_offset))
    return canvas


def build_orientation_canvas(
    car_crop: Image.Image,
    canvas_size: Tuple[int, int],
    fill_color: Tuple[int, int, int],
) -> Image.Image:
    """NEUTRAL canvas for the orientation model — fixed training placement."""
    return place_car_on_canvas(
        car_crop, canvas_size, fill_color,
        fill_ratio=ORIENTATION_INPUT_FILL_RATIO,
        vertical_anchor=ORIENTATION_INPUT_VERTICAL_ANCHOR,
    )


def build_final_canvas(
    car_crop: Image.Image,
    canvas_size: Tuple[int, int],
    fill_color: Tuple[int, int, int],
    fill_ratio: float,
    vertical_anchor: float,
) -> Image.Image:
    """FINAL canvas for diffusion — per-orientation placement from config."""
    return place_car_on_canvas(
        car_crop, canvas_size, fill_color,
        fill_ratio=fill_ratio, vertical_anchor=vertical_anchor,
    )


# ── diffusion working-resolution helpers (from the reference) ─────────────────

def snap(n: int, multiple: int = 32) -> int:
    """Snap down to a multiple (diffusion models want dims divisible by 32/64)."""
    return max(multiple, (n // multiple) * multiple)


def fit_longest_edge(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest edge == max_side (snapped), preserving aspect ratio.
    Never upscales."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if w >= h:
        new_w = snap(int(w * scale))
        new_h = snap(round(new_w * h / w))
    else:
        new_h = snap(int(h * scale))
        new_w = snap(round(new_h * w / h))
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.LANCZOS)


def stitch_for_kontext(car_img: Image.Image, bg_img: Image.Image) -> Image.Image:
    """Stitch car (left) + background (right) at equal height (Kontext backend only)."""
    car_w, car_h = car_img.size
    bg_w, bg_h = bg_img.size
    scale = car_h / bg_h
    new_bg_w = snap(round(bg_w * scale))
    bg_sized = bg_img.resize((new_bg_w, car_h), Image.LANCZOS)
    combined = Image.new("RGB", (car_w + new_bg_w, car_h))
    combined.paste(car_img, (0, 0))
    combined.paste(bg_sized, (car_w, 0))
    return combined
