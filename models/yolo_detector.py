"""YOLO car detection — returns the largest vehicle bbox (Stage 2)."""

from __future__ import annotations

import threading
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

from processing.exceptions import ModelLoadError
from utils.gpu_monitor import oom_guard

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


class YoloDetector:
    def __init__(
        self,
        weights: str,
        device: str,
        classes: Sequence[int],
        conf: float,
        iou: float,
    ) -> None:
        self.device = device
        self.classes = list(classes)
        self.conf = conf
        self.iou = iou
        # Shared across preprocess-pool threads → serialize inference for safety.
        self._lock = threading.Lock()
        try:
            self.model = YOLO(weights)
            self.model.to(device)
            # warmup so the first real image isn't paying lazy-init cost
            self.model(
                np.zeros((640, 640, 3), dtype=np.uint8),
                classes=self.classes, device=device, verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"[yolo] failed to load {weights}: {exc}") from exc

    def detect_largest(self, image: Image.Image) -> Optional[BBox]:
        """Return the largest detected vehicle bbox, or None if no vehicle found."""
        arr = np.asarray(image.convert("RGB"))
        with self._lock, oom_guard(stage="yolo.infer", device=self.device):
            results = self.model(
                arr, classes=self.classes, conf=self.conf, iou=self.iou,
                device=self.device, verbose=False,
            )
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        x1, y1, x2, y2 = xyxy[int(np.argmax(areas))]
        return int(x1), int(y1), int(x2), int(y2)
