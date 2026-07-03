"""Generic timm image classifier.

Both pipeline classifiers — the 3-way router and the 8-way orientation model —
are the same architecture (`tf_efficientnetv2_l.in21k`) with the same transform,
so they share this one wrapper (instantiated twice). Fixed-size input means it is
batchable, which is the throughput win for these stages.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from timm import create_model
from torchvision import transforms

from processing.exceptions import ModelLoadError
from utils.gpu_monitor import oom_guard


@dataclass(frozen=True)
class Prediction:
    index: int
    label: str
    confidence: float


class TimmClassifier:
    def __init__(
        self,
        weights_path: Path,
        arch: str,
        num_classes: int,
        class_names: Sequence[str],
        img_size: int,
        norm_mean: Tuple[float, float, float],
        norm_std: Tuple[float, float, float],
        device: str,
        dtype: torch.dtype,
        batch_size: int,
        name: str = "classifier",
    ) -> None:
        if len(class_names) != num_classes:
            raise ModelLoadError(
                f"[{name}] class_names ({len(class_names)}) != num_classes ({num_classes})"
            )
        self.name = name
        self.device = device
        self.dtype = dtype
        self.class_names = list(class_names)
        self.batch_size = batch_size
        # Shared across preprocess-pool threads → serialize inference for safety.
        self._lock = threading.Lock()

        try:
            model = create_model(arch, pretrained=False, num_classes=num_classes)
            state = torch.load(weights_path, map_location=device)
            # tolerate checkpoints saved as {"state_dict": ...} or raw state dicts
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001 — surface as a clean pipeline error
            raise ModelLoadError(f"[{name}] failed to load {weights_path}: {exc}") from exc

        self.model = model.eval().to(device=device, dtype=dtype)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(norm_mean), std=list(norm_std)),
        ])

    def _to_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        tensors = [self.transform(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors, dim=0)
        return batch.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict(self, images: Sequence[Image.Image]) -> List[Prediction]:
        """Classify a list of images. Internally chunked by `batch_size`."""
        results: List[Prediction] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            batch = self._to_batch(chunk)
            with self._lock, oom_guard(stage=f"{self.name}.infer", device=self.device):
                logits = self.model(batch)
            probs = torch.softmax(logits.float(), dim=1)
            conf, idx = probs.max(dim=1)
            for i, c in zip(idx.tolist(), conf.tolist()):
                results.append(Prediction(index=i, label=self.class_names[i], confidence=float(c)))
        return results

    def predict_one(self, image: Image.Image) -> Prediction:
        return self.predict([image])[0]
