"""Stage 1 — DINOv2 embedding + kNN pose retrieval (Module 1, merged + adapted).

Merges Module 1's ``perspective-estimation/embedding.py`` (DINOv2 embed + YOLO
car-crop) and ``retrieval.py`` (cosine kNN over the synthetic index, soft-averaged
to a continuous camera estimate with azimuth averaged CIRCULARLY) into one module.

The embedding, car-crop, cosine-kNN, and softmax circular-averaging maths are kept
EXACTLY as Module 1 had them. The only adaptations are plumbing:

  * config is read from the unified ``PipelineSettings`` (``retrieval`` sub-config +
    ``yolo_weights``) instead of Module 1's flat ``config`` module;
  * the ``Retriever`` loads ``emb.npy`` / ``meta.json`` from the configured in-package
    index dir (``settings.index_dir``);
  * ``embed(img, do_crop=...)`` is exposed so the unified caller can pass the SHARED
    YOLO crop with ``do_crop=False`` (avoiding a second, redundant detection). The
    ``do_crop=True`` path keeps Module 1's own ``car_crop`` for standalone use / the
    index builder.

The car-crop here makes the embedding key on VEHICLE POSE rather than the
background, which is what lets a synthetic library match real dealer photos.
"""
import os
import json
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import torch
import timm

from config.pipeline_config import get_settings

_YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

_CONT = ["elevation_deg", "distance_m", "cam_height_m", "focal_mm", "roll_deg"]


def _device() -> str:
    """The device DINOv2 lives on — the SAME device as the other small models.

    Read from ``settings.small_model_device`` so the retriever CO-LOCATES with the
    router/orientation/YOLO and its VRAM footprint is measured on the same device the
    capacity planner subtracts from (model_manager loads small models on
    ``small_model_device`` and measures the footprint on cuda:0). A module-level
    ``"cuda"`` constant would silently pin DINOv2 to cuda:0 regardless of where the
    other small models ran, corrupting the VRAM plan on a multi-GPU host.
    """
    dev = (get_settings().small_model_device or "").strip()
    if dev:
        return dev
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _embedder():
    cfg = get_settings().retrieval
    model = timm.create_model(cfg.dino_model, pretrained=True, num_classes=0,
                              dynamic_img_size=True)
    model.eval().to(_device())
    data_cfg = timm.data.resolve_data_config({}, model=model)
    data_cfg["input_size"] = (3, cfg.embed_size, cfg.embed_size)
    return model, timm.data.create_transform(**data_cfg, is_training=False)


@lru_cache(maxsize=1)
def _yolo():
    """Module 1's own YOLO for the ``do_crop=True`` path (standalone / index build).

    The unified per-image flow shares ONE detector and passes a pre-crop with
    ``do_crop=False``, so this lazy instance is only ever loaded if someone calls
    ``embed(..., do_crop=True)`` (or ``Retriever.estimate(..., do_crop=True)``).
    """
    from ultralytics import YOLO
    w = os.environ.get("MOTUVA_YOLO") or get_settings().yolo_weights
    if not w or not os.path.exists(str(w)):
        w = "yolov8s.pt"      # auto-download fallback (e.g. on a fresh pod)
    return YOLO(str(w))


def car_crop(pil, pad=None):
    """Crop to the largest vehicle; return the original if none detected."""
    cfg = get_settings().retrieval
    if pad is None:
        pad = cfg.crop_pad
    try:
        r = _yolo().predict(pil.convert("RGB"), verbose=False, conf=0.25, device=_YOLO_DEVICE)[0]
        best = None
        for b in r.boxes:
            if int(b.cls[0]) in cfg.vehicle_classes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                area = (x2 - x1) * (y2 - y1)
                if best is None or area > best[4]:
                    best = (x1, y1, x2, y2, area)
        if best is None:
            return pil
        x1, y1, x2, y2, _ = best
        W, H = pil.size; bw, bh = x2 - x1, y2 - y1
        return pil.crop((int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh)),
                         int(min(W, x2 + pad * bw)), int(min(H, y2 + pad * bh))))
    except Exception:
        return pil


@torch.no_grad()
def embed(pil, do_crop=True):
    """Return an L2-normalized DINOv2 embedding of the (optionally cropped) image.

    Pass ``do_crop=False`` with an already-cropped image (e.g. the shared YOLO
    crop) to skip the internal detection.
    """
    model, tf = _embedder()
    img = car_crop(pil) if do_crop else pil
    x = tf(img.convert("RGB")).unsqueeze(0).to(_device())
    f = torch.nn.functional.normalize(model(x), dim=1)
    return f[0].cpu().numpy().astype("float32")


@torch.no_grad()
def embed_batch(pils, do_crop=True):
    """Embed a list of PIL images in one forward pass (used by the index builder)."""
    model, tf = _embedder()
    imgs = [(car_crop(p) if do_crop else p).convert("RGB") for p in pils]
    x = torch.stack([tf(im) for im in imgs]).to(_device())
    f = torch.nn.functional.normalize(model(x), dim=1)
    return f.cpu().numpy().astype("float32")


@dataclass
class RetrievalResult:
    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    cam_height_m: float
    focal_mm: float
    roll_deg: float
    top1_sim: float
    mean_topk_sim: float
    elev_spread_deg: float
    neighbours: list = field(default_factory=list)


class Retriever:
    def __init__(self, index_dir=None):
        if index_dir is None:
            index_dir = get_settings().index_dir
        index_dir = str(index_dir)
        self.E = np.load(os.path.join(index_dir, "emb.npy")).astype("float32")
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def estimate(self, pil, k=None, temp=None, do_crop=True) -> RetrievalResult:
        """k-NN pose estimate for ``pil``.

        Pass ``do_crop=False`` with a pre-cropped image (the shared YOLO crop) so
        retrieval keys on the same crop the rest of the pipeline uses.
        """
        cfg = get_settings().retrieval
        if k is None:
            k = cfg.k_neighbours
        if temp is None:
            temp = cfg.softmax_temp

        q = embed(pil, do_crop=do_crop)
        sims = self.E @ q                                  # cosine (both normalized)
        idx = np.argsort(-sims)[:k]
        s = sims[idx]
        w = np.exp((s - s.max()) / temp); w = w / w.sum()
        cams = [self.meta[i]["camera"] for i in idx]

        vals = {key: float(np.sum(w * np.array([c[key] for c in cams]))) for key in _CONT}
        rad = np.deg2rad([c["azimuth_deg"] for c in cams])
        az = float(np.rad2deg(np.arctan2(np.sum(w * np.sin(rad)),
                                         np.sum(w * np.cos(rad)))) % 360.0)
        neighbours = [{"sim": round(float(sims[i]), 4),
                       "azimuth_deg": self.meta[i]["camera"]["azimuth_deg"],
                       "elevation_deg": self.meta[i]["camera"]["elevation_deg"]} for i in idx]

        return RetrievalResult(
            azimuth_deg=round(az, 2),
            elevation_deg=round(vals["elevation_deg"], 2),
            distance_m=round(vals["distance_m"], 2),
            cam_height_m=round(vals["cam_height_m"], 2),
            focal_mm=round(vals["focal_mm"], 2),
            roll_deg=round(vals["roll_deg"], 2),
            top1_sim=round(float(s[0]), 4),
            mean_topk_sim=round(float(s.mean()), 4),
            elev_spread_deg=round(float(np.std([c["elevation_deg"] for c in cams])), 2),
            neighbours=neighbours,
        )
