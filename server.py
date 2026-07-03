"""server.py — Motuva unified pipeline HTTP server (FastAPI, port 8000).

Loads the whole pipeline ONCE (ModelManager + warm Blender worker) and serves per-image
processing as a STREAMING response: each artifact is sent the instant it is produced
(length-prefixed), so the client writes files into ``OUTPUT/<stem>/`` one by one as they
appear. Processing is STRICTLY SERIAL — one image fully done (plate → composite →
returned) before the next — enforced by a global lock.

FLUX has been removed: exterior-full composites the remove.bg car cutout directly onto
the rendered plate.

Endpoints
---------
  POST /set_studio   form: studio (JSON)   → overrides the active studio look (persists)
  GET  /get_studio
  POST /process      file + form: debug    → streamed artifacts (see _process_stream)
  GET  /health

Streaming wire format (one frame per artifact):
  <json header line>\\n<raw bytes>
where header = {"name","ctype","size","meta"}. The client reads the header line, then
exactly ``size`` bytes. Body bytes may contain \\n (only the header is line-delimited).

Artifacts per image
-------------------
  exterior-full, always:  original, plate, composite
  exterior-full, debug:   + crop_raw, crop_resized (orientation drawn on it),
                          plate_marked (BIG cx,cy marker), cutout, meta.json
  interior / partial:     original, composite  (debug images are exterior-full only)

Run:
    cd motuva_pipeline
    python server.py
"""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator, Dict, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont

from config.pipeline_config import ImageClass, get_orientation_registry, get_settings
from models.model_manager import ModelManager
from pipeline import _camera_export_payload
from processing import interior, partial as partial_lane
from processing.background_remover import BackgroundRemover
from runtime.blender_pool import BlenderPool
from render.blender_worker import BlenderWorker
from stages import exterior_full
from utils.logging import configure_logging, get_logger, stage_timer

PORT = 8000

settings = get_settings()
_log = get_logger("server")

# Strictly-serial processing: one image fully done before the next.
PROCESS_LOCK = asyncio.Lock()

# Shared server state (filled by the background loader).
S = SimpleNamespace(
    ready=False,
    manager=None,            # ModelManager
    blender_pool=None,       # BlenderPool (owns the warm worker)
    pre_exec=None,           # ThreadPoolExecutor (preprocess tier)
    remover=None,            # BackgroundRemover (all remove.bg lanes)
    registry=None,           # VisualOccupancyRegistry
    bg_rgba=None,            # supplied background for the partial lane (optional)
    active_studio=dict(settings.blender.studio),   # studio look, overridable via /set_studio
    load_error=None,
)


# ─────────────────────────────────────────────────────────────────────────────
# Small image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _img_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    if fmt.upper() in ("JPEG", "JPG"):
        img.convert("RGB").save(buf, format="JPEG", quality=95)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _label_crop(crop: Image.Image, pre) -> Image.Image:
    """Draw the orientation prediction on the resized crop (helps diagnose azimuth)."""
    im = crop.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    snapped = getattr(pre.estimate, "azimuth_snapped", False)
    txt = (f"{pre.orientation.value}  conf {pre.orientation_confidence:.2f}  "
           f"az {pre.camera.get('azimuth', 0):.0f}{'  SNAP' if snapped else ''}")
    fs = max(16, im.width // 32)
    font = _font(fs)
    try:
        tw = d.textlength(txt, font=font)
    except Exception:  # noqa: BLE001 — older Pillow
        tw = fs * len(txt) * 0.6
    d.rectangle([0, 0, tw + 14, fs + 14], fill=(0, 0, 0))
    d.text((7, 6), txt, fill=(0, 255, 0), font=font)
    return im


def _mark_plate(plate: Image.Image, cx: float, cy: float) -> Image.Image:
    """Draw a BIG, clearly visible marker at the plate's (cx,cy) car-spot."""
    im = plate.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    r = max(14, max(im.size) // 40)           # big
    w = max(3, r // 4)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=w)
    d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0))
    d.line([cx - 2 * r, cy, cx + 2 * r, cy], fill=(255, 0, 0), width=max(2, r // 6))
    d.line([cx, cy - 2 * r, cx, cy + 2 * r], fill=(255, 0, 0), width=max(2, r // 6))
    return im


def _meta(pre, plate) -> Dict[str, object]:
    e = pre.estimate
    return {
        "orientation": pre.orientation.value,
        "orientation_confidence": round(pre.orientation_confidence, 4),
        "azimuth": pre.camera.get("azimuth"),
        "azimuth_snapped": getattr(e, "azimuth_snapped", None),
        "azimuth_reason": getattr(e, "azimuth_reason", None),
        "elevation": pre.camera.get("elevation"),
        "distance": pre.camera.get("distance"),
        "focal": pre.camera.get("focal"),
        "confidence": getattr(e, "confidence", None),
        "cx": round(plate.cx, 1),
        "cy": round(plate.cy, 1),
        "disc_m": round(pre.disc_m, 3),
        "pose": pre.pose,
        "recovered_length_m": round(pre.recovered_length_m, 2),
    }


def _frame(name: str, data: bytes, ctype: str = "image/png",
           meta: Optional[dict] = None) -> bytes:
    """One streamed artifact: a JSON header line + the raw bytes."""
    header = json.dumps(
        {"name": name, "ctype": ctype, "size": len(data), "meta": meta or {}}
    ).encode("utf-8")
    return header + b"\n" + data


# ─────────────────────────────────────────────────────────────────────────────
# Plate render (mirrors pipeline.Pipeline._render_plate, reusing the shared helpers)
# ─────────────────────────────────────────────────────────────────────────────

async def _render_plate(pre, stem: str, log):
    w, h = int(pre.orig_size[0]), int(pre.orig_size[1])
    out_jpg = Path(tempfile.gettempdir()) / "motuva_plates" / f"{stem}_plate.jpg"
    out_json = out_jpg.with_suffix(".json")
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    meta = await S.blender_pool.render(
        _camera_export_payload(pre.camera),
        pre.disc_m,
        str(out_jpg),
        str(out_json),
        photo_w=w,
        photo_h=h,
        studio=S.active_studio,
        long_edge=max(w, h),
        samples=settings.blender.render_samples,
    )
    plate_img = Image.open(out_jpg)
    plate_img.load()
    return exterior_full._coerce_plate_result((plate_img, meta), log)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image streaming processor (STRICTLY SERIAL via PROCESS_LOCK)
# ─────────────────────────────────────────────────────────────────────────────

async def _process_stream(content: bytes, filename: str, debug: bool) -> AsyncIterator[bytes]:
    stem = Path(filename).stem
    ext = (Path(filename).suffix or ".jpg").lower()
    orig_ctype = "image/png" if ext == ".png" else "image/jpeg"
    log = get_logger("server").bind(request_id=filename)
    loop = asyncio.get_running_loop()

    async with PROCESS_LOCK:                       # one image fully done before the next
        try:
            # original — always, first (the client already has it, but we echo it so the
            # output folder is self-contained).
            yield _frame("original" + ext, content, orig_ctype)

            image = Image.open(io.BytesIO(content)).convert("RGB")

            # classify → lane
            pred = await loop.run_in_executor(
                S.pre_exec, lambda: S.manager.router.predict_one(image)
            )
            lane = ImageClass(pred.label)
            log.info("route", lane=lane.value, confidence=round(pred.confidence, 4))

            if lane is ImageClass.EXTERIOR_FULL:
                # GeoCalib (inside preprocess) loads the full frame from disk → temp file.
                tmp_in = Path(tempfile.gettempdir()) / "motuva_in" / f"{stem}{ext}"
                tmp_in.parent.mkdir(parents=True, exist_ok=True)
                tmp_in.write_bytes(content)

                # tier 1 — preprocess
                pre = await loop.run_in_executor(
                    S.pre_exec,
                    partial(
                        exterior_full.preprocess, image, str(tmp_in),
                        yolo=S.manager.yolo,
                        orientation_model=S.manager.orientation,
                        retriever=S.manager.retriever,
                        settings=settings,
                        registry=S.registry,
                        log=log,
                    ),
                )
                if debug:
                    if pre.raw_crop is not None:
                        yield _frame("crop_raw.png", _img_bytes(pre.raw_crop),
                                     meta={"note": "YOLO bbox crop before resize"})
                    yield _frame(
                        "crop_resized.png", _img_bytes(_label_crop(pre.resized_crop, pre)),
                        meta={"orientation": pre.orientation.value,
                              "orientation_confidence": round(pre.orientation_confidence, 4)},
                    )

                # tier 2 — warm Blender plate render
                plate = await _render_plate(pre, stem, log)
                yield _frame("plate.jpg", _img_bytes(plate.plate, "JPEG"), "image/jpeg")  # always
                if debug:
                    yield _frame(
                        "plate_marked.jpg",
                        _img_bytes(_mark_plate(plate.plate, plate.cx, plate.cy), "JPEG"),
                        "image/jpeg",
                        meta={"cx": round(plate.cx, 1), "cy": round(plate.cy, 1)},
                    )

                # tier 3 — remove.bg cutout + manual composite onto the plate (no FLUX)
                crop_bytes = _img_bytes(pre.raw_crop, "PNG")
                cutout = await S.remover.remove(
                    crop_bytes, filename,
                    add_shadow=settings.removebg.exterior_add_shadow,
                )
                if debug:
                    cut_dbg = cutout.convert("RGBA").resize(pre.resized_crop.size, Image.LANCZOS)
                    yield _frame("cutout.png", _img_bytes(cut_dbg),
                                 meta={"note": "remove.bg cutout, occupancy-resized"})
                composite = await loop.run_in_executor(
                    S.pre_exec,
                    partial(exterior_full.composite_on_plate, pre, plate, cutout,
                            settings=settings, log=log),
                )
                yield _frame("composite.png", _img_bytes(composite), meta=_meta(pre, plate))  # always
                if debug:
                    yield _frame("meta.json",
                                 json.dumps(_meta(pre, plate), indent=2).encode("utf-8"),
                                 "application/json")

            elif lane is ImageClass.INTERIOR:
                out = await interior.process_interior(
                    content, filename, remover=S.remover, settings=settings, log=log
                )
                yield _frame("composite.png", _img_bytes(out))

            else:  # EXTERIOR_PARTIAL
                if S.bg_rgba is None:
                    raise RuntimeError(
                        "exterior-partial image needs a background — set "
                        "MOTOCUT_BACKGROUND_IMAGE on the server."
                    )
                out = await partial_lane.process_partial(
                    content, filename, remover=S.remover,
                    background_rgba=S.bg_rgba, settings=settings, log=log,
                )
                yield _frame("composite.png", _img_bytes(out))

            log.info("image.done", lane=lane.value)

        except Exception as exc:  # noqa: BLE001 — surface as a final error frame
            log.error("process.error", error=repr(exc), exc_info=True)
            msg = repr(exc).encode("utf-8")
            yield _frame("error.txt", msg, "text/plain", meta={"error": repr(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load models in the background so the port binds immediately
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=settings.log_level, environment=settings.environment)
    S.registry = get_orientation_registry()        # fail fast on a bad YAML
    loop = asyncio.get_running_loop()

    async def _load():
        try:
            from concurrent.futures import ThreadPoolExecutor
            _log.info("server.loading")
            S.manager = await loop.run_in_executor(None, ModelManager.build, settings)
            S.remover = BackgroundRemover(settings)
            S.pre_exec = ThreadPoolExecutor(
                max_workers=settings.preprocess_workers, thread_name_prefix="pre"
            )
            worker = BlenderWorker(settings)
            S.blender_pool = BlenderPool(worker)
            await S.blender_pool.start()
            if settings.background_image and Path(settings.background_image).exists():
                S.bg_rgba = Image.open(settings.background_image).convert("RGBA")
            S.ready = True
            _log.info("server.ready")
        except Exception as exc:  # noqa: BLE001
            S.load_error = repr(exc)
            _log.error("server.load_failed", error=repr(exc), exc_info=True)

    asyncio.create_task(_load())
    yield
    # shutdown
    try:
        if S.blender_pool is not None:
            await S.blender_pool.shutdown()
        if S.remover is not None:
            await S.remover.aclose()
        if S.pre_exec is not None:
            S.pre_exec.shutdown(wait=False)
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(title="Motuva Unified Pipeline Server", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/set_studio")
async def set_studio(studio: str = Form(...)):
    try:
        overrides = json.loads(studio) if studio.strip() else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"studio must be a JSON object: {exc}")
    if not isinstance(overrides, dict):
        raise HTTPException(400, "studio must be a JSON object of {key: value}.")
    # Active look = config default + the keys you sent. Send {} to reset to default.
    S.active_studio = {**settings.blender.studio,
                       **{k: v for k, v in overrides.items() if v not in (None, "")}}
    _log.info("studio.updated", studio=S.active_studio)
    return {"status": "ok", "studio": S.active_studio}


@app.get("/get_studio")
async def get_studio():
    return {"studio": S.active_studio}


@app.post("/process")
async def process(file: UploadFile = File(...), debug: bool = Form(False)):
    if not S.ready:
        detail = S.load_error or "model is still loading — retry shortly"
        raise HTTPException(503, detail)
    content = await file.read()
    return StreamingResponse(
        _process_stream(content, file.filename or "image.jpg", debug),
        media_type="application/octet-stream",
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": S.ready,
        "load_error": S.load_error,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "studio": S.active_studio,
        "background_set": S.bg_rgba is not None,
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
