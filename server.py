"""server.py — Motuva unified pipeline HTTP server (FastAPI, port 8000).

Loads the whole pipeline ONCE (ModelManager + warm Blender worker) and serves per-image
processing as a STREAMING response: each artifact is sent the instant it is produced
(length-prefixed), so the client writes files into ``OUTPUT/<stem>/`` one by one as they
appear. Processing is STRICTLY SERIAL — one image fully done (plate → composite →
returned) before the next — enforced by a global lock.

Exterior-full first composites the rembg car cutout directly onto the rendered plate,
then can optionally run a final FLUX.2 Klein image-edit refinement pass.

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
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator, Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config.pipeline_config import ImageClass, get_orientation_registry, get_settings
from models.model_manager import ModelManager
from pipeline import _camera_export_payload
from processing.flux_inpaint import FluxFillInpainter, default_request
from processing.flux_refine import FluxKleinRefiner, default_refine_request
from processing import interior, partial as partial_lane
from processing.background_remover import BackgroundRemover
from processing.rembg_segmenter import RembgSegmenter
from runtime.blender_pool import BlenderPool
from render.blender_worker import BlenderWorker
from stages import composite_refine, exterior_full, inpaint_masks
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
    rembg_segmenter=None,    # RembgSegmenter (server exterior-full only)
    inpainter=None,          # FluxFillInpainter (lazy, optional legacy path)
    flux_refiner=None,       # FluxKleinRefiner (lazy final image-edit pass)
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


def _gray_yolo_guide(pre, canvas_size, paste_top_left) -> Image.Image:
    """Gray full-frame guide with the resized YOLO crop at the final placement."""
    guide = Image.new("RGB", canvas_size, tuple(settings.canvas.fill_color))
    guide.paste(pre.resized_crop.convert("RGB"), paste_top_left)
    return guide


def _lock_background_after_flux(
    base: Image.Image,
    refined: Image.Image,
    inputs: inpaint_masks.InpaintInputs,
) -> Image.Image:
    """Keep prompt edits on the car; allow only shadow-like darkening on the floor."""
    solid_car = composite_refine.solidify_mask(
        inputs.car_mask,
        threshold=8,
        close_px=7,
        fill_holes=True,
    )
    car_core = solid_car.filter(ImageFilter.GaussianBlur(0.6))
    narrow_edge = composite_refine.mask_subtract(
        composite_refine.dilate_mask(solid_car, 7),
        composite_refine.erode_mask(solid_car, 1),
    ).filter(ImageFilter.GaussianBlur(1.2))
    narrow_edge = narrow_edge.point(lambda p: int(p * 0.72))
    car_edit = composite_refine.mask_union(car_core, narrow_edge)

    contact_context = Image.new("L", base.size, 0)
    bbox = solid_car.getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        car_w = max(1, x1 - x0)
        car_h = max(1, y1 - y0)
        pad_x = max(36, int(car_w * 0.075))
        above = max(4, int(car_h * 0.025))
        below = max(40, int(car_h * 0.14))
        draw = ImageDraw.Draw(contact_context)
        draw.rounded_rectangle(
            [
                max(0, x0 - pad_x),
                max(0, y1 - above),
                min(base.size[0], x1 + pad_x),
                min(base.size[1], y1 + below),
            ],
            radius=max(10, int(car_h * 0.025)),
            fill=255,
        )
        contact_context = contact_context.filter(
            ImageFilter.GaussianBlur(max(5.0, car_h * 0.014))
        )

    base_rgb = base.convert("RGB")
    refined_rgb = refined.resize(base.size, Image.LANCZOS).convert("RGB")
    merged = Image.composite(refined_rgb, base_rgb, car_edit)

    base_arr = np.array(base_rgb, dtype=np.float32)
    refined_arr = np.array(refined_rgb, dtype=np.float32)
    merged_arr = np.array(merged, dtype=np.float32)
    contact = np.array(contact_context, dtype=np.float32) / 255.0
    car_protect = np.array(
        solid_car.filter(ImageFilter.GaussianBlur(1.0)),
        dtype=np.float32,
    ) / 255.0
    contact *= 1.0 - car_protect

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    base_luma = np.dot(base_arr, weights)
    refined_luma = np.dot(refined_arr, weights)
    shadow = np.clip((base_luma - refined_luma) / 42.0, 0.0, 1.0) * contact
    shadow = np.clip(shadow, 0.0, 0.65)
    shadow_rgb = np.minimum(base_arr, refined_arr)
    merged_arr = merged_arr * (1.0 - shadow[:, :, None])
    merged_arr += shadow_rgb * shadow[:, :, None]

    return Image.fromarray(
        merged_arr.clip(0, 255).astype(np.uint8),
        mode="RGB",
    )


def _prompt_refine_base_without_shadow(
    plate: Image.Image,
    inputs: inpaint_masks.InpaintInputs,
) -> Image.Image:
    """Clean rembg cutout on the rendered plate, before prompt-generated shadow."""
    base = plate.convert("RGBA")
    base.alpha_composite(inputs.cutout_resized, inputs.paste_top_left)
    return base.convert("RGB")


def _meta(pre, plate, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    e = pre.estimate
    meta = {
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
    if extra:
        meta.update(extra)
    return meta


def _frame(name: str, data: bytes, ctype: str = "image/png",
           meta: Optional[dict] = None) -> bytes:
    """One streamed artifact: a JSON header line + the raw bytes."""
    header = json.dumps(
        {"name": name, "ctype": ctype, "size": len(data), "meta": meta or {}}
    ).encode("utf-8")
    return header + b"\n" + data


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _inpaint_request_from_form(
    *,
    inpaint: Optional[str],
    inpaint_mode: Optional[str],
    inpaint_prompt: Optional[str],
    inpaint_steps: Optional[str],
    inpaint_seed: Optional[str],
    inpaint_max_edge: Optional[str],
    body_opacity: Optional[str],
):
    req = default_request(settings)
    req.enabled = _parse_bool(inpaint, req.enabled)
    if inpaint_mode not in (None, ""):
        if inpaint_mode not in {"shadow", "shadow_edge", "shadow_edge_body"}:
            raise ValueError(
                "inpaint_mode must be one of: shadow, shadow_edge, shadow_edge_body"
            )
        req.mode = inpaint_mode
    if inpaint_prompt not in (None, ""):
        req.prompt = inpaint_prompt
    req.num_steps = int(_parse_int(inpaint_steps, req.num_steps))
    req.seed = _parse_int(inpaint_seed, req.seed)
    req.max_long_edge = int(_parse_int(inpaint_max_edge, req.max_long_edge))
    req.body_opacity = _parse_float(body_opacity, req.body_opacity)
    return req


def _flux_prompt_for_mode(mode: str) -> str:
    cfg = settings.flux_refine
    if cfg.prompt:
        return cfg.prompt
    if mode == "composite_only":
        return cfg.prompt_composite_only
    return cfg.prompt_with_reference


def _flux_refine_request_from_form(
    *,
    flux_refine: Optional[str],
    flux_refine_prompt: Optional[str],
    flux_refine_steps: Optional[str],
    flux_refine_seed: Optional[str],
    flux_refine_max_edge: Optional[str],
    flux_refine_guidance: Optional[str],
    flux_refine_strength: Optional[str],
    flux_refine_reference_mode: Optional[str],
):
    req = default_refine_request(settings)
    req.enabled = _parse_bool(flux_refine, req.enabled)
    if flux_refine_reference_mode not in (None, ""):
        if flux_refine_reference_mode not in {"both", "with_reference", "multi_reference", "composite_only"}:
            raise ValueError(
                "flux_refine_reference_mode must be one of: both, with_reference, composite_only"
            )
        req.reference_mode = flux_refine_reference_mode
    if flux_refine_prompt not in (None, ""):
        req.prompt = flux_refine_prompt
    else:
        req.prompt = _flux_prompt_for_mode(req.reference_mode)
    req.num_steps = int(_parse_int(flux_refine_steps, req.num_steps))
    req.seed = _parse_int(flux_refine_seed, req.seed)
    req.max_long_edge = int(_parse_int(flux_refine_max_edge, req.max_long_edge))
    req.guidance_scale = _parse_float(flux_refine_guidance, req.guidance_scale)
    if flux_refine_strength not in (None, ""):
        req.strength = _parse_float(flux_refine_strength, req.strength or 0.0)
    return req


def _flux_refine_modes(req) -> list[str]:
    if not req.enabled:
        return []
    if req.reference_mode == "both":
        return ["composite_only", "with_reference"]
    return [req.reference_mode]


def _flux_refine_req_for_mode(req, mode: str):
    prompt = req.prompt
    if req.reference_mode == "both" and req.prompt == _flux_prompt_for_mode("both"):
        prompt = _flux_prompt_for_mode(mode)
    return replace(req, reference_mode=mode, prompt=prompt)


def _flux_output_name(mode: str) -> str:
    if mode == "composite_only":
        return "40_final_klein_no_reference_prompt_edges_glass_shadow.png"
    if mode in {"with_reference", "multi_reference"}:
        return "41_final_klein_with_reference_restore_parts_edges_glass_shadow.png"
    return f"40_final_klein_{mode}_prompt_edges_glass_shadow.png"


def _flux_output_description(mode: str) -> str:
    if mode == "composite_only":
        return (
            "Final FLUX.2 Klein cleanup using only the current composite: prompt fixes "
            "edges, glass contamination, old reflections, and contact shadow."
        )
    if mode in {"with_reference", "multi_reference"}:
        return (
            "Final FLUX.2 Klein cleanup using composite plus clean background plate, "
            "placement guide, and source-car references: prompt can restore missing "
            "source details such as mirrors, replace window/reflection content with "
            "the current background, then fix edges and contact shadow."
        )
    return f"Final FLUX.2 Klein cleanup in {mode!r} mode."


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

async def _process_stream(
    content: bytes,
    filename: str,
    debug: bool,
    inpaint_req,
    flux_req,
) -> AsyncIterator[bytes]:
    stem = Path(filename).stem
    ext = (Path(filename).suffix or ".jpg").lower()
    orig_ctype = "image/png" if ext == ".png" else "image/jpeg"
    log = get_logger("server").bind(request_id=filename)
    loop = asyncio.get_running_loop()

    async with PROCESS_LOCK:                       # one image fully done before the next
        try:
            # original — always, first (the client already has it, but we echo it so the
            # output folder is self-contained).
            yield _frame("00_source_original" + ext, content, orig_ctype)

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
                        yield _frame("01_debug_yolo_source_crop.png", _img_bytes(pre.raw_crop),
                                     meta={"note": "YOLO bbox crop before resize"})
                    yield _frame(
                        "02_debug_yolo_resized_crop_with_pose.png", _img_bytes(_label_crop(pre.resized_crop, pre)),
                        meta={"orientation": pre.orientation.value,
                              "orientation_confidence": round(pre.orientation_confidence, 4)},
                    )

                # tier 2 — warm Blender plate render
                plate = await _render_plate(pre, stem, log)
                yield _frame("10_scene_empty_blender_plate.jpg", _img_bytes(plate.plate, "JPEG"), "image/jpeg")  # always
                if debug:
                    yield _frame(
                        "11_debug_scene_plate_center_marked.jpg",
                        _img_bytes(_mark_plate(plate.plate, plate.cx, plate.cy), "JPEG"),
                        "image/jpeg",
                        meta={"cx": round(plate.cx, 1), "cy": round(plate.cy, 1)},
                    )

                # tier 3 — rembg cutout + manual composite + optional FLUX.2 edit refine.
                rembg_result = await loop.run_in_executor(
                    S.pre_exec,
                    partial(S.rembg_segmenter.remove, pre.raw_crop),
                )
                inputs = await loop.run_in_executor(
                    S.pre_exec,
                    partial(
                        inpaint_masks.build_inputs,
                        pre,
                        plate,
                        rembg_result.cutout,
                        settings=settings,
                        mode=inpaint_req.mode,
                        log=log,
                    ),
                )

                gray_guide = _gray_yolo_guide(
                    pre,
                    plate.plate.size,
                    inputs.paste_top_left,
                )

                if debug:
                    yield _frame("20_debug_rembg_raw_cutout.png", _img_bytes(rembg_result.cutout),
                                 meta={"model": rembg_result.model_name})
                    yield _frame("21_debug_cutout_clean_alpha_glass_edges.png", _img_bytes(inputs.cutout_resized),
                                 meta={"note": "resized matte after alpha, edge, glass, and color cleanup"})
                    yield _frame("22_reference_gray_placement_guide.png", _img_bytes(gray_guide),
                                 meta={"note": "resized YOLO crop at final placement"})
                    if pre.raw_crop is not None:
                        yield _frame("23_reference_source_car_crop.png", _img_bytes(pre.raw_crop),
                                     meta={"note": "original YOLO crop reference"})
                    yield _frame("30_composite_local_cutout_with_placeholder_shadow.png", _img_bytes(inputs.manual_composite),
                                 meta={"note": "local rembg cleanup before final prompt-based Klein refinement"})

                prompt_refine_base = _prompt_refine_base_without_shadow(
                    plate.plate,
                    inputs,
                )
                if debug:
                    yield _frame("31_composite_prompt_input_no_shadow.png", _img_bytes(prompt_refine_base),
                                 meta={"note": "clean cutout on new plate; final shadow/glass/edge cleanup is prompt-generated by Klein"})

                composite = inputs.manual_composite
                if inpaint_req.enabled:
                    inpainted = await loop.run_in_executor(
                        S.pre_exec,
                        partial(S.inpainter.inpaint, inputs.manual_composite,
                                inputs.inpaint_mask, inpaint_req),
                    )
                    composite = await loop.run_in_executor(
                        S.pre_exec,
                        partial(
                            inpaint_masks.merge_inpaint_result,
                            inputs.manual_composite,
                            inpainted,
                            inputs,
                            mode=inpaint_req.mode,
                            body_opacity=inpaint_req.body_opacity,
                        ),
                    )
                    if debug:
                        yield _frame("35_optional_flux_fill_mask_repair_not_final.png", _img_bytes(composite),
                                     meta={"note": "optional old FLUX Fill stage before final Klein prompt refinement"})

                flux_outputs: Dict[str, Image.Image] = {}
                flux_prompts: Dict[str, str] = {}
                base_for_flux = prompt_refine_base if flux_req.enabled else composite
                for refine_mode in _flux_refine_modes(flux_req):
                    mode_req = _flux_refine_req_for_mode(flux_req, refine_mode)
                    flux_prompts[refine_mode] = mode_req.prompt
                    refined = await loop.run_in_executor(
                        S.pre_exec,
                        partial(
                            S.flux_refiner.refine,
                            base_for_flux,
                            gray_guide,
                            pre.raw_crop,
                            mode_req,
                            plate.plate,
                        ),
                    )
                    refined = _lock_background_after_flux(base_for_flux, refined, inputs)
                    flux_outputs[refine_mode] = refined
                    yield _frame(
                        _flux_output_name(refine_mode),
                        _img_bytes(refined),
                        meta={
                            "flux_refine_reference_mode": refine_mode,
                            "description": _flux_output_description(refine_mode),
                            "prompt": mode_req.prompt,
                            "background_reference": "10_scene_empty_blender_plate.jpg",
                            "background_lock": "tight_car_window_edge_plus_shadow_only_floor_darkening",
                        },
                    )

                selected_output = ""
                if flux_outputs:
                    selected_output = (
                        "with_reference"
                        if "with_reference" in flux_outputs
                        else next(reversed(flux_outputs.keys()))
                    )
                    composite = (
                        flux_outputs.get("with_reference")
                        or next(reversed(flux_outputs.values()))
                    )

                meta_extra = {
                    "rembg_model": rembg_result.model_name,
                    "inpaint_enabled": bool(inpaint_req.enabled),
                    "inpaint_mode": inpaint_req.mode,
                    "inpaint_prompt": inpaint_req.prompt,
                    "inpaint_seed": inpaint_req.seed,
                    "inpaint_steps": inpaint_req.num_steps,
                    "inpaint_max_long_edge": inpaint_req.max_long_edge,
                    "body_opacity": inpaint_req.body_opacity,
                    "flux_refine_enabled": bool(flux_req.enabled),
                    "flux_refine_model": settings.flux_refine.model_id,
                    "flux_refine_prompt": flux_req.prompt,
                    "flux_refine_seed": flux_req.seed,
                    "flux_refine_steps": flux_req.num_steps,
                    "flux_refine_max_long_edge": flux_req.max_long_edge,
                    "flux_refine_guidance_scale": flux_req.guidance_scale,
                    "flux_refine_strength": flux_req.strength,
                    "flux_refine_reference_mode": flux_req.reference_mode,
                    "flux_refine_background_reference": "10_scene_empty_blender_plate.jpg",
                    "flux_refine_modes": list(flux_outputs.keys()),
                    "flux_refine_prompts_by_mode": flux_prompts,
                    "final_klein_input": (
                        "31_composite_prompt_input_no_shadow.png"
                        if flux_req.enabled
                        else "30_composite_local_cutout_with_placeholder_shadow.png"
                    ),
                    "selected_final_source": selected_output or "local_composite",
                    "output_legend": {
                        "10_scene_empty_blender_plate.jpg": "clean rendered target background plate; with-reference Klein also receives this for glass/reflections",
                        "21_debug_cutout_clean_alpha_glass_edges.png": "local cleaned cutout before final Klein",
                        "22_reference_gray_placement_guide.png": "placement guide for car silhouette, scale, and tire contact points",
                        "23_reference_source_car_crop.png": "source car crop for identity, paint color, trim, mirrors, and missing details",
                        "30_composite_local_cutout_with_placeholder_shadow.png": "local cleaned rembg cutout on rendered plate before final prompt-based Klein refinement",
                        "31_composite_prompt_input_no_shadow.png": "actual clean composite sent into final Klein; shadow/reflection/window cleanup is prompt-generated",
                        "35_optional_flux_fill_mask_repair_not_final.png": "optional old FLUX Fill repair stage; not the final output",
                        "40_final_klein_no_reference_prompt_edges_glass_shadow.png": _flux_output_description("composite_only"),
                        "41_final_klein_with_reference_restore_parts_edges_glass_shadow.png": _flux_output_description("with_reference"),
                        "99_final_selected.png": "selected final output; prefers with-reference when both modes are requested",
                        "composite.png": "legacy alias of 99_final_selected.png",
                    },
                    "mask_pixels": inputs.mask_stats,
                    "paste_top_left": list(inputs.paste_top_left),
                    "placed_px": list(inputs.placed_size),
                    "clamped": inputs.clamped,
                }
                final_meta = _meta(pre, plate, meta_extra)
                yield _frame("99_final_selected.png", _img_bytes(composite),
                             meta=final_meta)  # always
                yield _frame("composite.png", _img_bytes(composite),
                             meta={**final_meta, "alias_of": "99_final_selected.png"})
                if debug:
                    yield _frame("meta.json",
                                 json.dumps(final_meta, indent=2).encode("utf-8"),
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
            S.rembg_segmenter = RembgSegmenter(settings)
            S.inpainter = FluxFillInpainter(settings)
            S.flux_refiner = FluxKleinRefiner(settings)
            S.pre_exec = ThreadPoolExecutor(
                max_workers=settings.preprocess_workers, thread_name_prefix="pre"
            )
            worker = BlenderWorker(settings)
            S.blender_pool = BlenderPool(worker)
            if not settings.blender.release_after_render:
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
        if S.rembg_segmenter is not None:
            await S.rembg_segmenter.aclose()
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
async def process(
    file: UploadFile = File(...),
    debug: bool = Form(False),
    inpaint: Optional[str] = Form(None),
    inpaint_mode: Optional[str] = Form(None),
    inpaint_prompt: Optional[str] = Form(None),
    inpaint_steps: Optional[str] = Form(None),
    inpaint_seed: Optional[str] = Form(None),
    inpaint_max_edge: Optional[str] = Form(None),
    body_opacity: Optional[str] = Form(None),
    flux_refine: Optional[str] = Form(None),
    flux_refine_prompt: Optional[str] = Form(None),
    flux_refine_steps: Optional[str] = Form(None),
    flux_refine_seed: Optional[str] = Form(None),
    flux_refine_max_edge: Optional[str] = Form(None),
    flux_refine_guidance: Optional[str] = Form(None),
    flux_refine_strength: Optional[str] = Form(None),
    flux_refine_reference_mode: Optional[str] = Form(None),
):
    if not S.ready:
        detail = S.load_error or "model is still loading — retry shortly"
        raise HTTPException(503, detail)
    try:
        inpaint_req = _inpaint_request_from_form(
            inpaint=inpaint,
            inpaint_mode=inpaint_mode,
            inpaint_prompt=inpaint_prompt,
            inpaint_steps=inpaint_steps,
            inpaint_seed=inpaint_seed,
            inpaint_max_edge=inpaint_max_edge,
            body_opacity=body_opacity,
        )
        flux_req = _flux_refine_request_from_form(
            flux_refine=flux_refine,
            flux_refine_prompt=flux_refine_prompt,
            flux_refine_steps=flux_refine_steps,
            flux_refine_seed=flux_refine_seed,
            flux_refine_max_edge=flux_refine_max_edge,
            flux_refine_guidance=flux_refine_guidance,
            flux_refine_strength=flux_refine_strength,
            flux_refine_reference_mode=flux_refine_reference_mode,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    content = await file.read()
    return StreamingResponse(
        _process_stream(content, file.filename or "image.jpg", debug, inpaint_req, flux_req),
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
        "rembg": {
            "model": settings.rembg.model_name,
            "loaded": bool(S.rembg_segmenter and S.rembg_segmenter.loaded),
        },
        "inpaint": {
            "enabled_default": bool(settings.inpaint.enabled),
            "model": settings.inpaint.model_id,
            "mode_default": settings.inpaint.mode,
            "loaded": bool(S.inpainter and S.inpainter.loaded),
        },
        "flux_refine": {
            "enabled_default": bool(settings.flux_refine.enabled),
            "model": settings.flux_refine.model_id,
            "reference_mode_default": settings.flux_refine.reference_mode,
            "loaded": bool(S.flux_refiner and S.flux_refiner.loaded),
        },
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
