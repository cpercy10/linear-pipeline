"""client.py — send local images to the Motuva server and save what it streams back.

Everything is configured by the DEFAULTS block below — just run:

    python client.py

For each image in INPUT_DIR it POSTs to the server's /process and writes every
artifact the server streams (original, plate, composite, and — when DEBUG — the
crop/cutout/plate-marker debug images + meta.json) into OUTPUT_DIR/<image-stem>/,
saving each file the instant it arrives. Processing is one image at a time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests


INPAINT_ENABLED = False
INPAINT_MODE = "shadow_edge"
INPAINT_STEPS = 28
INPAINT_MAX_EDGE = 1024
FLUX_REFINE_ENABLED = True
FLUX_REFINE_STEPS = 4
FLUX_REFINE_MAX_EDGE = 768
FLUX_REFINE_GUIDANCE = 1.0
FLUX_REFINE_STRENGTH = ""           # "" = model default
FLUX_REFINE_REFERENCE_MODE = "both"  # both | with_reference | composite_only
# ── DEFAULTS (edit these; no CLI args needed) ────────────────────────────────
SERVER_URL = "https://6ab7ri5k4gsatu-8000.proxy.runpod.net"   # your server (e.g. a https://xxxx-8000.proxy.runpod.net URL)
INPUT_DIR = r"C:\Users\SyedZulfiqarHaiderZa\Desktop\cars2"
OUTPUT_DIR = r"C:\Users\SyedZulfiqarHaiderZa\Desktop\cars2\Output-2"     # results land in OUTPUT_DIR/<stem>/
INPAINT_PROMPT  = (
    "Create a natural soft studio contact shadow beneath the tires and repair only "
    "the cutout edge transition into the floor and background. Preserve the car "
    "identity, paint color, silhouette, wheels, lights, glass, trim, and details."
)
FLUX_REFINE_PROMPTS = {
    "composite_only": (
        "Automotive composite repair only. Use the provided image as the locked final "
        "composition and improve only the pasted car integration. Keep the background, "
        "camera viewpoint, framing, road, sky, buildings, signs, and all environment "
        "details unchanged. Repair jagged cutout edges, missing or thin car edge pixels, "
        "small holes, alpha fringing, background bleed-through inside the car, weak tire "
        "grounding, absent contact shadow, color mismatch, exposure mismatch, glass "
        "contamination, paint reflections, and unnatural reflections according to the "
        "visible scene lighting. Preserve the same car silhouette, dimensions, viewing "
        "angle, camera perspective, proportions, body shape, wheel size, wheel position, "
        "ride height, glass shape, grille, lights, badges, trim, license area, model "
        "text, and original paint hue. Do not repaint, recolor, redesign, resize, "
        "rotate, warp, smooth away details, add new objects, remove background objects, "
        "or hallucinate background content."
    ),
    "with_reference": (
        "Automotive composite repair using references. Use the first image as the locked "
        "final composition to improve. Use the gray guide image only for the car's exact "
        "placement, silhouette, scale, tire contact points, and camera perspective. Use "
        "the cropped car reference only for car identity and missing details: same "
        "silhouette, dimensions, viewing angle, proportions, body shape, wheel size, "
        "wheel position, ride height, glass shape, grille, lights, badges, trim, license "
        "area, model text, and original paint hue. Keep the background scene, camera "
        "viewpoint, road, sky, buildings, signs, and environment details unchanged. "
        "Repair jagged edges, missing car parts, edge holes, alpha fringing, background "
        "bleed-through, weak/no contact shadow, color mismatch, exposure mismatch, glass "
        "contamination, old-environment reflections, and unnatural paint reflections. "
        "Remove only unwanted old-environment reflected objects and color contamination "
        "while preserving automotive gloss, metallic tone if present, natural panel "
        "shading, broad gradients, glass tint, chrome, trim, and specular highlights. "
        "Do not repaint, recolor, redesign, resize, rotate, warp, flatten paint, make "
        "it matte, add objects, remove background objects, or hallucinate background details."
    ),
}
INPAINT_SEED     = ""                # "" = random / server default
FLUX_REFINE_SEED = ""                # "" = random / server default
INPAINT_MAX_EDGE = 1024
BODY_OPACITY     = 0.35              # only used by shadow_edge_body
DEBUG      = True                    # True → also receive crop/cutout/plate-marker debug images

# Studio look overrides for /set_studio — set ONCE before the run (empty {} = use the
# server's config default). Valid values: render/studio_engine/configurator/catalog.json
STUDIO = {
    "room":      "flatwall",       # flatwall | cove | curved
    "light":     "panels",         # panels | led | none
    "turntable": "flush",           # flush | raised | none
    "floor":     "concrete_polished_fine",
    "wall":      "paint:1E2022",   # paint:RRGGBB to test wall colours
    "branding":  "none",
}

IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REQUEST_TIMEOUT = 1200              # seconds per image (Blender + remove.bg can be slow)
WAIT_READY_S    = 1200              # how long to wait for the server model to load
# ─────────────────────────────────────────────────────────────────────────────


class _StreamReader:
    """Buffered reader over requests' streamed bytes: readline() + readexactly(n)."""

    def __init__(self, resp: requests.Response, chunk: int = 65536) -> None:
        self._it = resp.iter_content(chunk_size=chunk)
        self._buf = b""

    def _fill(self) -> bool:
        try:
            self._buf += next(self._it)
            return True
        except StopIteration:
            return False

    def readline(self) -> bytes:
        while b"\n" not in self._buf:
            if not self._fill():
                line, self._buf = self._buf, b""
                return line
        i = self._buf.index(b"\n")
        line, self._buf = self._buf[:i], self._buf[i + 1:]
        return line

    def readexactly(self, n: int) -> bytes:
        while len(self._buf) < n:
            if not self._fill():
                break
        data, self._buf = self._buf[:n], self._buf[n:]
        return data


def wait_for_ready(base: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    announced = False
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=15)
            if r.status_code == 200:
                h = r.json()
                if h.get("model_ready"):
                    return h
                if h.get("load_error"):
                    raise SystemExit(f"server failed to load models: {h['load_error']}")
        except requests.RequestException:
            pass
        if not announced:
            print("waiting for the server model to finish loading…", flush=True)
            announced = True
        time.sleep(3)
    raise SystemExit(f"server at {base} not ready within {timeout:.0f}s")


def set_studio(base: str, studio: dict) -> None:
    r = requests.post(f"{base}/set_studio", data={"studio": json.dumps(studio)}, timeout=60)
    r.raise_for_status()
    print(f"studio set: {r.json().get('studio')}", flush=True)


def process_one(base: str, img_path: Path, out_root: Path, debug: bool) -> bool:
    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(img_path, "rb") as f:
        try:
            resp = requests.post(
                f"{base}/process",
                files={"file": (img_path.name, f, "application/octet-stream")},
                data={
                    "debug": "true" if debug else "false",
                    "inpaint": "true" if INPAINT_ENABLED else "false",
                    "inpaint_mode": INPAINT_MODE,
                    "inpaint_prompt": INPAINT_PROMPT,
                    "inpaint_steps": str(INPAINT_STEPS),
                    "inpaint_seed": str(INPAINT_SEED),
                    "inpaint_max_edge": str(INPAINT_MAX_EDGE),
                    "body_opacity": str(BODY_OPACITY),
                    "flux_refine": "true" if FLUX_REFINE_ENABLED else "false",
                    "flux_refine_prompt": FLUX_REFINE_PROMPTS.get(
                        FLUX_REFINE_REFERENCE_MODE, ""
                    ),
                    "flux_refine_steps": str(FLUX_REFINE_STEPS),
                    "flux_refine_seed": str(FLUX_REFINE_SEED),
                    "flux_refine_max_edge": str(FLUX_REFINE_MAX_EDGE),
                    "flux_refine_guidance": str(FLUX_REFINE_GUIDANCE),
                    "flux_refine_strength": str(FLUX_REFINE_STRENGTH),
                    "flux_refine_reference_mode": FLUX_REFINE_REFERENCE_MODE,
                },
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"    request failed: {exc}", flush=True)
            return False
    if resp.status_code != 200:
        print(f"    server {resp.status_code}: {resp.text[:200]}", flush=True)
        return False

    reader = _StreamReader(resp)
    ok = True
    while True:
        line = reader.readline()
        if not line:
            break
        try:
            header = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            print(f"    bad frame header: {line[:120]!r}", flush=True)
            ok = False
            break
        data = reader.readexactly(int(header["size"]))
        name = header["name"]
        (out_dir / name).write_bytes(data)

        meta = header.get("meta") or {}
        tag = ""
        if "orientation" in meta:
            tag = f"   orientation={meta['orientation']}"
        elif "cx" in meta:
            tag = f"   cx,cy=({meta['cx']},{meta['cy']})"
        if name.startswith("error"):
            ok = False
            print(f"    ✗ server error: {meta.get('error', data.decode('utf-8', 'replace'))}", flush=True)
        else:
            print(f"    ← {name} ({len(data)} B){tag}", flush=True)
    return ok


def main() -> None:
    base = SERVER_URL.rstrip("/")
    in_dir = Path(INPUT_DIR)
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in in_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise SystemExit(f"no images found in {in_dir}")

    print(f"connecting to {base} …", flush=True)
    health = wait_for_ready(base, WAIT_READY_S)
    print(f"server ready (gpu={health.get('gpu')}).", flush=True)

    if STUDIO:
        set_studio(base, STUDIO)

    print(f"sending {len(imgs)} images → {base}  (debug={DEBUG})", flush=True)
    done = 0
    for i, img in enumerate(imgs):
        print(f"[{i + 1}/{len(imgs)}] {img.name}", flush=True)
        try:
            done += bool(process_one(base, img, out_root, DEBUG))
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR {exc}", flush=True)
    print(f"\nfinished — ok: {done}/{len(imgs)}   output: {out_root}", flush=True)


if __name__ == "__main__":
    main()
