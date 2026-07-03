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

# ── DEFAULTS (edit these; no CLI args needed) ────────────────────────────────
SERVER_URL = "https://adwkhfepqo7bxo-8000.proxy.runpod.net"   # your server (e.g. a https://xxxx-8000.proxy.runpod.net URL)
INPUT_DIR  = r"E:\CorTechSols\Motorcut\Background Removal\Background-Replacement-Removebg-Latest\Ali-Cortechsols\Latest\Results\Images\Set 02"    # folder of images to send
OUTPUT_DIR = r"D:\Motocut\Code\motuva-pipline-removebgappraoch\Images\Output-1"     # results land in OUTPUT_DIR/<stem>/
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
                data={"debug": "true" if debug else "false"},
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
