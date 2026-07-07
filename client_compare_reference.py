"""Run Motuva reference-mode comparison in one server request per image.

This is an A/B testing client for the server's final FLUX refine pass.

Outputs are written as:

    OUTPUT_DIR/<image-stem>/40_final_klein_no_reference_prompt_edges_glass_shadow.png
    OUTPUT_DIR/<image-stem>/41_final_klein_with_reference_restore_parts_edges_glass_shadow.png

The default `both` mode makes the server run the heavy pipeline once, then reuse the
already-loaded FLUX refiner to produce both `composite_only` and `with_reference`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


# Defaults you can edit, or override from the command line.
SERVER_URL = "https://6ab7ri5k4gsatu-8000.proxy.runpod.net"
INPUT_DIR = r"C:\Users\SyedZulfiqarHaiderZa\Desktop\cars2"
OUTPUT_DIR = r"C:\Users\SyedZulfiqarHaiderZa\Desktop\cars2\Reference-Compare1"

# Ask the server for both outputs in one request by default.
REFERENCE_MODES = ("both",)

# Turn this on only if you also want the older mask-based FLUX Fill inpaint pass.
INPAINT_ENABLED = False
INPAINT_MODE = "shadow_edge"
INPAINT_STEPS = 28
INPAINT_SEED = ""
INPAINT_MAX_EDGE = 1024
BODY_OPACITY = 0.35

# Main A/B pass: FLUX.2 Klein final image-edit refine.
FLUX_REFINE_ENABLED = True
FLUX_REFINE_STEPS = 6
FLUX_REFINE_SEED = "1234"  # fixed seed makes A/B comparisons less noisy; "" = random
FLUX_REFINE_MAX_EDGE = 1024
FLUX_REFINE_GUIDANCE = 1.0
FLUX_REFINE_STRENGTH = ""  # "" = server/model default
FLUX_REFINE_PROMPT = ""    # "" = server picks the prompt for each reference mode

# Debug true saves plate, masks, guide, car reference, manual composite, and final output.
DEBUG = True

STUDIO = {
    "room": "flatwall",
    "light": "panels",
    "turntable": "flush",
    "floor": "concrete_polished_fine",
    "wall": "paint:1E2022",
    "branding": "none",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REQUEST_TIMEOUT = 1200
WAIT_READY_S = 1200


class _StreamReader:
    """Buffered reader over requests' streamed bytes."""

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


def _bool_form(value: bool) -> str:
    return "true" if value else "false"


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
            print("waiting for the server model to finish loading...", flush=True)
            announced = True
        time.sleep(3)
    raise SystemExit(f"server at {base} not ready within {timeout:.0f}s")


def set_studio(base: str, studio: dict) -> None:
    if not studio:
        return
    r = requests.post(f"{base}/set_studio", data={"studio": json.dumps(studio)}, timeout=60)
    r.raise_for_status()
    print(f"studio set: {r.json().get('studio')}", flush=True)


def _request_form(reference_mode: str, debug: bool) -> Dict[str, str]:
    form = {
        "debug": _bool_form(debug),
        "inpaint": _bool_form(INPAINT_ENABLED),
        "inpaint_mode": INPAINT_MODE,
        "inpaint_steps": str(INPAINT_STEPS),
        "inpaint_seed": str(INPAINT_SEED),
        "inpaint_max_edge": str(INPAINT_MAX_EDGE),
        "body_opacity": str(BODY_OPACITY),
        "flux_refine": _bool_form(FLUX_REFINE_ENABLED),
        "flux_refine_steps": str(FLUX_REFINE_STEPS),
        "flux_refine_seed": str(FLUX_REFINE_SEED),
        "flux_refine_max_edge": str(FLUX_REFINE_MAX_EDGE),
        "flux_refine_guidance": str(FLUX_REFINE_GUIDANCE),
        "flux_refine_strength": str(FLUX_REFINE_STRENGTH),
        "flux_refine_reference_mode": reference_mode,
    }
    if FLUX_REFINE_PROMPT:
        form["flux_refine_prompt"] = FLUX_REFINE_PROMPT
    return form


def process_one_mode(
    base: str,
    img_path: Path,
    out_dir: Path,
    reference_mode: str,
    debug: bool,
) -> Tuple[bool, Optional[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"    mode={reference_mode}", flush=True)

    with img_path.open("rb") as f:
        try:
            resp = requests.post(
                f"{base}/process",
                files={"file": (img_path.name, f, "application/octet-stream")},
                data=_request_form(reference_mode, debug),
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            print(f"      request failed: {exc}", flush=True)
            return False, {"error": str(exc)}

    if resp.status_code != 200:
        msg = f"server {resp.status_code}: {resp.text[:300]}"
        print(f"      {msg}", flush=True)
        return False, {"error": msg}

    reader = _StreamReader(resp)
    ok = True
    final_meta: Optional[dict] = None
    while True:
        line = reader.readline()
        if not line:
            break
        try:
            header = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            print(f"      bad frame header: {line[:120]!r}", flush=True)
            return False, {"error": "bad frame header"}

        data = reader.readexactly(int(header["size"]))
        name = str(header.get("name", "frame.bin"))
        meta = header.get("meta") or {}
        (out_dir / name).write_bytes(data)

        if name == "composite.png":
            final_meta = meta
        if name.startswith("error"):
            ok = False
            text = data.decode("utf-8", "replace")
            print(f"      server error: {meta.get('error', text)}", flush=True)
        else:
            print(f"      <- {name} ({len(data)} B)", flush=True)

    if final_meta is not None:
        (out_dir / "final_meta.json").write_text(
            json.dumps(final_meta, indent=2),
            encoding="utf-8",
        )
    return ok, final_meta


def iter_images(in_dir: Path) -> List[Path]:
    return sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run composite_only and with_reference server tests side by side."
    )
    parser.add_argument("--server", default=SERVER_URL, help="server base URL")
    parser.add_argument("--input", default=INPUT_DIR, help="input image directory")
    parser.add_argument("--output", default=OUTPUT_DIR, help="output directory")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(REFERENCE_MODES),
        choices=("both", "composite_only", "with_reference", "multi_reference"),
        help="reference modes to run",
    )
    parser.add_argument("--no-debug", action="store_true", help="save only normal frames")
    parser.add_argument("--skip-studio", action="store_true", help="do not call /set_studio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = str(args.server).rstrip("/")
    in_dir = Path(args.input)
    out_root = Path(args.output)
    debug = not bool(args.no_debug)

    if not in_dir.exists():
        raise SystemExit(f"input directory not found: {in_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    imgs = iter_images(in_dir)
    if not imgs:
        raise SystemExit(f"no images found in {in_dir}")

    print(f"connecting to {base}...", flush=True)
    health = wait_for_ready(base, WAIT_READY_S)
    print(
        "server ready "
        f"(gpu={health.get('gpu')}, "
        f"flux_refine={health.get('flux_refine')})",
        flush=True,
    )

    if not args.skip_studio:
        set_studio(base, STUDIO)

    summary = {
        "server": base,
        "input": str(in_dir),
        "output": str(out_root),
        "debug": debug,
        "modes": list(args.modes),
        "images": {},
    }

    print(f"processing {len(imgs)} image(s) with modes: {', '.join(args.modes)}", flush=True)
    for i, img in enumerate(imgs, start=1):
        print(f"[{i}/{len(imgs)}] {img.name}", flush=True)
        image_summary = {}
        for mode in args.modes:
            mode_dir = out_root / img.stem if mode == "both" else out_root / img.stem / mode
            ok, meta = process_one_mode(base, img, mode_dir, mode, debug)
            image_summary[mode] = {
                "ok": bool(ok),
                "output_dir": str(mode_dir),
                "meta": meta or {},
            }
        summary["images"][img.name] = image_summary

    (out_root / "_compare_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"done. outputs: {out_root}", flush=True)
    print(f"summary: {out_root / '_compare_summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
