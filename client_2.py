"""client_2.py — studio-shuffle testing client for the Motuva server.

Like ``client.py``, but for each input image it RANDOMISES the studio floor + wall
(room / turntable / branding are fixed in ``STUDIO_BASE`` below) and saves ONLY the
final composite to ``OUTPUT_DIR/<image-stem>.png`` — flat, no per-image subfolders.

Floor/wall choices are drawn from the package catalog
(``render/studio_engine/configurator/catalog.json``), so only real materials are
used. Selection is independent-random per image (repeats allowed).

Error handling (precision): if a render fails, the offending floor OR wall is
identified and BLACKLISTED — the GOOD side is kept and only the culprit is re-rolled.
Blacklists PERSIST across runs in ``material_blacklist.json`` (loaded at startup), so a
known-bad material is never re-tried in a later run either.
  * First we try to read the culprit straight out of the error message.
  * If that's inconclusive, an isolation render against the known-good default
    (``SAFE_FLOOR`` / ``SAFE_WALL``) pinpoints which side is bad.
  * If an image fails even with the known-good default floor+wall, the problem is the
    IMAGE itself (not the materials) → it is logged and SKIPPED.
  * Transient server/HTTP errors are treated as transient → the image is skipped, never
    blacklisted (so a server hiccup can't poison the persistent list).

Set-once note: the server's studio look is a single global; this client is strictly
sequential (one image fully done before the next), so calling /set_studio per image is
safe. Don't run two copies against the same server at once.

    python client_2.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import requests

# ── DEFAULTS (edit these; no CLI args) ───────────────────────────────────────
SERVER_URL = "https://adwkhfepqo7bxo-8000.proxy.runpod.net"   # your server (e.g. a https://xxxx-8000.proxy.runpod.net URL)
INPUT_DIR  = r"E:\CorTechSols\Motorcut\Background Removal\Background-Replacement-Removebg-Latest\Ali-Cortechsols\Latest\Results\Images\Set 02"    # folder of images to send
OUTPUT_DIR = r"D:\Motocut\Code\motuva-pipline-removebgappraoch\Images\Output-1"     # results land in OUTPUT_DIR/<stem>/

# Studio knobs YOU control (floor + wall are randomised per image; lighting stays at
# the server's config default since it isn't sent).
STUDIO_BASE = {
    "room":      "flatwall",
    "branding":  "none",
    "turntable": "flush",          # flush = on (visible ring); also valid: raised | none
}

# Known-good baseline — used ONLY to pinpoint a bad material / detect image-level
# errors. These are the server's config defaults and are assumed reliable.
SAFE_FLOOR = "concrete_polished_fine"
SAFE_WALL  = "paint:1E2022"

MAX_RETRIES  = 6                   # random-pair attempts per image before giving up
CATALOG_PATH = (Path(__file__).resolve().parent
                / "render" / "studio_engine" / "configurator" / "catalog.json")
# Persistent per-side blacklist of materials that errored — loaded at startup and
# appended atomically as new bad materials are found, so future runs skip them with no
# rediscovery cost. Human-editable JSON: delete an entry to give a material another go.
BLACKLIST_PATH = Path(__file__).resolve().parent / "material_blacklist.json"

IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REQUEST_TIMEOUT = 1200             # seconds per image (Blender + remove.bg can be slow)
WAIT_READY_S    = 1200             # how long to wait for the server model to load
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


def load_catalog(path: Path):
    """Return (floor_keys, wall_keys) from the package catalog — the only valid set."""
    if not path.exists():
        raise SystemExit(f"catalog not found: {path}")
    cat = json.loads(path.read_text(encoding="utf-8"))
    floors = [f["key"] for f in cat.get("floors", []) if f.get("key")]
    walls = [w["key"] for w in cat.get("walls", []) if w.get("key")]
    if not floors or not walls:
        raise SystemExit(f"catalog has no floors/walls: {path}")
    return floors, walls


def load_blacklist(path: Path) -> dict:
    """Load the persistent per-side blacklist as {"floors": {key: meta}, "walls": {...}}.
    A missing/corrupt file just starts empty (never blocks the run)."""
    bl = {"floors": {}, "walls": {}}
    if not path.exists():
        return bl
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for side in ("floors", "walls"):
            entries = data.get(side, {})
            if isinstance(entries, dict):
                bl[side] = {str(k): v for k, v in entries.items()}
            elif isinstance(entries, list):          # tolerate a bare list of keys
                bl[side] = {str(k): {} for k in entries}
    except Exception as exc:  # noqa: BLE001 — a corrupt file must not stop the run
        print(f"warning: could not read blacklist {path}: {exc} (starting empty)", flush=True)
    return bl


def save_blacklist(path: Path, blacklist: dict) -> None:
    """Atomic write (temp file + replace) so an interrupted run can't corrupt the list."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blacklist, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def add_to_blacklist(blacklist: dict, path: Path, side: str, key: str,
                     reason: str, image: str) -> None:
    """Blacklist a material (with an audit reason + triggering image) and persist now."""
    if key in blacklist[side]:
        return
    blacklist[side][key] = {"reason": reason, "image": image}
    save_blacklist(path, blacklist)


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


def render_once(base: str, img_path: Path, studio: dict):
    """Set the studio look, run /process, return (composite_bytes | None, error | None).

    Only the composite frame is captured; original/plate frames are consumed and
    ignored. ``debug=false`` so the server streams the minimum.
    """
    set_studio(base, studio)
    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{base}/process",
            files={"file": (img_path.name, f, "application/octet-stream")},
            data={"debug": "false"},
            stream=True,
            timeout=REQUEST_TIMEOUT,
        )
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"

    reader = _StreamReader(resp)
    composite = None
    error = None
    while True:
        line = reader.readline()
        if not line:
            break
        try:
            header = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            error = f"bad frame header: {line[:120]!r}"
            break
        data = reader.readexactly(int(header["size"]))
        name = header.get("name", "")
        if name.startswith("composite"):
            composite = data
        elif name.startswith("error"):
            meta = header.get("meta") or {}
            error = meta.get("error") or data.decode("utf-8", "replace")
    return composite, error


def attribute_culprit(error: str, floor: str, wall: str):
    """Best-effort: does the error message name the floor or the wall key?
    Returns 'floor' | 'wall' | None (None → fall back to an isolation render)."""
    if not error:
        return None
    e = error.lower()
    f_in = floor.lower() in e
    w_in = wall.lower() in e
    if f_in and not w_in:
        return "floor"
    if w_in and not f_in:
        return "wall"
    return None


def _pick(pool, blacklist):
    choices = [x for x in pool if x not in blacklist]
    return random.choice(choices) if choices else None


def process_image(base, img_path, out_dir, floors, walls, blacklist, bl_path):
    """Render one image with a random floor+wall, retrying + blacklisting on errors.

    On a material error the GOOD side is kept and only the culprit is re-rolled; the
    culprit is added to the persistent ``blacklist`` (saved immediately).
    Returns ('ok', floor, wall) | ('skip', reason).
    """
    stem = img_path.stem
    image_known_ok = False     # have we confirmed the image renders with safe materials?
    F = None
    W = None                   # kept across attempts; only the blamed side is reset to None

    for attempt_i in range(1, MAX_RETRIES + 1):
        if F is None:
            F = _pick(floors, blacklist["floors"])
        if W is None:
            W = _pick(walls, blacklist["walls"])
        if F is None or W is None:
            return ("skip", "no non-blacklisted floors/walls left")

        print(f"    try {attempt_i}: floor={F}  wall={W}", flush=True)
        comp, err = render_once(base, img_path, {**STUDIO_BASE, "floor": F, "wall": W})
        if comp is not None:
            (out_dir / f"{stem}.png").write_bytes(comp)
            print(f"    ✓ saved {stem}.png  (floor={F}, wall={W})", flush=True)
            return ("ok", F, W)

        # Transient server/HTTP errors are NOT material problems → skip, never blacklist
        # (so a server hiccup can't poison the persistent list).
        if err and err.startswith("HTTP"):
            return ("skip", f"transient server error: {err}")

        # ── attribute the failure, keeping the good side ─────────────────────
        culprit = attribute_culprit(err, F, W)
        if culprit == "floor":
            add_to_blacklist(blacklist, bl_path, "floors", F, "error names floor", img_path.name)
            print(f"    ✗ error names floor → blacklist floor {F} (keep wall)", flush=True)
            F = None
            continue
        if culprit == "wall":
            add_to_blacklist(blacklist, bl_path, "walls", W, "error names wall", img_path.name)
            print(f"    ✗ error names wall → blacklist wall {W} (keep floor)", flush=True)
            W = None
            continue

        # ── unparseable → isolation pinpoint ─────────────────────────────────
        snippet = (err or "")[:120]
        print(f"    ? error not attributable ({snippet}) → isolating…", flush=True)

        comp_fw, _ = render_once(base, img_path, {**STUDIO_BASE, "floor": F, "wall": SAFE_WALL})
        if comp_fw is not None:
            # F renders fine with the safe wall → the wall W was the culprit. Keep F.
            add_to_blacklist(blacklist, bl_path, "walls", W,
                             "isolation: floor ok with safe wall", img_path.name)
            print(f"    → floor {F} ok with safe wall → blacklist wall {W} (keep floor)", flush=True)
            W = None
            continue

        # (F, SAFE_WALL) failed too → either F is bad, or the IMAGE itself is bad.
        if not image_known_ok:
            comp_safe, err_safe = render_once(
                base, img_path, {**STUDIO_BASE, "floor": SAFE_FLOOR, "wall": SAFE_WALL}
            )
            if comp_safe is None:
                # Fails even with the known-good baseline → it's the image, not materials.
                reason = f"image-level error: {(err_safe or '')[:160]}"
                print(f"    ✗ fails with safe floor+wall too → IMAGE problem, skipping", flush=True)
                return ("skip", reason)
            image_known_ok = True

        # Image is fine and (F, safe wall) failed → F is the bad one. Keep W.
        add_to_blacklist(blacklist, bl_path, "floors", F,
                         "isolation: image ok with safe materials", img_path.name)
        print(f"    → image ok with safe materials → blacklist floor {F} (keep wall)", flush=True)
        F = None
        continue

    return ("skip", f"no success within {MAX_RETRIES} retries")


def main() -> None:
    base = SERVER_URL.rstrip("/")
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    floors, walls = load_catalog(CATALOG_PATH)
    print(f"catalog: {len(floors)} floors, {len(walls)} walls", flush=True)

    imgs = sorted(p for p in in_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise SystemExit(f"no images found in {in_dir}")

    print(f"connecting to {base} …", flush=True)
    health = wait_for_ready(base, WAIT_READY_S)
    print(f"server ready (gpu={health.get('gpu')}).", flush=True)

    blacklist = load_blacklist(BLACKLIST_PATH)
    if blacklist["floors"] or blacklist["walls"]:
        print(f"loaded blacklist: {len(blacklist['floors'])} floors, "
              f"{len(blacklist['walls'])} walls (from {BLACKLIST_PATH.name})", flush=True)
    saved = []
    skipped = []

    print(f"processing {len(imgs)} images → {out_dir}", flush=True)
    for i, img in enumerate(imgs):
        print(f"[{i + 1}/{len(imgs)}] {img.name}", flush=True)
        try:
            result = process_image(base, img, out_dir, floors, walls, blacklist, BLACKLIST_PATH)
        except requests.RequestException as exc:
            print(f"    request failed: {exc}", flush=True)
            result = ("skip", f"request error: {exc}")
        except Exception as exc:  # noqa: BLE001 — one bad image must not kill the run
            print(f"    ERROR {exc}", flush=True)
            result = ("skip", f"unexpected: {exc}")

        if result[0] == "ok":
            saved.append((img.name, result[1], result[2]))
        else:
            skipped.append((img.name, result[1]))

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n──────── summary ────────", flush=True)
    print(f"saved   : {len(saved)}/{len(imgs)}", flush=True)
    print(f"skipped : {len(skipped)}", flush=True)
    for name, reason in skipped:
        print(f"    - {name}: {reason}", flush=True)
    if blacklist["floors"]:
        print(f"blacklisted floors ({len(blacklist['floors'])}): {sorted(blacklist['floors'])}", flush=True)
    if blacklist["walls"]:
        print(f"blacklisted walls ({len(blacklist['walls'])}): {sorted(blacklist['walls'])}", flush=True)
    print(f"blacklist persisted to: {BLACKLIST_PATH}", flush=True)
    print(f"output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
