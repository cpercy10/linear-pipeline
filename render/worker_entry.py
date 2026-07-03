"""WARM Blender worker — runs INSIDE one long-lived Blender process.

Launched once by ``render/blender_worker.py`` as::

    blender -b assets/blender/motuva-studio-master.blend \
            --python render/worker_entry.py -- --port <PORT> [--samples N] [--long-edge N]

This is the persistent replacement for Module 2's cold ``blender -b … --python
render.py`` per-image launch. It does Module 2 ``render.py``'s import/override
dance against the vendored ``render/studio_engine`` ONCE at startup (sys.path
inject ``scripts`` + ``configurator``, repoint ``material_library`` to
``assets/blender/Materials/_downloaded_2k`` and RE-RUN its import-time scan,
import ``render_server``, force Cycles to GPU, replace ``render_server.setup``
with the CAR-FREE setup and run it once). Then it LOOPS, reading newline-
delimited JSON jobs from a localhost socket and replying with one JSON line per
job::

    request : {"id": <int>, "camera": {...export values...},
               "disc_diam": <float>, "photo_w": <int>, "photo_h": <int>,
               "studio": {...look...}, "samples": <int?>, "long_edge": <int?>,
               "out_jpg": <path>, "out_json": <path>,
               "studio_frame": <bool?>, "no_threeq_zoom": <bool?>}
    reply   : {"id": <int>, "ok": true,  "out_jpg": ..., "out_json": ...,
               "meta": {...}, "elapsed_ms": <float>}
       or   : {"id": <int>, "ok": false, "error": "<message>", "trace": "..."}

Special control messages:
    {"cmd": "ping"}     -> {"ok": true, "pong": true}
    {"cmd": "shutdown"} -> {"ok": true, "bye": true}  then the process exits.

CRITICAL — per-job state isolation: every job that mutates the scene
(turntable diameter, camera, resolution, samples) restores the saved values and
resets ``RS.S["last"]`` afterwards, exactly as Module 2 ``export_plate`` already
does, plus we re-assert ``RS.rm.TT_DIAM`` per job so disc sizing never leaks
across renders. The scene / room / baked materials loaded at startup stay
resident; only per-job overrides change.

The faithful per-photo framing + the 3/4 zoom-out behavior are preserved
verbatim from Module 2 ``render.py`` (see ``_is_three_quarter`` / ``THREEQ_*``).
"""

import bpy  # noqa: F401  (Blender's bundled interpreter provides this)
import sys
import os
import json
import math
import socket
import traceback
import datetime
import argparse
import time


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing (everything after the Blender "--" sentinel is ours)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="worker_entry")
    ap.add_argument("--port", type=int, required=True,
                    help="localhost TCP port the host client is listening on")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--samples", type=int, default=16,
                    help="default Cycles samples (per-job 'samples' overrides)")
    ap.add_argument("--long-edge", type=int, default=1024,
                    help="default export long edge (per-job 'long_edge' overrides)")
    ap.add_argument("--materials", default=None,
                    help="absolute path to Materials/_downloaded_2k (overrides default)")
    return ap.parse_args(argv)


ARGS = _parse_args()

# The vendored studio engine lives next to this file: render/studio_engine/{scripts,configurator}.
HERE = os.path.dirname(os.path.abspath(__file__))                       # …/motuva_pipeline/render
PKG_ROOT = os.path.dirname(HERE)                                        # …/motuva_pipeline
STUDIO_ENGINE = os.path.join(HERE, "studio_engine")
DEFAULT_MATERIALS = os.path.join(PKG_ROOT, "assets", "blender", "Materials", "_downloaded_2k")
MATERIALS = (
    ARGS.materials
    or os.environ.get("MOTUVA_LIB2K")
    or os.environ.get("MOTOCUT_MATERIALS_2K_DIR")
    or DEFAULT_MATERIALS
)


def _log(*a):
    print("[worker]", *a, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Module-2 import/override dance — done EXACTLY ONCE at startup.
#    Mirrors plate-rendering/render.py: put the REAL studio code on sys.path
#    BEFORE importing render_server (it imports render_master/material_library at
#    module load), repoint material_library to the Windows/in-package location and
#    RE-RUN its import-time scan, then import render_server and override its PKG /
#    LIB2K / export constants. Cycles is forced to GPU (NOT CPU, unlike render.py).
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(STUDIO_ENGINE, "scripts"))       # render_master, material_library, branding
sys.path.insert(0, os.path.join(STUDIO_ENGINE, "configurator"))  # render_server

import material_library as L  # noqa: E402
L.LIB_2K = MATERIALS
L.LIB_FULL = MATERIALS                                            # full-res masters binned; 2K is the live source
L.PBR = L._scan_pbr()                                            # rescan against the correct root
L.MATERIALS = L.PBR + L.BLENDERKIT + L.WALLMODELS
L._BY_KEY = {m["key"]: m for m in L.MATERIALS}                   # get() reads this — must be rebuilt
_log(f"material library: {len(L.floors())} floors, {len(L.walls())} walls "
     f"({'granite OK' if 'lib_floor-tiling-stonegranitetile-' in L._BY_KEY else 'granite MISSING'})")

import render_server as RS  # noqa: E402

RS.PKG = STUDIO_ENGINE                                           # keep export side-effects (exports/<ts>/) inside the engine dir
RS.LIB2K = MATERIALS                                            # _pbr_mat() reads this constant
RS.EXPORT_LONG_EDGE = int(ARGS.long_edge)
RS.EXPORT_SAMPLES = int(ARGS.samples)


def _setup_carfree():
    """render_server.setup() WITHOUT loading the car — verbatim from Module 2 render.py.

    Builds camera + room + the baked-material cache exactly as production and sets every
    car-related state entry to a safe empty so apply_and_render's car-hide loop is a no-op
    and the car-on framing branch is unreachable (export_plate forces car:False)."""
    rm = RS.rm
    rm.enable_gpu()
    RS.configure_render()
    RS.S["cam"] = rm.setup_camera()
    bpy.context.scene.camera = RS.S["cam"]
    rm.activate_room("flatwall")
    RS.S["rig"] = None
    RS.S["meshes"] = []
    RS.S["pts_local"] = []
    RS.S["zmin_local"] = 0.0
    RS.S["car_all"] = []      # car-hide loop iterates zero times
    RS.S["meshset"] = set()
    RS.S["ctop2"] = None
    RS.S["baked"] = {}
    for objs in RS.ROOM_TARGETS.values():
        for nm in objs["floor"] + objs["wall"]:
            o = bpy.data.objects.get(nm)
            if o and o.data.materials and o.data.materials[0]:
                m = o.data.materials[0]
                m.use_fake_user = True   # protect the plain canvas mats from the per-render purge
                RS.S["baked"][nm] = m
    _log("setup complete — CAR-FREE (camera + room + baked materials ready)")


RS.setup = _setup_carfree
RS.setup()

# Force Cycles onto the GPU (replacing render.py's hardcoded cycles.device='CPU').
# enable_gpu() above already enables the GPU devices + set device='GPU'; assert it
# again so a per-job apply_and_render purge can never silently fall back to CPU.
try:
    RS.rm.enable_gpu()
    bpy.context.scene.cycles.device = "GPU"
    _log("Cycles device forced to GPU")
except Exception as e:  # noqa: BLE001
    _log("GPU force failed:", e)


# ─────────────────────────────────────────────────────────────────────────────
# 3/4-ONLY zoom-out — preserved verbatim from Module 2 render.py.
# Front / rear / complete-side shots stay EXACTLY as the faithful plate logic
# produces them; only front-3/4 and rear-3/4 are pulled back a touch.
# ─────────────────────────────────────────────────────────────────────────────
THREEQ_ZOOMOUT = 1.08      # gentle pull-back on 3/4 angles
THREEQ_HALF_WIDTH = 30.0   # ± band (deg) around each 3/4 centre; keeps front/rear/side out


def _is_three_quarter(az):
    az = float(az) % 360.0
    for centre in (45.0, 135.0, 225.0, 315.0):           # rear-3/4 L/R and front-3/4 L/R
        if abs((az - centre + 180.0) % 360.0 - 180.0) <= THREEQ_HALF_WIDTH:
            return True
    return False


# Default empty-studio look, mirroring config.BlenderConfig.studio. Overridable per job.
DEFAULT_STUDIO = {
    "room": "flatwall",
    "light": "panels",
    "turntable": "flush",
    "floor": "lib_floor-tiling-stonegranitetile-",
    "wall": "paint:54585B",
    "branding": "none",
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-job render. Reuses RS.export_plate (faithful framing + 3/4 zoom-out), then
# writes plate.jpg + meta.json to the caller-supplied paths. Per-job state is
# fully reset so materials/objects/diameter do NOT leak across renders.
# ─────────────────────────────────────────────────────────────────────────────
def _render_job(job: dict) -> dict:
    cam_vals = dict(job.get("camera") or {})
    disc = job.get("disc_diam")
    pw = int(job.get("photo_w", cam_vals.get("photo_w", 1600)))
    ph = int(job.get("photo_h", cam_vals.get("photo_h", 1200)))
    studio = dict(job.get("studio") or DEFAULT_STUDIO)
    out_jpg = job["out_jpg"]
    out_json = job["out_json"]

    # Per-job render-quality overrides (default to the startup values).
    saved_le, saved_sa = RS.EXPORT_LONG_EDGE, RS.EXPORT_SAMPLES
    if job.get("long_edge"):
        RS.EXPORT_LONG_EDGE = int(job["long_edge"])
    if job.get("samples"):
        RS.EXPORT_SAMPLES = int(job["samples"])

    # Per-job turntable diameter — sized to THIS (resized) car. add_turntable() (run
    # during export_plate's scene sync) reads this module constant, so set it each job
    # and restore it after, so disc sizing can never leak into the next render.
    rm = RS.rm
    saved_diam = rm.TT_DIAM
    if disc:
        rm.TT_DIAM = float(disc)
        _log(f"disc_diam = {float(disc):.2f} m")

    # Build the export payload exactly like Module 2's client: studio look + the
    # continuous camera values + the photo aspect (plate renders at the input aspect).
    export = {**cam_vals, "photo_w": pw, "photo_h": ph}

    # 3/4 zoom-out: only for the faithful (non-studio) framing path, mirroring render.py.
    az = export.get("azimuth_deg", 135.0)
    if (not job.get("studio_frame", False)
            and _is_three_quarter(az)
            and not job.get("no_threeq_zoom", False)):
        export["distance_m"] = float(export.get("distance_m", 7.0)) * THREEQ_ZOOMOUT
        _log(f"3/4 angle az={float(az):.0f} -> zoom-out x{THREEQ_ZOOMOUT}")

    payload = {"action": "export_plate", **studio, "export": export}

    try:
        # RS.export_plate already: sets per-job resolution + samples from the photo
        # aspect, syncs the scene car-free, poses the camera directly from the values,
        # renders, builds meta (car_spot_px + pixels_per_metre + camera.resolution),
        # restores resolution/samples, and resets RS.S["last"]. We just relocate its
        # outputs to the caller's paths.
        data, meta = RS.export_plate(payload)
    finally:
        # Restore everything we overrode so the NEXT job starts from a clean baseline.
        rm.TT_DIAM = saved_diam
        RS.EXPORT_LONG_EDGE, RS.EXPORT_SAMPLES = saved_le, saved_sa
        RS.S["last"] = {}   # belt-and-braces: force a full scene re-sync next job

        # Per-job material/datablock reclaim. export_plate (do_render=False) returns
        # BEFORE render_server's own purge block, so on the warm path NOTHING frees the
        # per-render materials/textures/images created by get_material / _pbr_mat /
        # get_wall_material(base.copy()) for each job's floor/wall keys. In a long-lived
        # worker those accumulate unbounded and eventually OOM the process mid-batch.
        # Reclaim them here: clear the blend-material cache (its refs may dangle after
        # purge) and orphan-purge. The baked room mats + paint/concrete mats carry
        # use_fake_user=True, so they survive; only the per-job wall_/mat_ copies with no
        # fake user are reclaimed.
        try:
            RS.S["blendmat_cache"].clear()
            bpy.data.orphans_purge(do_local_ids=True, do_recursive=True)
        except Exception as e:  # noqa: BLE001 — purge must never fail a completed render
            _log("per-job purge skipped:", e)

    os.makedirs(os.path.dirname(os.path.abspath(out_jpg)), exist_ok=True)
    with open(out_jpg, "wb") as f:
        f.write(data)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    # Contract sanity check: the meta MUST carry the compositor anchors.
    for required in ("car_spot_px", "pixels_per_metre"):
        if required not in meta:
            raise RuntimeError(f"render meta missing '{required}' — cannot composite")
    if "camera" not in meta or "resolution" not in meta.get("camera", {}):
        raise RuntimeError("render meta missing camera.resolution — cannot composite")

    return {"out_jpg": out_jpg, "out_json": out_json, "meta": meta,
            "bytes": len(data)}


# ─────────────────────────────────────────────────────────────────────────────
# Socket loop — newline-delimited JSON request/response over localhost.
# Blender runs single-threaded; we serve exactly one connection (the host client)
# and process jobs strictly in order.
# ─────────────────────────────────────────────────────────────────────────────
def _serve():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Connect back to the host client, which is already listening on ARGS.port.
    deadline = time.time() + 60.0
    conn = None
    while time.time() < deadline:
        try:
            sock.connect((ARGS.host, ARGS.port))
            conn = sock
            break
        except OSError:
            time.sleep(0.2)
    if conn is None:
        _log(f"FATAL: could not connect to host {ARGS.host}:{ARGS.port}")
        sys.exit(2)

    # Announce readiness so the host can release startup callers.
    conn.sendall((json.dumps({"ready": True, "ok": True}) + "\n").encode("utf-8"))
    _log(f"connected to host {ARGS.host}:{ARGS.port} — ready for jobs")

    buf = b""
    fobj = conn.makefile("rwb")
    try:
        while True:
            line = fobj.readline()
            if not line:
                _log("host closed the connection — exiting")
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                _reply(fobj, {"ok": False, "error": f"bad json: {e}"})
                continue

            cmd = req.get("cmd")
            if cmd == "ping":
                _reply(fobj, {"ok": True, "pong": True, "id": req.get("id")})
                continue
            if cmd == "shutdown":
                _reply(fobj, {"ok": True, "bye": True, "id": req.get("id")})
                break

            job_id = req.get("id")
            t0 = time.perf_counter()
            try:
                result = _render_job(req)
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                _reply(fobj, {"id": job_id, "ok": True, "elapsed_ms": elapsed_ms,
                              **result})
                _log(f"job {job_id} OK ({result['bytes'] // 1024} KB, {elapsed_ms:.0f} ms)")
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                _log(f"job {job_id} FAILED: {e}\n{tb}")
                _reply(fobj, {"id": job_id, "ok": False, "error": str(e), "trace": tb})
    finally:
        try:
            fobj.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _reply(fobj, obj: dict):
    fobj.write((json.dumps(obj) + "\n").encode("utf-8"))
    fobj.flush()


if __name__ == "__main__":
    _serve()
