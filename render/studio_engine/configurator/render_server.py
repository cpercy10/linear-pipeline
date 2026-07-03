"""render_server.py — LIVE WebSocket render server for the new Motuva studio.

Adapts the proven v6 engine (Cycles 8-sample+denoise, persistent GPU data, JSON-in/JPEG-out,
coalesced live preview) to the NEW master via render_master + material_library + branding.

Run (live, Windows):  "C:\\Program Files\\Blender Foundation\\Blender 4.x\\blender.exe" ^
                 -b motuva-studio-master.blend --python-use-system-env --python configurator/render_server.py
             (Install Blender first and adjust the version folder in the path above.
              --python-use-system-env lets Blender see system packages; the logo pipeline needs PIL —
              install it into Blender's bundled Python: <blender>/4.x/python/bin/python.exe -m pip install pillow)
Self-test :  …same… --python configurator/render_server.py -- selftest
Listens ws://localhost:8765 — browser sends a selection dict, gets back a JPEG frame.
"""
import bpy, sys, os, json, tempfile, math, base64, datetime, subprocess, shutil, hashlib
from math import radians
from mathutils import Vector

# Windows migration (2026-06-15): repo root derived from this file's location; assets
# resolve from the in-repo bundle Motuva-Studio-Materials/ (was ~/Desktop/Blender Mac paths).
PKG    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
BUNDLE = os.path.join(PKG, "Motuva-Studio-Materials")
sys.path.insert(0, os.path.join(PKG, "scripts"))
import render_master as rm
import material_library as L
try:
    import branding
except Exception as e:
    branding = None
    print("[server] branding unavailable:", e, flush=True)
try:
    from logo_preprocess import preprocess_logo_bytes, LogoPreprocessError
except Exception as e:
    preprocess_logo_bytes = None
    print("[server] logo_preprocess unavailable:", e, flush=True)

PORT        = 8765
SAMPLES     = 10
PREVIEW_RES = (960, 720)    # 4:3 — the majority dealer-photo aspect (LOCKED Chris 2026-06-10). Hero matches
                            # exactly: rendered frame IS the displayed frame. Production plates render at the
                            # uploaded photo's aspect per job — the rig is aspect-agnostic (verified 4:3 + 16:9).
LIB2K       = os.path.join(BUNDLE, "Materials", "_downloaded_2k")
CAR         = os.path.join(BUNDLE, "cars", "car_006_audi_rs6_avant_2020_7701f4.blend")  # old clean-set RS6 (Chris pick 2026-06-10); heading 0
LOGO        = os.path.join(BUNDLE, "logo", "test_logos", "dealertalk_trimmed.png")  # dev fallback only — file NOT in bundle (drop it here or upload via UI)
SVG         = os.path.join(BUNDLE, "logo", "test_logos", "dealertalk.svg")           # dev fallback only — file NOT in bundle
POTRACE     = shutil.which("potrace") or "potrace"   # install potrace + add to PATH for 3D / back-lit logo styles (Windows)
# V14 envelope contract: every logo is CONTAINED in a fixed wall container at
# native aspect — never stretched, never fixed-width. Wordmarks fill the width,
# icons fill the height. 2.5:1 is the locked envelope aspect.
BRAND_W     = 3.4                  # container width on the wall (m)
BRAND_H     = 1.00                 # container height (m) — icons cap at 1.0m tall (LOCKED Chris 2026-06-10)
BRAND_Z     = 2.50                 # band centre above floor (Chris 2026-06-10; frame fit includes the band)
CAR_HEADING = 0     # car_006 RS6: verified by render — abs rot 180 = front view, so no offset (rigged asset needed -90)
FILL_RIG    = 0.80   # car ON = the REAL camera rig — the locked production framing. Must match the rig contract; never drift.
FILL_ROOM   = 0.55   # car OFF = neutral room-builder view, pulled back to take in the whole studio (UI only, not the rig).

ROOM_TARGETS = {
    "flatwall": {"floor": ["flatwall_floor"], "wall": ["flatwall_wall_back", "flatwall_wall_left", "flatwall_wall_right"]},
    "curved":   {"floor": ["curved_floor"],   "wall": ["curved_wall"]},
    "cove":     {"floor": ["canvas"],          "wall": []},
}
ANGLE_ROT = {"front": 180, "front_q": 135, "side": 90, "rear_q": 45, "rear": 0}
HEIGHT_Z  = {"low": 0.70, "eye": 1.35, "high": 3.30}

S = {"cam": None, "rig": None, "meshes": None, "tt": [], "brand": [], "comp": [], "blendmat_cache": {},
     "last": {}, "floor_name": "flatwall_floor", "ctop": None, "ctop2": None,
     "frame_cache": {}, "logo_raw": None, "logo_raw_ref": None}
FRAME_CACHE_MAX = 400   # ~60KB/frame → ~24MB ceiling. Renders are deterministic (verified
                        # by pixel-diff 2026-06-10) so a selection-keyed cache is safe.

# ── render config ──
def configure_render(res=PREVIEW_RES):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
    sc.render.use_persistent_data = False   # off: frees render buffers between frames so browsing doesn't climb in (unified) memory
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "JPEG"; sc.render.image_settings.quality = 85
    rm.enable_gpu()

def setup():
    rm.enable_gpu(); configure_render()
    S["cam"] = rm.setup_camera()
    bpy.context.scene.camera = S["cam"]
    rm.activate_room("flatwall")
    rig, meshes = rm.load_car(CAR, "flatwall_floor")
    S["rig"], S["meshes"] = rig, meshes
    # measure the car ONCE — every later settings change uses this cache, never a vertex sweep
    S["pts_local"], S["zmin_local"] = rm.cache_car_points(rig, meshes)
    print(f"[server] car measured once: {len(S['pts_local'])} framing pts cached", flush=True)
    # the whole car hierarchy (incl. non-mesh parts like suspension spring curves) — so "hide" hides ALL of it
    allc = []
    def _desc(o):
        allc.append(o)
        for c in o.children: _desc(c)
    _desc(rig)
    S["car_all"] = allc
    S["meshset"] = set(meshes)
    # remember each room surface's plain baked material so "no pick" can restore it (blank canvas)
    S["baked"] = {}
    for objs in ROOM_TARGETS.values():
        for nm in objs["floor"] + objs["wall"]:
            o = bpy.data.objects.get(nm)
            if o and o.data.materials and o.data.materials[0]:
                m = o.data.materials[0]
                m.use_fake_user = True   # protect the plain canvas mats from the per-render purge so "clear" can restore them
                S["baked"][nm] = m
    print("[server] setup complete — car + camera ready", flush=True)

# ── material loaders ──
def _tex(nt, mp, d, fn, noncolor=False):
    p = os.path.join(d, fn)
    if not os.path.exists(p): return None
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(p, check_existing=True)
    if noncolor: t.image.colorspace_settings.name = "Non-Color"
    nt.links.new(mp.outputs["Vector"], t.inputs["Vector"]); return t

def _pbr_mat(entry):
    key = "mat_" + entry["key"]
    if key in bpy.data.materials: return bpy.data.materials[key]
    d = os.path.join(LIB2K, entry["folder"]); uv = entry.get("uv", 6)
    m = bpy.data.materials.new(key); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (uv, uv, uv); nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
    col = _tex(nt, mp, d, "color.jpg")
    if col: nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    rgh = _tex(nt, mp, d, "roughness.jpg", True)
    if rgh: nt.links.new(rgh.outputs["Color"], bsdf.inputs["Roughness"])
    met = _tex(nt, mp, d, "metallic.jpg", True)
    if met: nt.links.new(met.outputs["Color"], bsdf.inputs["Metallic"])
    nrm = _tex(nt, mp, d, "normal.png", True)
    if nrm:
        nm = nt.nodes.new("ShaderNodeNormalMap"); nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m

def _blend_mat(entry):
    key = entry["key"]
    if key in S["blendmat_cache"]: return S["blendmat_cache"][key]
    path = entry.get("blend") or ""
    if not path or not os.path.exists(path):
        S["blendmat_cache"][key] = None; return None
    try:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.materials = list(src.materials)
        mats = [m for m in dst.materials if m]
        mat = mats[0] if mats else None
    except Exception as e:
        print(f"[server] blend_mat load failed {entry['name']}: {e}", flush=True); mat = None
    S["blendmat_cache"][key] = mat; return mat

def _paint_mat(hexstr):
    """Smooth matte paint/plaster wall — flat Principled BSDF tinted to a hex colour.
    No textures, no library lookup; cached + fake-user so the orphan purge can't nuke it."""
    hexstr = hexstr.lstrip("#").upper()
    key = "paint_" + hexstr
    if key in bpy.data.materials: return bpy.data.materials[key]
    r, g, b = (int(hexstr[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    s2l = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4   # sRGB→linear
    m = bpy.data.materials.new(key); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (s2l(r), s2l(g), s2l(b), 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9          # matte
    for spec in ("Specular IOR Level", "Specular"):
        if spec in bsdf.inputs: bsdf.inputs[spec].default_value = 0.25; break
    m.use_fake_user = True
    return m

def _concrete_mat(base_key, finish):
    """A real concrete texture (colour + normal detail kept) with the roughness map replaced by a
    constant so the surface reads as Matte (troweled microcement) or Polished (sealed, catches light)."""
    name = f"conc_{base_key}_{finish}"
    if name in bpy.data.materials: return bpy.data.materials[name]
    base = get_material(base_key)
    if not base: return None
    m = base.copy(); m.name = name; m.use_fake_user = True
    rough = 0.85 if finish == "matte" else 0.18
    if m.use_nodes:
        for n in m.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED":
                inp = n.inputs["Roughness"]
                for l in list(inp.links): m.node_tree.links.remove(l)   # drop the roughness map
                inp.default_value = rough
                if finish != "matte":
                    for spec in ("Specular IOR Level", "Specular"):
                        if spec in n.inputs: n.inputs[spec].default_value = 0.6; break
                break
    return m

MASTER_BLEND = os.path.join(BUNDLE, "old-configurator", "master-file.blend")  # source of the 8 'mblend:' walls; its textures point to a full-res lib NOT in the bundle → those walls load untextured (soft-fail, see migration notes)
def _master_blend_mat(name):
    """Append a named, pre-calibrated wall material from the previous master .blend, as-is."""
    cache = "mb_" + name
    if cache in bpy.data.materials: return bpy.data.materials[cache]
    try:
        with bpy.data.libraries.load(MASTER_BLEND, link=False) as (src, dst):
            dst.materials = [name] if name in src.materials else []
        mats = [m for m in dst.materials if m]
        m = mats[0] if mats else None
        if m: m.name = cache; m.use_fake_user = True
        return m
    except Exception as e:
        print(f"[server] master-blend load failed {name}: {e}", flush=True); return None

def get_material(key):
    if not key: return None
    if key.startswith("paint:"): return _paint_mat(key[6:])
    if key.startswith("concrete:"):
        _, base_key, finish = key.split(":"); return _concrete_mat(base_key, finish)
    if key.startswith("mblend:"): return _master_blend_mat(key[7:])
    e = L.get(key)
    if not e: return None
    if e["type"] == "pbr_folder":    return _pbr_mat(e)
    if e["type"] == "blend_material": return _blend_mat(e)
    return None   # blend_model handled separately

# ── wall texture scale ──────────────────────────────────────────────────────────
# Tiling comes from a TexCoord.UV → Mapping node over the wall's single 0–1 UV. Each material's
# `uv` hint is its content size calibrated for the 44 m floor; on the ~10 m wall we reuse that hint
# scaled down by WALL_K so the physical size matches the floor (marble = big slabs, concrete = fine).
# Blend materials (brick) carry no hint → BLEND_WALL_UV. WALL_ROT corrects the wall UV's orientation.
WALL_K        = 0.28          # floor-uv → wall-uv factor (wall ≈ 10 m vs 44 m floor)
WALL_ASPECT   = 1.0           # V-vs-U multiplier to correct wall stretch
WALL_ROT      = radians(90)   # rotate so brick courses / grain run horizontal
BLEND_WALL_UV = 6.0           # effective wall scale for blend materials with no uv hint

def get_wall_material(key):
    if not key: return None
    if key.startswith("paint:"): return _paint_mat(key[6:])   # flat colour — no tiling to scale
    base = get_material(key)
    if not base: return None
    name = "wall_" + key.replace(":", "_")
    if name in bpy.data.materials: return bpy.data.materials[name]
    m = base.copy(); m.name = name; m.use_fake_user = True
    e = L.get(key) if ":" not in key else None
    uv_hint = (e.get("uv") if e else None)
    su = (uv_hint * WALL_K) if uv_hint else BLEND_WALL_UV
    sv = su * WALL_ASPECT
    if m.use_nodes:
        for n in m.node_tree.nodes:
            if n.type == "MAPPING":
                n.inputs["Scale"].default_value = (su, sv, 1.0)
                n.inputs["Rotation"].default_value = (0.0, 0.0, WALL_ROT)
    return m

def set_surface(objname, mat):
    """Apply `mat`, or restore the room's plain baked material when mat is None (blank canvas)."""
    o = bpy.data.objects.get(objname)
    if not o: return
    use = mat or S["baked"].get(objname)
    if use:
        o.data.materials.clear(); o.data.materials.append(use)

# ── dealer logo upload pipeline (V17, ported verbatim from the old configurator) ──
# Upload PNG → key out solid background + trim to content (logo_preprocess) →
# Potrace the alpha silhouette into SVG curves. The trimmed PNG drives Painted /
# Multi-print / the 3D front face; the SVG drives the extruded Signage3D /
# Back-lit letters. One ordinary raster upload yields every style.

def _vectorise_trimmed_to_svg(png_path: str, svg_path: str,
                              alpha_threshold: int = 64) -> str:
    """Trimmed PNG (with alpha) → SVG via Potrace. Locked in V15: pixels with
    alpha > threshold are 'logo' (black in the BMP), below = background."""
    from PIL import Image
    import numpy as np

    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    alpha = arr[..., 3]

    bmp = np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
    bmp_img = Image.fromarray(bmp, mode="L")
    bmp_path = svg_path.replace(".svg", ".bmp")
    bmp_img.save(bmp_path)

    subprocess.run(
        [POTRACE, "-s",
         "--turdsize", "2",
         "--opttolerance", "0.2",
         "-o", svg_path,
         bmp_path],
        check=True,
    )
    try:
        os.unlink(bmp_path)
    except OSError:
        pass
    return svg_path


def _prep_dealer_logo(raw: bytes, treatment: str = "original"):
    """Run the locked upload pipeline on raw uploaded bytes. Returns
    (trimmed_png_path, svg_path). Paths are content-hashed so a NEW logo gets
    a NEW path — branding._img() loads with check_existing=True, and reusing
    one fixed path would serve the previous logo's stale datablock (the old
    server's documented 'swap the logo, still see the old one' bug).
    Non-PNG rasters (JPEG) are transcoded to PNG first so the locked
    preprocessor stays PNG-only. Potrace failure is non-fatal — Signage3D /
    Back-lit fall back to the test SVG; Painted / Multi-print still carry
    the dealer logo."""
    import io
    from PIL import Image
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        im = Image.open(io.BytesIO(raw)); im.load()
        buf = io.BytesIO(); im.convert("RGBA").save(buf, format="PNG")
        raw = buf.getvalue()
    trimmed = preprocess_logo_bytes(raw, treatment=treatment, pad=False)
    tag = hashlib.md5(trimmed).hexdigest()[:12]
    png_path = os.path.join(tempfile.gettempdir(), f"motuva_logo_{tag}.png")
    with open(png_path, "wb") as f:
        f.write(trimmed)
    tw, th = Image.open(io.BytesIO(trimmed)).size
    asp = (tw / th) if th else 2.5
    svg_path = os.path.join(tempfile.gettempdir(), f"motuva_logo_{tag}.svg")
    try:
        _vectorise_trimmed_to_svg(png_path, svg_path)
    except Exception as e:
        print(f"[logo] WARN: potrace failed ({e}). 3D styles unavailable for "
              f"this logo — flat styles still carry it.", flush=True)
        svg_path = None   # caller downgrades Signage3D/Back-lit to Painted (never the sample)
    print(f"[logo] preprocessed: {len(raw)}B in → {len(trimmed)}B trimmed, "
          f"aspect {asp:.2f} ({png_path})", flush=True)
    return png_path, svg_path, asp


# ── turntable / branding clear ──
def clear_turntable():
    for o in S["tt"]:
        try: bpy.data.objects.remove(o, do_unlink=True)
        except Exception: pass
    S["tt"] = []

def clear_branding():
    if branding and hasattr(branding, "_rm"):
        try: branding._rm()
        except Exception: pass
    for o in S["brand"]:
        try: bpy.data.objects.remove(o, do_unlink=True)
        except Exception: pass
    S["brand"] = []

def _bbox(meshes):
    deps = bpy.context.evaluated_depsgraph_get()
    mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
    for o in meshes:
        ev = o.evaluated_get(deps); m = ev.to_mesh()
        for v in m.vertices:
            w = ev.matrix_world @ v.co
            for i in range(3): mn[i] = min(mn[i], w[i]); mx[i] = max(mx[i], w[i])
        ev.to_mesh_clear()
    return mn, mx

def clear_composite():
    for o in S.get("comp", []):
        try: bpy.data.objects.remove(o, do_unlink=True)
        except Exception: pass
    S["comp"] = []

def apply_composite_wall(entry):
    """Append a wall-composition model, join it, stand it to a ~3.3 m band, and TILE it across the
    flat-wall back (array to fill ~13 m), sitting on the floor, centred on the car-spot, just in front
    of the plain wall. Tiling is what makes single panels read as a full feature wall."""
    path = entry.get("blend") or ""
    if not path or not os.path.exists(path):
        return False
    sc = bpy.context.scene
    before = set(bpy.data.objects)
    try:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
    except Exception as e:
        print("[server] composite load fail:", e, flush=True); return False
    objs = [o for o in dst.objects if o]
    for o in objs:
        if o.name not in sc.collection.objects:
            try: sc.collection.objects.link(o)
            except Exception: pass
    meshes = [o for o in objs if o.type == "MESH"]
    if not meshes:
        S["comp"] = [o for o in bpy.data.objects if o not in before]; return False
    # join to one mesh, bake transforms
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes: o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    try: bpy.ops.object.join()
    except Exception: pass
    M = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action="DESELECT"); M.select_set(True)
    bpy.context.view_layer.objects.active = M
    try: bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    except Exception: pass
    mn, mx = _bbox([M]); dims = mx - mn
    BAND = 3.3
    s = BAND / max(dims.z, 1e-3)
    M.scale = (s, s, s); bpy.context.view_layer.update()
    mn, mx = _bbox([M]); dims = mx - mn
    along_x = dims.x >= dims.y
    width = max(dims.x if along_x else dims.y, 0.1)
    cnt = max(1, math.ceil(13.0 / width))
    mod = M.modifiers.new("tile", "ARRAY")
    mod.use_relative_offset = True
    mod.relative_offset_displace = (1, 0, 0) if along_x else (0, 1, 0)
    mod.count = cnt
    bpy.context.view_layer.update()
    if along_x:                       # run the tiled row along Y (the wall width)
        M.rotation_euler[2] = math.radians(90); bpy.context.view_layer.update()
    mn, mx = _bbox([M])
    BACK_X = rm.CAR_SPOT.x - 6.0
    M.location.x += BACK_X - mn.x + 0.05
    M.location.y += rm.CAR_SPOT.y - (mn.y + mx.y) / 2
    M.location.z += rm.FLOOR_Z - mn.z
    bpy.context.view_layer.update()
    S["comp"] = [o for o in bpy.data.objects if o not in before]
    return True

def reground(floor_name, ctop):
    """Grounding from the load-time measurement — pure arithmetic, no vertex sweep.
    The rig only rotates about Z, so world z-min = rig.z + scale.z * zmin_local always."""
    rig = S["rig"]
    zmin = lambda: rig.location.z + rig.scale.z * S["zmin_local"]
    rig.location.z += rm.floor_z_at(rm.CAR_SPOT.x, rm.CAR_SPOT.y, floor_name) - zmin()
    rig.location.z += (ctop - rm.FLOOR_Z)
    bpy.context.view_layer.update()
    return zmin() + rm.STD_CAR[2]

# ── apply + render ──
def apply_and_render(payload, do_render=True):
    room   = payload.get("room") or "flatwall"
    if room not in ROOM_TARGETS: room = "flatwall"
    light  = payload.get("light") or "none"
    tt     = payload.get("turntable") or "none"
    angle  = payload.get("angle") or "front_q"
    height = payload.get("height") or "eye"
    floor_key, wall_key = payload.get("floor"), payload.get("wall")
    brand_style = payload.get("branding") or "none"
    car_on = bool(payload.get("car"))
    Lst = S["last"]; note = None

    # dealer logo — sent ONCE: a fresh upload arrives as logo_data (full base64); every
    # later frame carries only logo_ref (its signature). The raw bytes are cached server-
    # side; if the ref is unknown (server restarted) we ask the client to resend.
    # NO SAMPLE LOGO: branding without an uploaded logo renders as "none" (Chris ruling).
    treatment = payload.get("treatment") or "original"
    led       = payload.get("led") or "warm"
    logo_data = payload.get("logo_data")
    if logo_data:
        raw_b64 = logo_data.split(",")[-1]
        ref = f"{len(raw_b64)}:{raw_b64[-40:]}"
        S["logo_raw"], S["logo_raw_ref"] = raw_b64, ref
    else:
        ref = payload.get("logo_ref")
        if ref and ref != S.get("logo_raw_ref"):
            ref = None; note = "logo-needed"   # client re-sends the full file on this note
    logo_sig = (ref + ":" + treatment) if ref else None
    logo_png, logo_svg, logo_asp = None, None, None
    if ref:
        if logo_sig != S.get("logo_prep_sig"):
            S["logo_prep"], S["logo_prep_sig"] = None, logo_sig
            if preprocess_logo_bytes:
                try:
                    b = base64.b64decode(S["logo_raw"])
                    S["logo_prep"] = _prep_dealer_logo(b, treatment)
                except Exception as e:
                    print("[server] logo preprocess fail:", e, flush=True)
                    note = "logo-rejected"   # surfaced to the UI — never silently show the sample
        if S.get("logo_prep"):
            logo_png, logo_svg, logo_asp = S["logo_prep"]
    if brand_style != "none" and not logo_png:
        brand_style = "none"               # no uploaded logo = no branding, never the sample
    if brand_style in ("signage3d", "backlit") and logo_svg is None and logo_png:
        brand_style = "painted"            # vectorise failed — flat styles still carry their logo
        if not note: note = "logo-3d-failed"
    # contain-fit: wide logos take the container width, tall/square logos take
    # its height — the logo always fits inside BRAND_W × BRAND_H at native aspect
    brand_w = min(BRAND_W, BRAND_H * logo_asp) if logo_asp else BRAND_W

    # frame cache — deterministic renders keyed by the full visual selection; hits skip
    # the scene sync AND the render (S["last"] still mirrors the real scene state).
    cache_key = json.dumps({"room": room, "light": light, "tt": tt, "floor": floor_key,
                            "wall": wall_key, "brand": brand_style, "car": car_on,
                            "angle": angle, "height": height, "logo": logo_sig, "led": led},
                           sort_keys=True)
    if do_render and cache_key in S["frame_cache"]:
        S["frame_cache"][cache_key] = S["frame_cache"].pop(cache_key)   # LRU bump
        return S["frame_cache"][cache_key], note

    # Only do the work that actually changed — material/camera picks must NOT re-sync the
    # whole scene (that's the slowness). Room/lighting/turntable/geometry changes do re-sync.
    room_ch = room != Lst.get("room")
    if room_ch:
        S["floor_name"] = rm.activate_room(room)
    floor_name = S["floor_name"]
    if room_ch or light != Lst.get("light"):
        rm.activate_light(room, light)
    tg = ROOM_TARGETS[room]

    if room_ch or floor_key != Lst.get("floor"):
        fm = get_material(floor_key) if floor_key else None
        for t in tg["floor"]: set_surface(t, fm)
    if tg["wall"] and (room_ch or wall_key != Lst.get("wall")):
        clear_composite()
        bw = bpy.data.objects.get("flatwall_wall_back")
        we = L.get(wall_key) if (wall_key and ":" not in wall_key) else None   # synthetic keys (paint:/concrete:) skip the library
        if we and we["type"] == "blend_model":
            if room == "flatwall":
                ok = apply_composite_wall(we)
                if bw: bw.hide_render = ok           # composite replaces the plain back wall
                for t in tg["wall"]: set_surface(t, None)  # side walls stay plain
                if not ok: note = "composite-load-failed"
            else:
                note = "composite-flatwall-only"
                for t in tg["wall"]: set_surface(t, None)
        else:
            if bw: bw.hide_render = (room != "flatwall")   # only show the flat back wall in the Flat-Wall room
            wm = get_wall_material(wall_key) if wall_key else None
            for t in tg["wall"]: set_surface(t, wm)

    tt_ch = (tt != Lst.get("tt")) or room_ch
    if tt_ch:
        clear_turntable()
        if tt != "none":
            before = set(bpy.data.objects)
            S["ctop"] = rm.add_turntable(tt)
            S["tt"] = [o for o in bpy.data.objects if o not in before]
        else:
            S["ctop"] = rm.FLOOR_Z
    ctop = S["ctop"] if S["ctop"] is not None else rm.FLOOR_Z

    car_ch = car_on != Lst.get("car")
    if car_ch:
        # show only the real body meshes when car is on; hide everything else (gizmos, spring curves) always
        for o in S["car_all"]:
            o.hide_render = (o not in S["meshset"]) or (not car_on)

    # ground + frame (car ON → real grounding + mesh framing; car OFF → cheap fixed-box framing)
    need_frame = (room_ch or tt_ch or car_ch or angle != Lst.get("angle") or height != Lst.get("height")
                  or brand_style != Lst.get("brand"))   # logo band is part of the frame fit
    arc = rm.ROOMS[room]["safe_arc"]
    if car_on:
        if room_ch or tt_ch or car_ch:
            S["ctop2"] = reground(floor_name, ctop)
        if room_ch or car_ch or angle != Lst.get("angle"):
            S["rig"].rotation_euler[2] = radians(ANGLE_ROT.get(angle, 135) + CAR_HEADING)
            bpy.context.view_layer.update()
        if need_frame:   # the REAL rig — locked contract framing
            # cached points (ms) instead of the full vertex sweep; lock_h=False = centre the CAR's
            # silhouette both axes (the proven test-rig behaviour — fixes off-centre quarter shots)
            world_pts = [S["rig"].matrix_world @ p for p in S["pts_local"]]
            # contract: car AND logo both in frame AND both centred. For band styles (not
            # multiprint — that's wallpaper, clipping intended) add the logo band corners so
            # the fit contains both, and pass the band centre as the parallax-centring anchor.
            centre_pt = None
            if brand_style in ("painted", "signage3d", "backlit") and tg["wall"]:
                bz = rm.FLOOR_Z + BRAND_Z
                lh = brand_w / (logo_asp or 6.8) + 0.15          # logo height + margin
                CSp = rm.CAR_SPOT
                bx = (CSp.x - 6.0) if room == "flatwall" else (CSp.x - rm.CURVED_R)
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        world_pts.append(Vector((bx, CSp.y + sy*(brand_w/2 + 0.1), bz + sz*lh/2)))
                centre_pt = Vector((bx, CSp.y, bz))
            rm.set_shot(S["cam"], 180.0, arc, HEIGHT_Z.get(height, 1.35), 50.0, FILL_RIG,
                        car_z_top=S["ctop2"], pts=world_pts, lock_h=False, centre_pt=centre_pt)
    elif need_frame:      # neutral room-builder view — pulled back, not the rig
        rm.set_shot(S["cam"], 180.0, arc, HEIGHT_Z.get(height, 1.35), 50.0, FILL_ROOM,
                    car_z_top=rm.FLOOR_Z + rm.STD_CAR[2], meshes=None, lock_h=True)

    if (room_ch or brand_style != Lst.get("brand") or logo_sig != Lst.get("logo_sig")
            or led != Lst.get("led")):
        clear_branding()
        if branding and brand_style != "none" and tg["wall"]:
            try:
                band = rm.FLOOR_Z + BRAND_Z
                S["brand"] = branding.apply_branding(brand_style, logo_png, logo_png, logo_svg,
                                                     room=("flatwall" if room == "flatwall" else "curved"),
                                                     band_z=band, target_w=brand_w, led=led) or []
            except Exception as e:
                print("[server] branding failed:", e, flush=True)

    S["last"] = {"room": room, "light": light, "tt": tt, "floor": floor_key, "wall": wall_key,
                 "brand": brand_style, "car": car_on, "angle": angle, "height": height,
                 "logo_sig": logo_sig, "led": led}

    if not do_render:          # scene sync only (export sets its own camera + renders itself)
        return None, note

    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="cfg_"); os.close(fd)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    with open(path, "rb") as f: data = f.read()
    try: os.unlink(path)
    except OSError: pass
    S["frame_cache"][cache_key] = data
    if len(S["frame_cache"]) > FRAME_CACHE_MAX:
        S["frame_cache"].pop(next(iter(S["frame_cache"])))   # evict oldest
    # free memory: drop unreferenced materials/textures so browsing doesn't accumulate in (unified) RAM.
    # blend-material refs may dangle after purge → clear the cache so they reload on demand; baked mats
    # carry a fake user so they survive. pbr mats are cached by name and self-heal on next lookup.
    S["blendmat_cache"].clear()
    try: bpy.data.orphans_purge(do_local_ids=True, do_recursive=True)
    except Exception as e: print("[server] purge skipped:", e, flush=True)
    return data, note

EXPORT_RES_DEFAULT = (2560, 1920)
EXPORT_SAMPLES = 64
_EXPORT_SEQ = 0   # per-process counter → uniquifies the opt-in exports/<ts>/ dir name

EXPORT_LONG_EDGE = 2560   # plate render quality; aspect comes from the dealer photo

# ── Plate reframing (vertical lens-shift) — Phase 1: OPT-IN, default OFF ──────────────
# When PLATE_SHIFT_ON, export_plate slides the framing up so the car's ground-contact sits
# at PLATE_CONTACT_TARGET (fraction down the frame), trimming excess floor on floor-heavy
# plates WITHOUT changing perspective or grounding. Module attrs (overridable by job/env)
# so the live default stays OFF until validated. car_spot_px stays correct automatically
# (export_metadata projects through the shifted camera).
PLATE_SHIFT_ON       = os.environ.get("MOTUVA_PLATE_SHIFT", "1") in ("1", "true", "True")
PLATE_CONTACT_TARGET = float(os.environ.get("MOTUVA_PLATE_TARGET", "0.67"))
PLATE_SHIFT_CAP      = float(os.environ.get("MOTUVA_PLATE_CAP", "0.18"))
PLATE_DEADBAND       = float(os.environ.get("MOTUVA_PLATE_DEADBAND", "0.03"))
PLATE_SHIFT_PROBE    = 0.1   # internal probe step to measure the (linear) shift_y->frame slope

def export_plate(payload):
    """ONE production plate: the chosen studio, CAR OFF, camera posed DIRECTLY from the
    ML angle-reader's variable set (render_dataset.py label — azimuth / elevation /
    distance / cam_height / focal / roll). Engineers type the values in manually for the
    prototype; the trained model supplies them per dealer photo later. One photo = one
    set of values = one plate; 12 photos = this, 12 times.

    payload["export"]:
      azimuth_deg   — which way the car faces in the photo (dataset convention). The room
                      never rotates (turntable principle: camera stays in the safe arc,
                      the CAR presents the face), so azimuth doesn't move the background —
                      it's recorded in the metadata for the compositor.
      distance_m    — camera-to-car distance from the photo.
      cam_height_m  — camera height above the floor.
      elevation_deg — camera pitch down/up toward the car. Blank/None = aim at the
                      standard car volume's centre (derived).
      focal_mm      — lens.
      roll_deg      — handheld tilt (default 0).
      photo_w/photo_h — the dealer photo's pixel size; ONLY the aspect ratio is used
                      (plate renders at EXPORT_LONG_EDGE on the long side).
    Saves plate.jpg + metadata.json spec sheet into exports/<timestamp>/."""
    exp = payload.get("export") or {}
    az_car = float(exp.get("azimuth_deg", 135.0))
    dist   = max(2.0, float(exp.get("distance_m", 7.0)))
    h_m    = float(exp.get("cam_height_m", exp.get("height_m", 1.35)))
    focal  = max(20.0, float(exp.get("focal_mm", 50.0)))
    roll   = float(exp.get("roll_deg", 0.0))
    elev   = exp.get("elevation_deg", None)
    pw, ph = max(1, int(exp.get("photo_w", 1600))), max(1, int(exp.get("photo_h", 1200)))
    if pw >= ph: res = (EXPORT_LONG_EDGE, max(64, round(EXPORT_LONG_EDGE * ph / pw)))
    else:        res = (max(64, round(EXPORT_LONG_EDGE * pw / ph)), EXPORT_LONG_EDGE)

    sc = bpy.context.scene
    saved = (sc.render.resolution_x, sc.render.resolution_y, sc.cycles.samples)
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.cycles.samples = EXPORT_SAMPLES

    # sync the scene (room/light/turntable/floor/wall/branding) with the car hidden
    base = {**payload, "car": False}; base.pop("action", None)
    _, note = apply_and_render(base, do_render=False)
    room = base.get("room") or "flatwall"
    if room not in ROOM_TARGETS: room = "flatwall"
    ctop = S["ctop"] if S["ctop"] is not None else rm.FLOOR_Z

    # pose the camera DIRECTLY from the values — no fill-solving, no auto-centring.
    # Room azimuth is fixed at 180 (the safe-arc convention); the camera sits at the
    # photo's distance and height, pitched to the photo's elevation, rolled to match.
    cam = S["cam"]; cam.data.lens = focal
    aim_z_default = ctop + rm.STD_CAR[2] / 2          # standard car volume centre
    dz_guess = h_m + rm.FLOOR_Z - aim_z_default
    horiz = math.sqrt(max(dist*dist - dz_guess*dz_guess, 0.25))
    a = math.radians(180.0)
    cam.location = Vector((rm.CAR_SPOT.x - math.cos(a)*horiz,
                           rm.CAR_SPOT.y - math.sin(a)*horiz,
                           rm.FLOOR_Z + h_m))
    if elev is not None and str(elev) != "":
        aim_z = cam.location.z - math.tan(radians(float(elev))) * horiz
    else:
        aim_z = aim_z_default
    aim = Vector((rm.CAR_SPOT.x, rm.CAR_SPOT.y, aim_z))
    from mathutils import Euler
    q = (aim - cam.location).to_track_quat('-Z', 'Y')
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = q @ Euler((0, 0, radians(roll)), "XYZ").to_quaternion()
    bpy.context.view_layer.update()

    # --- plate reframing: vertical lens-shift to a target contact row (OPT-IN, default OFF) ---
    # Slides framing up so the car's ground-contact lands at PLATE_CONTACT_TARGET (fraction
    # down the frame), trimming excess floor / showing more wall — perspective & grounding
    # unchanged. Self-normalising (already-on-target plate -> shift 0), bounded by the cap so
    # the car is never cropped. The shift_y->frame slope is measured live (linear), so sign-safe.
    cam.data.shift_y = 0.0                       # reset each job (warm worker reuses the camera)
    _framing = {"shift_y": 0.0, "capped": False, "contact_target": None}
    if PLATE_SHIFT_ON:
        from bpy_extras.object_utils import world_to_camera_view
        _contact = Vector((rm.CAR_SPOT.x, rm.CAR_SPOT.y, rm.FLOOR_Z))
        _cy0 = world_to_camera_view(sc, cam, _contact).y          # Blender NDC y: 0=bottom, 1=top
        cam.data.shift_y = PLATE_SHIFT_PROBE                       # measure the linear slope
        _cy1 = world_to_camera_view(sc, cam, _contact).y
        cam.data.shift_y = 0.0
        _slope = (_cy1 - _cy0) / PLATE_SHIFT_PROBE
        _target_cy = 1.0 - PLATE_CONTACT_TARGET                    # "fraction down" -> NDC y
        _shift = 0.0; _capped = False
        if abs(_slope) > 1e-9 and abs(_target_cy - _cy0) > PLATE_DEADBAND:
            _raw = (_target_cy - _cy0) / _slope
            _shift = max(-PLATE_SHIFT_CAP, min(PLATE_SHIFT_CAP, _raw))
            _capped = abs(_raw) > PLATE_SHIFT_CAP + 1e-9
        cam.data.shift_y = _shift
        bpy.context.view_layer.update()
        _framing = {"shift_y": round(_shift, 4), "capped": bool(_capped),
                    "contact_target": PLATE_CONTACT_TARGET}

    # Render to a UNIQUE temp file and read the bytes back. Blender needs a filepath to
    # write_still to, but we must not accumulate an exports/<ts>/ tree per render: the
    # warm worker (worker_entry) relocates the bytes to the caller-supplied out_jpg, so
    # the engine-internal copy is pure disk growth — and a second-resolution timestamp
    # collides when two renders finish in the same wall-clock second. Temp-file render
    # avoids both. The durable exports/<ts>/ copy is OPT-IN (env flag) for standalone /
    # WebSocket use, and uses a uniquified dir name so same-second renders never clash.
    rfd, rpath = tempfile.mkstemp(suffix=".jpg", prefix="plate_"); os.close(rfd)
    sc.render.filepath = rpath
    bpy.ops.render.render(write_still=True)
    with open(rpath, "rb") as f: data = f.read()
    try: os.unlink(rpath)
    except OSError: pass

    meta = rm.export_metadata(cam, 180.0, h_m, focal, room)
    meta["ml_values"] = {"azimuth_deg": az_car, "elevation_deg": (float(elev) if elev not in (None, "") else
                                                                  round(math.degrees(math.atan2(cam.location.z - aim_z_default, horiz)), 2)),
                         "distance_m": dist, "cam_height_m": h_m, "focal_mm": focal, "roll_deg": roll,
                         "photo_size": [pw, ph], "plate_resolution": list(res)}
    meta["selection"] = {k: base.get(k) for k in
                         ("room", "light", "turntable", "floor", "wall", "branding", "treatment", "led")}
    if note: meta["pipeline_note"] = note
    meta["framing"] = _framing

    # Durable engine-internal copy: OPT-IN only (MOTUVA_EXPORT_DISK=1). Off by default so
    # the warm worker — which already writes the plate to the caller's paths — does not
    # double-write or grow exports/ unbounded. The dir name carries pid + a per-process
    # counter alongside the timestamp so two same-second renders never overwrite.
    if os.environ.get("MOTUVA_EXPORT_DISK") in ("1", "true", "True"):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        global _EXPORT_SEQ
        _EXPORT_SEQ += 1
        outdir = os.path.join(PKG, "exports", f"{ts}-{os.getpid()}-{_EXPORT_SEQ}")
        os.makedirs(outdir, exist_ok=True)
        fp = os.path.join(outdir, "plate.jpg")
        with open(fp, "wb") as f: f.write(data)
        with open(os.path.join(outdir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[export] plate → {fp} ({len(data)//1024} KB)", flush=True)
    else:
        print(f"[export] plate rendered ({len(data)//1024} KB) — "
              f"engine-internal disk copy skipped (MOTUVA_EXPORT_DISK unset)", flush=True)

    sc.render.resolution_x, sc.render.resolution_y, sc.cycles.samples = saved
    cam.rotation_mode = "XYZ"   # the preview rig drives rotation_euler — restore it
    S["last"] = {}   # reset diff so the next live preview re-applies cleanly
    return data, meta

# ── modes ──
def selftest():
    setup()
    # keys come from the live catalog — hand-picked keys go stale when the library is culled
    cat = json.load(open(os.path.join(PKG, "configurator", "catalog.json")))
    fl = [m["key"] for m in cat["floors"][:3]]
    wl = [m["key"] for m in cat["walls"][:2]]
    for i, pl in enumerate([
        {"room": "flatwall", "light": "panels", "turntable": "flush", "floor": fl[0], "wall": wl[0], "angle": "front_q", "height": "eye"},
        {"room": "curved", "light": "led", "turntable": "raised", "floor": fl[1], "wall": wl[1], "angle": "side", "height": "low"},
        {"room": "cove", "light": "none", "turntable": "none", "floor": fl[2], "angle": "front", "height": "eye"},
    ]):
        data, note = apply_and_render(pl)
        out = f"/tmp/worker_selftest_{i}_{pl['room']}.jpg"
        open(out, "wb").write(data)
        print(f"[selftest] {pl['room']:9} → {out}  ({len(data)//1024} KB) note={note}", flush=True)

def serve():
    import asyncio, websockets
    setup()
    async def handle(ws):
        async for msg in ws:
            try:
                payload = json.loads(msg)
                if payload.get("action") in ("export", "export_plate"):
                    data, meta = export_plate(payload)
                    await ws.send(data)                       # the plate (browser downloads it)
                    await ws.send(json.dumps({"exported": True, "meta": meta}))
                    continue
                data, note = apply_and_render(payload)
                await ws.send(data)
                if note: await ws.send(json.dumps({"note": note}))
            except Exception as e:
                print("[server] render error:", e, flush=True)
                await ws.send(json.dumps({"error": str(e)}))
    async def main():
        # ping disabled: long synchronous renders (car-on ~25s, export ~2min) block the loop and would
        # otherwise trip the keepalive timeout and drop the connection.
        async with websockets.serve(handle, "127.0.0.1", PORT, max_size=None, ping_interval=None):
            print(f"[server] live render → ws://localhost:{PORT}", flush=True)
            await asyncio.Future()
    asyncio.run(main())

if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    (selftest if "selftest" in argv else serve)()
