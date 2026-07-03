"""
render_master.py — Motuva master studio rig: multi-room, per-room safe arc, plate + metadata
=============================================================================================

Generalises the proven cove rig (~/Desktop/Blender/test-rig/render_studio.py — left untouched) into a
MULTI-ROOM rig driven by a per-room contract (see ROOM-CONTRACT.md). The rig is fixed; every room is
built to satisfy it. The one global that became per-room is the SAFE ARC.

Two hard rules carried from the cove (STUDIO-RIG.md):
  1. The camera lives in the room's backdrop arc and NEVER crosses it, or the room's edge/void enters frame.
     The arc is now PER ROOM (cove = wide 90-270; flat-wall = constrained).
  2. The car ROTATES to present each face (turntable); the camera does not circle it.

Rooms register in ROOMS below. Each declares: safe_arc, floor object name (for grounding), a builder
(None = geometry already in the .blend, e.g. the cove), and branding_zones.

Run
---
    # single plate, cove (baseline):
    blender --background motuva-studio-master.blend --python render_master.py -- \
        --room cove --az 180 --height 1.35 --out plate.jpg

    # single plate, flat-wall:
    blender --background motuva-studio-master.blend --python render_master.py -- \
        --room flatwall --az 180 --out plate.jpg

    # contact sheet across the arc (one session, montaged):
    blender --background motuva-studio-master.blend --python render_master.py -- \
        --room flatwall --sheet 120,140,160,180,200,220,240 --out flatwall_arc.jpg

NOTE: this script never saves the .blend. Procedural rooms (e.g. flat-wall) are built in-memory at render
time. A room is baked into motuva-studio-master.blend only once Chris approves its look.
"""

import bpy, math, os, sys, json
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ── Universal invariants (ROOM-CONTRACT §1 — identical in every room) ────────
CAR_SPOT = Vector((-0.08, -1.29))   # turntable centre on the floor
FLOOR_Z  = 0.16                     # flat shooting-pad height
TT_DIAM  = 6.0                      # turntable clear diameter
STD_CAR  = (4.60, 1.90, 1.45)       # standard car box (L×W×H) used to frame the plate

# ── Room registry (ROOM-CONTRACT §3) ─────────────────────────────────────────
# builder: None = geometry already in the .blend; else a fn that builds the room in-memory.
# floor:   object the rig ray-casts for grounding.
# safe_arc:(min,max) azimuth the camera may occupy.
# branding_zones: list of (name, centre(x,y,z), width, height, normal_axis) rects, or [].
def _room_flatwall():
    return build_flatwall()

def _room_curved():
    return build_curved()

ROOMS = {
    "cove": {
        "safe_arc": (90, 270),
        "floor": "canvas",
        "builder": None,
        "branding_zones": [],
    },
    "flatwall": {
        "safe_arc": (135, 225),          # constrained — confirm true limit from the arc sheet
        "floor": "flatwall_floor",
        "builder": _room_flatwall,
        "branding_zones": [
            # wide band lifted into the headroom above the car's shoulder — uses the empty upper wall
            ("wall_logo", (CAR_SPOT.x - 5.97, CAR_SPOT.y, FLOOR_Z + 2.05), 7.0, 1.1, "+X"),
        ],
    },
    "curved": {
        "safe_arc": (75, 285),               # wider than flat — no corners to catch
        "floor": "curved_floor",
        "builder": _room_curved,
        "branding_zones": [],                # branding on a curved wall is a later call (logo follows the curve)
    },
}

# ─────────────────────────────────────────────────────────────────────────────
def enable_gpu():
    try:
        pr = bpy.context.preferences.addons['cycles'].preferences
        # Prefer OPTIX on NVIDIA (faster on RTX/Ada) over plain CUDA; METAL for Apple,
        # HIP for AMD. The first backend that exposes a matching (non-CPU) device wins.
        chosen = None
        for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL'):
            try:
                pr.compute_device_type = backend; pr.get_devices()
                if any(d.type == backend for d in pr.devices):
                    chosen = backend; break
            except Exception:
                continue
        # Enable ONLY GPU-class devices. The CUDA/OPTIX backend also lists a CPU
        # pseudo-device; enabling it makes Cycles render hybrid GPU+CPU, stealing cores
        # from the preprocess pool / FLUX host — NOT the "force Cycles to GPU" behaviour
        # the worker claims.
        n_gpu = 0
        for d in pr.devices:
            d.use = (d.type != 'CPU')
            if d.use:
                n_gpu += 1
        bpy.context.scene.cycles.device = 'GPU'
        # Verify a GPU was actually enabled. If get_devices() reported only a CPU,
        # device='GPU' silently falls back to CPU (10x-slower renders discovered only
        # at run time). Surface it loudly at warm-up instead.
        if n_gpu == 0:
            print("GPU setup WARNING: no GPU compute device found "
                  f"(backend={chosen}); Cycles will fall back to CPU.", flush=True)
        else:
            print(f"GPU enabled: backend={chosen}, {n_gpu} GPU device(s) active",
                  flush=True)
    except Exception as e:
        print("GPU setup skipped:", e)

# ── Room activation ──────────────────────────────────────────────────────────
def _neutral_mat(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*rgb, 1)
    bsdf.inputs['Roughness'].default_value = 0.85
    return m

def build_flatwall():
    """Three-wall open-fronted room: a flat BACK wall on -X plus LEFT and RIGHT side walls running
    forward toward the open front (+X, the rig side). The open front is where the camera shoots in —
    the side walls fill the void that a single wall left at the edges of the arc. Mockup look only;
    materials/lighting are later contract layers. Returns the floor object name."""
    BACK_X  = CAR_SPOT.x - 6.0      # back wall 6 m behind the car (tightened from 8 m)
    FRONT_X = CAR_SPOT.x + 4.5      # side walls run forward to here; the front stays open for the rig
    HALF_W  = 6.5                   # side walls at car-spot.y ± this  (room ~13 m wide, tightened from 18)
    WALL_H  = 10.0                  # wall height (top edge never in frame)
    depth   = FRONT_X - BACK_X
    midX    = (BACK_X + FRONT_X) / 2
    yL, yR  = CAR_SPOT.y + HALF_W, CAR_SPOT.y - HALF_W

    bpy.ops.mesh.primitive_plane_add(size=44.0, location=(CAR_SPOT.x, CAR_SPOT.y, FLOOR_Z))
    floor = bpy.context.active_object; floor.name = "flatwall_floor"
    floor.data.materials.clear(); floor.data.materials.append(_neutral_mat("flatwall_floor_mat", (0.32, 0.32, 0.34)))

    wall_mat = _neutral_mat("flatwall_wall_mat", (0.55, 0.55, 0.57))
    def _wall(name, loc, rot, scale):
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc)
        w = bpy.context.active_object; w.name = name
        w.rotation_euler = rot; w.scale = scale
        w.data.materials.clear(); w.data.materials.append(wall_mat)
        return w

    _wall("flatwall_wall_back",  (BACK_X, CAR_SPOT.y, FLOOR_Z + WALL_H/2), (0, math.radians(90), 0), (WALL_H, 2*HALF_W, 1.0))
    _wall("flatwall_wall_left",  (midX,   yL,         FLOOR_Z + WALL_H/2), (math.radians(90), 0, 0), (depth, WALL_H, 1.0))
    _wall("flatwall_wall_right", (midX,   yR,         FLOOR_Z + WALL_H/2), (math.radians(90), 0, 0), (depth, WALL_H, 1.0))
    bpy.context.view_layer.update()
    return "flatwall_floor"

def _logo_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (0.86, 0.16, 0.05, 1)
    bsdf.inputs['Emission Color'].default_value = (0.90, 0.20, 0.06, 1)
    bsdf.inputs['Emission Strength'].default_value = 1.4
    return m

def _build_logo_strip(name, cx, cy, radius, theta_centre, arc_width, z_centre, height, segs=24):
    """Curved logo container that follows a wall arc (adapted from the old configurator's
    _ensure_logozone_curved_geometry). Verts along the arc at bottom + top edges; quad faces."""
    eff_r = radius - 0.03                                  # 3 cm in front of the wall (no z-fight)
    half  = arc_width / (2.0 * eff_r)
    t0, t1 = theta_centre - half, theta_centre + half
    zb, zt = z_centre - height/2.0, z_centre + height/2.0
    verts = [(cx + eff_r*math.cos(t0 + (t1-t0)*i/segs), cy + eff_r*math.sin(t0 + (t1-t0)*i/segs), zb)
             for i in range(segs+1)]
    verts += [(cx + eff_r*math.cos(t0 + (t1-t0)*i/segs), cy + eff_r*math.sin(t0 + (t1-t0)*i/segs), zt)
              for i in range(segs+1)]
    n = segs + 1
    faces = [(i, i+1, n+i+1, n+i) for i in range(segs)]
    mesh = bpy.data.meshes.new(name + "_mesh"); mesh.from_pydata(verts, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh); bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(_logo_mat(name + "_mat"))
    return obj

def add_wall_logo(room_name):
    """Drop a placeholder logo (emissive bar) so we can see when branding is in the tight frame and when
    the camera angle drops it out. Flat rooms get a flat plane; curved rooms get an arced strip that
    follows the wall radius (the container 'arcs to the wall')."""
    if room_name == "curved":
        # locked logo band (z ≈ 2.2 m, ~7 m on the arc, ~1.1 m tall), centred directly behind the car (θ=180°)
        _build_logo_strip("logo_curved", CAR_SPOT.x, CAR_SPOT.y, CURVED_R,
                          math.radians(180), 7.0, FLOOR_Z + 2.05, 1.1)
        return
    for (zname, c, w, h, _axis) in ROOMS[room_name]["branding_zones"]:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(c[0] + 0.03, c[1], c[2]))
        p = bpy.context.active_object; p.name = "logo_" + zname
        p.rotation_euler = (0, math.radians(90), 0); p.scale = (h, w, 1.0)
        p.data.materials.clear(); p.data.materials.append(_logo_mat("logo_" + zname + "_mat"))

# Curved room geometry — shared by the wall builder and the arced-logo builder
CURVED_R  = 8.0                     # backdrop radius from the car-spot
CURVED_H  = 10.0                    # wall height
CURVED_A0, CURVED_A1 = math.radians(55), math.radians(305)   # arc; gap at the front (+X) for the rig

def build_curved():
    """Curved room: flat floor pad + a concave curved backdrop wall standing on it (visible floor-to-wall
    line, no corners). Wraps the back ~250°, open at the front (+X) for the rig. Distinct from the cove
    (Infinity), whose floor sweeps up seamlessly with no floor line. Mockup look only. Returns floor name."""
    R   = CURVED_R
    H   = CURVED_H
    a0, a1 = CURVED_A0, CURVED_A1
    segs = 64
    cx, cy = CAR_SPOT.x, CAR_SPOT.y

    bpy.ops.mesh.primitive_plane_add(size=44.0, location=(cx, cy, FLOOR_Z))
    floor = bpy.context.active_object; floor.name = "curved_floor"
    floor.data.materials.clear(); floor.data.materials.append(_neutral_mat("curved_floor_mat", (0.32, 0.32, 0.34)))

    verts, faces = [], []
    for i in range(segs + 1):
        t = a0 + (a1 - a0) * i / segs
        x, y = cx + R*math.cos(t), cy + R*math.sin(t)
        verts.append((x, y, FLOOR_Z)); verts.append((x, y, FLOOR_Z + H))
    for i in range(segs):
        b0, t0, b1, t1 = 2*i, 2*i+1, 2*i+2, 2*i+3
        faces.append((b0, b1, t1, t0))
    mesh = bpy.data.meshes.new("curved_wall_mesh"); mesh.from_pydata(verts, [], faces); mesh.update()
    wall = bpy.data.objects.new("curved_wall", mesh); bpy.context.scene.collection.objects.link(wall)
    wall.data.materials.append(_neutral_mat("curved_wall_mat", (0.55, 0.55, 0.57)))
    bpy.context.view_layer.update()
    return "curved_floor"

# Baked room collections (created by bake_master.py). The rig toggles these; if a room isn't baked yet
# it falls back to building procedurally in-memory.
ROOM_COLLECTIONS = {
    "cove":     "Room_Infinity (cove)",
    "flatwall": "Room_Flat",
    "curved":   "Room_Curved",
}

# Switchable wall-light options baked per room (bake_lights.py). "none" = overhead only (no collection on).
# Infinity (cove) has no wall lights by design. Options: none | led | fins | panels.
LIGHT_COLLECTIONS = {
    ("flatwall", "led"):    "Light_Flat_Led",     ("flatwall", "fins"): "Light_Flat_Fins",
    ("flatwall", "panels"): "Light_Flat_Panels",
    ("curved",   "led"):    "Light_Curved_Led",   ("curved",   "fins"): "Light_Curved_Fins",
    ("curved",   "panels"): "Light_Curved_Panels",
}

def activate_light(room, option):
    """Show the chosen wall-light collection for this room; hide every other option. 'none'/cove → all off."""
    for cname in set(LIGHT_COLLECTIONS.values()):
        c = bpy.data.collections.get(cname)
        if c:
            for o in c.objects: o.hide_render = True; o.hide_viewport = True
    if option and option != "none":
        cname = LIGHT_COLLECTIONS.get((room, option))
        c = bpy.data.collections.get(cname) if cname else None
        if c:
            for o in c.objects: o.hide_render = False; o.hide_viewport = False
        elif room != "cove":
            print(f"⚠ no baked light option '{option}' for room '{room}' — overhead only")

def activate_room(name):
    """Show only this room's geometry (lights stay shared). Toggle baked collections if present, else
    build the room procedurally. Returns the floor object name."""
    room = ROOMS[name]
    baked = bpy.data.collections.get(ROOM_COLLECTIONS.get(name, ""))
    if not baked and room["builder"]:
        room["builder"]()                      # not baked — build in-memory

    # show the active room's collection, hide every other room's
    for rname, cname in ROOM_COLLECTIONS.items():
        c = bpy.data.collections.get(cname)
        if not c: continue
        hide = (rname != name)
        for o in c.objects:
            o.hide_render = hide; o.hide_viewport = hide

    # belt-and-braces for the procedural path: keep the cove backdrop hidden unless we're in the cove
    cove_backdrop = bpy.data.objects.get('canvas')
    if cove_backdrop:
        cove_backdrop.hide_render = (name != "cove")
        cove_backdrop.hide_viewport = (name != "cove")

    # ambient floor so a mockup room is never pure black (lighting is a later layer)
    if name != "cove" and bpy.context.scene.world:
        bg = bpy.context.scene.world.node_tree.nodes.get('Background')
        if bg: bg.inputs['Strength'].default_value = max(bg.inputs['Strength'].default_value, 0.15)
    return room["floor"]

# ── Grounding ────────────────────────────────────────────────────────────────
def floor_z_at(x, y, floor_name):
    """True floor-surface height under (x,y) — raycast the room's floor object only."""
    fl = bpy.data.objects.get(floor_name)
    if not fl: return FLOOR_Z
    deps = bpy.context.evaluated_depsgraph_get()
    fe = fl.evaluated_get(deps)
    mw = fl.matrix_world; mwi = mw.inverted()
    o_l = mwi @ Vector((x, y, 8.0))
    d_l = (mwi.to_3x3() @ Vector((0, 0, -1))).normalized()
    hit, loc, *_ = fe.ray_cast(o_l, d_l)
    return (mw @ loc).z if hit else FLOOR_Z

def assert_safe(az, arc):
    a = az % 360
    if not (arc[0] <= a <= arc[1]):
        print(f"⚠ azimuth {az}° is OUTSIDE this room's safe arc {arc} — the room edge/void will show. "
              f"Keep the camera in-arc and rotate the CAR instead.")

# ── Car (optional QA preview) ────────────────────────────────────────────────
def car_meshes(objs): return [o for o in objs if o.type == 'MESH']

def world_z_min(meshes):
    deps = bpy.context.evaluated_depsgraph_get(); mn = 1e9
    for o in meshes:
        ev = o.evaluated_get(deps); m = ev.to_mesh()
        for v in m.vertices: mn = min(mn, (ev.matrix_world @ v.co).z)
        ev.to_mesh_clear()
    return mn

def cache_car_points(rig, meshes, cap=2000):
    """Measure the car ONCE at load. Returns (pts_local, zmin_local) in rig-local space:
    a ~cap-point subsample of the body for framing, and the exact lowest z for grounding.
    The rig only rotates about Z and translates (uniform scale), so afterwards:
      world framing points = rig.matrix_world @ p   (cap points, not millions)
      world z-min          = rig.location.z + rig.scale.z * zmin_local   (Z-rotation preserves z)
    Kills the per-settings-change full-vertex sweeps (8-25s → ms)."""
    deps = bpy.context.evaluated_depsgraph_get()
    inv = rig.matrix_world.inverted()
    pts, zmin = [], 1e9
    per_mesh = max(1, cap // max(1, len(meshes)))
    for o in meshes:
        ev = o.evaluated_get(deps); m = ev.to_mesh()
        M = inv @ ev.matrix_world
        n = len(m.vertices)
        step = max(1, n // per_mesh)
        for i in range(0, n, step):
            pts.append(M @ m.vertices[i].co)
        for v in m.vertices:                       # exact z-min — full sweep, but only ONCE
            z = (M @ v.co).z
            if z < zmin: zmin = z
        ev.to_mesh_clear()
    return pts, zmin


def load_car(car_blend, floor_name, length=4.75):
    with bpy.data.libraries.load(car_blend, link=False) as (src, dst):
        dst.objects = list(src.objects)
    objs = [o for o in dst.objects if o]; roots = []
    for o in objs:
        bpy.context.scene.collection.objects.link(o)
        if o.parent is None: roots.append(o)
    # Rigged cars (cars_bk) ship posed in a drift/steer stance via an armature — front wheels turned to
    # lock, looks broken on quarter/front angles. Force every armature to its REST pose → wheels straight.
    for o in objs:
        if o.type == 'ARMATURE':
            o.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    meshes = car_meshes(objs)
    # Rigged cars (cars_bk) carry control-gizmo meshes that sit below the wheels and skew grounding,
    # framing and scale. Measure from real body geometry only; hide the gizmos from render.
    SKIP = ("ctrl", "steering", "drive", "master", "gizmo", "widget", "handle", "bone", "_empty", "root")
    body = [o for o in meshes if not any(k in o.name.lower() for k in SKIP)]
    if not body: body = meshes
    for o in meshes:
        if o not in body:
            o.hide_render = True; o.hide_viewport = True
    meshes = body
    deps = bpy.context.evaluated_depsgraph_get()
    mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
    for o in meshes:
        ev = o.evaluated_get(deps); m = ev.to_mesh()
        for v in m.vertices:
            w = ev.matrix_world @ v.co
            for i in range(3): mn[i] = min(mn[i], w[i]); mx[i] = max(mx[i], w[i])
        ev.to_mesh_clear()
    bpy.ops.object.empty_add(location=((mn.x+mx.x)/2, (mn.y+mx.y)/2, (mn.z+mx.z)/2))
    rig = bpy.context.active_object; rig.name = "CAR_RIG"
    for r in roots: r.parent = rig; r.matrix_parent_inverse = rig.matrix_world.inverted()
    rig.scale = ([length / max(mx.x-mn.x, mx.y-mn.y)] * 3)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
    for o in meshes:
        ev = o.evaluated_get(deps); m = ev.to_mesh()
        for v in m.vertices:
            w = ev.matrix_world @ v.co
            for i in range(3): mn[i] = min(mn[i], w[i]); mx[i] = max(mx[i], w[i])
        ev.to_mesh_clear()
    rig.location.x += CAR_SPOT.x - (mn.x+mx.x)/2
    rig.location.y += CAR_SPOT.y - (mn.y+mx.y)/2
    bpy.context.view_layer.update()
    rig.location.z += floor_z_at(CAR_SPOT.x, CAR_SPOT.y, floor_name) - world_z_min(meshes)
    bpy.context.view_layer.update()
    return rig, meshes

# ── Turntable (optional visible disc) ────────────────────────────────────────
def add_turntable(style="flush"):
    R = TT_DIAM / 2
    if style == "flush":
        bpy.ops.mesh.primitive_torus_add(location=(CAR_SPOT.x, CAR_SPOT.y, FLOOR_Z-0.015),
            major_radius=R, minor_radius=0.045, major_segments=128, minor_segments=12)
        o = bpy.context.active_object; _tt_mat(o, (0.05, 0.05, 0.06)); return FLOOR_Z
    if style == "raised":
        bpy.ops.mesh.primitive_cylinder_add(radius=R, depth=0.07,
            location=(CAR_SPOT.x, CAR_SPOT.y, FLOOR_Z+0.035), vertices=128)
        plat = bpy.context.active_object; _tt_mat(plat, (0.55, 0.55, 0.57))
        return FLOOR_Z + 0.07
    return FLOOR_Z

def _tt_mat(o, rgb):
    m = bpy.data.materials.new("tt"); m.use_nodes = True
    m.node_tree.nodes['Principled BSDF'].inputs[0].default_value = (*rgb, 1)
    o.data.materials.clear(); o.data.materials.append(m)

# ── Camera rig: backdrop-arc + auto-frame to 80% fill + centre both axes ─────
def setup_camera():
    cd = bpy.data.cameras.new("rig"); cam = bpy.data.objects.new("rig", cd)
    bpy.context.scene.collection.objects.link(cam); bpy.context.scene.camera = cam
    return cam

def _frame_points(meshes, car_z_top):
    if meshes:
        deps = bpy.context.evaluated_depsgraph_get(); pts = []
        for o in meshes:
            ev = o.evaluated_get(deps); m = ev.to_mesh()
            pts += [ev.matrix_world @ v.co for v in m.vertices]; ev.to_mesh_clear()
        return pts
    L, W, H = STD_CAR
    cz0 = FLOOR_Z; cz1 = FLOOR_Z + H
    return [Vector((CAR_SPOT.x+sx*L/2, CAR_SPOT.y+sy*W/2, z))
            for sx in (-1, 1) for sy in (-1, 1) for z in (cz0, cz1)]

def set_shot(cam, az, arc, height, lens=50, fill=0.80, car_z_top=None, meshes=None, iters=9, lock_h=False, pts=None, centre_pt=None):
    """lock_h=True: keep the horizontal frame centre on the turntable axis (the car-spot) instead of
    panning to the car's projected silhouette. The car spins about that axis so it stays centred, and the
    wall logo (centred behind the spot) lands dead-centre on every shot. Vertical fit + fill unchanged.
    pts: pre-computed world-space framing points (from cache_car_points) — pass these instead of meshes
    to skip the full per-vertex sweep.
    centre_pt: a world point (the wall logo anchor) that must ALSO land on the horizontal frame centre.
    Pan alone can't centre car and logo together on quarter shots (different depths, off-axis body) —
    so the camera steps SIDEWAYS, photographer-style, until the parallax lines the anchor up behind the
    car's silhouette centre, then the pan centres both. Lateral step capped ±1.5m to stay in-arc."""
    assert_safe(az, arc)
    scene = bpy.context.scene; cam.data.lens = lens
    a = math.radians(az); d = 11.0; lat = 0.0
    lat_dir = Vector((math.sin(a), -math.cos(a), 0.0))
    car_z_top = car_z_top if car_z_top is not None else FLOOR_Z + STD_CAR[2]
    aim = Vector((CAR_SPOT.x, CAR_SPOT.y, (FLOOR_Z + car_z_top) / 2))
    pts = pts if pts is not None else _frame_points(meshes, car_z_top)
    for _ in range(iters):
        cam.location = Vector((CAR_SPOT.x - math.cos(a)*d, CAR_SPOT.y - math.sin(a)*d,
                               FLOOR_Z + height)) + lat_dir * lat
        cam.rotation_euler = (aim - cam.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        xs = []; ys = []
        for p in pts:
            c = world_to_camera_view(scene, cam, p)
            if c.z > 0: xs.append(c.x); ys.append(c.y)
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        d *= max(maxx-minx, maxy-miny) / fill
        cxn, cyn = (minx+maxx)/2, (miny+maxy)/2
        fw = d * cam.data.sensor_width / lens
        fh = fw * scene.render.resolution_y / scene.render.resolution_x
        R = cam.matrix_world.to_3x3()
        aim = aim + (R @ Vector((0,1,0)))*((cyn-0.5)*fh)          # vertical centring always
        if not lock_h:
            aim = aim + (R @ Vector((1,0,0)))*((cxn-0.5)*fw)     # horizontal pan to car silhouette (default)
            if centre_pt is not None:
                cw = world_to_camera_view(scene, cam, centre_pt).x
                lat = max(-1.5, min(1.5, lat - (cw - cxn) * fw * 0.6))   # parallax step toward alignment
    return cam

# ── Metadata spec sheet ──────────────────────────────────────────────────────
def export_metadata(cam, az, height, lens, room_name):
    scene = bpy.context.scene; rx, ry = scene.render.resolution_x, scene.render.resolution_y
    def to_px(p):
        c = world_to_camera_view(scene, cam, p); return (round(c.x*rx, 1), round((1-c.y)*ry, 1))
    spot3 = Vector((CAR_SPOT.x, CAR_SPOT.y, FLOOR_Z))
    Rm = cam.matrix_world.to_3x3()
    right = (Rm @ Vector((1,0,0))); right_floor = Vector((right.x, right.y, 0)).normalized()
    toward = (cam.location - spot3); toward_floor = Vector((toward.x, toward.y, 0)).normalized()
    p0 = to_px(spot3); pl = to_px(spot3 + right_floor); pd = to_px(spot3 + toward_floor)
    pxm_lat   = math.dist(p0, pl); pxm_depth = math.dist(p0, pd)
    room = ROOMS[room_name]
    zones = []
    for (zname, c, w, h, _axis) in room["branding_zones"]:
        cv = Vector(c)
        zones.append({"name": zname, "centre_m": [round(v, 3) for v in c],
                      "width_m": w, "height_m": h, "centre_px": to_px(cv)})
    return {
        "room": room_name,
        "safe_arc": list(room["safe_arc"]),
        "camera": {"azimuth_deg": az, "height_m": round(FLOOR_Z+height, 3),
                   "lens_mm": lens, "sensor_mm": cam.data.sensor_width, "resolution": [rx, ry]},
        "car_spot_px": list(p0),
        "pixels_per_metre": {"lateral": round(pxm_lat, 1), "toward_camera": round(pxm_depth, 1)},
        "floor_z": FLOOR_Z,
        "branding_zones": zones,
        "note": "Place car ground-contact centre at car_spot_px; scale so car_length_m * "
                "pixels_per_metre.lateral = width in px. Wheels sit on the floor.",
    }

# ── Render + montage ─────────────────────────────────────────────────────────
def render(out_path, samples=140, res=(1280, 960)):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; enable_gpu()
    scene.cycles.samples = samples
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

def montage(paths, labels, cols, out_path):
    """Tile rendered JPGs into one contact sheet using numpy + Blender's image IO (no PIL/IM needed)."""
    import numpy as np
    cells = []
    cw = ch = None
    for p in paths:
        img = bpy.data.images.load(p, check_existing=False)
        w, h = img.size
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)  # bottom-up, linear, RGBA
        cells.append(arr); cw, ch = w, h
        bpy.data.images.remove(img)
    rows = (len(cells) + cols - 1) // cols
    pad = 8
    W = cols * cw + (cols + 1) * pad
    H = rows * ch + (rows + 1) * pad
    canvas = np.full((H, W, 4), [0.02, 0.02, 0.02, 1.0], dtype=np.float32)  # linear dark grey
    for i, arr in enumerate(cells):
        r = i // cols; c = i % cols
        # numpy buffer is bottom-up; place from the bottom so reading-order matches az order top-left→
        y0 = (rows - 1 - r) * ch + (rows - r) * pad
        x0 = c * cw + (c + 1) * pad
        canvas[y0:y0+ch, x0:x0+cw, :] = arr
    out_img = bpy.data.images.new("contact", width=W, height=H, alpha=True)
    out_img.colorspace_settings.name = 'sRGB'
    out_img.pixels = canvas.ravel()
    out_img.file_format = 'JPEG'
    out_img.filepath_raw = os.path.abspath(out_path)
    out_img.save()
    print("contact sheet:", out_path, f"({cols}×{rows}, az order L→R, T→B:", labels, ")")

# ── CLI ──────────────────────────────────────────────────────────────────────
def _args():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    a = {"room": "cove", "az": 180.0, "height": 1.35, "lens": 50.0, "car": None, "car_blend": None,
         "turntable": "flush", "out": "out.jpg", "fill": 0.80, "sheet": None, "matrix": None, "logo": None,
         "light": "none", "logo_style": None, "logo_img": None, "logo_svg": None,
         "heights": None, "samples": None, "rots": None}
    it = iter(argv)
    for k in it:
        key = k.lstrip("-").replace("-", "_")
        if key in a: a[key] = next(it)
    a["az"], a["height"], a["lens"], a["fill"] = float(a["az"]), float(a["height"]), float(a["lens"]), float(a["fill"])
    if a["car"] is not None: a["car"] = float(a["car"])
    return a

def main():
    a = _args()
    a["out"] = os.path.abspath(a["out"])   # Blender's cwd may not be the package dir — resolve up front
    res = (1280, 960)
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = res
    floor_name = activate_room(a["room"])
    activate_light(a["room"], a["light"])
    arc = ROOMS[a["room"]]["safe_arc"]
    if a["logo"]: add_wall_logo(a["room"])
    if a["logo_style"]:
        import branding
        _logo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Motuva-Studio-Materials", "logo", "test_logos")  # Windows: dev fallback (file not in bundle)
        _li = a["logo_img"] or os.path.join(_logo_dir, "dealertalk_trimmed.png")
        _ls = a["logo_svg"] or os.path.join(_logo_dir, "dealertalk.svg")
        branding.apply_branding(a["logo_style"], _li, _li, _ls, room=a["room"])
    car_z_top = add_turntable(a["turntable"])
    meshes = None; rig = None
    if a["car_blend"]:
        rig, meshes = load_car(a["car_blend"], floor_name)
        rig.location.z += (car_z_top - FLOOR_Z)
        rig.rotation_euler[2] = math.radians(a["car"] or 0.0)
        bpy.context.view_layer.update()
        car_z_top = world_z_min(meshes) + STD_CAR[2]
    cam = setup_camera()

    if a["matrix"]:
        # camera fixed facing the wall (--az, default 180); CAR spins to present each face (cols),
        # camera height steps low / eye / high-down (rows). Locked 80% framing throughout.
        rots    = [0, 45, 90, 135, 180]          # rear → rear-¾ → side → front-¾ → front
        heights = [0.70, 1.35, 3.50]             # low · eye-level · high looking down
        if a.get("heights"): heights = [float(x) for x in a["heights"].split(",")]
        if a.get("rots"): rots = [float(x) for x in a["rots"].split(",")]
        tmpdir = os.path.join(os.path.dirname(os.path.abspath(a["out"])) or ".", "_matrix_frames")
        os.makedirs(tmpdir, exist_ok=True)
        paths, labels = [], []
        for hgt in heights:
            for rot in rots:
                if rig:
                    rig.rotation_euler[2] = math.radians(rot); bpy.context.view_layer.update()
                set_shot(cam, a["az"], arc, hgt, a["lens"], a["fill"], car_z_top=car_z_top, meshes=meshes)
                fp = os.path.join(tmpdir, f"h{int(hgt*100)}_r{rot}.jpg")
                render(fp, samples=int(a["samples"]) if a["samples"] else 140, res=(640, 480))
                paths.append(fp); labels.append(f"h{hgt}/r{rot}")
        montage(paths, labels, len(rots), a["out"])
        print("matrix:", tmpdir, "| cols=car-rotation(rear→front), rows=height(low/eye/high)")
        return

    if a["sheet"]:
        azis = [float(x) for x in a["sheet"].split(",")]
        tmpdir = os.path.join(os.path.dirname(os.path.abspath(a["out"])) or ".", "_arc_frames")
        os.makedirs(tmpdir, exist_ok=True)
        paths, labels = [], []
        for az in azis:
            set_shot(cam, az, arc, a["height"], a["lens"], a["fill"], car_z_top=car_z_top, meshes=meshes)
            fp = os.path.join(tmpdir, f"az{int(az)}.jpg")
            render(fp, res=(640, 480))
            paths.append(fp); labels.append(f"az{int(az)}")
        cols = min(4, len(paths))
        montage(paths, labels, cols, a["out"])
        print("frames:", tmpdir); print("arc (safe):", arc, "| rendered:", labels)
        return

    set_shot(cam, a["az"], arc, a["height"], a["lens"], a["fill"], car_z_top=car_z_top, meshes=meshes,
             lock_h=bool(a["logo_style"]))   # branded shots centre the logo on the turntable axis
    render(a["out"], res=res)
    meta = export_metadata(cam, a["az"], a["height"], a["lens"], a["room"])
    with open(os.path.splitext(a["out"])[0] + ".json", "w") as f: json.dump(meta, f, indent=2)
    print("rendered:", a["out"]); print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
