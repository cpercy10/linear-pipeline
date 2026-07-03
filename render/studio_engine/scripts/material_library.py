"""Motuva Studio — MATERIAL LIBRARY (single source of truth).

The curated set of floor + wall finishes appropriate to a PREMIUM car photo studio.
Built from our existing Poliigon library (~/Desktop/Blender/Materials/_downloaded) + targeted
CC0 additions from ambientCG (microcement / clean plaster / subtle terrazzo). Materials that
don't belong in a premium car studio (carpet, acoustic foam, rammed earth, worn metal, decorative
glass, curtains, brick, small busy tiles, herringbone, loud coloured stone) are deliberately EXCLUDED.

Consumed by: render_master / mock_materials (apply a finish) and the configurator (list choices).
Each entry resolves to a folder of PBR maps: color.jpg · roughness.jpg · normal.png · (ao.jpg) · (metallic.jpg).

Fields:
  key      stable id used by the configurator
  name     dealer-facing display name
  cat      'floor' | 'wall'
  family   concrete | microcement | epoxy | terrazzo | marble | porcelain | wood | plaster | stone
  tone     light | mid | dark | warm | white | black
  finish   matte | satin | gloss   (default reflection character; tunable at render)
  uv       default texture tiling scale on the room plane
  tier     signature | premium | standard
  folder   directory name under the library root
  src      poliigon | ambientcg
"""
import os

# Windows migration (2026-06-15): assets resolve from the in-repo bundle
# (Motuva-Studio-Materials/) instead of the original ~/Desktop/Blender Mac paths.
PKG      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
BUNDLE   = os.path.join(PKG, "Motuva-Studio-Materials")
LIB_FULL = os.path.join(BUNDLE, "Materials", "_downloaded")      # full-res masters (NOT in bundle; optional final-export fallback)
LIB_2K   = os.path.join(BUNDLE, "Materials", "_downloaded_2k")   # live render source

def resolve_dir(key, res="2k"):
    """Absolute path to a material's map folder. res='2k' for working renders, 'full' for final export."""
    m = get(key)
    root = LIB_2K if res == "2k" else LIB_FULL
    return os.path.join(root, m["folder"])

# ── FLOORS ───────────────────────────────────────────────────────────────────
FLOORS = [
    # microcement / concrete — the modern-studio core
    dict(key="micro_light",   name="Microcement — Light",   family="microcement", tone="light", finish="satin", uv=8,  tier="signature", folder="floor-microcement-acg-concrete034", src="ambientcg"),
    dict(key="micro_dark",    name="Microcement — Dark",    family="microcement", tone="dark",  finish="satin", uv=8,  tier="premium",   folder="floor-microcement-acg-concrete030", src="ambientcg"),
    dict(key="concrete_polished", name="Polished Concrete", family="concrete",    tone="mid",   finish="gloss", uv=10, tier="signature", folder="floor-concrete-concretefloorpolished-9237", src="poliigon"),
    dict(key="concrete_polished_fine", name="Polished Concrete — Fine", family="concrete", tone="light", finish="satin", uv=10, tier="premium", folder="floor-concrete-concretepolished001", src="poliigon"),
    dict(key="concrete_poured", name="Poured Concrete",     family="concrete",    tone="light", finish="matte", uv=10, tier="standard",  folder="floor-concrete-concretefloorpoured-8476", src="poliigon"),
    dict(key="concrete_industrial", name="Industrial Concrete", family="concrete", tone="mid",  finish="matte", uv=10, tier="standard",  folder="floor-concrete-concretefloorpaintedworn-11981", src="poliigon"),
    # epoxy / resin — high-gloss
    dict(key="epoxy_1",       name="Epoxy Resin",           family="epoxy",       tone="mid",   finish="gloss", uv=6,  tier="premium",   folder="floor-epoxy-poliigon-epoxy-12658", src="poliigon"),
    dict(key="epoxy_2",       name="Epoxy Resin — Deep",    family="epoxy",       tone="dark",  finish="gloss", uv=6,  tier="premium",   folder="floor-epoxy-poliigon-epoxy-12661", src="poliigon"),
    dict(key="epoxy_blackwhite", name="Epoxy Resin — Mono", family="epoxy",       tone="black", finish="gloss", uv=6,  tier="premium",   folder="floor-epoxy-poliigon-epoxy-12661-blackwhite", src="poliigon"),
    # terrazzo
    dict(key="terrazzo_grey", name="Terrazzo — Grey",       family="terrazzo",    tone="mid",   finish="satin", uv=7,  tier="premium",   folder="floor-terrazzo-acg-terrazzo004", src="ambientcg"),
    dict(key="terrazzo_venetian", name="Venetian Terrazzo", family="terrazzo",    tone="mid",   finish="gloss", uv=8,  tier="premium",   folder="floor-concrete-tilesterrazzovenetianpolishedgrey001", src="poliigon"),
    # marble — premium
    dict(key="marble_white",  name="White Dolomite Marble", family="marble",      tone="white", finish="satin", uv=4,  tier="signature", folder="floor-marble-dolomitewhitemichelangelo002", src="poliigon"),
    dict(key="marble_carrara",name="Carrara Marble",        family="marble",      tone="white", finish="satin", uv=3,  tier="signature", folder="floor-tiling-tilesmarblecarrarahoned001", src="poliigon"),
    dict(key="marble_statuario", name="Statuario Marble",   family="marble",      tone="white", finish="satin", uv=3,  tier="premium",   folder="floor-tiling-tilesmarblestatuariohoned001", src="poliigon"),
    dict(key="marble_grey",   name="Grey Fleury Marble",    family="marble",      tone="mid",   finish="satin", uv=4,  tier="premium",   folder="floor-marble-marblegreyflueryhoned001", src="poliigon"),
    dict(key="marble_cipollino", name="Cipollino Marble",   family="marble",      tone="mid",   finish="satin", uv=4,  tier="premium",   folder="floor-marble-marblecipollinohoned001", src="poliigon"),
    dict(key="marble_nero",   name="Nero Belvedere Marble", family="marble",      tone="black", finish="gloss", uv=4,  tier="premium",   folder="floor-marble-marblenerobelvederehoned001", src="poliigon"),
    dict(key="marble_slab",   name="Marble Slab",           family="marble",      tone="light", finish="satin", uv=3,  tier="standard",  folder="floor-marble-stonemarbleslab-12652", src="poliigon"),
    # porcelain
    dict(key="porcelain_grey",name="Porcelain — Grey",      family="porcelain",   tone="mid",   finish="satin", uv=4,  tier="premium",   folder="floor-tiling-tilesporcelaingrey001", src="poliigon"),
    # wood — warm / boutique
    dict(key="wood_walnut",   name="Walnut Wood",           family="wood",        tone="warm",  finish="satin", uv=6,  tier="standard",  folder="floor-wood-woodflooringwalnut002", src="poliigon"),
    dict(key="wood_oak",      name="Light Oak Wood",        family="wood",        tone="warm",  finish="matte", uv=6,  tier="standard",  folder="floor-wood-woodflooringcirceo001", src="poliigon"),
]

# ── WALLS / BACKDROPS ──────────────────────────────────────────────────────────
WALLS = [
    dict(key="wall_white_clean", name="Clean White",        family="plaster",     tone="white", finish="matte", uv=4, tier="signature", folder="wall-plaster-acg-paintedplaster017", src="ambientcg"),
    dict(key="wall_grey_smooth", name="Smooth Grey",        family="plaster",     tone="mid",   finish="matte", uv=4, tier="signature", folder="wall-plaster-acg-plaster004", src="ambientcg"),
    dict(key="wall_warm_white",  name="Warm White",         family="plaster",     tone="warm",  finish="matte", uv=4, tier="premium",   folder="wall-plaster-acg-plaster001", src="ambientcg"),
    dict(key="wall_plaster_1",   name="Smooth Plaster",     family="plaster",     tone="light", finish="matte", uv=4, tier="standard",  folder="wall-plaster-plastersmooth001", src="poliigon"),
    dict(key="wall_plaster_2",   name="Smooth Plaster — Grey", family="plaster",  tone="mid",   finish="matte", uv=4, tier="standard",  folder="wall-plaster-plastersmooth002", src="poliigon"),
    dict(key="wall_painted",     name="Painted Plaster",    family="plaster",     tone="light", finish="matte", uv=4, tier="standard",  folder="wall-plaster-plasterpainted-7664", src="poliigon"),
    dict(key="wall_natural",     name="Natural Paint",      family="plaster",     tone="white", finish="matte", uv=4, tier="standard",  folder="wall-plaster-plasternaturalpaint001", src="poliigon"),
    dict(key="wall_lime",        name="Lime Plaster — Warm",family="plaster",     tone="warm",  finish="matte", uv=4, tier="premium",   folder="wall-plaster-plasterlime002", src="poliigon"),
    dict(key="wall_venetian",    name="Venetian Plaster",   family="plaster",     tone="mid",   finish="satin", uv=4, tier="premium",   folder="wall-plaster-plastervenetianfinish001", src="poliigon"),
    dict(key="wall_stucco",      name="Stucco Lime",        family="plaster",     tone="warm",  finish="matte", uv=4, tier="standard",  folder="wall-plaster-plasterstuccolime-10562", src="poliigon"),
    dict(key="wall_concrete_raw",name="Raw Concrete",       family="concrete",    tone="mid",   finish="matte", uv=5, tier="premium",   folder="wall-plaster-concretewallold-8454", src="poliigon"),
    dict(key="wall_concrete_clad",name="Concrete Cladding", family="concrete",    tone="mid",   finish="matte", uv=4, tier="premium",   folder="wall-tiling-concretecladdingvertical001", src="poliigon"),
    dict(key="wall_travertine",  name="Travertine Facade",  family="stone",       tone="warm",  finish="matte", uv=4, tier="premium",   folder="wall-tiling-stonetravertinefacade-10927", src="poliigon"),
]

# ── All PBR-folder materials (AUTO-SCANNED) ────────────────────────────────────
# The curated FLOORS/WALLS lists above are an OVERLAY (display name / finish / tier / uv) on top
# of a FULL scan of the library folder — so EVERY material from the previous Blender is included,
# nothing dropped. Inclusive now; curate down in the configurator later.
for _m in FLOORS: _m.update(cat="floor")
for _m in WALLS:  _m.update(cat="wall")
_OVERLAY = {m["folder"]: m for m in FLOORS + WALLS}
_FAMILY_FIX = {"tiling": "tile"}

def _scan_pbr():
    out = []
    # discover from the 2K library (the live render source — full-res masters binned
    # 2026-06-10, optional); fall back to full-res only if 2K is somehow absent.
    scan_dir = LIB_2K if os.path.isdir(LIB_2K) else LIB_FULL
    if not os.path.isdir(scan_dir):
        return [dict(m, type="pbr_folder", relief=m.get("relief", "flat")) for m in _OVERLAY.values()]
    for folder in sorted(os.listdir(scan_dir)):
        d = os.path.join(scan_dir, folder)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "color.jpg")):
            continue
        if folder in _OVERLAY:
            m = dict(_OVERLAY[folder])
        else:
            parts = folder.split("-")
            cat = parts[0] if parts and parts[0] in ("floor", "wall") else "floor"
            fam = _FAMILY_FIX.get(parts[1], parts[1]) if len(parts) > 1 else "?"
            m = dict(key=f"lib_{folder[:30]}", name=folder.replace("-", " "), cat=cat, family=fam,
                     tone="?", finish="satin", uv=8, tier="candidate", folder=folder, src="library")
        m.setdefault("type", "pbr_folder"); m.setdefault("relief", "flat")
        out.append(m)
    return out

PBR = _scan_pbr()

# ── BlenderKit Pro materials (auto-loaded) ─────────────────────────────────────
# Pulled by fetch_blenderkit_materials.py → catalog.json. These are .blend materials
# (the artist's real node setup — procedural/fluid/epoxy come through as-designed).
# Inclusive set ("add all candidates"); curate down later in the configurator UI.
import json as _json
_BK_CATALOG = os.path.join(BUNDLE, "Materials", "_blenderkit", "catalog.json")
_BK_DIR     = os.path.dirname(_BK_CATALOG)

def _load_blenderkit():
    if not os.path.exists(_BK_CATALOG):
        return []
    out = []
    for it in _json.load(open(_BK_CATALOG)):
        out.append(dict(
            key=f"bk_{it['id'][:8]}", name=it.get("name") or it["id"][:8],
            cat=it["cat"], family=it.get("family", "?"), tone="?",
            finish="satin" if it["cat"] == "floor" else "matte",
            relief=it.get("relief", "flat"), uv=4, tier="candidate",
            type="blend_material", asset_id=it["id"],
            # catalog.json stores absolute Mac paths — resolve by basename under the local bundle dir
            blend=os.path.join(_BK_DIR, os.path.basename(it["blend"])) if it.get("blend") else "",
            thumb=it.get("thumb_file", ""), src="blenderkit",
        ))
    return out

# ── BlenderKit Pro wall COMPOSITION models (auto-loaded) ───────────────────────
# Modelled wall assemblies (slat / fluted / 3D panel / stone section) — geometry + material.
# A third wall type alongside flat + dimensional materials. type=blend_model; append whole model.
_WALLMODEL_CATALOG = os.path.join(BUNDLE, "Materials", "_blenderkit_models", "wall_models.json")  # NOT in bundle → wall-models load empty (handled below)

def _load_wallmodels():
    if not os.path.exists(_WALLMODEL_CATALOG):
        return []
    out = []
    for it in _json.load(open(_WALLMODEL_CATALOG)):
        out.append(dict(
            key=f"wm_{it['id'][:8]}", name=it.get("name") or it["id"][:8],
            cat="wall", family=it.get("family", "composite"), tone="?", finish="matte",
            relief="composite", uv=1, tier="candidate", type="blend_model",
            asset_id=it["id"], blend=it.get("blend", ""), thumb=it.get("thumb_file", ""), src="blenderkit",
        ))
    return out

BLENDERKIT = _load_blenderkit()
WALLMODELS = _load_wallmodels()
MATERIALS = PBR + BLENDERKIT + WALLMODELS
_BY_KEY = {m["key"]: m for m in MATERIALS}

def get(key):     return _BY_KEY[key]
def floors():     return [m for m in MATERIALS if m["cat"] == "floor"]
def walls():      return [m for m in MATERIALS if m["cat"] == "wall"]
def dimensional():return [m for m in MATERIALS if m.get("relief") == "dimensional"]
def composites():  return [m for m in MATERIALS if m.get("relief") == "composite"]
def by_tier(t):   return [m for m in MATERIALS if m["tier"] == t]
def by_family(f): return [m for m in MATERIALS if m["family"] == f]
def by_source(s): return [m for m in MATERIALS if m["src"] == s]

def categories():
    from collections import Counter
    out = {}
    for cat in ("floor", "wall"):
        fams = Counter(m["family"] for m in MATERIALS if m["cat"] == cat)
        out[cat] = dict(sorted(fams.items(), key=lambda kv: -kv[1]))
    return out

if __name__ == "__main__":
    print(f"floors {len(floors())} · walls {len(walls())} "
          f"(flat-mat + dimensional {len(dimensional())} + composite models {len(composites())}) · total {len(MATERIALS)}")
    print(f"  PBR-folder (incl. ALL old Blender materials): {len(PBR)} · BlenderKit materials: {len(BLENDERKIT)} · wall composites: {len(WALLMODELS)}")
    cats = categories()
    for cat in ("floor", "wall"):
        print(f"\n{cat.upper()}S by family:")
        for fam, n in cats[cat].items():
            print(f"  {fam:14} {n}")
