# Motuva Remove-BG Pipeline

A self-contained Python package that turns dealer car photos into background-replaced
**studio composites** at the original image dimensions — **without any diffusion model**.
For exterior-full cars it renders a per-car studio "plate" in Blender at the photo's
perspective, cuts the car out with **remove.bg**, and **manually alpha-composites** the
cutout onto the plate at the recovered ground-contact placement. It can be driven two
ways: a **batch runner** over a directory, or a **streaming FastAPI server**.

> This directory is the **remove-bg variant** of the pipeline. FLUX / diffusion has been
> removed entirely (the FLUX version lives in a separate directory). The exterior-full
> lane preserves the car *exactly as shot* — no relighting or reflection cleanup — and
> pastes it onto the rendered plate.

---

## 1. The pieces

| Piece | What it is | Role |
|---|---|---|
| **Perspective estimation** | YOLO crop → DINOv2 k-NN pose **retrieval** + **GeoCalib** elevation + a confidence **gate** | Recovers camera values (azimuth / elevation / distance / cam-height / focal / roll) from one photo |
| **Plate rendering** | A headless **Blender** "studio" run as a persistent **warm GPU worker** | Renders a *car-free* ground plate (floor + wall + turntable) at the photo's perspective and input dimensions |
| **remove.bg + manual composite** | remove.bg car cutout + a plain alpha-paste onto the plate | Cuts the car out and pastes it onto the plate at the exact recovered placement (no diffusion) |

## 2. Routing — three lanes

The 3-way router (`models/timm_classifier.py`) picks the lane (`pipeline.py`). **Every
lane now uses remove.bg:**

- **`interior`** → remove.bg, then flatten to a white background. Terminal.
- **`exterior-partial`** → remove.bg, then composite the car over a **supplied** background image. Terminal.
- **`exterior-full`** → the fused path: render a per-car studio plate, then remove.bg + manual composite onto it.

The confidence gate is **advisory / logged only** — it never branches the flow.

## 3. The exterior-full path

For one image, across the resource tiers:

1. **YOLO bbox** (shared) → RGB bbox crop.
2. **Orientation** (timm, 8-way) + **DINOv2 retrieval** (pose) + **GeoCalib** (elevation) → **gate** → **camera values**. Retrieval gives a continuous azimuth; the orientation model **corrects it only when the circular |Δ| > 45°**.
3. **Occupancy resize** — the crop is **uniformly scaled** toward an area-based **visual-occupancy** target per pose class (see `config/orientation.yaml`).
4. **Disc** sized from the recovered footprint → **plate** rendered in Blender at the **input dimensions** (so plate pixels == canvas pixels 1:1). The plate meta carries `car_spot_px` (the turntable centre `cx,cy`) + `pixels_per_metre`.
5. **remove.bg** the YOLO crop → RGBA cutout, resized to the occupancy dims.
6. **Manual composite** — the cutout is pinned by its **wheel-footprint centre** onto the plate's turntable centre `cx,cy`, slid **toward the camera** by a per-pose **forward bias**, clamped fully on-plate, and **alpha-pasted onto the plate** → result at exact input dimensions. See `stages/anchor.py` + `stages/exterior_full.py` (`composite_on_plate`).

## 4. Architecture: concurrency (single GPU)

Two resource tiers plus a network lane:

- **Preprocess pool** (ThreadPoolExecutor) — YOLO / orientation / DINOv2 / GeoCalib / resize / the manual composite paste.
- **Blender pool** — a single warm-worker slot (scene + materials loaded once; Cycles on GPU). Serializes renders against each other.
- **remove.bg** — network-bound, gated by `BackgroundRemover`'s own semaphore.

There is **no diffusion model and no VRAM co-residency planning** — the warm Blender
worker is the only heavy GPU user.

## 5. Package layout

```
motuva-pipline-removebgappraoch/
  config/         PipelineSettings (pydantic-settings) + orientation.yaml (occupancy targets)
  models/         timm router+orientation, shared YOLO, DINOv2 retriever, GeoCalib, model manager
  processing/     image ops, validators, exceptions, background remover, interior + partial lanes
  perspective/    estimate (fuser), gate (+azimuth correction), footprint (disc), anchor_geometry (S0 wheel finder)
  stages/         occupancy_resize, anchor (composite onto plate), exterior_full (the fused lane)
  render/         warm Blender worker (host client + in-Blender entry) + vendored studio_engine
  runtime/        blender_pool (single warm-worker slot)
  utils/          structured logging (stage_timer), metrics, gpu monitor
  assets/         weights/, index/ (emb.npy + meta.json), blender/ (master.blend + materials)  [gitignored]
  pipeline.py     per-image orchestration (lane dispatch + the tiers)
  runner.py       batch over the input directory
  main.py         CLI entry (batch)
  server.py       streaming FastAPI server  |  client.py / client_2.py  HTTP clients for it
```

## 6. Prerequisites (install / provide before running)

See **SETUP.md** for step-by-step pod prep. In short you must provide:

1. **Blender 5.x** installed, with its Cycles GPU device visible → set `MOTOCUT_BLENDER_EXE`.
2. **Python deps** incl. PyTorch for your CUDA and **GeoCalib** (a git dep in `requirements.txt`). No diffusers / transformers / gated model needed.
3. **remove.bg API key** (`REMOVE_BG_API_KEY`) — now required for **all** lanes (interior, partial, and the exterior-full car cutout).
4. **Model weights** in `assets/weights/` (classifier + orientation `.pth`) and the **retrieval index** in `assets/index/` (`emb.npy` + `meta.json`). These are large and **gitignored** — copy them out-of-band.
5. For the `exterior-partial` lane only: a **background image** (`MOTOCUT_BACKGROUND_IMAGE`).

## 7. Configuration

A single `PipelineSettings` (pydantic-settings, env prefix `MOTOCUT_`) merges every
tunable; see `config/pipeline_config.py`. It is populated from process env + an
**`a.env`** file at the package root (copy `.env.example` → `a.env`). Asset paths
default **into the package** and are env-overridable.

- Car **sizing**: `config/orientation.yaml` → `target_visual_occupancy` per pose class.
- Car **placement / framing**: `AnchorConfig` → per-pose `forward_bias_*` (and `sensor_mm`).
- **Contact shadow**: exterior-full uses a local plate-aware contact shadow by default. Keep `RemoveBgConfig.exterior_add_shadow` / `MOTOCUT_REMOVEBG__EXTERIOR_ADD_SHADOW` false unless you intentionally want remove.bg's baked shadow instead.

## 8. What to update before running (checklist)

Edit **`a.env`** (copied from `.env.example`):

- [ ] `REMOVE_BG_API_KEY` — for **all** lanes.
- [ ] `MOTOCUT_BLENDER_EXE` — absolute path to Blender 5.x **on this machine/pod**.
- [ ] `MOTOCUT_INPUT_DIR` / `MOTOCUT_OUTPUT_DIR` — batch I/O (or pass `--input/--output`).
- [ ] `MOTOCUT_BACKGROUND_IMAGE` — the partial-lane background (only if you have partial images).

Also confirm the **assets are present** (`assets/weights/*.pth`, `assets/index/emb.npy`).

## 9. Running

**Batch over a directory:**
```bash
cp .env.example a.env          # then fill it in (see the checklist above)
# run from the package directory so config/, utils/, etc. resolve on sys.path:
python main.py --input /path/to/input --output /path/to/output
```
The runner scans the input dir, drives `pipeline.process_path` across the tiers, writes
each output at the input filename + dimensions, and prints a per-stage timing summary
(p50/p95/mean/max) plus lane counts.

**Streaming server (interactive / debug):**
```bash
python server.py               # FastAPI on :8000  (/process, /set_studio, /health)
python client.py               # drives the server over HTTP, saving debug frames
python client_2.py             # studio-shuffle tester (randomises floor/wall per image)
```
The server processes one image at a time and can stream intermediate frames
(`crop_raw`, `crop_resized`, `plate`, `plate_marked`, `cutout`, `composite`).

## 10. Placement tuning

Two independent dials, both per pose group, both live without code changes:

- **Size** — `config/orientation.yaml` `target_visual_occupancy` (fraction of canvas area the car covers).
- **Forward framing** — `AnchorConfig.forward_bias_*` (fraction of the disc radius the car is slid toward the camera):
  - `MOTOCUT_ANCHOR__FORWARD_BIAS_FRONTREAR`
  - `MOTOCUT_ANCHOR__FORWARD_BIAS_THREEQ` (the four 3/4 corners)
  - `MOTOCUT_ANCHOR__FORWARD_BIAS_SIDE`
  - `0.0` = pure grounding (car centred on the turntable, no forward shift).

Grounding (wheels on the floor) is automatic and per-car; the forward bias is the "how
much foreground" look.

## 11. Compute notes

Single GPU; Blender's Cycles renders on the **GPU** via the warm worker. Since there is
no diffusion model, VRAM demand is modest — the pipeline runs on far smaller cards than
the FLUX version. The trade-off: the car is composited **exactly as shot** (no
relighting or reflection harmonization), so the plate should be chosen to sit naturally
under a real photo.
