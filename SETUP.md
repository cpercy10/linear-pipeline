# Motuva Remove-BG Pipeline — Pod Setup

Three steps to get a single-GPU pod ready to run the batch. Run everything from the
package directory (it must be on `sys.path` — the top-level subpackages `config`,
`utils`, etc. are imported absolutely).

All large assets (model weights, the retrieval `emb.npy` + `meta.json`, the Blender
master `.blend` + materials, and the studio engine) are **vendored into the package**
under `assets/` and `render/studio_engine/`. You do **not** export them separately — the
config defaults point at them. You still install the two external runtimes (Blender,
GeoCalib) and supply one secret.

> FLUX / diffusion has been removed — there is no Hugging Face gated model and no
> `HF_TOKEN`. The exterior-full lane composites the remove.bg car cutout directly onto
> the rendered plate.

---

## 1. Install Blender 5.x and point the pipeline at it

The plate-render stage drives a real Blender executable as a warm worker. On a RunPod
pod:
```bash
apt update
apt install -y wget xz-utils libx11-6 libxi6 libxrender1 libxext6 libsm6 libxkbcommon-x11-0 libxxf86vm1 libgl1 libegl1 libdbus-1-3
wget https://download.blender.org/release/Blender5.0/blender-5.0.0-linux-x64.tar.xz
tar -xf blender-5.0.0-linux-x64.tar.xz
mv blender-5.0.0-linux-x64 blender-5.0
ln -sf /workspace/blender-5.0/blender /usr/local/bin/blender
blender --version
```

Then set:
```bash
export MOTOCUT_BLENDER_EXE=/workspace/blender-5.0/blender   # path to the executable
```

(Windows dev box default is `D:\Blender\B\blender.exe`.) The master `.blend`, materials
(`Materials/_downloaded_2k`), and the studio engine (`scripts` + `configurator`) are
already in-package — no extra paths needed unless you want to override
`MOTOCUT_MASTER_BLEND` / `MOTOCUT_MATERIALS_2K_DIR` / `MOTOCUT_STUDIO_ENGINE_DIR`.

> The warm worker forces Cycles onto the **GPU**. Confirm the pod's Blender sees the
> CUDA/OptiX device (`blender -b --python-expr "import bpy; print([d.name for d in bpy.context.preferences.addons['cycles'].preferences.devices])"`).

## 2. Install the Python dependencies (including GeoCalib)

Install PyTorch for the pod's CUDA first, then the rest. GeoCalib is a git dependency
and is included in `requirements.txt`:

```bash
pip install "torch>=2.7" "torchvision>=0.22" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# requirements.txt pulls: geocalib @ git+https://github.com/cvg/GeoCalib.git
```

No diffusers / transformers / accelerate are needed (FLUX is gone).

## 3. Secret + the partial-lane background

Copy the env template to **`a.env`** (the loader reads `a.env`, not `.env`) and fill in
the operational values:

```bash
cp .env.example a.env
# REMOVE_BG_API_KEY=...                     # REQUIRED for ALL lanes (incl. exterior-full cutout)
# MOTOCUT_BLENDER_EXE=/path/to/blender      # the Blender 5.x executable
# MOTOCUT_BACKGROUND_IMAGE=/path/to/bg.jpg  # exterior-partial lane only (optional)
```

Also set the batch I/O dirs (`MOTOCUT_INPUT_DIR`, `MOTOCUT_OUTPUT_DIR`) here or pass
them on the CLI.

---

## Smoke check

```bash
python -c "import sys; sys.path.insert(0,'.'); \
from config.pipeline_config import get_settings, get_orientation_registry, validate_orientation_azimuth_map; \
s=get_settings(); print('weights ok:', s.classifier_weights.exists(), s.orientation_weights.exists()); \
print('index ok:', (s.index_dir/'emb.npy').exists()); \
print('blend ok:', s.master_blend.exists()); \
print('remove.bg key set:', bool(s.remove_bg_api_key)); \
print('occupancy front:', get_orientation_registry().get('front')); \
print('azimuth map ok:', validate_orientation_azimuth_map(s.index_dir)['ok'])"
```

You should see every asset resolve `True`, the remove.bg key set, and the azimuth-map
validation report `ok: True` (8 tight buckets).
