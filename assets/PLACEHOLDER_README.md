# Assets

The large runtime assets (model weights, the retrieval index, and the Blender studio
scene + materials) are **not committed to GitHub**. Only this `PLACEHOLDER_README.md`
is tracked, so the `assets/` folder still exists in the repo.

## How to get the assets

1. Download the **`assets`** folder from Google Drive:
   https://drive.google.com/drive/folders/15nluqIPoetdybSRWoA8xLp2xmidd1xHl?usp=drive_link
2. Replace this `assets/` folder with the downloaded one (keep it at the same path:
   `<repo>/assets/`).
3. Inside the Drive's `assets/blender/Materials/` there is a **`.zip`**
   (`_downloaded_2k/_downloaded_2k.zip`). **Extract it in place before running** — the
   renderer reads the unzipped PBR material folders, not the archive.

## Directory structure

```
assets/
├── blender/
│   ├── Materials/
│   │   └── _downloaded_2k/             # PBR materials — EXTRACT _downloaded_2k.zip here first
│   └── motuva-studio-master.blend      # Blender studio master scene
├── index/
│   ├── emb.npy                         # DINOv2 retrieval embeddings
│   └── meta.json                       # per-row camera ground truth
├── weights/
│   ├── interiorvsfullvspartial.pth     # 3-way router (interior / exterior-partial / exterior-full)
│   └── orientation-model.pth           # 8-way orientation classifier
└── PLACEHOLDER_README.md               # this file
```
