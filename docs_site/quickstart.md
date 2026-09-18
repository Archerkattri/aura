# Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gpu,assets]"   # from source
# or from PyPI:
pip install aura-splat
```

The `data/` tree is gitignored, so a fresh clone has no scenes — fetch one
first, or reproduce the paper with no data or GPU at all via `REPRODUCE.md`
(everything runs from committed artifacts).

```bash
# 0. Fetch a scene (skip if you already have a capture).
bash scripts/fetch_scene.sh truck data/tanks/truck

# 1. Build a capture manifest from COLMAP.
aura colmap-to-capture-manifest data/tanks/truck/sparse/0 \
  --root data/tanks/truck --image-dir data/tanks/truck/images \
  --output outputs/truck-manifest.json --point-seeded

# 2. Train carriers.
aura train-gsplat outputs/truck-manifest.json --output outputs/truck.aura --scale 1.0

# 3. Use the asset.
aura render     outputs/truck.aura --backend torch --output docs/view.ppm
aura export-splat outputs/truck.aura --output docs/truck.glb
aura ray-query  outputs/truck.aura --origin 0 0 0 --direction 0 0 1
```

!!! note "Environment isolation"
    For CUDA-first work use the GPU environment when available. The DBS-Beta
    fork installs under the `gsplat` package name and is kept isolated in
    `.dbs_venv` — never mix the two.
