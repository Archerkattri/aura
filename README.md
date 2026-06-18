# AURA

AURA means **Adaptive Unified Radiance Asset**.

AURA-Core is the native reconstruction engine for the post-3DGS captured-scene
representation:

```text
Photogrammetry -> NeRF -> 3D Gaussian Splatting -> AURA
```

AURA is not a 3DGS wrapper. Gaussian splats are allowed as ingest evidence or a
baseline only. The native path trains adaptive bounded carriers directly from
posed capture assets and exports a queryable `.aura` scene package.

## Current Status

Internal completion estimate: **85%**.

Implemented now:

- native carrier registry and payloads for surface, volume, beta, gabor,
  neural residual, semantic, and Gaussian fallback carriers;
- `.aura` package writer, loader, validator, JSON schemas, chunks, confidence
  maps, edit metadata, and semantic graph artifacts;
- capture manifests for posed image/depth/mask/normal assets;
- COLMAP sparse camera/pose/point import into capture manifests;
- packed capture tensor loading for PNG, PPM/PGM, COLMAP depth maps, COLMAP
  normal maps, and optional `imageio` assets;
- tiled PyTorch capture optimization through `aura train`;
- device-resident capture asset batching, mask-aware pixel sampling, camera ray
  construction, confidence target handling, carrier validity checks, gradient
  clipping, and reusable packed training batches for the torch optimization
  path;
- ordered front-to-back native torch carrier compositing with color, alpha,
  transmittance, depth, normal, confidence, material, semantic, residual, and
  ordered hit outputs;
- trainable native carrier parameters for color, opacity/density, shape,
  normals, confidence, residual scale, and frequency fields where supported;
- configurable image, depth, query, normal, mask, and confidence loss weights;
- mask-derived confidence targets so occluded or partial evidence can train
  carrier confidence instead of forcing every sampled ray to full certainty;
- deterministic packed batch preparation, so repeated optimization iterations
  reuse the same bounded capture/ray buffers instead of rebuilding them every
  step;
- adaptive split, promote, merge, and demote decisions during torch training;
- checkpoint and resume metadata for training runs;
- deterministic CPU/torch package rendering, query demos, ray-query scoring,
  and visual metric helpers;
- packaged CUDA kernel and PyTorch extension sources with build/status probes.

Still missing before this can be called production:

- compiled CUDA renderer dispatch parity and runtime benchmarks;
- production GPU BVH/traversal instead of the current AABB-centered tensor path;
- carrier-complete CUDA kernels with measured parity against the torch renderer
  for surface, volume, beta, gabor, neural residual, semantic, and Gaussian
  fallback carriers;
- larger real-scene benchmarks against COLMAP, NeRF, and 3DGS baselines;
- production EXR/video streaming and long-run memory tests.

## Install

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gpu,assets]"
```

On the GPU machine, expose CUDA device 0:

```bash
export CUDA_VISIBLE_DEVICES=0
aura torch-renderer-status
aura cuda-kernel-build-report --build
```

## Train AURA

The primary path is capture manifest -> native AURA carriers -> optimized
`.aura` package:

```bash
aura write-capture-manifest-template --output outputs/capture-manifest.json
aura inspect-capture-tensors data/custom-captures/<scene>/capture-manifest.json
aura plan-capture-sampling data/custom-captures/<scene>/capture-manifest.json \
  --tile-size 256 --pixel-stride 8 --max-targets-per-frame 4096
aura train data/custom-captures/<scene>/capture-manifest.json \
  --device cuda \
  --output outputs/scene.aura \
  --iterations 8 \
  --tile-size 256 \
  --pixel-stride 8 \
  --max-targets-per-frame 4096 \
  --max-targets-per-batch 1024 \
  --image-loss-weight 1.0 \
  --depth-loss-weight 1.0 \
  --query-loss-weight 1.0 \
  --normal-loss-weight 1.0 \
  --mask-loss-weight 1.0 \
  --confidence-loss-weight 1.0 \
  --checkpoint-interval 2
```

Resume from a checkpoint or trained package:

```bash
aura train data/custom-captures/<scene>/capture-manifest.json \
  --device cuda \
  --output outputs/scene-resumed.aura \
  --resume-from outputs/scene.aura/checkpoints/iter_000001.aura
```

Training writes `training_report.json`, optional checkpoint packages, and the
optimized native package under the selected output directory. Reports include
the effective loss weights, per-step loss curves, source windows, and checkpoint
metadata.

## Render And Query

```bash
aura validate-package outputs/scene.aura
aura inspect-package outputs/scene.aura
aura render outputs/scene.aura --backend torch --device cuda --output outputs/scene.ppm --width 256 --height 256
aura benchmark-reference outputs/scene.aura --width 64 --height 64
aura benchmark-visual outputs/scene.aura outputs/reference.ppm --min-psnr 30
```

Use `--backend cuda --require-cuda` only after the CUDA renderer extension
builds, imports, passes parity against the torch renderer, and has runtime
benchmarks on the target machine.

## Fixture Smoke Tests

These commands exercise the small deterministic package and reconstruction
fixtures without requiring external datasets:

```bash
aura build-native-demo --output-dir outputs/native-demo.aura
aura render outputs/native-demo.aura --backend torch --output outputs/native-demo.ppm
aura benchmark-ray-query outputs/native-demo.aura --native-demo-expectations
aura write-training-frames-demo --output outputs/training-frames.json
aura reconstruct-demo --frames outputs/training-frames.json --output-dir outputs/reconstruct-demo.aura --iterations 6 --render-backend torch
python -m pytest
```

## Data Layout

Keep datasets and third-party baselines out of git:

```text
data/
  custom-captures/<scene>/
    capture-manifest.json
    images/
    depth/
    masks/
    normals/
    colmap/
third_party/
  gaussian-splatting/
  gsplat/
  nerfstudio/
```

Generate capture manifests from COLMAP sparse models when available:

```bash
aura colmap-to-capture-manifest data/custom-captures/<scene>/colmap \
  --root data/custom-captures/<scene> \
  --output outputs/capture-from-colmap.json
```

## Repository Map

```text
src/aura/
  cli.py               command-line training, rendering, ingest, benchmarks
  core.py              reconstruction contracts and adaptive evolution policy
  torch_renderer.py    native torch render batches, compositing, objectives
  torch_optimizer.py   tiled capture optimization and checkpoint snapshots
  torch_kernels.py     carrier parameter tensors and differentiable responses
  cuda_renderer.py     CUDA renderer ABI and dispatch boundary
  cuda/                packaged CUDA and PyTorch extension sources
  ingest/              capture, COLMAP, 3DGS, depth, and semantic adapters
  package.py           .aura package IO and validation
  scene.py             ray-query traversal and response assembly
tests/                 deterministic contract, optimizer, renderer, CLI tests
docs/                  handoff notes, datasets, schemas, and research notes
```

`src/aura/ingest/` is an adapter boundary. 3DGS exports become `EvidenceSample`
records and only survive as Gaussian fallback carriers when native carrier
assignment does not justify a stronger representation.

## Baselines

Use 3DGS, gsplat, nerfstudio, COLMAP, LLFF/nerfstudio fixtures, Mip-NeRF 360,
Tanks and Temples, and Deep Blending as baselines or datasets. Keep their
outputs under ignored `data/`, `third_party/`, or `outputs/` directories.

## Development Rules

- Keep generated `.aura` packages, datasets, checkpoints, renders, and secrets
  out of git.
- Prefer implementation work in renderer, optimizer, CUDA dispatch, carrier
  evolution, and real benchmarks over new status documents.
- Prefix commit subjects with the current completion percentage, for example
  `73% feat: train neural opacity from masks`.
- Commit as `Archerkattri <krishiattriwork@gmail.com>`.
