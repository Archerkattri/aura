# AURA

AURA means **Adaptive Unified Radiance Asset**.

This repo is the CPU-only contract scaffold for a post-3DGS captured-scene
representation:

> Photogrammetry -> NeRF -> 3D Gaussian Splatting -> AURA

AURA is not "Gaussian splatting but certified." It is a proposed asset contract
where a captured scene is represented by adaptive local carriers that expose a
common ray-query, confidence, edit, LOD, and export interface.

## Current Status

This repo contains the non-GPU MVP contract layer:

- carrier registry;
- evidence-to-carrier assignment;
- bounded AURA elements and chunks;
- CPU reference ray-query response;
- simple front-to-back scene query;
- `.aura` package writer;
- glTF/USD exchange-target metadata;
- fixture CLI commands and tests.

It does **not** contain a real renderer, trainer, CUDA kernel, BVH, 3DGS
bootstrap, radiance-cell optimizer, or benchmark result yet.

## Install

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

## CPU/GPU Safety

For local scaffold work:

```bash
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

On the GPU machine, enable GPU only when implementing real rendering/training.

## Quick Smoke Commands

```bash
aura write-demo-package --output-dir outputs/demo.aura
aura query-demo --x 0.0 --y 0.0
python -m pytest
```

The demo package is a schema/contract fixture, not a renderable research result.

## What AURA Is

An AURA scene should eventually contain:

- **surface carriers** for high-confidence opaque geometry;
- **volume carriers** for fuzzy, uncertain, translucent, or soft regions;
- **Beta-like bounded kernels** for compact adaptive local support;
- **Gabor/frequency carriers** for high-frequency texture;
- **neural residual carriers** for view-dependent effects;
- **Gaussian fallback carriers** when vanilla splats are sufficient;
- **semantic/object carriers** for editing and language/object grouping;
- **ray-query acceleration** for shadows, reflections, picking, collision, and
  engine interaction;
- **asset export** to native `.aura` plus glTF/USD fallbacks.

## Data And Baselines To Download

Keep data and external baselines under ignored `data/` or sibling ignored
`third_party/` paths. Do not vendor large datasets into the repo.

```text
data/
  mipnerf360/
  tanks-and-temples/
  deep-blending/
  nerfstudio-fixtures/
  custom-captures/
third_party/
  gaussian-splatting/
  gsplat/
  nerfstudio/
  colmap-scripts/
```

Recommended benchmark/baseline sources:

| Source | Purpose |
| --- | --- |
| Mip-NeRF 360 | standard novel-view benchmark |
| Tanks and Temples | larger real scenes and geometry stress |
| Deep Blending | indoor view synthesis stress |
| LLFF / nerfstudio fixtures | small smoke scenes |
| Original 3DGS implementation | baseline/teacher/initializer |
| gsplat or nerfstudio | practical splat training/rendering baseline |
| COLMAP | camera poses and sparse point initialization |
| 3DGRT / EVER / Volumetric 3DGS papers | ray-query and volumetric correctness baselines |
| Beta/Gabor/Splat-the-Net/Radiance Meshes papers | carrier registry competitors/substrates |

## Expected First GPU Milestone

Do not start with full AURA. Start with:

1. train/load one small 3DGS baseline scene;
2. export Gaussian means/opacities/covariances;
3. build an AURA element scaffold from those samples;
4. preserve primary-view quality approximately;
5. add one ray-query demo: depth/first-hit/transmittance;
6. export a native `.aura` package and glTF/USD fallback metadata.

## Repository Map

```text
src/aura/
  asset.py       manifest/capability models
  assignment.py  evidence-to-carrier selection
  carriers.py    carrier registry
  cli.py         fixture CLI
  elements.py    bounded elements/chunks
  exchange.py    glTF/USD exchange target metadata
  package.py     native .aura package writer
  ray.py         ray and ray-query response contracts
  scene.py       CPU reference scene query
tests/           contract tests
docs/            no-GPU and handoff docs
```

## Paper Claim Boundary

Safe claim:

> AURA is a proposed adaptive traceable radiance asset contract for post-3DGS
> captured scenes.

Unsafe claims until implementation exists:

- replacement for 3DGS in quality or speed;
- physically correct inverse rendering;
- full PBR material recovery;
- robust dynamic scenes;
- real-time renderer complete;
- engine-ready production format.

