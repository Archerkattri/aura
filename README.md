# AURA

AURA means **Adaptive Unified Radiance Asset**.

This repo is the GPU-ready contract scaffold for a post-3DGS captured-scene
representation:

> Photogrammetry -> NeRF -> 3D Gaussian Splatting -> AURA

AURA is not "Gaussian splatting but certified." It is a proposed asset contract
where a captured scene is represented by adaptive local carriers that expose a
common ray-query, confidence, edit, LOD, and export interface.

## Current Status

This repo contains the GPU-ready MVP contract layer:

- carrier registry;
- native carrier payload models for surface, volume, beta, gabor, neural,
  Gaussian fallback, and semantic carriers;
- payload/carrier consistency validation;
- evidence-to-carrier assignment;
- evidence-to-element adaptive decomposition;
- package-level confidence maps and edit metadata;
- bounded AURA elements and chunks;
- carrier-aware reference ray-query response;
- simple front-to-back scene query;
- tiny JSON/ASCII/binary little-endian PLY 3DGS export reader for means/opacities/covariances;
- quaternion-aware PLY covariance conversion from 3DGS log-scales;
- direct 3DGS export/directory import adapter;
- `.aura` package writer;
- `.aura` package loader/validator;
- explicit `.aura` format/version compatibility checks;
- JSON package inspection output and JSON Schema documents;
- runtime JSON Schema validation for package files;
- deterministic orthographic package preview rendering and image metrics;
- strict-JSON render comparison metrics for regression checks;
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
python -m pip install -e ".[dev]"
```

## GPU Runtime

Expose the primary CUDA device for GPU development:

```bash
export CUDA_VISIBLE_DEVICES=0
```

The current fixture commands do not train or render yet, but new model,
rendering, and ray-query adapters should target CUDA by default.

## Quick Smoke Commands

```bash
aura write-demo-package --output-dir outputs/demo.aura
aura write-splat-demo-package --input tests/fixtures/tiny_3dgs_export.ply --output-dir outputs/splat-demo.aura
aura import-3dgs third_party/gaussian-splatting/output/<scene> --output-dir outputs/<scene>.aura
# imports direct PLY/JSON exports or point_cloud/iteration_*/point_cloud.ply layouts
aura validate-package outputs/splat-demo.aura
# validates JSON Schemas, cross-file references, schema version, and package counts
aura inspect-package outputs/splat-demo.aura
# prints the same package summary as stable JSON
aura render-package outputs/splat-demo.aura --output outputs/splat-demo.ppm --width 128 --height 128
# writes a deterministic PPM preview for package validation
aura compare-renders outputs/baseline.ppm outputs/splat-demo.ppm --min-psnr 35
# prints strict JSON MSE/PSNR metrics and exits nonzero if the threshold fails
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

Do not start with full AURA. Start with native decomposition and one small
evidence source:

1. define a tiny mixed evidence scene with surface, volume, beta, gabor,
   neural, semantic, and gaussian fallback regions;
2. decompose evidence into typed AURA elements with carrier payloads;
3. validate package/load/query behavior for the mixed native scene;
4. ingest one small 3DGS baseline scene as evidence, not as the final center;
5. preserve primary-view quality approximately;
6. add one GPU ray-query demo: depth/first-hit/transmittance;
7. export a native `.aura` package and glTF/USD fallback metadata.

## Repository Map

```text
src/aura/
  asset.py       manifest/capability models
  assignment.py  evidence-to-carrier selection
  carriers.py    carrier registry
  carrier_payloads.py native carrier payload contracts
  cli.py         fixture CLI
  decomposition.py evidence samples to mixed native AURA elements
  elements.py    bounded elements/chunks
  exchange.py    glTF/USD exchange target metadata
  package.py     native .aura package writer/loader/validator
  ray.py         ray and ray-query response contracts
  render.py      deterministic orthographic preview and image metrics
  schema.py      native package format and supported schema versions
  scene.py       reference scene query
  ingest/
    baselines.py  3DGS export discovery and import adapter
    splats.py     JSON/PLY 3DGS evidence reader and AURA conversion
tests/           contract tests
docs/            GPU handoff and dataset docs
docs/schemas/    JSON Schemas for native .aura package files
```

The `ingest/` package is intentionally an adapter boundary. 3DGS splats are
treated as evidence samples that can populate AURA Gaussian fallback payloads;
they are not the center of the native representation.

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
