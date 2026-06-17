# AURA

AURA means **Adaptive Unified Radiance Asset**.

This repo is being developed into **AURA-Core**, an end-to-end native
reconstruction engine for the post-3DGS captured-scene representation:

> Photogrammetry -> NeRF -> 3D Gaussian Splatting -> AURA

AURA is not "Gaussian splatting but certified" and not a 3DGS package wrapper.
The target is a reconstruction system that trains adaptive local radiance
carriers directly from images, video, poses, depth, and priors. Each region
should become the simplest bounded carrier that explains its geometry,
appearance, uncertainty, semantics, and interaction needs while exposing one
ray-query, editing, confidence, LOD, and export contract.

## Current Status

Current code is still an early AURA-Core scaffold. It already contains the
native representation contract pieces:

- carrier registry;
- native carrier payload models for surface, volume, beta, gabor, neural,
  Gaussian fallback, and semantic carriers;
- payload/carrier consistency validation;
- evidence-to-carrier assignment;
- evidence-to-element adaptive decomposition;
- package-level confidence maps and edit metadata;
- semantic/object graph package artifact;
- bounded AURA elements and chunks;
- carrier-aware reference ray-query response;
- simple front-to-back scene query;
- tiny JSON/ASCII/binary little-endian PLY 3DGS export reader for means/opacities/covariances;
- quaternion-aware PLY covariance conversion from 3DGS log-scales;
- AURA-Ingest adapters that convert 3DGS, depth, semantic mask, and sparse
  point priors into `EvidenceSample` contracts;
- `.aura` package writer;
- `.aura` package loader/validator;
- explicit `.aura` format/version compatibility checks;
- JSON package inspection output and JSON Schema documents;
- runtime JSON Schema validation for package files;
- deterministic orthographic package preview rendering and image metrics;
- reproducible benchmark plans plus CPU reference package/query/render timing metrics;
- strict-JSON render comparison metrics for regression checks;
- package-backed glTF/USD exchange-target metadata;
- fixture CLI commands and tests.

It does **not** yet contain the full AURA-Core reconstruction engine: image/video
data loading, pose/depth bootstrapping, differentiable carrier optimization,
adaptive split/merge/promote training, CUDA kernels, BVH, or end-to-end
benchmark results.

See `docs/AURA_CORE_RESEARCH.md` for the current research direction and why the
next milestone must be native reconstruction rather than more package polish.

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
aura build-native-demo --output-dir outputs/native-demo.aura
aura validate-package outputs/native-demo.aura
# validates mixed native carriers, typed payloads, confidence maps, edit metadata, and chunks
aura inspect-package outputs/native-demo.aura
# prints the native package summary as stable JSON
aura render-package outputs/native-demo.aura --output outputs/native-demo.ppm --width 128 --height 128
# writes a deterministic PPM preview for package validation
aura query-demo --x -0.5 --y -0.5
# queries the native mixed-carrier fixture
aura inspect-rays outputs/native-demo.aura --native-demo-probes
# prints material-aware occlusion/shadow/reflection-ready ray-query inspections
aura benchmark-reference outputs/native-demo.aura --width 32 --height 32
# runs CPU reference package/query/render timing metrics
aura benchmark-reference outputs/native-demo.aura --include-ablations
# runs carrier assignment ablation metrics
aura benchmark-core --iterations 6
# compares AURA-Core adaptive reconstruction against a static-carrier fixture run
aura migration-plan outputs/native-demo.aura
# prints package schema migration status

# AURA-Core reconstruction path, to be built next:
# aura reconstruct-demo --output-dir outputs/reconstruct-demo.aura
# runs posed-ray losses and adaptive split/promote/merge/demote carriers, without 3DGS

# AURA-Ingest bootstrap path for 3DGS evidence:
aura write-splat-demo-package --input tests/fixtures/tiny_3dgs_export.ply --output-dir outputs/splat-demo.aura
aura import-3dgs third_party/gaussian-splatting/output/<scene> --output-dir outputs/<scene>.aura
# imports direct PLY/JSON exports or point_cloud/iteration_*/point_cloud.ply layouts
aura compare-renders outputs/baseline.ppm outputs/splat-demo.ppm --min-psnr 35
# prints strict JSON MSE/PSNR metrics and exits nonzero if the threshold fails
aura benchmark-plan
# prints benchmark and carrier ablation plan JSON
aura ingest-adapters
# prints AURA-Ingest sources and their EvidenceSample contracts
python -m pytest
```

The native demo package is a schema/contract fixture, not a renderable research
result.

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

## Expected AURA-Core Milestone

Do not add more 3DGS convenience before the native engine exists. The next
milestone is a small end-to-end AURA-Core reconstruction fixture:

1. load or synthesize posed training images, depth, and masks;
2. initialize native AURA evidence cells without 3DGS;
3. render a CPU reference prediction and compute image/depth/ray-query losses;
4. adaptively split/promote/merge/demote carriers based on residuals and confidence;
5. optimize native carrier parameters for a few deterministic fixture steps;
6. export a native `.aura` package and training report;
7. compare the result against COLMAP/NeRF/3DGS-style baselines.

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
  exchange.py    package-backed glTF/USD exchange target metadata
  package.py     native .aura package writer/loader/validator
  ray.py         ray and ray-query response contracts
  render.py      deterministic orthographic preview and image metrics
  schema.py      native package format and supported schema versions
  semantic.py    object graph nodes and relationships
  benchmark.py   benchmark and ablation plan contracts
  core.py        AURA-Core reconstruction engine contracts
  scene.py       reference scene query
  ingest/
    baselines.py  3DGS export discovery and import adapter
    splats.py     JSON/PLY 3DGS evidence reader and AURA conversion
tests/           contract tests
docs/            GPU handoff and dataset docs
docs/schemas/    JSON Schemas for native .aura package files
```

The `ingest/` package is intentionally an adapter boundary. 3DGS splats are
converted to `EvidenceSample` records before decomposition, then become
Gaussian fallback carriers only when the evidence does not justify a stronger
native carrier. Depth priors, semantic masks, COLMAP sparse points, and future
PixelSplat/IDESplat adapters are also represented as evidence contracts rather
than direct renderer-specific elements.

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
