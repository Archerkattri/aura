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
- carrier-aware reference ray-query response with linear chunk and BVH
  traversal metrics;
- CPU reference interaction probes for hit points, shadow transmittance,
  reflection directions, and collision proxy distances;
- simple front-to-back scene query;
- tiny JSON/ASCII/binary little-endian PLY 3DGS export reader for means/opacities/covariances;
- quaternion-aware PLY covariance conversion from 3DGS log-scales;
- AURA-Ingest adapters that convert 3DGS, depth, semantic mask, and sparse
  point priors into `EvidenceSample` contracts;
- COLMAP binary/text sparse-model pose/intrinsics import to `AURA_CAPTURE_MANIFEST`;
- COLMAP sparse point depth-layer priors for native region initialization;
- COLMAP dense depth-map links and deterministic depth summaries for capture
  manifests;
- COLMAP normal-map links and average-normal summaries for normal-aware native
  priors;
- depth asset statistics that seed deterministic multi-region native
  surface-prior `TrainingRegion` evidence during manifest-to-training
  conversion;
- mask asset coverage that seeds native semantic/object `TrainingRegion`
  evidence during manifest-to-training conversion;
- semantic graph aggregation across multiple regions with the same object label;
- `.aura` package writer;
- `.aura` package loader/validator;
- explicit `.aura` format/version compatibility checks;
- JSON package inspection output and JSON Schema documents;
- runtime JSON Schema validation for package files;
- schema-validated AURA-Core posed frame and native evidence-region inputs;
- per-pixel capture asset tensors for PNG, PPM/PGM, COLMAP depth maps, COLMAP
  normal maps, and optional `imageio` EXR/HDR/video backends;
- torch/CUDA capture asset batching for manifest image/depth/mask/normal tensors
  via `torch_capture_asset_batch`;
- per-pixel capture training target generation from image/depth/mask/normal
  tensors via `capture_tensors_to_render_targets` and
  `torch_capture_training_batch`;
- torch reference rendering from capture training batches via
  `torch_render_capture_training_batch`;
- torch reference optimization steps via `torch_optimize_capture_batch`, using
  the batched native AURA forward contract for loss reporting and bounded
  carrier color updates;
- explicit torch carrier kernel specs for surface, volume, beta, gabor, neural,
  semantic, and Gaussian fallback reference semantics;
- capture-manifest reconstruction with `--load-assets` feeds per-pixel tensor
  targets into the CPU reference optimization loop;
- model-scored native feature proposals from image/depth/mask/normal tensor
  features, producing high-frequency and compact-detail `TrainingRegion`
  evidence before decomposition;
- deterministic capture asset summaries built from the same tensor path for
  manifest-backed native training fixtures;
- deterministic orthographic package preview rendering and reference
  MSE/PSNR/SSIM/LPIPS-proxy image metrics;
- CPU differentiable reference ray samples with image/depth/query losses,
  normal-target/query losses, color/depth gradients, and ray-query contract
  outputs for native AURA-Core fixture optimization;
- residual-driven confidence updates and confidence maps on optimized native
  carriers;
- optional PyTorch renderer contract with batched native first-hit/depth/color,
  transmittance, opacity, confidence, normal, material, semantic, residual,
  provenance, and query-loss outputs when installed with `aura-core[gpu]`;
- reproducible benchmark plans plus CPU reference package/query/render timing,
  confidence-quality, and interaction-quality metrics;
- ray-query correctness scoring for first-hit, carrier, depth, transmittance,
  semantic, material, normal, and residual contract checks;
- strict-JSON render comparison metrics for regression checks;
- package-backed glTF/USD exchange-target metadata;
- fixture CLI commands and tests.

It does **not** yet contain the full AURA-Core reconstruction engine:
production EXR/video tensor loading, carrier-complete GPU differentiable
optimization, CUDA kernels, production GPU BVH traversal, autograd carrier
updates, or end-to-end benchmark results.

See `docs/AURA_CORE_RESEARCH.md` for the current research direction and why the
next milestone must be native reconstruction rather than more package polish.

## Install

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# GPU development machine:
python -m pip install -e ".[dev,gpu]"
# Real capture tensor loading extras:
python -m pip install -e ".[dev,gpu,assets]"
```

## GPU Runtime

Expose the primary CUDA device for GPU development:

```bash
export CUDA_VISIBLE_DEVICES=0
aura torch-renderer-status
```

The PyTorch renderer is currently a native AABB first-hit prototype with
payload-aware reference semantics for transmittance, confidence, residual flags,
and semantic IDs. It is not yet carrier-complete CUDA rendering.

## Quick Smoke Commands

```bash
aura build-native-demo --output-dir outputs/native-demo.aura
aura validate-package outputs/native-demo.aura
# validates mixed native carriers, typed payloads, confidence maps, edit metadata, and chunks
aura inspect-package outputs/native-demo.aura
# prints the native package summary as stable JSON
aura export-report outputs/native-demo.aura
# reports native AURA preservation versus glTF/USD fallback losses for runtime engine workflows
aura torch-kernel-report
# reports native carrier autograd/CUDA readiness; surface, volume, beta, gabor, and neural have autograd, CUDA is still required
aura render-package outputs/native-demo.aura --output outputs/native-demo.ppm --width 128 --height 128
# writes a deterministic PPM preview for package validation
aura query-demo --x -0.5 --y -0.5
# queries the native mixed-carrier fixture
aura inspect-rays outputs/native-demo.aura --native-demo-probes
# prints material-aware occlusion, shadow, reflection, and collision ray-query inspections
aura benchmark-reference outputs/native-demo.aura --width 32 --height 32
# runs CPU reference package/query/render timing, confidence-quality, and interaction-quality metrics
aura benchmark-visual outputs/native-demo.aura outputs/native-reference.ppm --baseline-label native_self --min-psnr 40
# compares a package render against a teacher/reference PPM using MSE/PSNR/SSIM/LPIPS-proxy/FPS JSON metrics
aura benchmark-reference outputs/native-demo.aura --include-ablations
# runs carrier assignment ablation metrics
aura benchmark-ray-query outputs/native-demo.aura --native-demo-expectations
# scores native ray-query correctness against expected first-hit/depth/material/semantic probes
aura benchmark-core --iterations 6
# compares AURA-Core adaptive reconstruction against a static-carrier fixture run
aura migration-plan outputs/native-demo.aura
# prints package schema migration status

# AURA-Core reconstruction path:
aura write-training-frames-demo --output outputs/training-frames.json
# writes posed color/depth/semantic frames plus native evidence regions
aura reconstruct-demo --frames outputs/training-frames.json --output-dir outputs/reconstruct-demo.aura --iterations 6
# runs posed image/depth/query losses and adaptive split/promote/merge/demote carriers, without 3DGS

# Real-capture manifest path:
aura write-capture-manifest-template --output outputs/capture-manifest.json
# writes the schema-backed image/depth/mask/normal/camera/evidence manifest template
aura capture-manifest-to-training outputs/capture-manifest.json --output outputs/training-from-capture.json
# validates the manifest and converts it to the AURA-Core training dataset contract
aura reconstruct-capture-manifest outputs/capture-manifest.json --output-dir outputs/reconstruct-capture.aura --iterations 6
# runs the current CPU reference reconstruction path from a real-capture manifest

# Asset-backed fixture captures:
aura inspect-capture-assets data/custom-captures/<scene>/capture-manifest.json
# loads existing PNG, PPM/PGM, or COLMAP depth-map assets and prints deterministic summaries
aura inspect-capture-tensors data/custom-captures/<scene>/capture-manifest.json
# prints per-frame image/depth/mask/normal tensor shape, backend, and sample metadata
aura capture-manifest-to-training data/custom-captures/<scene>/capture-manifest.json --output outputs/training-from-capture-assets.json --load-assets
# replaces target color/depth summaries from PNG, PPM/PGM, or COLMAP depth-map assets
aura reconstruct-capture-manifest data/custom-captures/<scene>/capture-manifest.json --load-assets --pixel-stride 8 --max-targets-per-frame 1024
# feeds sampled per-pixel capture tensor targets into the CPU reference reconstruction loop
aura torch-optimize-capture-manifest data/custom-captures/<scene>/capture-manifest.json --device cuda --pixel-stride 8 --max-targets-per-frame 1024 --iterations 6
# runs the torch reference optimization scaffold from native capture tensor batches

# COLMAP pose/intrinsics ingest:
aura colmap-to-capture-manifest data/custom-captures/<scene>/colmap --root data/custom-captures/<scene> --output outputs/capture-from-colmap.json
# converts COLMAP cameras/images/points3D .bin or .txt files, plus standard stereo/depth_maps when present, into the native capture manifest contract

# AURA-Ingest bootstrap path for 3DGS evidence:
aura write-splat-demo-package --input tests/fixtures/tiny_3dgs_export.ply --output-dir outputs/splat-demo.aura
aura import-3dgs third_party/gaussian-splatting/output/<scene> --output-dir outputs/<scene>.aura
# imports direct PLY/JSON exports or point_cloud/iteration_*/point_cloud.ply layouts
aura compare-renders outputs/baseline.ppm outputs/splat-demo.ppm --min-psnr 35
# prints strict JSON MSE/PSNR/SSIM/LPIPS-proxy metrics and exits nonzero if the threshold fails
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
milestone is a small end-to-end AURA-Core reconstruction path:

1. write or import an `AURA_CAPTURE_MANIFEST` with image/depth/mask/normal paths,
   camera intrinsics, posed frame summaries, and native seed regions;
2. convert that manifest into the `AURA_TRAINING_FRAMES` contract;
3. initialize native AURA evidence cells from those region specs without 3DGS;
4. replace the current CPU reference prediction with differentiable rendering;
5. compute image/depth/ray-query losses from real frames;
6. adaptively split/promote/merge/demote carriers based on residuals and confidence;
7. optimize native carrier parameters on GPU;
8. export a native `.aura` package and training report;
9. compare the result against COLMAP/NeRF/3DGS-style baselines.

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
  optimize.py    CPU differentiable reference ray samples and gradients
  package.py     native .aura package writer/loader/validator
  ray.py         ray and ray-query response contracts
  render.py      deterministic orthographic preview and image metrics
  schema.py      native package format and supported schema versions
  semantic.py    object graph nodes and relationships
  benchmark.py   benchmark and ablation plan contracts
  core.py        AURA-Core reconstruction engine contracts
  scene.py       reference scene query and chunk traversal
  ingest/
    baselines.py  3DGS export discovery and import adapter
    colmap.py     COLMAP binary/text camera/pose/sparse-point manifest importer
    capture.py    image/depth/mask/normal/camera manifest to AURA-Core training dataset
    splats.py     JSON/PLY 3DGS evidence reader and AURA conversion
tests/           contract tests
docs/            GPU handoff, production handoff, and dataset docs
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
