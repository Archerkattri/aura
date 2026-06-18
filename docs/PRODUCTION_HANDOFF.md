# AURA-Core Production Handoff

This repo is a private AURA-Core scaffold. The next machine should use GPU
resources to turn the reference contracts into the real reconstruction engine.

## Setup

```bash
git clone https://github.com/Archerkattri/aura.git
cd aura
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,gpu,assets]"
export CUDA_VISIBLE_DEVICES=0
```

## First Verification On The GPU Machine

```bash
python -m pytest -q
aura write-capture-manifest-template --output outputs/capture-manifest.json
aura capture-manifest-to-training outputs/capture-manifest.json --output outputs/training-from-capture.json
aura reconstruct-capture-manifest outputs/capture-manifest.json --output-dir outputs/reconstruct-capture.aura --iterations 6
aura validate-package outputs/reconstruct-capture.aura
aura inspect-package outputs/reconstruct-capture.aura
aura readiness-report
aura production-gate-report outputs/reconstruct-capture.aura
aura torch-kernel-report
aura cuda-kernel-build-report --build
```

For manifests whose `image_path`, `depth_path`, `mask_path`, and `normal_path`
files exist as PNG, PPM/PGM, or COLMAP dense-map assets, also run:

```bash
aura inspect-capture-assets data/custom-captures/<scene>/capture-manifest.json
aura inspect-capture-tensors data/custom-captures/<scene>/capture-manifest.json
aura plan-capture-sampling data/custom-captures/<scene>/capture-manifest.json --tile-size 256 --pixel-stride 8 --max-targets-per-frame 1024
aura capture-manifest-to-training data/custom-captures/<scene>/capture-manifest.json --output outputs/training-from-capture-assets.json --load-assets
aura reconstruct-capture-manifest data/custom-captures/<scene>/capture-manifest.json --load-assets --tile-size 256 --pixel-stride 8 --max-targets-per-frame 1024 --output-dir outputs/reconstruct-capture-assets.aura --iterations 6
aura torch-optimize-capture-manifest data/custom-captures/<scene>/capture-manifest.json --device cuda --tile-size 256 --pixel-stride 8 --max-targets-per-frame 4096 --max-targets-per-batch 1024 --output-dir outputs/torch-optimize-capture.aura --iterations 6
```

The capture reconstruction and torch optimization commands reuse one
`load_capture_asset_tensors` result for summaries, native region proposals, and
per-pixel targets through `capture_tensors_to_training_dataset`. Keep future
tiled, memory-mapped, or GPU-native loaders on that single-batch contract rather
than re-decoding assets in each stage. Both reports include
`captureSamplingPlan`, which records the tile size, stride, and sampled/masked
pixel counts the GPU implementation should reproduce.
The torch optimization command consumes `CapturePackedRenderBatch` descriptors
instead of one monolithic target list; its report includes packed batch counts,
target counts, batch indices, target offsets, and source windows so CUDA kernels
can be parity-tested against the same deterministic tiled stream.

For COLMAP sparse models, generate the capture manifest with:

```bash
aura colmap-to-capture-manifest data/custom-captures/<scene>/colmap --root data/custom-captures/<scene> --output outputs/capture-from-colmap.json
```

## Real Data Layout

Do not commit data, third-party repos, checkpoints, outputs, or secrets.

```text
data/
  custom-captures/<scene>/
    images/
    depth/
    masks/
    colmap/
third_party/
  gaussian-splatting/
  gsplat/
  nerfstudio/
  colmap-scripts/
outputs/
```

## Current AURA-Core Contract

The current ingest path is:

```text
AURA_CAPTURE_MANIFEST
  -> TrainingDataset
  -> TrainingFrame + TrainingRegion
  -> EvidenceSample
  -> adaptive typed AuraElement carriers
  -> .aura package + training_report.json
```

3DGS is allowed as one evidence initializer or baseline, but it must stay under
`aura.ingest`. Do not make 3DGS the native representation. Splats must become
`EvidenceSample` records before decomposition.

## Production Tasks

Run `aura readiness-report` before claiming production status. The report is a
native AURA audit of implemented scaffolds versus missing production pillars,
including native carriers, package validation, PyTorch reference support, CUDA
kernel status, renderer/trainer gaps, and benchmark claim boundaries.
Run `aura production-gate-report <package>` on any package being used for a
claim. Benchmark reports also emit `productionGate`. Treat
`productionReady: false` as authoritative: current CPU reference and visual
smoke outputs are blocked from production interpretation while CUDA renderer
readiness is unavailable, the visual score is a self-reference comparison, or
the benchmark package has not exercised native AURA carrier families beyond
Gaussian fallback.
The report intentionally separates the legacy `cuda_kernels` metadata-only
renderer report from the callable `aura.cuda_renderer` boundary. A callable CPU
or torch fallback proves the launch/output contract can be exercised; it is not
CUDA acceleration and keeps production CUDA claims blocked until compiled CUDA
dispatch is available, parity-tested, and benchmarked.
The packaged `aura_render_rays_kernel` renderer source symbol is the ABI target
for that future dispatch. The repo now also packages `cuda/aura_bindings.cpp`,
which exposes a `render_rays(...)` PyTorch extension binding for packed tensor
dispatch into `aura_render_rays_launcher`; source availability alone must not
clear any production gate until the extension is compiled/imported on CUDA
hardware and passes parity plus speed gates.
`benchmark-reference` and `production-gate-report` also emit
`cudaRendererAbiParity`, a CPU oracle that packs native scene/ray buffers for
the packaged renderer ABI and compares first-hit indices against
`AuraScene.traverse_ray`. Passing this parity check means the flat-buffer ABI is
deterministic enough for compiled CUDA parity tests; its own
`productionReady: false` value remains a production-gate blocker until the
compiled CUDA renderer dispatch is available and benchmarked.

1. Replace the optional payload-aware PyTorch ordered-compositing reference path
   and CPU differentiable reference renderer with a carrier-complete
   PyTorch/CUDA renderer over the same `TrainingFrame` and `TrainingRegion`
   contracts.
2. Replace the CPU reference optimization loop with a GPU loop that uses
   `torch_capture_training_batch` and `torch_render_capture_training_batch` for
   forward passes, gradients, and carrier updates.
3. Replace the packed host capture tensor buffers with tiled, memory-mapped, or
   GPU-native asset loading for full-resolution image/video datasets.
4. Train and validate capture proposal weights with
   `train_capture_proposal_model` on labeled COLMAP/capture image, depth, mask,
   and normal features, then replace the lightweight logistic contract with a
   neural region proposal backend once real labels exist.
5. Build and import the packaged CUDA/PyTorch extension, verify that
   `aura_render_rays_kernel`, `aura_render_rays_launcher`, and `render_rays`
   are present, then run CPU/torch/CUDA parity over `cuda_render_rays`.
6. Replace the torch autograd carrier specs with CUDA kernels for every carrier.
   Surface, volume, beta, gabor, neural residual, semantic, and Gaussian
   fallback carriers have tested torch autograd paths only; `aura
   torch-kernel-report` must report `productionReady: true` before claiming
   this is complete.
7. Replace the cached CPU reference chunk BVH with a production BVH/GPU
   traversal path for secondary rays.
8. Benchmark against COLMAP/textured mesh, NeRF/nerfstudio, original 3DGS,
   2DGS, ray-traced GS, and radiance-mesh/neural-primitive baselines.
   Start with dataset manifests under `data/` and third-party baselines under
   `third_party/`; benchmark outputs belong under ignored `outputs/`.
9. Replace the current deterministic LPIPS-proxy metric with a learned LPIPS
   backend and report PSNR/SSIM/LPIPS/FPS, but make the paper claim around scene
   behavior: ray-query correctness, collision proxy quality, editing, relighting
   confidence, semantic grouping, runtime export, and engine workflow.
10. Clear the benchmark `productionGate` only after production CUDA renderer
   readiness is true and visual benchmarks compare against external teacher or
   baseline renders rather than package self-reference renders.

## Paper Claim Boundary

Safe current claim:

> AURA-Core is a scaffold for a native adaptive radiance reconstruction engine
> that converts captures into queryable runtime assets.

Do not claim:

- better PSNR than 3DGS;
- real-time performance;
- physical PBR recovery;
- production-ready engine integration;
- robustness on real datasets;
- a complete successor to 3DGS.

Those require GPU-side implementation and benchmarks.
