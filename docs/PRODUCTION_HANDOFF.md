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

1. Replace the fixture prediction loop in `src/aura/core.py` with a real
   differentiable renderer over the same `TrainingFrame` and `TrainingRegion`
   contracts.
2. Add image/depth loading for manifest `image_path`, `depth_path`, and
   `mask_path`.
3. Add COLMAP pose/intrinsics import that writes `AURA_CAPTURE_MANIFEST`.
4. Add GPU kernels or a PyTorch prototype for surface, volume, beta, gabor,
   neural residual, semantic, and Gaussian fallback carriers.
5. Implement BVH/chunk traversal for secondary ray queries.
6. Benchmark against COLMAP/textured mesh, NeRF/nerfstudio, original 3DGS,
   2DGS, ray-traced GS, and radiance-mesh/neural-primitive baselines.
7. Report PSNR/SSIM/LPIPS/FPS, but make the paper claim around scene behavior:
   ray-query correctness, collision proxy quality, editing, relighting
   confidence, semantic grouping, runtime export, and engine workflow.

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
