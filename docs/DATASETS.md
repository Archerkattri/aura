# Datasets and Baselines

All datasets and external baseline repositories belong under ignored directories
and must not be committed to the repository.

## Directory Convention

```text
data/
  mipnerf360/
  tanks-and-temples/
  deep-blending/
  llff/
  nerfstudio-fixtures/
  custom-captures/
third_party/
  gaussian-splatting/
  gsplat/
  nerfstudio/
```

## Supported Datasets

| Dataset | Purpose |
|---|---|
| Mip-NeRF 360 | Standard novel-view synthesis quality comparison |
| Tanks and Temples | Outdoor / geometry stress tests |
| Deep Blending | Indoor view-synthesis stress tests |
| LLFF | Smaller smoke / regression scenes |
| Custom captures | AURA asset behavior on proprietary scenes |

## Baseline Methods

| Baseline | Purpose |
|---|---|
| Original 3DGS | Visual quality reference and evidence ingest source |
| gsplat / nerfstudio | Practical modern training and rendering harness |
| 2DGS | Geometry-oriented splat baseline |
| 3DGRT | Ray-query Gaussian baseline |
| EVER | Volumetrically consistent 3DGS comparison |
| Beta/Gabor splatting | Carrier-specific competitors |
| Splat the Net / Radiance Meshes | Closest post-3DGS substrate comparisons |

## Benchmark Commands

```bash
# Reference benchmark against a built package (no external data required)
aura benchmark-reference outputs/scene.aura --include-ablations

# Visual quality comparison against a teacher or baseline render
aura benchmark-visual outputs/scene.aura data/baselines/<scene>/reference.ppm

# Real-scene benchmark against an external baseline directory
aura benchmark-real-scene outputs/scene.aura \
  --reference-dir data/baselines/<scene> \
  --baseline-label 3dgs \
  --min-psnr 25

# CUDA vs. torch runtime throughput and parity
aura benchmark-cuda-runtime
```

`benchmark-real-scene` falls back to deterministic fixtures when
`--reference-dir` is omitted, since datasets are kept out of git.

## Importing 3DGS Exports

`aura import-3dgs` accepts either a direct `.ply` / fixture `.json` export or a
standard original-3DGS output directory containing
`point_cloud/iteration_*/point_cloud.ply`. When multiple iterations exist, the
latest numeric iteration is selected. Directories with several unrelated `.ply`
files are rejected to avoid ambiguity.

PLY scale fields follow the original 3DGS log-scale convention. Rotation
quaternions are applied when computing world covariance. Imported splats become
`EvidenceSample` records and are assigned Gaussian fallback carriers only where
native carrier evidence is insufficient.

## Capture Asset Contracts

Capture manifests may reference image, depth, mask, and normal assets. The
standard path loads PNG, PPM/PGM, and COLMAP dense maps into packed float
buffers. Optional backends (`imageio[assets]`) handle EXR, HDR, and video
frames.

```python
from aura.ingest.capture import load_capture_asset_tensors

tensors = load_capture_asset_tensors(
    manifest,
    max_loaded_bytes=4 * 1024**3,   # cap total decoded batch
    max_frame_bytes=256 * 1024**2,  # cap per-frame decode
)
```

For large captures, use tiled sampling:

```bash
aura plan-capture-sampling data/custom-captures/<scene>/capture-manifest.json \
  --tile-size 256 --pixel-stride 8 --max-targets-per-frame 4096
```

`plan_capture_tensor_sampling` records deterministic tile counts and
`capture_tensors_to_packed_render_batches` materializes bounded batches. Each
packed batch includes `sourceWindows` — the exact tile target ranges used —
so a streaming or GPU loader can reproduce the same row-major sampling without
keeping an unbounded per-pixel target list.
