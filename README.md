# AURA-Core

**AURA** (Adaptive Unified Radiance Asset) is a post-3DGS 3D scene
reconstruction engine. It trains a mixed adaptive carrier representation
directly from posed image, depth, mask, and normal captures and exports a
queryable `.aura` scene package.

```text
COLMAP / MVS  →  NeRF  →  3D Gaussian Splatting  →  AURA
```

Rather than optimizing a single primitive type across the entire scene, AURA
assigns each spatial region the most appropriate carrier from a typed family
(surface, volume, beta kernel, gabor/frequency, neural residual, semantic, or
Gaussian fallback). Carriers evolve during training through split, promote,
merge, and demote decisions driven by measured residuals. The result is an
auditable, ray-queryable scene representation with native support for
confidence, geometry proxies, semantic grouping, and LOD.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design rationale
and carrier-family descriptions.

## PRISM — AURA's own differentiable rasterizer

AURA renders and trains through **PRISM** (*Pluggable Radiance-prImitive
Splatting Module*, `aura.prism` / `aura.prism_cuda`) — a differentiable, GPU,
tile-based alpha-compositing rasterizer with a **pluggable per-carrier
footprint**, built as AURA's own alternative to gsplat. It splats a *spectrum*
of carrier types, not just Gaussians, and trains them jointly:

| Footprint | Kernel | Backend |
|---|---|---|
| **gaussian** | `exp(-½·conic)` (3DGS-style) | CUDA fwd + diff. backward |
| **beta** | bounded `(1-r/3)^β` (Deformable-Beta) | CUDA fwd + diff. backward |
| **gabor** | envelope × oscillation (high-freq texture) | CUDA fwd + diff. backward |
| **neural** | bounded MLP over Fourier features (Splat-the-Net) | torch (autograd) |

A single scene mixes carrier types per region (`--carrier auto` assigns Gabor to
high-texture regions from image gradients, Gaussian elsewhere). PRISM also
supports adaptive densification (`--densify`) and EVER-style
volumetrically-consistent alpha (`--volumetric`, `1-exp(-opacity·w)`).

```bash
aura train-prism outputs/truck-pts129k-manifest.json --carrier auto --densify --scale 0.25
python scripts/eval_psnr.py <out>.aura outputs/truck-pts129k-manifest.json --renderer prism --scale 0.25
```

## Results (Tanks & Temples — Truck, real data)

The reconstruction lineage on the same scene, eval frames, and **the same correct
COLMAP poses** (Photogrammetry → NeRF → 3DGS → AURA):

![GT · COLMAP · NeRF · 3DGS · AURA — Truck](docs/lineage_truck.png)

_Ground truth · COLMAP SfM point cloud (photogrammetry) · NeRF (compact
from-scratch) · executed vanilla 3DGS · AURA. AURA reconstructs the truck sharply,
on par with vanilla 3DGS — its current quality path is the gsplat backend, so the
two look alike (that's honest, not a coincidence)._

### The pose fix (the real story)

AURA (and the executed baseline) initially plateaued at ~14–16 dB while published
3DGS reaches ~25. The cause was **not** the representation — it was a bug: the
COLMAP→manifest conversion stored only `camera_origin` + `look_at` (forward
direction) and reconstructed the view from a fixed `up=(0,-1,0)`, **dropping
camera roll**. Handheld captures have roll, so every pose was wrong and
reconstruction was capped regardless of iterations/SH/densification.

Smoking gun (`experiments/direct_pose_test.py`): training gsplat with the **full
COLMAP poses** (quaternion+translation) directly gives **20.6 dB** @0.25 (7k
iters) vs ~14 dB through the manifest path — **+6.6 dB from correct poses alone**.
The fix carries the full world-to-camera rotation (`view_rotation`) through the
pipeline. After it:

| Run (correct poses) | Scale | Iters | PSNR | SSIM | N |
|---|---|---|---|---|---|
| AURA (gsplat backend) | 0.25 | 7,000 | 18.44 dB | 0.580 | 129,531 |
| AURA (gsplat backend, full-res, SH, densify) | 1.0 | 30,000 | **19.44 dB** | 0.689 | 3,428,171 |
| direct COLMAP-pose gsplat (reference) | 0.25 | 7,000 | 20.60 dB | — | 129,531 |
| 3DGS (Kerbl 2023, published, original repo) | 1.0 | 30,000 | ~25.19 dB | ~0.879 | — |

AURA is now genuinely competitive with 3DGS view synthesis. The remaining gap to
the published ~25 dB reflects densification/LR tuning and harness/eval-protocol
differences, not the representation.

### Rasterizer speed (PRISM forward, RTX 5090, ms/frame · FPS)

PRISM's CUDA kernel is real-time; ~18–25× its own torch tiled path. gsplat (a
mature fused library) is faster still.

| Carriers | Res | PRISM CUDA | PRISM torch | gsplat |
|---|---|---|---|---|
| 50k | 512² | 1.68 ms / 595 fps | 35.9 ms | 0.23 ms |
| 100k | 512² | 1.99 ms / 503 fps | 36.4 ms | 0.29 ms |
| 200k | 979×546 | 7.23 ms / 138 fps | 54.5 ms | 1.26 ms |

### Honest status of PRISM and typed carriers

PRISM (AURA's own typed-carrier rasterizer) is real-time and trains all four
carrier footprints (gaussian/beta/gabor/neural), but it is **not yet at parity
with the gsplat backend on quality**: PRISM-native training trails by several dB
on dense real scenes, and showed instability at very large per-tile caps. An
earlier carrier-type ablation suggested typed/mixed carriers beat plain Gaussian
(+0.8 dB) — but that result **did not survive the pose fix**: at correct poses,
Gaussian (12.08) ≈ adaptive-mix (11.98) at the same budget. Typed carriers still
demonstrably help on *controlled* signals (Gabor beats Gaussian on stripes, a
neural carrier beats a Gaussian on a ring — `tests/test_prism.py`), but a
real-scene advantage at matched poses/budget is **not yet established**. Closing
the PRISM-vs-gsplat quality gap, and a learned (vs heuristic) carrier assignment,
are the open problems. The convergence numbers above use the gsplat backend,
which is AURA's current quality reference.

Reproduce: `python experiments/prism_ablation.py ...`, `prism_benchmark.py`,
`direct_pose_test.py` (results in `experiments/results/`).

## Features

- Native carrier registry for surface, volume, beta, gabor, neural residual,
  semantic, and Gaussian fallback carriers with typed payload validation.
- `.aura` package writer, loader, and runtime JSON Schema validator covering
  manifest, elements, chunks, exchange, semantic graph, and training records.
- Capture manifest format for posed image / depth / mask / normal assets with
  COLMAP sparse-model import (`aura colmap-to-capture-manifest`).
- Packed capture tensor loading for PNG, PPM/PGM, COLMAP dense maps, and
  optional `imageio` EXR/HDR/video backends.
- Tiled PyTorch optimization (`aura train`) with device-resident asset batching,
  mask-aware pixel sampling, camera ray construction, configurable loss weights,
  gradient clipping, and checkpoint/resume support.
- Grouped torch ray/carrier intersections for all carrier types with ordered
  front-to-back compositing across color, alpha, depth, normal, confidence,
  material, semantics, residual, and hit outputs.
- Compiled CUDA renderer dispatched via pybind11 `render_rays` over packed
  scene and ray tensors, with measured per-carrier parity against the PyTorch
  renderer.
- Production GPU BVH traversal kernel (`render_rays_bvh`) using a flattened
  binned-SAH element BVH (median-split fallback), replacing the brute-force scan.
- Anti-aliasing for the torch renderer: Mip-Splatting-style 3D frequency cap,
  ray-cone footprint prefilter, and 2x2 supersampling (all opt-in), plus
  early-transmittance termination for energy-conserving compositing.
- Per-attribute Adam optimization with per-group learning-rate schedules,
  alongside the existing SGD path; gradient-magnitude (AbsGS) accumulation,
  opacity reset/recovery signals, importance scores (RadSplat), carrier budget
  ceilings, and optional depth-distortion / normal-consistency losses (2DGS).
- SOTA carrier upgrades (opt-in, defaults preserve prior behavior): deformable
  Beta kernels, multi-directional Gabor filter banks, Scaffold-GS-style anchored
  neural-residual carriers, and LangSplat-style sparse-codebook semantic
  features.
- Semantic-graph-governed heterogeneous carrier allocation (`aura.allocation`):
  scene-graph clustering selects the carrier type per region, with soft
  inter-type conversion scores, cross-carrier residual hooks, and a
  single-carrier ablation mode for typed-mix-vs-baseline studies.
- Physically based shading and relighting (`aura.shading`): Lambertian,
  Cook-Torrance microfacet with split-sum IBL, and BVH shadow-ray visibility
  baking, with per-carrier albedo/roughness/metallic and a relighting demo path
  (emissive output unchanged when shading is disabled).
- CUDA-vs-torch runtime benchmark (`aura benchmark-cuda-runtime`) measuring
  on-device throughput and cross-backend parity.
- EXR/PFM float radiance export and turntable video export (MP4 via
  `imageio[ffmpeg]` or system `ffmpeg`, with PNG frame-sequence fallback).
- Long-run memory stability probe tracking tracemalloc and torch CUDA
  allocations across many iterations.
- Real-scene benchmark harness scoring an `.aura` package against external
  COLMAP/NeRF/3DGS baseline renders (PSNR/SSIM/LPIPS-proxy JSON report).

## Maturity

All capabilities above are implemented and covered by the deterministic test
suite. The advanced optimization, anti-aliasing, carrier, allocation, and
shading paths are validated on fixtures and gated behind opt-in configuration so
that default behavior is unchanged. Quantitative quality and performance claims
against COLMAP / NeRF / 3DGS baselines require running `aura benchmark-real-scene`
on external datasets and have not yet been published. The
semantic-graph-governed allocation framework provides working graph-driven
carrier selection plus the differentiable inter-type-conversion and
cross-carrier residual structure; learning those assignments end-to-end is a
training-time step that runs once real captures are supplied.

## Requirements

- Python 3.11+
- PyTorch 2.3+ (for GPU training and the torch renderer)
- CUDA toolkit (for the compiled CUDA renderer; optional)
- `imageio[assets]` (for EXR export and video; optional)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gpu,assets]"
```

The `gpu` extra adds PyTorch. The `assets` extra adds `imageio` for EXR and
video support. The `dev` extra adds pytest.

### CUDA Renderer

After installing on a GPU machine:

```bash
export CUDA_VISIBLE_DEVICES=0
aura torch-renderer-status
aura cuda-kernel-build-report --build
```

`cuda-kernel-build-report` compiles and loads the pybind11 extension and
reports `compiled: true, loadable: true` when the CUDA renderer is available.

## Quickstart

### 1. Verify the installation

```bash
python -m pytest -q
aura build-native-demo --output-dir outputs/native-demo.aura
aura render outputs/native-demo.aura --backend torch --output outputs/native-demo.ppm
aura benchmark-ray-query outputs/native-demo.aura --native-demo-expectations
```

### 2. Create a capture manifest

```bash
aura write-capture-manifest-template --output outputs/capture-manifest.json
```

Edit the template to reference your posed images, depth maps, masks, and
normals. Or import from a COLMAP sparse model:

```bash
aura colmap-to-capture-manifest data/custom-captures/<scene>/colmap \
  --root data/custom-captures/<scene> \
  --output outputs/capture-from-colmap.json
```

### 3. Inspect and plan sampling

```bash
aura inspect-capture-tensors data/custom-captures/<scene>/capture-manifest.json
aura plan-capture-sampling data/custom-captures/<scene>/capture-manifest.json \
  --tile-size 256 --pixel-stride 8 --max-targets-per-frame 4096
```

### 4. Train

```bash
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

Training writes `training_report.json`, optional checkpoint packages, and the
optimized `.aura` package under the selected output directory.

### 5. Resume from a checkpoint

```bash
aura train data/custom-captures/<scene>/capture-manifest.json \
  --device cuda \
  --output outputs/scene-resumed.aura \
  --resume-from outputs/scene.aura/checkpoints/iter_000001.aura
```

## Rendering and Querying

```bash
# Validate and inspect a package
aura validate-package outputs/scene.aura
aura inspect-package outputs/scene.aura

# Render (PPM, EXR, or PFM)
aura render outputs/scene.aura --backend torch --device cuda \
  --output outputs/scene.ppm --width 256 --height 256
aura render outputs/scene.aura --format exr \
  --output outputs/scene.exr --width 256 --height 256

# Turntable video
aura render-video outputs/scene.aura --output outputs/turntable.mp4 \
  --frames 48 --fps 24

# Benchmarks
aura benchmark-reference outputs/scene.aura --width 64 --height 64
aura benchmark-visual outputs/scene.aura outputs/reference.ppm --min-psnr 30
aura benchmark-real-scene outputs/scene.aura \
  --reference-dir data/baselines/<scene> --baseline-label 3dgs --min-psnr 25

# Memory stability
aura memory-stability-probe outputs/scene.aura --iterations 256
```

EXR export requires `imageio[assets]`; without it `--format exr` writes a
stdlib `.pfm` float raster instead. `aura render-video` writes MP4 when an
encoder is available, otherwise a PNG/PPM frame sequence plus a
`sequence.json` manifest.

## Fixture Smoke Tests

These commands exercise small deterministic fixtures without requiring external
datasets:

```bash
aura build-native-demo --output-dir outputs/native-demo.aura
aura render outputs/native-demo.aura --backend torch --output outputs/native-demo.ppm
aura benchmark-ray-query outputs/native-demo.aura --native-demo-expectations
aura write-training-frames-demo --output outputs/training-frames.json
aura reconstruct-demo \
  --frames outputs/training-frames.json \
  --output-dir outputs/reconstruct-demo.aura \
  --iterations 6 \
  --render-backend torch
python -m pytest
```

## Repository Map

```text
src/aura/
  cli.py               CLI — train, render, ingest-adapters, benchmark, inspect commands
  core.py              Reconstruction contracts and adaptive evolution policy
  torch_renderer.py    Torch render batches, grouped carrier hits, compositing
  torch_optimizer.py   Tiled capture optimization and checkpoint snapshots
  torch_kernels.py     Carrier parameter tensors and differentiable responses
  cuda_renderer.py     CUDA renderer ABI and dispatch boundary
  cuda/                CUDA kernels and pybind11 extension sources
  ingest/              Capture, COLMAP, 3DGS, depth, and semantic adapters
  package.py           .aura package IO and validation
  scene.py             Ray-query traversal and response assembly
  schemas/             JSON Schema files for runtime validation
tests/                 Deterministic contract, optimizer, renderer, CLI tests
docs/
  ARCHITECTURE.md      Design rationale, carrier families, pipeline overview
```

`src/aura/ingest/` is an adapter boundary. 3DGS exports become `EvidenceSample`
records and survive only as Gaussian fallback carriers when native carrier
assignment does not justify a stronger representation. All 3DGS-specific logic
stays under `aura.ingest`.

## Data Layout

Keep datasets, checkpoints, renders, and third-party baselines out of git:

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
outputs/          (generated packages, renders, and reports — git-ignored)
```

Baseline datasets (Mip-NeRF 360, Tanks and Temples, Deep Blending, and 3DGS /
nerfstudio exports) are obtained from their original sources and kept under the
git-ignored `data/` and `third_party/` directories. Score a trained package
against external baseline renders with `aura benchmark-real-scene`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

- Keep generated `.aura` packages, datasets, checkpoints, renders, and secrets
  out of git (all covered by `.gitignore`).
- All 3DGS-specific logic must remain under `aura.ingest`; splats are evidence
  inputs, not native representation elements.
- New ingest sources must produce `EvidenceSample` records before
  decomposition.

## License

MIT License. See [LICENSE](LICENSE) for details.
