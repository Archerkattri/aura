# Datasets And Baselines - AURA

All datasets and external baseline repos belong under ignored directories.

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

## Priority Benchmarks

| Dataset | Purpose |
| --- | --- |
| Mip-NeRF 360 | standard NVS quality comparison |
| Tanks and Temples | outdoor/geometry stress |
| Deep Blending | indoor view-synthesis stress |
| LLFF | smaller smoke/regression scenes |
| Custom room/courtyard capture | AURA asset behavior demo |

## Priority Baselines

| Baseline | Purpose |
| --- | --- |
| Original 3DGS | visual quality and teacher/initializer |
| gsplat / nerfstudio | practical modern training/rendering harness |
| 2DGS | geometry-oriented splat baseline |
| 3DGRT | ray-query Gaussian baseline |
| EVER / volumetrically consistent 3DGS | volumetric correctness comparison |
| Beta/Gabor splatting | carrier-specific competitors |
| Splat the Net / Radiance Meshes | closest post-3DGS substrate comparisons |

## Benchmark Harnesses

The current reproducible harness is `aura benchmark-reference <package>`,
optionally with `--include-ablations`. It reports CPU reference package/query/
render timing, native carrier coverage, runtime export readiness, interaction
quality, ray-query correctness, and `cudaRendererAbiParity`.
`cudaRendererAbiParity` is a CPU oracle for the packaged CUDA renderer ABI:
it validates deterministic flat-buffer inputs and first-hit output parity, but
it keeps `productionReady: false` until compiled CUDA dispatch exists.

Use `aura benchmark-visual <package> <teacher.ppm>` only with teacher or
baseline renders generated from ignored dataset/baseline directories when
making visual-quality comparisons. Self-reference previews remain smoke tests.
Run `aura production-gate-report <package>` beside every benchmark result used
for a claim; the gate must remain blocked while CUDA renderer readiness,
external visual baselines, or native carrier coverage are incomplete.

## First Data Rule

Start with one tiny mixed native AURA fixture, then one small trained 3DGS scene
as an ingest evidence source. AURA should first prove that adaptive typed
carriers can be packaged, loaded, queried, and inspected before adding large
benchmark suites.

The current scaffold accepts fixture JSON, ASCII PLY, and binary little-endian
PLY Gaussian splat exports. PLY scale fields follow the original 3DGS log-scale
convention, and rotation quaternions are applied when computing world
covariance. Full checkpoint loading and renderer-specific checkpoint adapters
remain explicit next steps.

The `aura import-3dgs` command accepts either a direct `.ply`/fixture `.json`
export or a common original-3DGS output directory containing
`point_cloud/iteration_*/point_cloud.ply`. When multiple standard iterations
exist, the latest numeric iteration is selected. Ambiguous arbitrary directories
with several unrelated `.ply` files are rejected.

## Real Capture Tensor Contracts

Capture manifests may reference image, depth, mask, and normal assets. The
stdlib path loads PNG, PPM/PGM, and COLMAP dense maps into packed float buffers;
optional asset backends handle EXR/HDR/video when installed. Loaded
`CaptureTensor` and `CaptureFrameTensors` records report `loadedBytes` so
callers can audit host memory before moving a batch to CPU or torch training.

Use `load_capture_asset_tensors(manifest, max_loaded_bytes=..., max_frame_bytes=...)`
when loading real captures from scripts. The first limit caps the decoded
manifest batch, and the second caps each decoded frame. Tiled sampling remains
the path for large captures: `plan_capture_tensor_sampling(...)` records
deterministic tile counts, and `capture_tensors_to_packed_render_batches(...)`
materializes bounded array-backed batches. Each packed batch now includes
`sourceWindows`, which identify the exact tile target ranges used to build that
batch so a streaming or GPU loader can reproduce the same row-major sampling
without keeping an unbounded per-pixel target list.
