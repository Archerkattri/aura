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

## First Data Rule

Start with one small scene and a fixture export. AURA should first prove the
contract can ingest a trained splat scene and answer ray queries before adding
large benchmark suites.

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
