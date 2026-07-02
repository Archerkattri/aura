# AURA

**Adaptive Unified Radiance Asset**

AURA turns posed captures into a typed, queryable, relightable, engine-ready
radiance asset. It keeps the fast Gaussian/DBS-Beta renderers where they are
strong, then adds the asset layer they do not provide: typed carrier metadata,
confidence, semantics, relighting, ray queries, PRISM extension footprints, and
standards-based export.

<p align="center">
  <img src="docs/truck_orbit.gif" width="82%" alt="AURA reconstruction orbit on Truck"><br>
  <em>Truck reconstructed as an AURA asset.</em>
</p>

<p align="center">
  <img src="docs/aura_capability_reel.gif" width="82%" alt="AURA capability reel: reconstruction, depth, confidence, semantics, and query"><br>
  <em>Current asset operations: reconstruction, depth, confidence, semantics, and open-vocabulary query.</em>
</p>

## Why AURA Exists

Photogrammetry gives sparse geometry. NeRF gives a continuous radiance field.
3DGS gives high-quality real-time splats. AURA keeps that progress and moves the
output closer to a usable scene asset:

- **Typed carriers:** Gaussian, Beta, Gabor, and neural footprints under one
  contract.
- **Asset behavior:** ray query, confidence, semantic identity, relighting, and
  export are first-class operations.
- **Engine compatibility:** export to `KHR_gaussian_splatting` GLB and USD
  preview.
- **PRISM extension layer:** PRISM adds non-Gaussian footprints on top of the
  primary gsplat/DBS-Beta quality path. It is not a replacement for gsplat or
  DBS-Beta.

![How COLMAP, NeRF, 3DGS, and AURA represent a scene](docs/how_it_works.png)

## Current Status

| Area | Status | Evidence |
|---|---|---|
| Local publication validation | **11/11 gates pass** | `experiments/results/publication_validation_2026-06-25.json` |
| Dataset audit | **8/8 local scenes complete** | Truck + 7 extracted Mip-NeRF 360 roots |
| Typed quality | **Beta beats fixed Gaussian on every audited scene** | mean +0.80 dB PSNR |
| Compactness | **Beta reaches Gaussian quality with about half the carriers on Truck** | 500k Beta > 1M fixed Gaussian |
| PRISM | **Complete for its intended additive role** | Gaussian/Beta stay primary; Gabor/neural route to PRISM |
| Engine export | **GLB/USD bridge validated** | `KHR_gaussian_splatting` GLB + USD runtime report |
| Real-scene FPS | **Trained checkpoints exceed 30 FPS (measured on RTX 5090)** | Truck DBS-Beta/fixed-Gaussian + 3DGUT Truck/Room |
| External baselines | **Local + official same-split table complete** | COLMAP, NeRF, 3DGS, 2DGS-style, ray-traced-GS-style, official 2DGS, official 3DGUT |
| SOTA A/B upgrades | **Local artifact-backed A/B ready** | DINOv3, VGGT, Depth Anything 3, 3DGUT, official 2DGS |
| Calibrated confidence (P0) | **Certified per-carrier confidence, validated on 4 real scenes** | `docs/P0_CALIBRATED_CONFIDENCE.md` |

**Claim boundary:** the external baseline gate is closed for local
artifact-backed validation. Official 2DGS and 3DGUT have now been built and run
as 30k same-split GPU validation rows on all 8 audited scenes. Every local SOTA
A/B provider now has passing artifact-backed evidence. AURA can claim local
artifact-backed A/B readiness; official leaderboard claims remain out of scope.

## Results

### Multi-Scene Typed-Carrier Quality

Across all local benchmark scenes, AURA's Beta carriers beat the fixed-Gaussian
control.

| Scene | AURA Beta PSNR | Fixed Gaussian PSNR | Delta |
|---|---:|---:|---:|
| bicycle | 25.15 | 24.84 | +0.30 |
| bonsai | 34.03 | 32.27 | +1.76 |
| counter | 30.32 | 28.81 | +1.51 |
| garden | 27.27 | 26.64 | +0.63 |
| kitchen | 32.37 | 31.29 | +1.09 |
| room | 32.78 | 32.29 | +0.49 |
| stump | 26.64 | 26.46 | +0.19 |
| truck | 26.39 | 25.96 | +0.43 |

**Mean gain: +0.80 dB PSNR.**

![All local benchmark scenes](docs/dataset_scene_grid.png)

![AURA Beta vs fixed Gaussian across 8 scenes](docs/multiscene.png)

![Per-scene PSNR gains](docs/multiscene_delta.png)

The rendered scene media in this README is limited to the local Truck and Train
assets currently present in the workspace.

### Local Train Evidence

Train is included as local image-sequence and COLMAP-sparse evidence rather than
as a trained DBS-Beta AURA checkpoint.

| Train image sweep | Train sparse depth |
|---|---|
| ![Train orbit](docs/train_orbit.gif) | ![Train sparse depth](docs/train_depth_orbit.gif) |

### Truck Compactness

| Representation | PSNR | SSIM | LPIPS | Carriers |
|---|---:|---:|---:|---:|
| fixed Gaussian | 26.02 | 0.890 | 0.128 | 1.0 M |
| AURA Beta | **26.35** | **0.896** | **0.122** | 1.0 M |
| AURA Beta | 26.07 | 0.890 | 0.139 | **0.5 M** |

Beta wins at matched carrier count and reaches comparable quality with about half
the carriers.

![Ground truth vs fixed Gaussian vs adaptive Beta](docs/beta_vs_gauss_truck.png)

![Compactness curve](docs/compactness_curve.png)

### Publication Gate Snapshot

| Gate | Result |
|---|---|
| Local multi-scene quality | pass |
| Downloaded dataset audit | pass |
| PRISM additive contract | pass |
| PRISM CUDA throughput smoke | pass |
| Real trained-scene FPS | pass |
| Engine/viewer export integration | pass |
| Viewer/export structural compatibility | pass |
| Learned LPIPS on CUDA | pass |
| External method baseline table | pass |
| Secondary-ray/reflection validation | pass |
| Inverse-material validation | pass |

Run:

```bash
aura publication-validation-report --output experiments/results/publication_validation.json
```

Latest durable report:

```text
experiments/results/publication_validation_2026-06-25.json
publicationReady: true
passedGateCount: 11
remainingGateIds: []
```

## What The Asset Can Do

### Render And Query

AURA keeps high-quality primary rendering on the mature gsplat/DBS-Beta path and
exposes a unified ray-query payload over the asset.

```bash
aura render scene.aura --backend torch --output view.ppm
aura ray-query scene.aura --origin 0 0 0 --direction 0 0 1
```

![Expected-depth orbit](docs/truck_depth_orbit.gif)

### Relight

Carriers carry surface/material fields used by the relighting layer. The same
asset can be previewed under changed lighting without changing geometry.

```bash
aura relight-preview scene.aura scene/manifest.json --output relit.ppm
```

![Relighting sweep](docs/relight_sweep.gif)

### Calibrated Confidence (P0 Killer Property)

Each carrier stores a confidence value derived from multi-view support, useful for
inspection, filtering, and downstream tools that need to distinguish well-observed
geometry from speculative structure.

```bash
aura confidence scene.aura scene/manifest.json
```

![Confidence heatmap](docs/confidence_truck.png)

Beyond the raw heuristic, AURA now ships **calibrated, certified per-carrier
confidence** — the property a bare 3DGS/DBS splat does not have. Isotonic
calibration maps the multi-view signal to a reliability an engine can trust, a
distribution-free **conformal pruning certificate** bounds the reliability lost by
dropping carriers below a threshold, and the calibrated value travels with the
asset as `_AURA_CONFIDENCE` in the `KHR_gaussian_splatting` export. Authoritative
write-up: [`docs/P0_CALIBRATED_CONFIDENCE.md`](docs/P0_CALIBRATED_CONFIDENCE.md).

```bash
aura calibrate-confidence <package> <reliability.npz>
```

Validated end-to-end on **four real scenes** — Truck (129k carriers) and three
Mip-NeRF 360 scenes: Garden (outdoor, 120k), Kitchen (indoor, 120k), Room (indoor,
107k). The export-time feature (train-view color agreement) predicts held-out
reliability; the shipped view-count heuristic and opacity do not:

| signal vs held-out reliability (corr) | Truck | Garden | Kitchen | Room |
|---|---:|---:|---:|---:|
| **train-view color agreement** (export-time feature) | **0.91** | **0.93** | **0.98** | **0.96** |
| view-count heuristic (shipped) | −0.05 | −0.13 | −0.01 | 0.05 |
| opacity (engine pruning default) | −0.18 | 0.16 | 0.08 | 0.05 |
| calibration ECE (raw → calibrated) | 0.59→0.001 | 0.55→0.002 | 0.56→0.001 | 0.46→0.002 |

The killer property is **selection AUC** (mean retained reliability across pruning
budgets): calibrated confidence lands within **1–4% of the oracle ceiling** on
every scene and beats opacity, the raw heuristic, and random at every budget
(calibrated 0.58–0.72 vs opacity 0.37–0.53, itself at or below random). At a
10%-keep budget calibrated confidence retains 0.77–0.90 reliability vs opacity's
0.31–0.49. Opacity — the usual engine pruning default — is a poor pruning signal
on all four scenes (its *correlation* is negative on Truck but near-zero on the
Mip-360 scenes; either way it is at or below random for selection).

![Four-scene selection AUC: calibrated confidence vs opacity vs oracle ceiling](assets/p0_selection_auc.png)

**Pruning sweep (Room, held-out view).** The same property, rendered. As we prune
100%→10% of carriers, the reliability of the *kept* carriers (bottom meters) is
the P0 axis: calibrated-confidence pruning (left) tracks the oracle ceiling —
retained reliability rises to **0.90 at a 10%-keep budget** — while opacity
pruning (right) stays flat near random (**~0.50**).

![Pruning sweep: calibrated-confidence vs opacity carrier pruning, Room held-out view](assets/pruning_sweep.gif)

**Honest caveat (verified, not the naive story).** The *rendered* image degrades
*faster* under confidence pruning than under opacity pruning — opacity holds a
higher render PSNR at every budget (30%-keep: **22.7 dB opacity vs 18.7 dB
confidence** below). This is structural, not a bug: opacity *is* the
alpha-compositing blend weight, so keeping the highest-opacity carriers preserves
the pixels you see almost by construction (which is why opacity pruning is the
3DGS standard). The point of P0 is the other axis — opacity keeps a good-looking
render but **unreliable** carriers and ships no guarantee, whereas calibrated
confidence keeps the carriers that agree with held-out observations and comes with
a distribution-free certificate. The two signals optimize different things.

![Pruning to 30% of carriers: full vs calibrated-confidence@30% vs opacity@30%](assets/pruning_30pct.png)

Regenerate both with `experiments/make_pruning_sweep_gif.py --scene room --frame 8`.

The property also **survives an occlusion-aware reliability label**
(`--label depth_aware`, which counts a carrier only in held-out views where it is
the visible front surface): calibrated confidence stays within 1–9% of the oracle
and still beats opacity on all four scenes. Reproduce with
`experiments/per_carrier_reliability.py` followed by
`experiments/calibrate_confidence.py`; authoritative write-up in
[`docs/P0_CALIBRATED_CONFIDENCE.md`](docs/P0_CALIBRATED_CONFIDENCE.md).

### Semantics And Open-Vocabulary Search

AURA lifts multi-view DINO features onto carriers and uses CLIP-style text
queries for group-level retrieval. The same-split SOTA A/B pass now promotes a
DINOv3 small/timm CUDA path after increasing the semantic cluster budget to 12:
truck, wheel, ground, and building resolve to four distinct groups while the
aggregate query margin remains above the DINOv2 baseline. The report still
records that DINOv2 has the stronger wheel-only margin.

![Semantic carrier segmentation](docs/semantic_distill_truck.png)

![Open-vocabulary query for a wheel](docs/semantic_query_truck.png)

| Stronger semantic A/B | DINOv2 | DINOv3 |
|---|---|---|
| 14-view carrier groups | ![DINOv2 stride-16 semantic groups](docs/semantic_distill_dinov2_stride16_ab.png) | ![DINOv3 k12 stride-16 semantic groups](docs/semantic_distill_dinov3_k12_stride16.png) |
| Wheel query highlight | ![DINOv2 stride-16 query](docs/semantic_query_dinov2_stride16_ab.png) | ![DINOv3 k12 stride-16 query](docs/semantic_query_dinov3_k12_stride16.png) |

### Export

The export path writes real engine-facing assets instead of leaving results as
an experiment-only checkpoint.

```bash
aura export-splat scene.aura --output scene.glb
aura export-usd scene.aura --output scene.usda
aura export-usd scene.aura --schema --output scene.usda   # OpenUSD 26.03 splat schema
aura validate-package scene.aura
aura inspect-package scene.aura
```

Supported export surfaces:

- `KHR_gaussian_splatting` GLB with position, color/opacity, rotation, scale, and
  SH payloads.
- USD export: a dependency-free ASCII preview bridge for scene-graph and DCC
  workflows, plus the official **OpenUSD 26.03** `UsdVolParticleField3DGaussianSplat`
  schema via `--schema` (native splat prim with a confidence vendor channel;
  requires `usd-core`).
- `.aura` package plus `carriers.npz` sidecar for fast local rendering/eval.

Latest engine integration artifact:

```text
experiments/results/engine_integration_validation_2026-06-25.json
experiments/results/viewer_compatibility_validation_2026-06-25.json
docs/engine_exports/aura_splat.glb
docs/engine_exports/aura_scene.usda
```

## PRISM

PRISM is the **Pluggable Radiance-prImitive Splatting Module**. Its job is to add
typed extension footprints that the primary quality backends do not cover.

| Carrier | Default path | Role |
|---|---|---|
| Gaussian | gsplat | primary quality rasterization |
| Beta | DBS-Beta | primary typed-carrier quality path |
| Gabor | PRISM | additive high-frequency extension |
| Neural | PRISM | additive experimental extension |

PRISM is therefore an extension layer in AURA, not an alternative quality backend
for Gaussian/Beta scenes.

![PRISM extension stack](docs/prism_extension_stack.png)

![PRISM footprint families](docs/prism_footprints.png)

Validation artifact:

```text
experiments/results/prism_additive_validation_2026-06-24.json
experiments/results/production_fps_sweep_2026-06-25.json
experiments/results/real_scene_fps_sweep_2026-06-25.json
```

It verifies that Gaussian/Beta route to the primary backend, Gabor/neural route
to PRISM, and the PRISM extension changes the rendered image. The FPS sweep keeps
that role boundary machine-readable: PRISM is additive, while gsplat/DBS-Beta
remain the primary quality backends.

## External Baselines

The local same-split baseline artifact is complete:

```text
experiments/results/external_baselines_2026-06-24.json
```

| Baseline row | PSNR | SSIM | LPIPS | Boundary |
|---|---:|---:|---:|---|
| COLMAP sparse SfM | 8.9952 | 0.049027 | 0.757455 | local CUDA smoke |
| compact NeRF | 8.6726 | 0.126395 | 0.971559 | local 1-iter CUDA smoke |
| 3DGS / gsplat-control | 26.0172 | 0.890420 | 0.127743 | executed fixed-Gaussian control |
| 2DGS-style surfel | 10.7072 | 0.177134 | 0.645361 | local smoke/protocol row |
| ray-traced-GS-style | 6.7688 | 0.066934 | 0.822136 | local smoke/protocol row |
| official 2DGS Truck | 25.1223 | 0.873086 | 0.173525 | official repo, 30k steps, Truck scene native |
| official 3DGUT Truck | 25.3198 | 0.878045 | 0.183758 | official repo, 30k steps, Truck scene native |
| official 2DGS Room | 30.5354 | 0.906617 | 0.243403 | official repo, 30k steps, Mip-NeRF 360 Room images_2 |
| official 3DGUT Room | 31.4958 | 0.918965 | 0.296945 | official repo, 30k steps, Mip-NeRF 360 Room downsample_factor=2 |
| official 2DGS Bicycle | 24.5921 | 0.711770 | 0.306886 | official repo, 30k steps, Mip-NeRF 360 Bicycle images_2 |
| official 3DGUT Bicycle | 24.3068 | 0.696055 | 0.359877 | official repo, 30k steps, Mip-NeRF 360 Bicycle downsample_factor=2 |
| official 2DGS Bonsai | 31.2977 | 0.931000 | 0.226856 | official repo, 30k steps, Mip-NeRF 360 Bonsai images_2 |
| official 3DGUT Bonsai | 32.4276 | 0.944540 | 0.251687 | official repo, 30k steps, Mip-NeRF 360 Bonsai downsample_factor=2 |
| official 2DGS Counter | 28.0533 | 0.893028 | 0.229328 | official repo, 30k steps, Mip-NeRF 360 Counter images_2 |
| official 3DGUT Counter | 29.1397 | 0.910729 | 0.257860 | official repo, 30k steps, Mip-NeRF 360 Counter downsample_factor=2 |
| official 2DGS Garden | 26.6861 | 0.833891 | 0.164357 | official repo, 30k steps, Mip-NeRF 360 Garden images_2 |
| official 3DGUT Garden | 26.3824 | 0.801139 | 0.241828 | official repo, 30k steps, Mip-NeRF 360 Garden downsample_factor=2 |
| official 2DGS Kitchen | 30.2164 | 0.915704 | 0.147227 | official repo, 30k steps, Mip-NeRF 360 Kitchen images_2 |
| official 3DGUT Kitchen | 30.8491 | 0.926038 | 0.159499 | official repo, 30k steps, Mip-NeRF 360 Kitchen downsample_factor=2 |
| official 2DGS Stump | 26.0513 | 0.749460 | 0.293722 | official repo, 30k steps, Mip-NeRF 360 Stump images_2 |
| official 3DGUT Stump | 26.3474 | 0.758430 | 0.360993 | official repo, 30k steps, Mip-NeRF 360 Stump downsample_factor=2 |

Official replacement sources are recorded in:

```text
experiments/results/external_baseline_sources_2026-06-24.json
experiments/results/official_multiscene_baselines_2026-06-25.json
```

The official multi-scene collector records completed and missing rows so the
paper package does not blur local evidence with leaderboard coverage. Current
completed counts are official 2DGS 8/8 scenes, official 3DGUT 8/8 scenes, and
local gsplat-control 3DGS 8/8 scenes.

The current SOTA A/B artifact is:

```text
experiments/results/sota_ab_validation_2026-06-25.json
sotaReady: true
promotedProviderIds: 3dgrut_3dgut_official, dinov3_small_timm, official_2dgs
remaining blocker: none for local artifact-backed A/B readiness; official leaderboard claims remain out of scope
```

The publication bundle draft and claim table are in:

```text
docs/publication_bundle_2026-06-25.md
docs/submission_readiness_2026-06-25.md
docs/paper_outline_2026-06-25.md
```

Submission verdict: AURA/PRISM is ready to package as a paper artifact and
internal preprint draft with local artifact-backed SOTA A/B readiness. It is not
an official leaderboard SOTA claim.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gpu,assets]"
```

For CUDA-first local work, use the existing GPU environments when available:

```bash
source .gpu_venv/bin/activate
```

The DBS-Beta fork installs under the `gsplat` package name and is kept isolated
in `.dbs_venv`.

## Quick Start

```bash
# 1. Build a capture manifest from COLMAP.
aura colmap-to-capture-manifest data/tanks/truck/sparse/0 \
  --root data/tanks/truck \
  --image-dir data/tanks/truck/images \
  --output outputs/truck-manifest.json \
  --point-seeded

# 2. Train or import carriers.
aura train-gsplat outputs/truck-manifest.json --output outputs/truck.aura --scale 1.0

# 3. Use the asset.
aura render outputs/truck.aura --backend torch --output docs/view.ppm
aura export-splat outputs/truck.aura --output docs/truck.glb
aura ray-query outputs/truck.aura --origin 0 0 0 --direction 0 0 1
```

## Reproduce The Evidence

Most headline artifacts are generated from scripts in `experiments/`.

```bash
bash scripts/fetch_scene.sh truck data/tanks/truck
bash experiments/dbs_truck_ablation.sh
bash experiments/dbs_compactness_sweep.sh
bash experiments/run_multiscene.sh 7000 1
python experiments/collect_multiscene.py
python experiments/audit_multiscene.py
python experiments/prism_additive_validation.py
python experiments/prism_benchmark.py --out experiments/results/production_fps_sweep_2026-06-25.json
python experiments/engine_integration_validation.py
python experiments/viewer_compatibility_validation.py
.dbs_venv/bin/python experiments/real_scene_fps_sweep.py
python experiments/secondary_reflection_validation.py
python experiments/inverse_material_validation.py
python experiments/external_baseline_smokes.py --device cuda
python experiments/render_tandt_scene_gifs.py
python experiments/make_readme_visuals.py
python experiments/make_p0_selection_auc_figure.py
```

Regenerate the publication report:

```bash
aura publication-validation-report \
  --output experiments/results/publication_validation_2026-06-25.json
```

## Gallery

| AURA | PRISM / Evidence |
|---|---|
| **Reconstruction**<br>![Truck orbit](docs/truck_orbit.gif) | **Method map**<br>![Method map](docs/how_it_works.png) |
| **Depth query**<br>![Depth orbit](docs/truck_depth_orbit.gif) | **PRISM stack**<br>![PRISM stack](docs/prism_extension_stack.png) |
| **Relighting**<br>![Relighting](docs/relight_sweep.gif) | **PRISM footprints**<br>![PRISM footprints](docs/prism_footprints.png) |
| **Confidence**<br>![Confidence](docs/confidence_truck.png) | **Benchmark scene grid**<br>![Dataset grid](docs/dataset_scene_grid.png) |
| **Semantics**<br>![Semantics](docs/semantic_distill_truck.png) | **8-scene quality**<br>![Multiscene quality](docs/multiscene.png) |
| **Open-vocabulary query**<br>![Query](docs/semantic_query_truck.png) | **Per-scene gains**<br>![Multiscene delta](docs/multiscene_delta.png) |

## Repository Map

```text
src/aura/
  carrier_io.py             fast carriers.npz sidecar
  gltf_splat.py             KHR_gaussian_splatting export
  hybrid.py                 primary backend + PRISM extension routing
  prism.py                  torch PRISM rasterizer
  prism_cuda.py             CUDA PRISM path
  publication.py            artifact-backed publication gate report
  readiness.py              stricter production-readiness boundary
  relight.py                relighting layer
  confidence.py             per-carrier confidence
  calibration.py            calibrated confidence + conformal pruning certificate (P0)
  carrier_query.py          ray-query payloads
  schemas/                  .aura package schemas

scripts/                    dataset, eval, baseline, and DBS bridge utilities
experiments/                reproduction scripts and figure/GIF generators
tests/                      contract, renderer, validation, and CLI tests
docs/                       README figures and GIFs
```

## License

MIT. See [LICENSE](LICENSE).
