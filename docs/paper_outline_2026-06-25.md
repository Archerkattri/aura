# AURA Paper Outline

Date: 2026-06-25

## Working Title

AURA: Adaptive Unified Radiance Assets for Traceable Captured Scenes

## Abstract

Captured radiance fields now render real scenes at high quality, but their
outputs are still difficult to use as inspectable, editable, relightable, and
engine-facing scene assets. AURA converts posed captures into typed radiance
assets by keeping mature Gaussian and DBS-Beta renderers as the primary quality
path, then adding a unified carrier contract for query payloads, confidence,
semantic grouping, relighting, PRISM extension footprints, and standards-based
export. Across eight locally audited real-capture scenes, AURA Beta carriers
outperform fixed-Gaussian controls by a mean +0.80 dB PSNR, while Truck
compactness evidence shows comparable quality at about half the carriers. The
current artifact package validates PRISM as an additive extension layer, CUDA
throughput, learned perceptual metrics, same-split external baseline rows,
secondary-ray readiness, material-field relighting, and GLB/USD export. The paper
claims local artifact-backed readiness and same-split replacement evidence, not
official leaderboard superiority.

## Contributions

1. A typed radiance-asset contract that exposes rendering, ray-query, confidence,
   semantics, relighting, and export from one package.
2. A production role split where gsplat/DBS-Beta remain the Gaussian/Beta quality
   path and PRISM adds Gabor/neural extension footprints.
3. Local 8-scene evidence that Beta carriers beat fixed-Gaussian controls, plus
   Truck compactness evidence at reduced carrier count.
4. A submission artifact matrix covering FPS, LPIPS, external same-split rows,
   SOTA A/B upgrades, export validation, and explicit claim boundaries.

## Sections

### 1. Introduction

- Radiance-field capture has moved from NeRF to real-time Gaussian splatting.
- The missing layer is asset behavior: queries, confidence, semantics,
  relighting, interoperability, and explicit boundaries.
- State the AURA thesis as an asset system, not a new single kernel.

### 2. Related Work

- COLMAP/SfM as pose and sparse-geometry infrastructure.
- NeRF/MERF and continuous neural radiance fields.
- 3DGS and derived Gaussian-splatting quality systems.
- 2DGS and geometry-oriented splatting.
- 3DGUT / ray-traced Gaussian work for nonlinear cameras and secondary rays.
- Dense feature lifting with DINOv2/DINOv3 and open-vocabulary querying.
- Scene interchange through glTF Gaussian splatting and USD-style bridges.

### 3. Method

- AURA package schema and carrier sidecar.
- Typed carriers: Gaussian, Beta, Gabor, neural.
- Primary backend routing: Gaussian to gsplat, Beta to DBS-Beta.
- PRISM extension routing: Gabor/neural as additive layers.
- Ray-query payloads and confidence.
- Semantic group lifting and text query.
- Material/relighting fields.
- Export surfaces: `.aura`, `KHR_gaussian_splatting` GLB, USD bridge.

### 4. Experiments

- Dataset audit: Truck plus seven extracted Mip-NeRF 360 scene roots.
- Multi-scene Beta-vs-fixed-Gaussian quality.
- Truck compactness sweep.
- External rows: local COLMAP/NeRF/3DGS/2DGS-style/ray-traced-GS-style plus
  official 2DGS and 3DGUT same-split rows.
- PRISM additive routing and CUDA FPS.
- Real-scene FPS sweeps.
- Learned LPIPS and visual examples.
- Semantics: DINOv2 vs DINOv3 A/B evidence.
- Export and viewer-compatibility structural checks.

### 5. Limitations

- No official leaderboard superiority claim.
- Official rows are same-split replacement evidence, not a leaderboard entry.
- PRISM is an additive extension layer, not the primary quality renderer for
  Gaussian/Beta scenes.
- Current material and reflection artifacts are readiness/contract validations,
  not full photoreal inverse rendering.
- Third-party GUI viewer validation remains structural until installed viewer
  checks are recorded.

### 6. Reproducibility

- Point to `docs/submission_readiness_2026-06-25.md`.
- Include result JSON checksums or archived copies in the artifact bundle.
- Include README media and figure-generation commands.
- Include exact CUDA environment notes for `.gpu_venv` and `.dbs_venv`.

## Reviewer-Safe Claim Text

Use:

> AURA is locally publication-ready as an artifact-backed typed radiance-asset
> system.

Avoid:

> AURA is the official state of the art on 3D reconstruction or novel-view
> synthesis benchmarks.

Use:

> PRISM extends the gsplat/DBS-Beta quality path with additive non-Gaussian
> footprints.

Avoid:

> PRISM replaces gsplat or DBS-Beta.
