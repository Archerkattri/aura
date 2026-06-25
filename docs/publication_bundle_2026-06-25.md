# AURA Publication Bundle

Date: 2026-06-25

This bundle is the paper/package starting point for the current AURA evidence
state. It separates claims that are backed by durable local artifacts from claims
that remain out of scope until third-party or leaderboard-grade checks are run.
The current submission-readiness memo is `docs/submission_readiness_2026-06-25.md`;
the paper outline is `docs/paper_outline_2026-06-25.md`.

## Abstract Draft

AURA converts posed image captures into typed radiance assets that keep mature
Gaussian and DBS-Beta renderers for primary visual quality while adding native
asset behavior: typed carrier metadata, confidence, semantic grouping, ray-query
payloads, relighting, PRISM extension footprints, and standards-based export.
Across eight locally downloaded real-capture scenes, AURA's adaptive Beta
carriers outperform fixed-Gaussian controls by a mean +0.80 dB PSNR. The current
validation suite also verifies PRISM as an additive extension layer, CUDA FPS
behavior, learned perceptual metrics, secondary-ray readiness, material-field
relighting, engine-facing GLB/USD export, and same-split external-method
baseline coverage.

## Claim Table

| Claim | Status | Primary artifact |
|---|---|---|
| Beta beats fixed Gaussian on all local scenes | backed | `experiments/results/multiscene.json` |
| Local dataset audit covers every downloaded scene root | backed | `experiments/results/multiscene_audit.json` |
| PRISM is additive to gsplat/DBS-Beta, not a replacement | backed | `experiments/results/prism_additive_validation_2026-06-24.json` |
| PRISM CUDA throughput exceeds 30 FPS in synthetic sweeps | backed | `experiments/results/production_fps_sweep_2026-06-25.json` |
| Trained Truck DBS-Beta renders at interactive FPS | backed | `experiments/results/real_scene_fps_sweep_2026-06-25.json` |
| AURA exports KHR Gaussian-splat GLB and USD bridge assets | backed | `experiments/results/engine_integration_validation_2026-06-25.json` |
| Exported GLB/USD files satisfy local structural viewer contracts | backed | `experiments/results/viewer_compatibility_validation_2026-06-25.json` |
| Official 2DGS and 3DGUT same-split replacement rows exist for all 8 audited scenes | backed | `experiments/results/official_multiscene_baselines_2026-06-25.json` |
| Same-split baseline table includes COLMAP, NeRF, 3DGS, 2DGS, and ray-traced-GS rows | backed | `experiments/results/external_baselines_2026-06-24.json` |
| DINOv3/VGGT/Depth Anything 3/3DGUT/official-2DGS A/B upgrades are artifact-backed | backed | `experiments/results/sota_ab_validation_2026-06-25.json` |

## Explicit Exclusions

- No official leaderboard superiority claim over COLMAP, NeRF, 2DGS, 3DGS, or
  ray-traced-GS.
- No claim that every publication scene has production-resolution FPS evidence.
- No third-party GUI viewer render claim unless Blender, USDView, PlayCanvas,
  Babylon, or three.js validation is run and recorded.
- No claim of full inverse-material recovery from unconstrained captures.
- No claim that PRISM replaces gsplat or DBS-Beta for Gaussian/Beta primary
  quality.

## Submission Verdict

AURA/PRISM is ready to package as a paper artifact and internal preprint draft.
It should be framed as a typed radiance-asset system with local artifact-backed
SOTA A/B readiness, not as an official leaderboard SOTA claim. The strongest
current statement is that all local publication-validation gates pass and the
official 2DGS/3DGUT same-split replacement table is complete for the 8 audited
scenes.

## Method Figure Set

| Figure | File | Purpose |
|---|---|---|
| Representation map | `docs/how_it_works.png` | COLMAP vs NeRF/MERF vs 3DGS vs AURA |
| PRISM stack | `docs/prism_extension_stack.png` | Shows PRISM as additive extension layer |
| PRISM footprints | `docs/prism_footprints.png` | Gaussian/Beta/Gabor/neural footprint families |
| Dataset grid | `docs/dataset_scene_grid.png` | Local scene coverage |
| Multi-scene PSNR | `docs/multiscene.png` | Beta vs fixed-Gaussian quality table visual |
| Per-scene gain | `docs/multiscene_delta.png` | Delta PSNR by local scene |
| Truck compactness | `docs/beta_vs_gauss_truck.png` | Qualitative Beta vs Gaussian panel |
| Truck orbit | `docs/truck_orbit.gif` | Reconstruction media |
| Depth orbit | `docs/truck_depth_orbit.gif` | Query/depth media |
| Semantic query | `docs/semantic_query_truck.png` | Open-vocabulary search media |

## Reproducibility Appendix

Run these from the repo root with CUDA available:

```bash
python experiments/prism_additive_validation.py
python experiments/prism_benchmark.py --out experiments/results/production_fps_sweep_2026-06-25.json
python experiments/engine_integration_validation.py
python experiments/viewer_compatibility_validation.py
.dbs_venv/bin/python experiments/real_scene_fps_sweep.py
aura publication-validation-report --output experiments/results/publication_validation_2026-06-25.json
aura readiness-report
```

The optional official baseline expansion is intentionally separated because it
uses external repos under `/tmp/aura_sota_repos` and long 30k training runs.
The current collector artifact records official 2DGS 8/8 scenes, official 3DGUT
8/8 scenes, and local gsplat-control 3DGS 8/8 scenes.

## Recommended Paper Structure

1. Introduction: radiance fields are visually strong but weak as reusable scene
   assets.
2. Related Work: COLMAP/SfM, NeRF/MERF, 3DGS, DBS-Beta, 2DGS, 3DGUT, semantic
   feature lifting, scene interchange.
3. Method: typed carriers, PRISM extension routing, query contract, confidence,
   relighting, export.
4. Experiments: local 8-scene Beta-vs-Gaussian, compactness, external baseline
   table, FPS, export validation.
5. Limitations: official leaderboard scope, third-party viewer scope,
   inverse-material scope, PRISM role boundary.
6. Reproducibility: artifact map and command appendix.
