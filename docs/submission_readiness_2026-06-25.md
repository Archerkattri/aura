# AURA Submission Readiness

Date: 2026-06-25

This note is the submission-facing status for AURA/PRISM. It uses the local
publication artifacts as evidence and keeps leaderboard claims separate from
artifact-backed readiness.

## Verdict

AURA/PRISM is ready to package as a paper artifact and internal preprint draft.
It is not ready to claim official benchmark SOTA superiority.

The honest headline is:

> AURA is a typed radiance-asset system that keeps gsplat/DBS-Beta as the primary
> quality path, uses PRISM as an additive non-Gaussian extension layer, and
> validates the resulting asset behavior across local quality, query, relighting,
> semantics, export, FPS, and same-split external-baseline artifacts.

The honest SOTA phrasing is:

> AURA has local artifact-backed SOTA A/B readiness for its selected upgrades
> and complete same-split official 2DGS/3DGUT replacement rows for the audited
> scenes. It does not claim official leaderboard superiority.

## Evidence Matrix

| Area | Current evidence | Submission status |
|---|---|---|
| Local publication gates | `experiments/results/publication_validation_2026-06-25.json` reports `publicationReady: true`, `passedGateCount: 11`, `remainingGateIds: []` | ready |
| Dataset coverage | `experiments/results/multiscene_audit.json` covers the 8 local scene roots | ready for local claim |
| Typed quality | `experiments/results/multiscene.json` shows Beta beating fixed Gaussian on all 8 local scenes, mean +0.80 dB PSNR | ready for local claim |
| Compactness | Truck compactness sweep shows 500k Beta above 1M fixed Gaussian PSNR | ready as single-scene evidence |
| PRISM role | `experiments/results/prism_additive_validation_2026-06-24.json` verifies Gaussian/Beta stay primary and Gabor/neural route to PRISM | ready |
| PRISM throughput | `experiments/results/production_fps_sweep_2026-06-25.json` and `experiments/results/real_scene_fps_sweep_2026-06-25.json` | ready as measured local FPS |
| Export | `experiments/results/engine_integration_validation_2026-06-25.json` and `experiments/results/viewer_compatibility_validation_2026-06-25.json` | ready as structural validation |
| External rows | `experiments/results/official_multiscene_baselines_2026-06-25.json` has official 2DGS and 3DGUT rows for all 8 audited scenes | ready as same-split replacement evidence |
| SOTA upgrades | `experiments/results/sota_ab_validation_2026-06-25.json` reports `sotaReady: true` for local A/B readiness | ready with boundary |

## Claim Boundary

AURA can claim:

- Beta carriers beat fixed-Gaussian controls on every audited local scene.
- The downloaded local dataset audit is complete for the 8 scene roots currently
  present.
- PRISM is complete for its intended additive role over gsplat/DBS-Beta.
- AURA has measured CUDA/FPS, LPIPS, secondary-ray readiness, inverse-material
  validation, and export-compatibility artifacts.
- AURA has same-split official 2DGS and 3DGUT rows for all 8 audited scenes.
- DINOv3, VGGT, Depth Anything 3, official 2DGS, and 3DGUT are represented in the
  local SOTA A/B evidence package.

AURA cannot claim yet:

- official external leaderboard superiority over COLMAP, NeRF, 2DGS, 3DGS, or
  ray-traced Gaussian baselines;
- production-resolution FPS across every publication scene;
- third-party GUI viewer rendering without installed viewer/checker artifacts;
- photorealistic reflected-image benchmark quality;
- full inverse-material recovery from unconstrained captures.

## Current External Context

The current surrounding field supports the claim boundary:

- 3DGS remains the mature real-time radiance-field baseline:
  https://github.com/graphdeco-inria/gaussian-splatting
- 2DGS is a relevant geometry-accurate Gaussian-splatting baseline:
  https://dl.acm.org/doi/10.1145/3641519.3657428
- 3DGUT is a relevant Gaussian-splatting/ray-query baseline for nonlinear
  cameras and secondary rays:
  https://research.nvidia.com/labs/toronto-ai/3DGUT/
- DINOv3 is current dense-feature evidence for the semantic-provider upgrade:
  https://ai.meta.com/research/publications/dinov3/
- NerfBaselines documents why unified same-split evaluation matters:
  https://github.com/nerfbaselines/nerfbaselines

## Paper Package

Use the following submission shape:

1. **Problem:** captured radiance fields are visually strong but weak as reusable
   scene assets.
2. **Method:** typed carriers, primary quality backends, PRISM additive extension
   routing, query/confidence/semantic/material/export contracts.
3. **Evidence:** local 8-scene typed-quality win, compactness, same-split
   external rows, FPS, learned LPIPS, export validation, SOTA A/B artifact.
4. **Boundary:** no official leaderboard-superiority claim; PRISM is not a
   Gaussian/Beta replacement; material/reflection claims are readiness claims,
   not full photoreal inverse-rendering claims.
5. **Artifact:** include the result JSONs, README media, reproduction commands,
   and this submission-readiness note.

## Remaining Non-Blocking Work

These items should be treated as post-submission or reviewer-hardening work, not
as open local publication gates:

- run third-party GUI viewer checks in Blender/USDView/three.js or equivalent;
- expand production-resolution FPS sweeps across every publication scene;
- run full learned-LPIPS tables for every official replacement row;
- add official leaderboard submissions if the paper wants a superiority claim;
- deepen reflection/inverse-material benchmarks beyond readiness artifacts.
