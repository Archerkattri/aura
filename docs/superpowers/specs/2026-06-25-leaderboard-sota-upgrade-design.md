# Leaderboard SOTA Upgrade Design

Date: 2026-06-25

## Goal

Move AURA from local publication readiness to real leaderboard-SOTA readiness by
running fair, repeatable GPU ablations for each candidate method/library upgrade
and promoting only upgrades that beat the current AURA rows under the same
dataset split, metrics, and hardware protocol.

## Current State

AURA already has local publication gates, official same-split 2DGS/3DGUT rows,
and a local SOTA A/B artifact. That is not enough for a leaderboard claim because
the current artifacts are partly collectors and summaries:

- `experiments/sota_ab_validation.py` summarizes measured rows.
- `experiments/collect_official_multiscene_baselines.py` reads completed `/tmp`
  runs.
- `src/aura/gsplat_renderer.py` uses current `gsplat.rasterization()` and
  `DefaultStrategy`, but does not expose newer gsplat-main features such as HiGS,
  native CUDA MCMC perturbation, AccuTile, PPISP, or extended 3DGUT signals.
- 3DGUT is benchmark evidence, not yet a first-class AURA runtime/backend.

## Candidate Upgrades

The upgrade ladder is ordered by expected impact and verification value:

1. **Leaderboard harness:** a machine-readable A/B result schema and runner
   contract that records method, scene, split, command, environment, GPU,
   metrics, artifact paths, and promotion decision.
2. **gsplat source/main:** evaluate unreleased gsplat features against the
   installed `gsplat 1.5.3` baseline.
3. **MCMC densification:** compare MCMC-style Gaussian optimization against
   `DefaultStrategy` for quality, speed, and stability.
4. **3DGUT/3DGRUT backend:** promote secondary-ray/distorted-camera support from
   external evidence into AURA backend coverage.
5. **RadSplat-style teacher/pruning:** test a radiance-field teacher, pruning,
   and test-time filtering path for quality/FPS gains.
6. **Inference stack:** compare HiGS, StopThePop-style consistency/culling, and
   OMG-style compactness for runtime FPS, memory, size, and quality retention.
7. **In-the-wild appearance path:** evaluate Splatfacto-W-style appearance
   embeddings only on captures with photometric variation or transient objects.

## Architecture

Add a small leaderboard-readiness layer rather than hard-coding each experiment
inside README scripts:

- `src/aura/leaderboard.py` defines immutable dataclasses for scene specs,
  method specs, run metrics, environment records, A/B decisions, and reports.
- `experiments/leaderboard_ablation.py` writes a durable JSON report. It starts
  with existing measured artifacts and grows into actual method launchers.
- Tests enforce strict claim boundaries: a report can be `leaderboardReady` only
  when every required scene has measured rows and a promoted method beats the
  baseline on the declared primary metric without missing artifacts.

## Data Flow

1. A scene spec declares dataset name, scene id, split, image scale, and expected
   evaluation rows.
2. A method spec declares the backend or external repo, command, environment,
   and whether the row is a baseline or candidate.
3. A run row records PSNR, SSIM, LPIPS, FPS, model size, training seconds,
   render milliseconds, and artifact paths.
4. The report groups rows by `(scene, task)`, compares candidates to the
   baseline, and records promotion decisions.
5. Publication/readiness gates consume this report once the first real ablation
   set is complete.

## Claim Policy

Allowed:

- "AURA has run fair same-split ablations for these candidate upgrades."
- "This candidate is promoted for this scene/task under this metric."
- "AURA is leaderboard-ready for this benchmark once the report covers every
  required scene and metric."

Not allowed:

- "AURA is leaderboard SOTA" unless the report covers the full benchmark and
  beats external baselines under the same protocol.
- "A candidate is better" when it only has fixture rows, missing artifacts, or
  incompatible splits.

## Testing

Use TDD for every code slice:

- schema tests first for report serialization and promotion decisions;
- failing test for missing-scene gating;
- failing test for candidate promotion only when metrics and artifacts pass;
- targeted pytest after each slice;
- full pytest before push when code touches shared validation paths.

## First Implementation Slice

Implement the report schema and deterministic comparison logic only. Do not
launch long training jobs in the first slice. The first slice should make it
impossible for fixture or partial rows to produce a leaderboard-SOTA claim.
