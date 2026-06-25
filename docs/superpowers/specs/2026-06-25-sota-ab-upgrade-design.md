# AURA SOTA A/B Upgrade Design

## Goal

Make AURA/PRISM SOTA-ready without weakening evidence quality: every proposed
library or method upgrade must be measured against the current baseline, and the
default path should only change when the upgrade wins or when it is required as
official replacement evidence for a stronger publication claim.

## Scope

This design covers optional adapters and validation gates for:

- semantic features: DINOv3 preferred, current semantic feature path as fallback;
- geometry priors: VGGT and Depth Anything 3 as optional camera/depth/pointmap
  providers, COLMAP/current capture tensors as fallback;
- ray tracing and secondary rays: 3DGRUT/3DGRT/3DGUT as optional official backend,
  current AURA CUDA/torch ray query as fallback;
- external baselines: official 2DGS and 3DGRUT rows as replacements for local
  smoke/protocol rows;
- reporting: a SOTA readiness report distinct from the current local publication
  readiness report.

The core package must remain installable with its existing light dependency set.
Large, gated, or conflict-prone projects are optional providers discovered at
runtime or invoked through explicit external-runner manifests.

## Architecture

Add a small provider registry under `aura.sota` with pure-Python contracts and no
hard dependency on DINOv3, VGGT, Depth Anything 3, 2DGS, or 3DGRUT. Providers
report availability, version, device, license/auth status, and whether their
results are real execution, imported official artifacts, or local fallback smoke.

Add an A/B evaluator that compares candidate artifacts to baseline artifacts by
task-specific metrics:

- semantics: feature coverage, feature dimension, query consistency, GPU use,
  and provider status;
- geometry priors: valid camera/depth outputs, pose/depth coverage, GPU use, and
  downstream render/eval readiness;
- ray/baseline backends: PSNR/SSIM/LPIPS/FPS where available, plus ray-query or
  secondary-ray evidence;
- official baseline replacements: same-split/full-split flag, official-code flag,
  and whether the smoke row can be replaced for publication claims.

The evaluator must produce a machine-readable JSON report and a readiness pillar
that says exactly which upgrades were tested, which won, which stayed fallback,
and which remain blocked by missing install/auth/data.

## Promotion Rules

An upgrade becomes the default only if one of these is true:

1. It beats the current default on the primary metric and does not regress any
   hard constraint such as GPU execution, schema compatibility, or claim boundary.
2. It provides official replacement evidence for a publication claim that local
   smoke evidence cannot support, even if direct quality is similar.
3. It is a compatibility upgrade required by an already-selected provider.

If an upgrade is newer but not measurably better, it remains available as an
optional provider and the report records why it was not promoted.

## Error Handling

Missing optional packages, missing gated weights, missing CUDA, or missing official
run artifacts must not crash normal AURA usage. They must produce explicit blocked
provider statuses in the SOTA report. SOTA readiness fails only when a claim needs
that provider and the provider is unavailable or loses its A/B comparison.

## Testing

Use TDD for each unit:

- provider contracts and registry;
- deterministic fixture providers for baseline/candidate comparisons;
- report generation and promotion logic;
- readiness integration;
- CLI report command.

GPU and external official repos are validated through artifact contracts so tests
can run without downloading gated weights. Real GPU runs are separate validation
commands that update artifacts when available.

## Claim Boundary

The current local publication package remains valid. The new SOTA-ready claim only
passes when official or current-SOTA providers are actually executed or imported
with traceable artifacts. DINOv3, VGGT, Depth Anything 3, 2DGS, and 3DGRUT must be
named as optional SOTA providers unless their artifacts are present and passing.
