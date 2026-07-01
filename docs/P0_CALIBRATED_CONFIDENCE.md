# P0 killer property — calibrated, certified per-carrier confidence

**Decision (2026-07-01):** AURA's P0 differentiator vs. 3DGS is **calibrated,
certified, exported per-carrier confidence**, chosen over inverse-rendering
relighting because (a) it builds on AURA's existing confidence + KHR-export
strengths, (b) its calibration + certificate core is CPU-completable and
validated now, and (c) "confidence-annotated exported splats with a reliability
guarantee" is a property no other splat representation ships — a concrete answer
to audit blocker **B1** (no demonstrated killer property) and major **M3**
(confidence was an uncalibrated heuristic).

## The claim

A plain 3DGS/DBS checkpoint has no notion of per-primitive trust. AURA exports,
per carrier, a **calibrated** confidence `c ∈ [0,1]` — meaning carriers reported
at confidence `p` are reliable ≈`p` of the time — plus a **distribution-free
pruning certificate**: "drop everything below threshold `τ`, losing at most `ε`
reliability mass, with confidence `1-α`." This makes level-of-detail, streaming,
and pruning decisions *certified* rather than heuristic, and it travels with the
asset via the `_AURA_CONFIDENCE` KHR vendor attribute.

## What is implemented and CPU-validated (`src/aura/calibration.py`)

1. **Isotonic calibration** (`IsotonicConfidenceCalibrator`, PAVA). Monotone map
   from the raw multi-view heuristic (`aura.confidence`, `1-exp(-views/12)`) to a
   calibrated reliability. Preserves ordering; drops ECE sharply (test:
   miscalibrated raw → calibrated ECE < 0.05).
2. **Conformal pruning certificate** (`conformal_prune_certificate`). Split-
   conformal risk control with a distribution-free Hoeffding UCB on the retained
   set's mean unreliability; returns the most inclusive threshold `τ` that keeps
   mean unreliability ≤ `ε` at confidence `1-α`.
3. **Downstream demonstration** (`selection_quality_curve`). Confidence-guided
   carrier selection retains more true reliability at every budget than random/
   opacity-guided selection (test: AUC_conf > AUC_rand).
4. **Export wiring** (`attach_calibrated_confidence`). Replaces the heuristic
   `confidence` with the calibrated value and sets `confidence_calibrated=True`,
   so `aura.gltf_splat` ships the trustworthy number.

Tests: `tests/test_calibration.py` (10 tests, CPU/numpy, all pass).

## The one GPU step (deferred — 5090s busy with gaussianfeels on 2026-07-01)

Calibration needs a **held-out per-carrier reliability label** `y_i`. The honest
signal is per-carrier held-out rendered error. Produce it on the training box:

- For a trained scene, hold out a view split. For each carrier, estimate its
  contribution error via **leave-one-carrier-out** (or the cheaper
  **gradient/alpha-weighted attribution** of held-out L1/SSIM to each carrier).
- Label `y_i = 1` (or continuous `1 - normalized_error_i`) when the held-out
  residual attributable to carrier `i` is below tolerance.
- Fit the calibrator on a calibration view-split, evaluate ECE + the pruning
  certificate on a disjoint test split, and run `selection_quality_curve`
  against opacity- and random-guided pruning to confirm the killer property on
  real Mip-NeRF-360 / Truck scenes.

Suggested command scaffold (to add under `experiments/`):

```
# on the GPU box, .gpu_venv active, GPUs verified idle:
python experiments/per_carrier_reliability.py --scene truck --holdout 0.2 \
    --attribution alpha_weighted --out reliability_truck.npz
python experiments/calibrate_confidence.py --reliability reliability_truck.npz \
    --report calib_truck.json   # ECE, certificate, selection AUCs
```

## Why this is the successor property, not scaffolding

3DGS beat NeRF on speed+parity+simplicity. AURA cannot out-render DBS. But a
*trustworthy, budget-controllable* asset — where a downstream engine can drop
30% of carriers with a certified quality bound instead of guessing — is a
capability the representation itself provides and a bare splat cannot. That is
the defensible successor axis; this module is its foundation.
