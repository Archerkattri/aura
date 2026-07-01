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

## Real-scene result (Truck, 129,531 trained carriers) — 2026-07-01

Ran end-to-end on the trained Truck scene (`outputs/truck-sidecar.aura`, 219 train
/ 32 held-out views). Reliability label = per-carrier held-out colour agreement
(`experiments/per_carrier_reliability.py`); calibration + certificate + selection
on a disjoint carrier split (`experiments/calibrate_confidence.py`,
`outputs/calib_truck.json`).

**The export-time feature works, the shipped heuristic does not.** A carrier's
train-view colour agreement predicts its held-out reliability at **r = 0.94**;
AURA's current view-count heuristic only **r = 0.27** (it saturates at ~0.99 on a
densely-captured scene), and **opacity is _negatively_ correlated (r = −0.19)** —
so pruning by opacity, the usual engine default, actively removes reliable
carriers.

**Calibration (ECE, disjoint eval split):**

| confidence | ECE |
|---|---|
| raw view-count heuristic (shipped) | **0.68** |
| train-agreement feature, uncalibrated | 0.020 |
| **calibrated** | **0.0017** |

**Certified pruning:** at ε = 0.6, α = 0.1 the conformal certificate keeps 71.7%
of carriers with the retained set's mean unreliability certified ≤ ε.

**Killer property — mean retained held-out reliability across pruning budgets
(AUC; higher is better):**

| selection signal | AUC | @10%-keep |
|---|---|---|
| **calibrated confidence** | **0.528** | **0.765** |
| oracle ceiling | 0.544 | — |
| raw heuristic | 0.410 | — |
| random | 0.313 | — |
| opacity (engine default) | 0.268 | 0.167 |

Calibrated-confidence-guided pruning is within 3% of the oracle and **~2×** the
opacity default. This is the concrete B1 answer: a certified, budget-controllable,
exported reliability that lets a consumer drop carriers with a guarantee — a
capability a bare 3DGS/DBS splat does not have.

**Multi-scene: it generalises.** Repeated end-to-end on **Garden** (Mip-NeRF-360
outdoor, 120k carriers, 161 train / 24 held-out views) — a very different scene
from Truck — with the same pipeline:

| metric | Truck | Garden |
|---|---|---|
| corr(train-agreement, reliability) | 0.94 | **0.95** |
| corr(view-count heuristic, reliability) | 0.27 | 0.31 |
| corr(opacity, reliability) | −0.19 | **−0.18** |
| ECE raw heuristic → calibrated | 0.68 → 0.0017 | 0.81 → **0.0048** |
| selection AUC: calibrated | 0.528 | 0.352 |
| selection AUC: oracle ceiling | 0.544 | 0.361 |
| selection AUC: opacity (engine default) | 0.268 | **0.135** |

On both scenes calibrated confidence lands within ~2–3% of the oracle ceiling and
roughly doubles the opacity default, and opacity is consistently the *worst*
signal (below random) — pruning by opacity removes reliable carriers. The
absolute reliability level is lower on Garden (sparser 120k carriers over a large
outdoor scene → more floaters), but the ordering and the calibrated-vs-opacity gap
are identical. This removes the single-scene caveat.

**Remaining caveats:** the reliability label is held-out colour agreement, so an
interior carrier occluded across the held-out views can score low (mitigated by a
robust median + a min-observation gate). Next: an occlusion-aware depth-ordered
attribution as a second reliability signal, and more Mip-NeRF-360 scenes.

## Reproduce (accuracy job — safe on shared GPUs, see gpu-usage-policy)

```
# .gpu_venv active; produces the held-out reliability signal from the trained scene
OMP_NUM_THREADS=2 .gpu_venv/bin/python experiments/per_carrier_reliability.py \
    --aura outputs/truck-sidecar.aura --manifest outputs/truck-pts129k-manifest.json \
    --out outputs/reliability_truck.npz
# calibration + certificate + selection report (CPU)
.gpu_venv/bin/python experiments/calibrate_confidence.py \
    --reliability outputs/reliability_truck.npz --report outputs/calib_truck.json
```

## Next reliability signal (optional strengthening)

The current label is held-out colour agreement. A stronger, occlusion-aware
signal is **depth-ordered per-carrier contribution error**: render the scene,
attribute held-out L1/SSIM to each carrier by its transmittance-weighted
contribution (front-to-back), and label by that. It removes the occlusion caveat
above. Also repeat across the Mip-NeRF-360 scenes (`data/mipnerf360` present) once
their carriers are trained, to show the property is not Truck-specific.

## Why this is the successor property, not scaffolding

3DGS beat NeRF on speed+parity+simplicity. AURA cannot out-render DBS. But a
*trustworthy, budget-controllable* asset — where a downstream engine can drop
30% of carriers with a certified quality bound instead of guessing — is a
capability the representation itself provides and a bare splat cannot. That is
the defensible successor axis; this module is its foundation.
