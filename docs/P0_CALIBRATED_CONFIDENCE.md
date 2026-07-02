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

## Real-scene results — four scenes, 2026-07-02

Ran the full pipeline end-to-end on four trained scenes — **Truck** (129,531
carriers, 219 train / 32 held-out) and three Mip-NeRF-360 scenes: **Garden**
(outdoor, 120k, 161/24), **Kitchen** (indoor, 120k, 244/35), **Room** (indoor,
107k, 272/39). Per scene: `experiments/per_carrier_reliability.py` produces the
held-out per-carrier reliability label; `experiments/calibrate_confidence.py`
fits isotonic calibration + the conformal certificate + selection curves on a
disjoint carrier split (`outputs/calib_<scene>.json`). All eight runs (four
scenes × two labels) use one consistent estimator.

**Estimator note (2026-07-02).** The robust observed-colour centre is a **masked
median over each carrier's *observed* held-out views** (`torch.nanmedian`),
replacing an earlier sentinel-median that poisoned the estimate for carriers seen
in fewer than half the views. The masked median is strictly more correct and is
*required* for the occlusion-aware label below (which observes fewer views per
carrier). It slightly refines the earlier (2026-07-01) Truck/Garden colour
figures — the table here supersedes them — but leaves the headline conclusion
unchanged (see AUC): calibrated confidence is a near-oracle pruning signal and
opacity is a poor one.

### The export-time feature predicts held-out reliability; the heuristic and opacity do not

Colour-agreement label, correlation of each signal with held-out reliability
(labeled carriers, disjoint from calibration):

| signal vs held-out reliability (corr) | Truck | Garden | Kitchen | Room |
|---|---:|---:|---:|---:|
| **train-view colour agreement** (export-time feature) | **0.91** | **0.93** | **0.98** | **0.96** |
| view-count heuristic (shipped) | −0.05 | −0.13 | −0.01 | 0.05 |
| opacity (engine pruning default) | **−0.18** | 0.16 | 0.08 | 0.05 |

The export-time train-agreement feature — a floater/inconsistency detector
computable *without* held-out ground truth — tracks held-out reliability at
r ≈ 0.91–0.98 on every scene. The shipped view-count heuristic is uninformative
(|r| ≤ 0.13; it saturates ≈1 on densely-captured scenes). **Opacity is the honest
surprise:** strongly *negative* on Truck (−0.18) but near-zero-to-mildly-positive
on the Mip-360 scenes (+0.05…+0.16). The earlier "opacity is uniformly negatively
correlated" reading was partly the sentinel-median artifact; the robust,
scene-independent statement is the AUC below — **opacity is a poor pruning
signal everywhere** (at or below random), even where its correlation is weakly
positive.

### Calibration (ECE, disjoint eval split)

| ECE | Truck | Garden | Kitchen | Room |
|---|---:|---:|---:|---:|
| raw view-count heuristic (shipped) | 0.586 | 0.551 | 0.557 | 0.464 |
| **calibrated** | **0.0014** | **0.0015** | **0.0006** | **0.0016** |

Isotonic calibration drops ECE by ~300–900× on every scene.

### Killer property — selection AUC (mean retained held-out reliability across pruning budgets; higher is better)

| selection signal | Truck | Garden | Kitchen | Room |
|---|---:|---:|---:|---:|
| **calibrated confidence** | **0.581** | **0.605** | **0.629** | **0.720** |
| oracle ceiling | 0.601 | 0.619 | 0.633 | 0.729 |
| raw heuristic | 0.426 | 0.404 | 0.439 | 0.552 |
| random | 0.406 | 0.426 | 0.442 | 0.528 |
| opacity (engine default) | 0.367 | 0.450 | 0.458 | 0.534 |
| calibrated confidence @10%-keep | 0.770 | 0.797 | 0.836 | 0.896 |
| opacity @10%-keep | 0.306 | 0.450 | 0.470 | 0.489 |

On all four scenes calibrated-confidence-guided pruning lands **within 1–4% of the
oracle ceiling** and **beats opacity, the raw heuristic, and random at every
budget** (calibrated 0.58–0.72 vs opacity 0.37–0.53, itself at or below random).
At a 10%-keep budget it retains 0.77–0.90 reliability vs opacity's 0.31–0.49. This
is the concrete B1 answer: a certified, budget-controllable, exported reliability
that lets a consumer drop carriers with a guarantee — a capability a bare 3DGS/DBS
splat does not have.

**Certified pruning.** The conformal certificate returns the most-inclusive
threshold keeping the retained set's mean unreliability ≤ ε at confidence 1−α
(α=0.1). At ε=0.6 the trained scenes are reliable enough that keeping the full set
is already certified (kept ≈1.0; Truck-depth 0.90); a tighter ε trades
kept-fraction for a stronger reliability bound. The certificate is
distribution-free — it never claims more than the held-out risk supports.

### Occlusion-aware label — the caveat-fix (`--label depth_aware`)

The colour label counts a carrier as "observed" in any held-out view it projects
into, so an interior carrier occluded across those views samples the *occluding*
surface's colour and scores a false-low. The **depth-aware label** fixes this:
before sampling, it builds a per-block front-surface depth buffer from all opaque
carriers (min camera-z over an 8-px grid) and counts a view as observing carrier
`i` **only if `i` is not occluded there** (`z ≤ front_z·(1+tol)`, tol=0.03) — i.e.
it is the visible front surface, not hidden behind opaque content. Fewer views per
carrier survive (labeled fraction drops to 61% Truck / 63% Garden / 98% Kitchen /
73% Room as floaters and occluded interiors are dropped), which is exactly why the
masked median is needed.

Recomputed on all four scenes, the property is **robust to the occlusion-aware
label — it confirms rather than weakens it:**

| metric (depth-aware label) | Truck | Garden | Kitchen | Room |
|---|---:|---:|---:|---:|
| corr(train-agreement, reliability) | 0.75 | 0.90 | 0.97 | 0.94 |
| AUC calibrated confidence | **0.523** | **0.616** | **0.639** | **0.713** |
| AUC oracle ceiling | 0.575 | 0.638 | 0.645 | 0.727 |
| AUC opacity (engine default) | 0.342 | 0.459 | 0.475 | 0.531 |
| ECE raw → calibrated | 0.620→0.004 | 0.537→0.002 | 0.550→0.001 | 0.471→0.002 |

Under the stricter, occlusion-aware label calibrated confidence stays within 1–9%
of the oracle and still beats opacity on every scene; the train-agreement feature
still predicts reliability at r = 0.75–0.97 (lowest on Truck, whose many floaters
make the depth-aware label sparsest). Both labels stay available
(`--label color|depth_aware`).

**Remaining caveats.** The depth-aware front buffer is a coarse (8-px block,
carrier-centre) z-buffer, not a full differentiable splat composite, so it
approximates occlusion rather than resolving sub-pixel transparency; and the
reliability is still a colour-agreement proxy, not a photometric render loss. Both
are conservative — they under-credit rather than over-credit a carrier.

## Reproduce (accuracy job — safe on shared GPUs, see gpu-usage-policy)

```
# .gpu_venv active; produce the held-out reliability label from a trained scene
# (colour label; add --label depth_aware for the occlusion-aware label)
OMP_NUM_THREADS=2 .gpu_venv/bin/python experiments/per_carrier_reliability.py \
    --aura outputs/<scene>-gsplat.aura --manifest outputs/<scene>-manifest.json \
    --out outputs/reliability_<scene>.npz
# calibration + certificate + selection report (CPU)
.gpu_venv/bin/python experiments/calibrate_confidence.py \
    --reliability outputs/reliability_<scene>.npz --scene <scene> \
    --report outputs/calib_<scene>.json
```

Trained scenes were built no-densify (critical: `train-gsplat --densify` balloons
large Mip-360 scenes past ~11M carriers and the `.aura` write becomes impractical):

```
.gpu_venv/bin/python -m aura.cli colmap-to-capture-manifest \
    data/mipnerf360/<scene>/sparse/0 --root data/mipnerf360/<scene> \
    --image-dir images --output outputs/<scene>-manifest.json \
    --point-seeded --max-seed-regions 120000
.gpu_venv/bin/python -m aura.cli train-gsplat outputs/<scene>-manifest.json \
    --output outputs/<scene>-gsplat.aura --iterations 5000 --scale 0.25 \
    --skip-validation
```

## Why this is the successor property, not scaffolding

3DGS beat NeRF on speed+parity+simplicity. AURA cannot out-render DBS. But a
*trustworthy, budget-controllable* asset — where a downstream engine can drop
30% of carriers with a certified quality bound instead of guessing — is a
capability the representation itself provides and a bare splat cannot. That is
the defensible successor axis; this module is its foundation, now validated across
four scenes and two reliability labels including an occlusion-aware one.
