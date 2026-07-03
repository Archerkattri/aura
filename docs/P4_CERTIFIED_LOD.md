# P4 — certified level-of-detail / streaming

P4 makes AURA's pruning certificate *do something*. P0 shipped a calibrated,
certified per-carrier confidence and a single conformal pruning certificate ("drop
below `τ`, lose at most `ε` reliability mass at confidence `1−α`"). P4 turns that
one certificate into an **ordered, multi-level streaming plan**: a consumer receives
carriers in descending calibrated-confidence order and may **stop at any published
level** with a stated, distribution-free bound on the reliability mass it has
discarded by stopping there.

This is the capability a bare 3DGS/DBS splat cannot offer: a level-of-detail /
streaming ladder where every rung carries a finite-sample guarantee, not a heuristic.

Artifacts: `src/aura/lod.py` (plan + apply + eval), `experiments/lod_certified_eval.py`
→ `outputs/lod_certified.json`, CLI `aura lod-plan`, `tests/test_lod.py`.

---

## Method

### 1. Sort
Carriers are sorted by **calibrated** confidence (the isotonic-calibrated value from
`aura.calibration`, not the raw view-count heuristic), descending, with a stable
tiebreak by carrier index. This is the streaming order.

### 2. Levels
`K` keep-fractions `f_1 < f_2 < … < f_K` (default `[0.10, 0.25, 0.50, 1.00]`). For
level `k`, the boundary threshold `τ_k` is the confidence of the `round(f_k·n)`-th
carrier in the sorted order; the retained set is `{ i : conf_i ≥ τ_k }`. `τ_k` is
non-increasing in `f_k` (keeping more admits lower-confidence carriers). It is stored
at full precision — the isotonic calibrator produces plateaus (many carriers share a
knot value) and the boundary often lands on one, so rounding `τ_k` would move the
`conf < τ_k` cut across an entire plateau and silently change the retained set.

### 3. The certified quantity — discarded reliability mass
Stopping at level `k` keeps the top-`f_k` and **drops** the rest. What you forfeit is
the reliability of the dropped carriers. Define the per-carrier discard loss over the
whole conformal set

```
ℓ_i^k = reliability_i · 1[ conf_i < τ_k ]      ∈ [0, 1]
loss_k = mean_i ℓ_i^k = (1/n) Σ_{i dropped} reliability_i
```

`loss_k` is the per-capita reliability mass thrown away by pruning to level `k`. It is
**0 at `f = 1.00`** (nothing is dropped) and grows as you prune harder; it is small
exactly when the dropped carriers are unreliable — which is the calibrated-confidence
promise. Because `ℓ^k` is a bounded `[0,1]` loss, its mean is certified by the same
finite-sample Hoeffding upper confidence bound the pruning certificate uses
(`aura.calibration.conformal_mean_upper_bound`, factored out of
`conformal_prune_certificate` so both share one audited bound):

```
ε_k = mean_cal(ℓ^k) + sqrt( log(1/α') / (2 · n_cal) )
```

computed on the scene's **calibration half only**.

### 4. Family-wise validity (Bonferroni)
`K` levels are `K` simultaneous certificates from one conformal set. Quoting a naive
per-level `1−α` would be optimistic. Each level is certified at

```
α' = α / K      (Bonferroni)
```

so by the union bound **all `K` bounds hold simultaneously with family-wise
confidence `1−α`** across the whole plan. Bonferroni *widens* each interval
(`log(K/α) > log(1/α)`), i.e. it is strictly more conservative than an uncorrected
bound — the honest direction. The 100% level is trivial (`ε = 0` by definition: no
carrier is pruned, so the discarded mass is exactly zero with no sampling
uncertainty) and is flagged `trivial: true`; it makes no probabilistic claim and only
tightens the family-wise statement.

### Plan serialization
`certified_lod_plan(confidence, reliability, *, levels, alpha, scene)` returns a
JSON-able dict:

```json
{
  "scene": "truck", "alpha": 0.1, "correction": "bonferroni",
  "alpha_per_level": 0.025, "family_wise_confidence": 0.9,
  "sort": "calibrated_confidence_desc", "n_cal": 63939, "n_levels": 4,
  "loss": "discarded_reliability_mass",
  "levels": [
    {"keep_fraction": 0.10, "tau": 0.7203…, "epsilon_certified": 0.3342…, "n_cal": 63939, "n_kept_cal": 6394, "trivial": false},
    …
    {"keep_fraction": 1.00, "tau": 0.0019…, "epsilon_certified": 0.0, "n_cal": 63939, "n_kept_cal": 63939, "trivial": true}
  ]
}
```

`apply_lod_plan(carriers, plan, level)` returns the streamed subset
`{ i : conf_i ≥ τ_level }` (every array with leading dim `N` is filtered; scalars pass
through; numpy and torch supported). `evaluate_lod_plan(plan, conf_eval, rel_eval)`
measures the empirical discarded mass at each level on a held-out set.

---

## Evaluation

`experiments/lod_certified_eval.py` builds each scene's plan from the committed
`outputs/reliability_<scene>.npz` using the **same seed-0 50/50 calibration/eval
carrier split as `experiments/calibrate_confidence.py`** (the calibrator is fit and
the plan is built on the calibration half only; the evaluation half never touches plan
construction). On the eval half it measures the empirical discarded reliability mass
at each `τ_k` and checks it against `ε_k`.

`α = 0.1` family-wise, `α' = 0.025` Bonferroni over `K = 4` levels. `τ` = boundary
confidence; `ε_cert` = certified bound (cal half); `emp_loss` = empirical discarded
mass (eval half); `holds` = `emp_loss ≤ ε_cert`.

**Truck** (n_cal = n_eval = 63 939, mean eval reliability 0.407)

| keep | τ | ε_certified | empirical_loss_eval | holds |
|---:|---:|---:|---:|:--:|
| 0.10 | 0.7203 | 0.3342 | 0.3281 | ✓ |
| 0.25 | 0.5977 | 0.2363 | 0.2320 | ✓ |
| 0.50 | 0.3939 | 0.1105 | 0.1051 | ✓ |
| 1.00 | 0.0019 | 0.0000 | 0.0000 | ✓ (trivial) |

**Garden** (n_cal = n_eval = 56 590, mean eval reliability 0.426)

| keep | τ | ε_certified | empirical_loss_eval | holds |
|---:|---:|---:|---:|:--:|
| 0.10 | 0.7427 | 0.3474 | 0.3401 | ✓ |
| 0.25 | 0.6180 | 0.2454 | 0.2377 | ✓ |
| 0.50 | 0.4193 | 0.1226 | 0.1164 | ✓ |
| 1.00 | 0.0021 | 0.0000 | 0.0000 | ✓ (trivial) |

**Kitchen** (n_cal = n_eval = 60 000, mean eval reliability 0.443)

| keep | τ | ε_certified | empirical_loss_eval | holds |
|---:|---:|---:|---:|:--:|
| 0.10 | 0.7761 | 0.3643 | 0.3598 | ✓ |
| 0.25 | 0.6340 | 0.2547 | 0.2511 | ✓ |
| 0.50 | 0.4425 | 0.1257 | 0.1216 | ✓ |
| 1.00 | 0.0018 | 0.0000 | 0.0000 | ✓ (trivial) |

**Room** (n_cal = 53 404, n_eval = 53 405, mean eval reliability 0.529)

| keep | τ | ε_certified | empirical_loss_eval | holds |
|---:|---:|---:|---:|:--:|
| 0.10 | 0.8561 | 0.4432 | 0.4390 | ✓ |
| 0.25 | 0.7420 | 0.3178 | 0.3117 | ✓ |
| 0.50 | 0.5597 | 0.1598 | 0.1514 | ✓ |
| 1.00 | 0.0010 | 0.0000 | 0.0000 | ✓ (trivial) |

**All 16 bounds hold** (12 non-trivial + 4 trivial), on every scene and level. The
Hoeffding margins are small (~0.005–0.006 at `n_cal ≈ 5–6 × 10⁴`) yet always suffice:
the calibration and evaluation halves are exchangeable draws of one scene, so the
eval discarded mass sits just below the cal-half bound at every rung. `ε_k` and the
empirical loss both fall monotonically to 0 as `keep → 1.00`.

Read the numbers correctly: keeping only 10% of carriers discards a *large* share of
the scene's total reliability mass (`ε ≈ 0.33–0.44`) even though the kept 10% are each
individually very reliable (P0: 0.77–0.90 mean retained reliability at 10%-keep). Both
are true — there are simply many moderately-reliable carriers, so a hard prune forfeits
much of the aggregate mass. The certificate bounds that forfeited mass honestly.

### An engineering finding (recorded, not tuned away)
An intermediate implementation rounded `τ_k` to 6 decimals before serializing it.
Because the isotonic calibrator plateaus and the level boundary lands on a plateau
(e.g. 1 240 Truck eval carriers share the exact boundary confidence), rounding `τ`
*up* pushed the whole plateau from kept to dropped and produced three spurious
"violations" (truck@0.25, garden@0.10, garden@0.25). The fix is to store `τ` at full
precision and round only for display. This is a genuine correctness subtlety of
plateau-valued calibrators, captured in `tests/test_lod.py`.

---

## Honest scope

- **`ε_k` bounds a *reliability-mass* proxy, not rendered PSNR.** The reliability
  label is the colour-agreement / occlusion-aware proxy from P0/P2. As those results
  establish, **opacity — not reliability — is the render-PSNR-preserving prune
  signal** (opacity is the alpha-blend weight, so keeping high-opacity carriers
  preserves the image by construction; see `docs/P0_CALIBRATED_CONFIDENCE.md` and
  `docs/P2_FULLRES_RENDERLOSS.md`). A certified-LOD stream trades on *trustworthiness*
  of the retained carriers, not on preserving the alpha-blended render. This is the
  same caveat baked into the P0 pruning-sweep figure and is not repaired here.
- **Distribution-free but exchangeability-dependent.** The guarantee is finite-sample
  and assumes the calibration and evaluation carriers are exchangeable (same scene,
  seed-0 random split). Under a distribution shift between the conformal set and the
  streamed carriers the bound can be violated — `tests/test_lod.py` includes a
  synthetic case where an anti-correlated eval set makes `holds` honestly `False`.
  Cross-scene deployment therefore needs a small local conformal set per scene
  (P1's finding: `docs/P1_CROSS_SCENE.md`).
- **Family-wise, not per-level.** The `1−α` confidence is over the whole plan of `K`
  levels (Bonferroni). Do not quote a level's `ε_k` as an independent `1−α` bound.

---

## Reproduce

```bash
# Evaluation over all four scenes -> outputs/lod_certified.json
PYTHONPATH=src .gpu_venv/bin/python experiments/lod_certified_eval.py

# CLI: print a scene's plan as JSON (seed-0 calibration half, matching the eval)
PYTHONPATH=src .gpu_venv/bin/python -m aura.cli lod-plan \
    outputs/reliability_truck.npz --scene truck

# Tests (CI-safe synthetic + real-data)
PYTHONPATH=src .gpu_venv/bin/python -m pytest tests/test_lod.py -q
PYTHONPATH=src .gpu_venv/bin/python -m pytest -m "not gpu and not local_data" -q   # CI selection
PYTHONPATH=src .gpu_venv/bin/python -m pytest tests/test_lod.py -m local_data -q   # real reliability_*.npz
```
