"""Tests for certified LOD/streaming plans (CPU, numpy-only).

The plan sorts carriers by calibrated confidence, publishes K keep-fraction levels,
and certifies the discarded reliability mass at each level with a Bonferroni
family-wise (1 - alpha) conformal bound. Synthetic-only here (CI-runnable); the one
test that touches committed outputs/*.npz is marked ``local_data``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.calibration import conformal_mean_upper_bound  # noqa: E402
from aura.lod import (  # noqa: E402
    DEFAULT_LEVELS,
    apply_lod_plan,
    certified_lod_plan,
    evaluate_lod_plan,
)


def _synthetic(n=2000, seed=0, anticorr=False):
    """A monotone reliability-vs-confidence synthetic conformal set."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.0, 1.0, n)
    base = (1.0 - conf) if anticorr else conf
    rel = np.clip(base + rng.normal(0, 0.05, n), 0.0, 1.0)
    return conf, rel


# --------------------------------------------------------------------------- #
# Schema / plan structure
# --------------------------------------------------------------------------- #

def test_plan_schema_and_provenance():
    conf, rel = _synthetic()
    plan = certified_lod_plan(conf, rel, scene="unit")
    assert plan["scene"] == "unit"
    assert plan["correction"] == "bonferroni"
    assert plan["sort"] == "calibrated_confidence_desc"
    assert plan["alpha"] == pytest.approx(0.1)
    assert plan["alpha_per_level"] == pytest.approx(0.1 / len(DEFAULT_LEVELS))
    assert plan["family_wise_confidence"] == pytest.approx(0.9)
    assert plan["n_cal"] == conf.shape[0]
    assert [lvl["keep_fraction"] for lvl in plan["levels"]] == [0.1, 0.25, 0.5, 1.0]
    for lvl in plan["levels"]:
        for key in ("keep_fraction", "tau", "epsilon_certified", "n_cal", "trivial"):
            assert key in lvl


def test_plan_is_json_serialisable():
    import json

    conf, rel = _synthetic()
    plan = certified_lod_plan(conf, rel, scene="s")
    round_trip = json.loads(json.dumps(plan))
    assert round_trip["levels"][-1]["trivial"] is True


# --------------------------------------------------------------------------- #
# Ordering / monotonicity
# --------------------------------------------------------------------------- #

def test_tau_non_increasing_with_keep_fraction():
    # Larger keep-fraction admits lower-confidence carriers => lower boundary tau.
    conf, rel = _synthetic()
    plan = certified_lod_plan(conf, rel)
    taus = [lvl["tau"] for lvl in plan["levels"]]
    assert all(taus[i] >= taus[i + 1] - 1e-12 for i in range(len(taus) - 1))


def test_epsilon_non_increasing_with_keep_fraction():
    # Keeping more prunes fewer carriers => less discarded reliability mass.
    conf, rel = _synthetic()
    plan = certified_lod_plan(conf, rel)
    eps = [lvl["epsilon_certified"] for lvl in plan["levels"]]
    assert all(eps[i] >= eps[i + 1] - 1e-12 for i in range(len(eps) - 1))
    assert plan["levels"][-1]["epsilon_certified"] == 0.0  # 100% level


def test_full_keep_level_is_trivial():
    conf, rel = _synthetic()
    plan = certified_lod_plan(conf, rel)
    trivial = plan["levels"][-1]
    assert trivial["keep_fraction"] == 1.0
    assert trivial["trivial"] is True
    assert trivial["epsilon_certified"] == 0.0
    assert trivial["n_kept_cal"] == conf.shape[0]


# --------------------------------------------------------------------------- #
# Bonferroni accounting
# --------------------------------------------------------------------------- #

def test_bonferroni_alpha_prime_used_in_certificate():
    # Each level's epsilon must be the Hoeffding UCB at alpha' = alpha/K on the
    # per-carrier discard loss (reliability where pruned, else 0) — reconstruct it.
    conf, rel = _synthetic()
    alpha = 0.1
    plan = certified_lod_plan(conf, rel, alpha=alpha)
    K = len(plan["levels"])
    assert plan["alpha_per_level"] == pytest.approx(alpha / K)
    for lvl in plan["levels"]:
        if lvl["trivial"]:
            continue
        dropped = conf < float(lvl["tau"])
        discard_loss = np.where(dropped, np.clip(rel, 0, 1), 0.0)
        expected = conformal_mean_upper_bound(discard_loss, alpha / K)
        assert lvl["epsilon_certified"] == pytest.approx(expected)


def test_more_levels_widens_each_certificate():
    # More simultaneous levels => smaller alpha' => larger Hoeffding margin, so the
    # same keep-fraction gets a strictly more conservative (larger) epsilon.
    conf, rel = _synthetic()
    few = certified_lod_plan(conf, rel, levels=[0.5, 1.0])           # K=2
    many = certified_lod_plan(conf, rel, levels=[0.1, 0.25, 0.5, 1.0])  # K=4
    eps_few = next(l["epsilon_certified"] for l in few["levels"] if l["keep_fraction"] == 0.5)
    eps_many = next(l["epsilon_certified"] for l in many["levels"] if l["keep_fraction"] == 0.5)
    assert eps_many > eps_few


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_determinism():
    conf, rel = _synthetic(seed=7)
    a = certified_lod_plan(conf, rel, scene="x")
    b = certified_lod_plan(conf, rel, scene="x")
    assert a == b


def test_stable_tiebreak_on_confidence_ties():
    # All-equal confidence must not crash and must keep everything (conf >= tau).
    conf = np.full(50, 0.5)
    rel = np.linspace(0, 1, 50)
    plan = certified_lod_plan(conf, rel)
    for lvl in plan["levels"]:
        # tau == 0.5, so conf >= tau keeps all; discarded mass is 0 (nothing < tau).
        assert lvl["epsilon_certified"] == pytest.approx(
            conformal_mean_upper_bound(np.zeros(50), plan["alpha_per_level"])
            if not lvl["trivial"] else 0.0)


# --------------------------------------------------------------------------- #
# apply_lod_plan subset correctness
# --------------------------------------------------------------------------- #

def test_apply_lod_plan_subset_numpy():
    n = 100
    conf = np.linspace(0, 1, n)
    carriers = {
        "means": np.arange(n * 3, dtype="float32").reshape(n, 3),
        "confidence": conf.astype("float32"),
        "sh_degree": 2,  # scalar, must pass through untouched
    }
    plan = certified_lod_plan(conf, np.clip(conf, 0, 1))
    lvl = next(l for l in plan["levels"] if l["keep_fraction"] == 0.5)
    tau = float(lvl["tau"])
    subset = apply_lod_plan(carriers, plan, 0.5)
    expected = conf >= tau
    assert subset["means"].shape[0] == int(expected.sum())
    assert np.array_equal(subset["means"], carriers["means"][expected])
    assert np.array_equal(subset["confidence"], carriers["confidence"][expected])
    assert subset["sh_degree"] == 2  # scalar preserved
    assert subset["_lod_kept"] == int(expected.sum())
    assert subset["_lod_tau"] == pytest.approx(tau)


def test_apply_lod_plan_full_level_keeps_all():
    n = 40
    conf = np.linspace(0.1, 0.9, n)
    carriers = {"means": np.zeros((n, 3)), "confidence": conf}
    plan = certified_lod_plan(conf, conf)
    subset = apply_lod_plan(carriers, plan, 1.0)
    assert subset["_lod_kept"] == n


def test_apply_lod_plan_unknown_level_raises():
    conf, rel = _synthetic(n=200)
    plan = certified_lod_plan(conf, rel)
    with pytest.raises(KeyError):
        apply_lod_plan({"confidence": conf}, plan, 0.42)


def test_apply_lod_plan_torch_tensor():
    torch = pytest.importorskip("torch")
    n = 60
    conf = torch.linspace(0.0, 1.0, n)
    carriers = {"means": torch.arange(n * 3).float().reshape(n, 3), "confidence": conf}
    plan = certified_lod_plan(conf.numpy(), conf.numpy())
    lvl = next(l for l in plan["levels"] if l["keep_fraction"] == 0.25)
    subset = apply_lod_plan(carriers, plan, 0.25)
    expected = (conf >= float(lvl["tau"]))
    assert int(subset["means"].shape[0]) == int(expected.sum().item())


# --------------------------------------------------------------------------- #
# Empty / degenerate inputs
# --------------------------------------------------------------------------- #

def test_empty_conformal_set_raises():
    with pytest.raises(ValueError):
        certified_lod_plan(np.array([]), np.array([]))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        certified_lod_plan(np.zeros(5), np.zeros(4))


def test_bad_keep_fraction_raises():
    conf, rel = _synthetic(n=100)
    with pytest.raises(ValueError):
        certified_lod_plan(conf, rel, levels=[0.0, 1.0])
    with pytest.raises(ValueError):
        certified_lod_plan(conf, rel, levels=[1.5])


def test_single_carrier_does_not_crash():
    plan = certified_lod_plan(np.array([0.7]), np.array([0.9]), levels=[0.5, 1.0])
    assert plan["n_cal"] == 1
    assert plan["levels"][-1]["trivial"] is True


# --------------------------------------------------------------------------- #
# Empirical validity + honest failure
# --------------------------------------------------------------------------- #

def test_bounds_hold_on_exchangeable_eval():
    # Exchangeable cal/eval halves of one iid draw: every bound should hold.
    conf, rel = _synthetic(n=6000, seed=1)
    half = conf.shape[0] // 2
    plan = certified_lod_plan(conf[:half], rel[:half], scene="ex")
    evals = evaluate_lod_plan(plan, conf[half:], rel[half:])
    assert all(e["holds"] for e in evals)


def test_certificate_honestly_fails_under_distribution_shift():
    # Plan built where dropped (low-conf) carriers are UNreliable (small discard
    # loss). Evaluated on a set where the SAME low-conf carriers are RELIABLE
    # (anti-correlated) => discarded mass explodes and the bound must be reported
    # as violated, not silently passed.
    n = 4000
    conf = np.linspace(0.0, 1.0, n)
    cal_rel = conf.copy()            # reliability tracks confidence
    eval_rel = 1.0 - conf           # reliability anti-tracks confidence
    plan = certified_lod_plan(conf, cal_rel, scene="shift")
    evals = evaluate_lod_plan(plan, conf, eval_rel)
    non_trivial = [e for e in evals if not e["trivial"]]
    assert any(e["holds"] is False for e in non_trivial)  # honestly reported
    # The aggressive-prune level (keep 10%, drop the reliable low-conf tail) fails.
    lvl10 = next(e for e in evals if e["keep_fraction"] == 0.1)
    assert lvl10["empirical_loss_eval"] > lvl10["epsilon_certified"]
    assert lvl10["holds"] is False
    # The trivial 100% level never fails (nothing dropped).
    assert evals[-1]["holds"] is True


def test_family_wise_error_within_alpha():
    # Over many exchangeable trials the family-wise violation rate (ANY level
    # violated) must stay within the stated alpha. With finite-sample Hoeffding
    # margins it is comfortably below alpha (the bound is conservative).
    alpha = 0.1
    n = 800
    trials = 300
    fw_violations = 0
    for seed in range(trials):
        conf, rel = _synthetic(n=n, seed=1000 + seed)
        half = n // 2
        plan = certified_lod_plan(conf[:half], rel[:half], alpha=alpha)
        evals = evaluate_lod_plan(plan, conf[half:], rel[half:])
        if any(not e["holds"] for e in evals):
            fw_violations += 1
    assert fw_violations / trials <= alpha


# --------------------------------------------------------------------------- #
# Real committed reliability artifacts (excluded from CI)
# --------------------------------------------------------------------------- #

@pytest.mark.local_data
@pytest.mark.parametrize("scene", ["truck", "garden", "kitchen", "room"])
def test_certified_lod_holds_on_real_scene(scene):
    path = Path(__file__).resolve().parent.parent / "outputs" / f"reliability_{scene}.npz"
    if not path.exists():
        pytest.skip(f"missing {path}")
    from aura.calibration import IsotonicConfidenceCalibrator

    d = np.load(path)
    labeled = d["labeled"]
    feat = d["train_agree"][labeled].astype("float64")
    rel = d["reliability"][labeled].astype("float64")
    m = feat.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(m)
    cal_idx, ev_idx = perm[: m // 2], perm[m // 2:]
    calibrator = IsotonicConfidenceCalibrator().fit(feat[cal_idx], rel[cal_idx])
    plan = certified_lod_plan(calibrator.predict(feat[cal_idx]), rel[cal_idx], scene=scene)
    evals = evaluate_lod_plan(plan, calibrator.predict(feat[ev_idx]), rel[ev_idx])
    assert all(e["holds"] for e in evals), [e for e in evals if not e["holds"]]
