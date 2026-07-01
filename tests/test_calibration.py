"""Tests for calibrated, certified per-carrier confidence (CPU, numpy-only)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.calibration import (  # noqa: E402
    IsotonicConfidenceCalibrator,
    attach_calibrated_confidence,
    conformal_prune_certificate,
    expected_calibration_error,
    pava_isotonic,
    selection_quality_curve,
)


def test_pava_is_monotone_and_optimal():
    # Non-decreasing input is returned unchanged.
    y = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(pava_isotonic(y), y)
    # A single violator gets pooled to the block mean.
    fit = pava_isotonic(np.array([0.0, 1.0, 0.0, 1.0]))
    assert np.all(np.diff(fit) >= -1e-12)  # monotone non-decreasing
    # Pool of [1,0] -> 0.5 each, so middle two become 0.5.
    assert fit[1] == pytest.approx(0.5) and fit[2] == pytest.approx(0.5)


def test_pava_empty():
    assert pava_isotonic(np.array([])).shape[0] == 0


def test_calibration_reduces_ece_on_miscalibrated_raw():
    rng = np.random.default_rng(0)
    n = 4000
    # True reliability is a monotone-but-nonlinear function of a latent signal;
    # the raw heuristic (a squashed count) is order-correct but miscalibrated.
    latent = rng.uniform(0, 1, n)
    true_p = latent**2                      # true reliability probability
    y = (rng.uniform(0, 1, n) < true_p).astype(float)
    raw = 1.0 - np.exp(-4.0 * latent)       # heuristic squash (miscalibrated)
    cal = IsotonicConfidenceCalibrator().fit(raw, y)
    ece_raw = expected_calibration_error(raw, y)
    ece_cal = cal.ece(raw, y)
    assert ece_cal < ece_raw
    assert ece_cal < 0.05                    # calibrated to within ~5%


def test_calibrator_preserves_order():
    raw = np.array([0.1, 0.3, 0.3, 0.7, 0.9])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    cal = IsotonicConfidenceCalibrator().fit(raw, y)
    pred = cal.predict(np.array([0.0, 0.2, 0.5, 0.8, 1.0]))
    assert np.all(np.diff(pred) >= -1e-12)   # monotone


def test_calibrator_requires_fit():
    with pytest.raises(RuntimeError):
        IsotonicConfidenceCalibrator().predict(np.array([0.5]))


def test_conformal_certificate_controls_risk():
    rng = np.random.default_rng(1)
    n = 5000
    conf = rng.uniform(0, 1, n)
    # Reliability increases with confidence: unreliability ~ 1 - conf.
    reliability = (rng.uniform(0, 1, n) < conf).astype(float)
    cert = conformal_prune_certificate(conf, reliability, epsilon=0.15, alpha=0.1)
    assert cert.certified
    # The retained set's empirical unreliability must respect the budget.
    assert cert.empirical_risk <= cert.epsilon
    assert 0.0 < cert.kept_fraction <= 1.0
    # Certified tau should keep the high-confidence carriers.
    assert cert.tau >= 0.5


def test_conformal_certificate_uncertifiable_budget():
    # No threshold can get mean unreliability near zero if every carrier is bad.
    rng = np.random.default_rng(2)
    conf = rng.uniform(0, 1, 500)
    reliability = np.zeros(500)  # everything unreliable
    cert = conformal_prune_certificate(conf, reliability, epsilon=0.01, alpha=0.1)
    assert not cert.certified


def test_selection_curve_confidence_beats_random():
    rng = np.random.default_rng(3)
    n = 3000
    conf = rng.uniform(0, 1, n)
    reliability = (rng.uniform(0, 1, n) < conf).astype(float)
    random_score = rng.uniform(0, 1, n)
    _, _, auc_conf = selection_quality_curve(conf, reliability)
    _, _, auc_rand = selection_quality_curve(random_score, reliability)
    # Confidence-guided selection retains more true reliability at fixed budgets.
    assert auc_conf > auc_rand


def test_attach_calibrated_confidence_marks_and_replaces():
    raw = np.array([0.2, 0.5, 0.9], dtype="float32")
    carriers = {"means": np.zeros((3, 3)), "confidence": raw}
    cal = IsotonicConfidenceCalibrator().fit(
        np.array([0.1, 0.5, 0.9]), np.array([0.0, 0.5, 1.0])
    )
    out = attach_calibrated_confidence(carriers, cal)
    assert out["confidence_calibrated"] is True
    assert out["confidence"].shape == (3,)
    # Original dict untouched (shallow copy semantics).
    assert "confidence_calibrated" not in carriers


def test_attach_requires_raw():
    with pytest.raises(KeyError):
        attach_calibrated_confidence({"means": np.zeros((2, 3))},
                                     IsotonicConfidenceCalibrator())
