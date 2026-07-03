"""Tests for the P2 render-loss reliability rule (CPU / numpy, self-contained).

The GPU part of ``experiments/render_loss_reliability.py`` (the exact blend-weight
attribution via colour backward passes) needs CUDA, but the reliability DEFINITION
— squash the blend-weighted RMS colour distance ``sqrt(SE_i/W_i)`` exactly like the
colour label, and leave occluded/invisible carriers (blend weight below a floor)
UNLABELLED — is pure numpy and is what the head-to-head conclusions rest on. These
tests pin that rule on synthetic ``(SE, W)`` accumulators with known answers.
"""
import numpy as np

from experiments.render_loss_reliability import weighted_error_reliability


def test_zero_error_is_full_reliability():
    # SE = 0 => dist 0 => exp(0) = 1 for visible carriers.
    se = np.zeros(4)
    w = np.array([1.0, 2.0, 0.5, 10.0])
    rel, labeled = weighted_error_reliability(se, w, beta=4.0, weight_floor=1e-4)
    assert np.allclose(rel, 1.0)
    assert labeled.all()


def test_monotone_decreasing_in_error():
    w = np.ones(5)
    se = np.array([0.0, 0.01, 0.04, 0.09, 0.25])  # dist = sqrt(se) = 0,.1,.2,.3,.5
    rel, _ = weighted_error_reliability(se, w, beta=4.0)
    assert np.all(np.diff(rel) < 0)                 # more error -> less reliable
    assert np.isclose(rel[1], np.exp(-4.0 * 0.1), atol=1e-6)


def test_invisible_carriers_are_unlabelled_not_mislabelled():
    # blend weight below floor => unlabelled, reliability forced to 0 (not exp(0)=1).
    se = np.array([0.0, 0.0])
    w = np.array([1e-9, 1.0])
    rel, labeled = weighted_error_reliability(se, w, beta=4.0, weight_floor=1e-4)
    assert not labeled[0] and labeled[1]
    assert rel[0] == 0.0 and np.isclose(rel[1], 1.0)


def test_reliability_in_unit_interval():
    rng = np.random.default_rng(0)
    w = rng.uniform(0.0, 5.0, 200)
    se = rng.uniform(0.0, 2.0, 200)
    rel, _ = weighted_error_reliability(se, w, beta=4.0)
    assert rel.min() >= 0.0 and rel.max() <= 1.0


def test_dist_uses_weighted_rms_scale():
    # dist_i = sqrt(SE_i / W_i): the same SE at higher weight is a smaller RMS.
    se = np.array([1.0, 1.0])
    w = np.array([1.0, 4.0])
    rel, _ = weighted_error_reliability(se, w, beta=1.0)
    assert np.isclose(rel[0], np.exp(-1.0 * 1.0))       # dist 1
    assert np.isclose(rel[1], np.exp(-1.0 * 0.5))       # dist 0.5
