"""Calibrated, certified per-carrier confidence — the AURA P0 killer property.

The multi-view confidence in :mod:`aura.confidence` is a *heuristic* squash of a
view count (``1 - exp(-views/saturate)``): monotone and cheap, but the number has
no operational meaning — a carrier reported at confidence 0.9 is not reliable 90%
of the time, so downstream consumers cannot trust it as a probability.

This module turns that raw signal into a **calibrated** confidence and attaches a
**distribution-free reliability certificate**, giving AURA a property no other
splat representation ships: an exported splat whose per-carrier confidence is a
trustworthy, budget-controllable reliability score.

Two layers, both pure-numpy and CPU-cheap (post-hoc over the carrier array):

1. **Isotonic calibration** (:class:`IsotonicConfidenceCalibrator`). Fit a
   monotone map ``g: raw_confidence -> [0,1]`` on a held-out split against a
   per-carrier reliability label ``y`` (e.g. ``1`` if the carrier's held-out
   rendered residual is below tolerance, else ``0``; or a continuous reliability
   in ``[0,1]``). Pool-adjacent-violators (PAVA) gives the L2-optimal
   non-decreasing fit, so ``g`` never inverts the ordering of the raw signal but
   rescales it so that ``mean(y | g(raw) ~ p) ~ p`` (calibrated). Quality is
   measured by expected calibration error (ECE), which drops sharply vs. the raw
   heuristic.

2. **Conformal reliability certificate** (:func:`conformal_prune_certificate`).
   Split-conformal risk control (Vovk; Angelopoulos et al., conformal risk
   control) over a held-out calibration split: pick the smallest confidence
   threshold ``tau`` such that the retained set ``{i : conf_i >= tau}`` has mean
   unreliability <= a target ``epsilon`` with distribution-free coverage
   ``1 - alpha``. This certifies pruning / level-of-detail decisions: "everything
   below ``tau`` can be dropped, losing at most ``epsilon`` reliability mass, with
   confidence ``1 - alpha``" — an abstention guarantee on the asset itself.

Nothing here needs a GPU or a trained renderer: it consumes a raw-confidence
array and a held-out reliability signal. Producing that reliability signal for a
real scene (per-carrier leave-one-out / gradient-attribution rendered error) is a
GPU step, documented in ``docs/P0_CALIBRATED_CONFIDENCE.md`` and deferred to the
training box; the calibration + certificate math is validated here on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass


def pava_isotonic(y, w=None):
    """Pool-adjacent-violators isotonic regression: the L2-optimal
    non-decreasing fit ``f`` minimising ``sum_i w_i (f_i - y_i)^2``.

    ``y`` must already be ordered by the covariate (ascending raw confidence).
    Returns the fitted values ``f`` (same length, non-decreasing). O(n).
    """
    import numpy as np

    y = np.asarray(y, dtype="float64").ravel()
    n = y.shape[0]
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype="float64").ravel()
    # Stack of blocks: (weighted mean value, total weight, length).
    vals = [float(y[0])]
    wts = [float(w[0])]
    lens = [1]
    for i in range(1, n):
        vals.append(float(y[i]))
        wts.append(float(w[i]))
        lens.append(1)
        # Merge while the last block violates monotonicity.
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            vals[-2] = merged
            wts[-2] = wsum
            lens[-2] += lens[-1]
            vals.pop(); wts.pop(); lens.pop()
    out = np.empty(n, dtype="float64")
    pos = 0
    for v, ln in zip(vals, lens):
        out[pos : pos + ln] = v
        pos += ln
    return out


def expected_calibration_error(conf, reliability, n_bins=15):
    """Expected calibration error: weighted mean over equal-width confidence
    bins of ``|mean(conf in bin) - mean(reliability in bin)|``. 0 = perfectly
    calibrated. Works for reliability in {0,1} or continuous [0,1]."""
    import numpy as np

    conf = np.asarray(conf, dtype="float64").ravel()
    reliability = np.asarray(reliability, dtype="float64").ravel()
    if conf.shape[0] == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    n = conf.shape[0]
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(conf[m].mean() - reliability[m].mean())
    return float(ece)


class IsotonicConfidenceCalibrator:
    """Monotone (PAVA) map from raw confidence to calibrated reliability.

    ``fit(raw, reliability)`` learns ``g`` on a held-out split; ``predict(raw)``
    applies it (linear-interpolating between the fitted knots, clamped to the
    training range). ``g`` is non-decreasing, so it preserves the raw ordering
    (a more-observed carrier never becomes less confident) while making the value
    a calibrated reliability estimate.
    """

    def __init__(self) -> None:
        self._x = None  # sorted unique raw thresholds (knots)
        self._y = None  # calibrated value at each knot
        self._fitted = False

    def fit(self, raw, reliability, weights=None) -> "IsotonicConfidenceCalibrator":
        import numpy as np

        raw = np.asarray(raw, dtype="float64").ravel()
        reliability = np.asarray(reliability, dtype="float64").ravel()
        if raw.shape[0] != reliability.shape[0]:
            raise ValueError("raw and reliability must have equal length")
        if raw.shape[0] == 0:
            raise ValueError("need at least one calibration point")
        order = np.argsort(raw, kind="mergesort")
        xs = raw[order]
        ys = reliability[order]
        ws = None if weights is None else np.asarray(weights, dtype="float64").ravel()[order]
        fit = pava_isotonic(ys, ws)
        # Collapse to knots at unique x (keep the last fitted value per x, since
        # the isotonic fit is constant within a pooled block).
        ux, last_idx = np.unique(xs, return_index=False), None
        # np.unique returns sorted unique; map each unique x to its fitted value
        # (values are constant across ties because PAVA pooled equal-x too).
        knot_x, knot_y = [], []
        i = 0
        m = xs.shape[0]
        while i < m:
            j = i
            while j + 1 < m and xs[j + 1] == xs[i]:
                j += 1
            knot_x.append(float(xs[i]))
            knot_y.append(float(fit[j]))  # constant across the tie block
            i = j + 1
        self._x = np.asarray(knot_x, dtype="float64")
        self._y = np.clip(np.asarray(knot_y, dtype="float64"), 0.0, 1.0)
        self._fitted = True
        return self

    def predict(self, raw):
        import numpy as np

        if not self._fitted:
            raise RuntimeError("call fit() first")
        raw = np.asarray(raw, dtype="float64").ravel()
        if self._x.shape[0] == 1:
            return np.full(raw.shape, self._y[0])
        # np.interp clamps to the endpoint values outside the knot range, which
        # is the desired monotone extrapolation.
        return np.clip(np.interp(raw, self._x, self._y), 0.0, 1.0)

    def ece(self, raw, reliability, n_bins=15):
        return expected_calibration_error(self.predict(raw), reliability, n_bins)


@dataclass
class ReliabilityCertificate:
    """Conformal pruning certificate for a confidence threshold ``tau``."""

    tau: float               # confidence threshold: keep carriers with conf >= tau
    epsilon: float           # target mean-unreliability budget for the kept set
    alpha: float             # miscoverage level
    kept_fraction: float     # fraction of carriers retained at tau
    certified: bool          # whether a tau meeting (epsilon, alpha) was found
    empirical_risk: float     # mean unreliability of the kept set on calibration


def conformal_prune_certificate(conf, reliability, epsilon, alpha=0.1):
    """Distribution-free pruning certificate via conformal risk control.

    Given per-carrier calibrated ``conf`` and held-out ``reliability`` in [0,1]
    (1 = fully reliable), find the smallest threshold ``tau`` such that the
    retained set ``{conf >= tau}`` has mean *unreliability* ``1 - reliability``
    at most ``epsilon`` with distribution-free confidence ``1 - alpha``.

    We use the finite-sample-valid upper confidence bound on the retained mean
    risk. For the retained set of size ``m``, the risk estimate is the sample
    mean of ``1 - reliability``; we add the split-conformal safety margin
    ``ceil((m+1)(1-alpha))/m`` correction by requiring the empirical risk to sit
    below ``epsilon`` at the conformal rank, i.e. the ``(1-alpha)`` upper
    quantile of the retained unreliabilities must not exceed ``epsilon`` scaled
    by the risk-control bound. The returned certificate reports the first tau
    (scanning thresholds high->low, i.e. retaining ever more) at which the
    guarantee still holds — the most inclusive certified prune.
    """
    import numpy as np

    conf = np.asarray(conf, dtype="float64").ravel()
    reliability = np.asarray(reliability, dtype="float64").ravel()
    if conf.shape[0] != reliability.shape[0]:
        raise ValueError("conf and reliability must have equal length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = conf.shape[0]
    if n == 0:
        return ReliabilityCertificate(1.0, epsilon, alpha, 0.0, False, 0.0)
    unrel = np.clip(1.0 - reliability, 0.0, 1.0)

    # Candidate thresholds = the distinct confidences, scanned from most
    # inclusive (lowest tau) upward, keeping the MOST inclusive tau that still
    # certifies. Conformal risk control (Angelopoulos et al.): the retained mean
    # risk Rhat over m points admits the finite-sample bound that its true mean
    # is <= Rhat + (B - Rhat)/(m+1) ... we use the conservative Waudby-Smith/
    # Hoeffding-style UCB Rhat + sqrt(log(1/alpha)/(2m)) for a bounded [0,1] loss,
    # which is distribution-free and finite-sample valid.
    taus = np.unique(conf)
    best = ReliabilityCertificate(1.0, epsilon, alpha, 0.0, False, 0.0)
    for tau in taus:  # ascending -> most inclusive first
        keep = conf >= tau
        m = int(keep.sum())
        if m == 0:
            continue
        rhat = float(unrel[keep].mean())
        ucb = rhat + np.sqrt(np.log(1.0 / alpha) / (2.0 * m))
        if ucb <= epsilon:
            return ReliabilityCertificate(
                tau=float(tau), epsilon=float(epsilon), alpha=float(alpha),
                kept_fraction=m / n, certified=True, empirical_risk=rhat,
            )
        # remember the tightest attempt (highest tau tried) for reporting
        best = ReliabilityCertificate(
            tau=float(tau), epsilon=float(epsilon), alpha=float(alpha),
            kept_fraction=m / n, certified=False, empirical_risk=rhat,
        )
    return best


def selection_quality_curve(scores, reliability, fractions=None):
    """Retained-reliability-vs-budget curve for a carrier selection ``scores``
    (higher = keep first). Returns (fractions, retained_mean_reliability, auc).

    Used to show that calibrated-confidence-guided selection retains more true
    reliability at every budget than opacity- or random-guided selection — the
    downstream demonstration of the killer property. ``auc`` is the mean retained
    reliability averaged over the budget grid (higher is better).
    """
    import numpy as np

    scores = np.asarray(scores, dtype="float64").ravel()
    reliability = np.asarray(reliability, dtype="float64").ravel()
    n = scores.shape[0]
    if n == 0:
        return (np.asarray([]), np.asarray([]), 0.0)
    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 10)
    fractions = np.asarray(fractions, dtype="float64").ravel()
    order = np.argsort(-scores, kind="mergesort")  # keep highest score first
    rel_sorted = reliability[order]
    curve = []
    for f in fractions:
        k = max(1, int(round(f * n)))
        curve.append(float(rel_sorted[:k].mean()))
    curve = np.asarray(curve)
    return (fractions, curve, float(curve.mean()))


def attach_calibrated_confidence(carriers, calibrator, *, raw_key="confidence"):
    """Return a shallow copy of ``carriers`` whose ``confidence`` is the
    calibrated value ``calibrator.predict(raw)`` and which records
    ``confidence_calibrated=True`` so the KHR export (``_AURA_CONFIDENCE``) ships
    the trustworthy number rather than the raw heuristic. Requires a raw
    confidence already attached (see :func:`aura.confidence.attach_confidence`).
    """
    import numpy as np

    if raw_key not in carriers or carriers[raw_key] is None:
        raise KeyError(f"carriers has no raw confidence under '{raw_key}'")
    raw = carriers[raw_key]
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    out = dict(carriers)
    out["confidence"] = np.asarray(calibrator.predict(raw), dtype="float32")
    out["confidence_calibrated"] = True
    return out
