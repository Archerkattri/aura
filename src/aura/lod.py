"""Certified level-of-detail / streaming plans — making the pruning certificate *do* something.

AURA's P0 killer property is a **calibrated, certified** per-carrier confidence
(:mod:`aura.calibration`). The conformal pruning certificate answers "drop below
threshold ``tau`` and lose at most ``epsilon`` reliability mass at confidence
``1 - alpha``". This module turns that single certificate into an *ordered,
multi-level streaming plan*: a consumer receives carriers in descending calibrated
confidence and may **stop at any published level** with a stated, distribution-free
bound on the reliability mass it has discarded by stopping there.

**What is certified.** Sort carriers by calibrated confidence (descending). Level
``k`` publishes a keep-fraction ``f_k`` (e.g. keep the top 10%), the confidence
threshold ``tau_k`` at that boundary, and a certified bound ``epsilon_k`` on the
**lost-reliability mass** — the per-carrier reliability discarded by pruning to
level ``k``::

    loss_k = mean_i ( reliability_i * 1[ conf_i < tau_k ] )     (over the population)

This is the reliability you forfeit by keeping only the top ``f_k``. It is ``0`` at
the 100% level (nothing is dropped) and grows as you prune harder. Because the
per-carrier term is a bounded ``[0,1]`` loss, its mean is certified with the same
finite-sample Hoeffding bound the pruning certificate uses
(:func:`aura.calibration.conformal_mean_upper_bound`) — computed on the scene's
**calibration half only**, so the evaluation half never informs the plan.

**Family-wise validity (Bonferroni over the non-trivial levels).** A ``K``-level plan
makes ``K`` simultaneous statements, but only the ``R`` **non-trivial** levels
(keep-fraction ``< 1``) are *random* certificates drawn from one conformal set. The
100% level is **deterministic**: ``epsilon = 0`` by definition — no carrier is pruned,
so the discarded mass is exactly zero with no sampling uncertainty — and is flagged
``trivial: true``. By the union bound the family-wise error is at most the sum of the
per-level errors, and the deterministic level contributes exactly ``0``, so it needs
no error budget. We therefore split ``alpha`` over the ``R`` **random** levels only —
``alpha' = alpha / R`` — and still guarantee family-wise ``1 - alpha`` (``R`` random
intervals each at ``alpha'`` union-bound to ``R * alpha' = alpha``, plus one
deterministic statement that never fails). This is strictly tighter than the naive
``alpha / K``: with the default four levels ``R = 3``, so ``alpha' = alpha / 3``
(``0.0333`` at ``alpha = 0.1``) instead of ``alpha / 4 = 0.025``, yielding smaller
certified ``epsilon_k`` at the same honest ``1 - alpha`` guarantee.

**Honest scope.** ``epsilon_k`` bounds *reliability* mass (a colour-agreement /
occlusion-aware reliability proxy), **not** rendered PSNR. As the repo's P0/P2
results establish, opacity — not reliability — is the render-PSNR-preserving prune
signal; a certified-LOD stream trades on trustworthiness, not on preserving the
alpha-blended image. See ``docs/P4_CERTIFIED_LOD.md`` and ``docs/P0_CALIBRATED_CONFIDENCE.md``.

Pure numpy, CPU-cheap: it consumes a calibrated-confidence array and a held-out
reliability signal and emits a JSON-able plan.
"""
from __future__ import annotations

DEFAULT_LEVELS = (0.10, 0.25, 0.50, 1.00)


def _as_1d(a, name):
    import numpy as np

    a = np.asarray(a, dtype="float64").ravel()
    return a


def _stable_desc_order(confidence):
    """Indices that sort ``confidence`` descending with a stable tiebreak by index.

    ``argsort`` of the negated signal with a stable ``mergesort`` keeps original
    index order among equal confidences, so the plan is deterministic.
    """
    import numpy as np

    return np.argsort(-confidence, kind="mergesort")


def _level_boundary_tau(sorted_conf, keep_fraction):
    """Confidence threshold at a keep-fraction boundary on ``sorted_conf`` (descending).

    Keeping the top ``round(f * n)`` carriers means the boundary confidence is the
    ``n_keep``-th largest value, so ``conf >= tau`` retains at least that many (more
    only under ties at the boundary). Returns ``(tau, n_keep)``.
    """
    n = sorted_conf.shape[0]
    n_keep = int(round(keep_fraction * n))
    n_keep = max(1, min(n, n_keep))
    tau = float(sorted_conf[n_keep - 1])
    return tau, n_keep


def certified_lod_plan(confidence, reliability, *, levels=DEFAULT_LEVELS,
                       alpha=0.1, scene=None):
    """Build a Bonferroni-corrected certified LOD/streaming plan on a conformal set.

    Parameters
    ----------
    confidence : array [n]
        **Calibrated** per-carrier confidence for the conformal (calibration) set —
        the output of a fitted :class:`~aura.calibration.IsotonicConfidenceCalibrator`
        on the calibration carriers. Carriers are streamed in descending order of
        this value.
    reliability : array [n]
        Per-carrier reliability in ``[0,1]`` (``1`` = fully reliable) for the *same*
        conformal set. Only this calibration half informs the plan.
    levels : sequence of float
        Keep-fractions in ``(0, 1]`` (default ``[0.10, 0.25, 0.50, 1.00]``). Of the
        ``K = len(levels)`` levels, the ``R`` with keep-fraction ``< 1`` are *random*
        certificates; each is certified at ``alpha' = alpha / R``. Any level with
        keep-fraction ``>= 1`` is the *deterministic* trivial level (``epsilon = 0``,
        no interval) and consumes no error budget.
    alpha : float
        Family-wise miscoverage. Per-level miscoverage for the random levels is
        ``alpha / R`` where ``R`` is the number of non-trivial levels (Bonferroni over
        the random levels only; the trivial level is deterministic — union bound).
    scene : str, optional
        Recorded in the plan for provenance.

    Returns
    -------
    dict
        JSON-able plan: ``scene``, ``alpha``, ``correction: "bonferroni"``,
        ``alpha_per_level``, ``family_wise_confidence``, ``sort:
        "calibrated_confidence_desc"``, ``n_cal``, and ``levels`` — one entry per
        keep-fraction with ``keep_fraction``, ``tau``, ``epsilon_certified``,
        ``n_cal``, ``n_kept_cal`` and ``trivial``.
    """
    import numpy as np

    from .calibration import conformal_mean_upper_bound

    confidence = _as_1d(confidence, "confidence")
    reliability = np.clip(_as_1d(reliability, "reliability"), 0.0, 1.0)
    if confidence.shape[0] != reliability.shape[0]:
        raise ValueError("confidence and reliability must have equal length")
    n = confidence.shape[0]
    if n == 0:
        raise ValueError("empty conformal set — cannot certify a plan over 0 carriers")
    levels = [float(f) for f in levels]
    if not levels:
        raise ValueError("levels must be non-empty")
    for f in levels:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {f}")

    K = len(levels)
    # Only the NON-TRIVIAL levels (keep-fraction < 1) are random certificates that
    # spend error budget; a keep-fraction >= 1 level prunes nothing and is a
    # deterministic epsilon=0 statement (no interval). Bonferroni over the random
    # levels only — union bound: R random intervals at alpha/R plus one deterministic
    # statement give family-wise 1-alpha. Tighter than the naive alpha/K.
    n_nontrivial = sum(1 for f in levels if f < 1.0)
    alpha_per_level = alpha / n_nontrivial if n_nontrivial > 0 else alpha

    order = _stable_desc_order(confidence)
    sorted_conf = confidence[order]

    level_entries = []
    for f in levels:
        if f >= 1.0:
            # Keep everything: no carrier is pruned, so the discarded reliability
            # mass is *exactly* 0 with no sampling uncertainty. tau is the minimum
            # confidence so that conf >= tau keeps all carriers.
            level_entries.append({
                "keep_fraction": round(f, 6),
                "tau": float(sorted_conf[-1]),
                "epsilon_certified": 0.0,
                "n_cal": int(n),
                "n_kept_cal": int(n),
                "trivial": True,
            })
            continue
        tau, n_keep = _level_boundary_tau(sorted_conf, f)
        # tau is stored at FULL precision: the isotonic calibrator produces
        # plateaus (many carriers share a knot value), and the boundary frequently
        # lands on one. Rounding tau would move the `conf < tau` cut across the
        # whole plateau and silently change the drop set, breaking the bound. Round
        # only for display, never for the stored threshold.
        dropped = confidence < tau
        # Per-carrier discard loss over the WHOLE conformal set: reliability if the
        # carrier is pruned at this level, else 0. Its mean is the lost-reliability
        # mass; the Hoeffding UCB at alpha' is the certified bound.
        discard_loss = np.where(dropped, reliability, 0.0)
        epsilon = conformal_mean_upper_bound(discard_loss, alpha_per_level)
        level_entries.append({
            "keep_fraction": round(f, 6),
            "tau": tau,
            "epsilon_certified": epsilon,
            "n_cal": int(n),
            "n_kept_cal": int(n_keep),
            "trivial": False,
        })

    return {
        "scene": scene,
        "alpha": alpha,
        "correction": "bonferroni",
        "alpha_per_level": alpha_per_level,
        "family_wise_confidence": round(1.0 - alpha, 6),
        "sort": "calibrated_confidence_desc",
        "n_cal": int(n),
        "n_levels": K,
        "n_nontrivial_levels": int(n_nontrivial),
        "loss": "discarded_reliability_mass",
        "levels": level_entries,
    }


def _find_level(plan, level):
    """Return the plan level entry matching keep-fraction ``level`` (float-tolerant)."""
    for lvl in plan["levels"]:
        if abs(float(lvl["keep_fraction"]) - float(level)) <= 1e-9:
            return lvl
    published = [lvl["keep_fraction"] for lvl in plan["levels"]]
    raise KeyError(f"level {level} not in plan (published: {published})")


def apply_lod_plan(carriers, plan, level):
    """Return the carrier subset retained at ``level`` for streaming.

    ``carriers`` is a mapping with a ``confidence`` array (the calibrated
    confidence). The subset is ``{i : confidence_i >= tau_level}``; every array-like
    value with leading dimension ``N`` is filtered to that mask (scalars such as
    ``sh_degree`` are passed through). The returned dict adds ``_lod_level``,
    ``_lod_tau`` and ``_lod_kept`` for provenance. Supports numpy arrays and torch
    tensors.
    """
    import numpy as np

    lvl = _find_level(plan, level)
    tau = lvl["tau"]
    if "confidence" not in carriers or carriers["confidence"] is None:
        raise KeyError("carriers has no 'confidence' array to stream by")
    conf = carriers["confidence"]
    conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
    conf_np = conf_np.ravel()
    n = conf_np.shape[0]
    keep = np.ones(n, dtype=bool) if tau is None else (conf_np >= float(tau))

    out = {}
    for key, value in carriers.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] == n:
            if hasattr(value, "detach"):  # torch tensor
                import torch

                out[key] = value[torch.as_tensor(keep, device=value.device)]
            else:
                out[key] = np.asarray(value)[keep]
        else:
            out[key] = value
    out["_lod_level"] = float(level)
    out["_lod_tau"] = None if tau is None else float(tau)
    out["_lod_kept"] = int(keep.sum())
    return out


def evaluate_lod_plan(plan, confidence_eval, reliability_eval):
    """Measure each level's empirical lost-reliability mass on a held-out eval set.

    Applies the plan's ``tau_k`` (built on the calibration half) to a *disjoint*
    evaluation half and computes ``empirical_loss_eval = mean(reliability *
    1[conf < tau_k])`` — the same discard-loss the certificate bounds, now estimated
    on fresh data. ``holds`` is ``empirical_loss_eval <= epsilon_certified``.

    Returns a list of per-level dicts ``{keep_fraction, tau, epsilon_certified,
    empirical_loss_eval, holds, trivial}``.
    """
    import numpy as np

    confidence_eval = _as_1d(confidence_eval, "confidence_eval")
    reliability_eval = np.clip(_as_1d(reliability_eval, "reliability_eval"), 0.0, 1.0)
    if confidence_eval.shape[0] != reliability_eval.shape[0]:
        raise ValueError("confidence_eval and reliability_eval must have equal length")

    results = []
    for lvl in plan["levels"]:
        tau = lvl["tau"]
        eps = float(lvl["epsilon_certified"])
        trivial = bool(lvl.get("trivial", False))
        if trivial:
            # Nothing pruned ⇒ zero discarded mass, deterministically.
            emp = 0.0
        else:
            dropped = confidence_eval < float(tau)
            emp = float(np.where(dropped, reliability_eval, 0.0).mean()) if confidence_eval.size else 0.0
        results.append({
            "keep_fraction": lvl["keep_fraction"],
            "tau": tau,
            "epsilon_certified": eps,
            "empirical_loss_eval": emp,
            "holds": bool(emp <= eps + 1e-12),
            "trivial": trivial,
        })
    return results
