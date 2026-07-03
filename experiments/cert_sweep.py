"""P1b — certificate operating study across all scenes and both labels.

The P0 paper reported the conformal pruning certificate at a single epsilon=0.6 and
only sketched the epsilon-sweep on Truck. This script extends that sweep to all four
scenes ({truck, garden, kitchen, room}) and both reliability labels ({color, depth}).

For each scene x label it uses the exact P0 in-scene protocol of
``calibrate_confidence.py``: fit the isotonic calibrator on the 50/50 seed-0 cal
half, then evaluate the certificate on the held-out eval half with the calibrated
confidence. It sweeps epsilon over a grid (default 0.30 -> 0.65) at fixed alpha=0.1
and records, per epsilon:
  * the certified threshold tau (keep carriers with conf >= tau),
  * whether a certificate exists,
  * the kept fraction, and
  * the empirical mean unreliability of the kept set.

kept_fraction is monotone non-decreasing in epsilon (a looser budget keeps more), so
each scene has a threshold epsilon* below which the certificate first becomes
selective (kept < 1.0). We report that onset per scene x label.

Pure CPU / numpy over the committed ``outputs/reliability_*.npz``; seed-0
deterministic; no GPU, no retraining, no dataset access.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCENES = ["truck", "garden", "kitchen", "room"]
LABELS = {"color": "", "depth": "_depth"}


def build_cert_sweep_report(outdir, *, alpha=0.1, eps_lo=0.30, eps_hi=0.65,
                            eps_step=0.01, seed=0) -> dict:
    """Compute the certificate operating study (epsilon-sweep, all scenes x both
    labels) from the ``reliability_*.npz`` files under ``outdir``. Pure CPU/numpy;
    returns the JSON-serialisable report dict."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from aura.calibration import (
        IsotonicConfidenceCalibrator,
        conformal_prune_certificate,
    )

    outdir = Path(outdir)
    # inclusive grid, rounded to avoid float dust in the JSON keys
    n_steps = int(round((eps_hi - eps_lo) / eps_step)) + 1
    eps_grid = [round(eps_lo + i * eps_step, 3) for i in range(n_steps)]

    report = {
        "experiment": "P1b_certificate_operating_study",
        "scenes": SCENES,
        "labels": list(LABELS),
        "alpha": alpha,
        "seed": seed,
        "epsilon_grid": eps_grid,
        "protocol": "in-scene P0 protocol: calibrator fit on the 50/50 seed-0 cal "
                    "half, certificate evaluated on the eval half with calibrated "
                    "confidence (matches calibrate_confidence.py / calib_<scene>.json)",
        "selectivity_note": "selectivity_onset_epsilon = the largest epsilon in the "
                            "grid whose certified kept_fraction < 1.0 (kept_fraction is "
                            "monotone non-decreasing in epsilon); None if the full set "
                            "is kept across the whole grid",
        "by_scene": {},
    }

    for scene in SCENES:
        report["by_scene"][scene] = {}
        for label_name, suffix in LABELS.items():
            d = np.load(outdir / f"reliability_{scene}{suffix}.npz")
            labeled = d["labeled"]
            feat = d["train_agree"][labeled].astype("float64")
            rel = d["reliability"][labeled].astype("float64")
            m = feat.shape[0]

            rng = np.random.default_rng(seed)
            perm = rng.permutation(m)
            half = m // 2
            cal_idx, ev_idx = perm[:half], perm[half:]

            g = IsotonicConfidenceCalibrator().fit(feat[cal_idx], rel[cal_idx])
            conf_ev = g.predict(feat[ev_idx])
            rel_ev = rel[ev_idx]

            sweep = []
            onset = None  # largest epsilon whose kept_fraction < 1.0
            for eps in eps_grid:
                cert = conformal_prune_certificate(conf_ev, rel_ev, epsilon=eps, alpha=alpha)
                selective = cert.kept_fraction < 1.0 - 1e-9
                sweep.append({
                    "epsilon": eps,
                    "tau": round(cert.tau, 4),
                    "certified": bool(cert.certified),
                    "kept_fraction": round(cert.kept_fraction, 4),
                    "empirical_risk_kept": round(cert.empirical_risk, 4),
                    "selective": bool(selective),
                })
                if selective:
                    onset = eps  # grid ascending -> ends at the largest selective eps

            report["by_scene"][scene][label_name] = {
                "n_labeled": int(m),
                "n_eval": int(ev_idx.shape[0]),
                "mean_reliability_eval": round(float(rel_ev.mean()), 4),
                "selectivity_onset_epsilon": onset,
                "always_selective": bool(sweep[-1]["selective"]),  # selective even at eps_hi
                "never_selective": onset is None,
                "sweep": sweep,
            }

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--report", default="outputs/cert_sweep.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps-lo", type=float, default=0.30)
    ap.add_argument("--eps-hi", type=float, default=0.65)
    ap.add_argument("--eps-step", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    report = build_cert_sweep_report(
        a.outdir, alpha=a.alpha, eps_lo=a.eps_lo, eps_hi=a.eps_hi,
        eps_step=a.eps_step, seed=a.seed)
    eps_grid = report["epsilon_grid"]

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"epsilon grid: {eps_grid[0]} -> {eps_grid[-1]} step {a.eps_step} "
          f"({len(eps_grid)} points), alpha={a.alpha}")
    for scene in SCENES:
        for label_name in LABELS:
            r = report["by_scene"][scene][label_name]
            onset = r["selectivity_onset_epsilon"]
            onset_s = f"eps={onset}" if onset is not None else "never (kept=1.0 across grid)"
            print(f"  {scene:8s} {label_name:5s}: selectivity onset {onset_s}"
                  f" | mean reliability {r['mean_reliability_eval']:.3f}")
    print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
