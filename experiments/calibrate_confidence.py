"""P0 calibration report on the real Truck scene.

Consumes the held-out reliability signal (per_carrier_reliability.py) and runs the
CPU calibration + certificate core (aura.calibration) on a carrier split disjoint
from the one the calibrator is fit on. Reports:
  * ECE of the raw view-count heuristic vs the calibrated confidence;
  * the conformal pruning certificate (drop below tau, lose <= epsilon reliability
    at confidence 1-alpha);
  * selection-quality AUC (mean retained held-out reliability across budgets) for
    calibrated confidence vs opacity vs the raw heuristic vs random -- the killer
    property is calibrated-confidence-guided pruning beating the alternatives.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reliability", default="outputs/reliability_truck.npz")
    ap.add_argument("--feature", default="train_agree",
                    choices=["train_agree", "raw_conf"],
                    help="export-time feature to calibrate into confidence")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.6,
                    help="pruning risk budget: mean unreliability of kept set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="truck", help="scene name recorded in report")
    ap.add_argument("--report", default="outputs/calib_truck.json")
    a = ap.parse_args()

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from aura.calibration import (
        IsotonicConfidenceCalibrator,
        conformal_prune_certificate,
        expected_calibration_error,
        selection_quality_curve,
    )

    d = np.load(a.reliability)
    labeled = d["labeled"]
    feat = d[a.feature][labeled]
    raw_conf = d["raw_conf"][labeled]
    reliability = d["reliability"][labeled]
    opacity = d["opacity"][labeled]
    m = feat.shape[0]

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(m)
    half = m // 2
    cal_idx, ev_idx = perm[:half], perm[half:]

    calibrator = IsotonicConfidenceCalibrator().fit(feat[cal_idx], reliability[cal_idx])
    conf_ev = calibrator.predict(feat[ev_idx])          # calibrated confidence
    rel_ev = reliability[ev_idx]

    # Calibration quality: raw heuristic vs calibrated (on the eval split).
    ece_raw = expected_calibration_error(raw_conf[ev_idx], rel_ev)
    ece_feat = expected_calibration_error(feat[ev_idx], rel_ev)
    ece_cal = expected_calibration_error(conf_ev, rel_ev)

    # Conformal pruning certificate on the eval split.
    cert = conformal_prune_certificate(conf_ev, rel_ev, epsilon=a.epsilon, alpha=a.alpha)

    # Selection quality: mean retained reliability across budgets (higher better).
    fracs = np.linspace(0.1, 1.0, 10)
    _, cal_curve, auc_cal = selection_quality_curve(conf_ev, rel_ev, fracs)
    _, opa_curve, auc_opa = selection_quality_curve(opacity[ev_idx], rel_ev, fracs)
    _, raw_curve, auc_raw = selection_quality_curve(raw_conf[ev_idx], rel_ev, fracs)
    rnd = rng.uniform(size=rel_ev.shape[0])
    _, _, auc_rnd = selection_quality_curve(rnd, rel_ev, fracs)
    # Oracle (sort by the true held-out reliability): the achievable ceiling.
    _, _, auc_oracle = selection_quality_curve(rel_ev, rel_ev, fracs)

    report = {
        "scene": a.scene,
        "label": str(d["label"]) if "label" in d else "color",
        "carriers_labeled": int(m),
        "feature": a.feature,
        "calibration": {
            "ece_raw_heuristic": round(ece_raw, 4),
            "ece_feature_uncalibrated": round(ece_feat, 4),
            "ece_calibrated": round(ece_cal, 4),
        },
        "pruning_certificate": {
            "tau": round(cert.tau, 4),
            "epsilon": cert.epsilon,
            "alpha": cert.alpha,
            "certified": cert.certified,
            "kept_fraction": round(cert.kept_fraction, 4),
            "empirical_risk_kept": round(cert.empirical_risk, 4),
        },
        "selection_auc_retained_reliability": {
            "calibrated_confidence": round(auc_cal, 4),
            "opacity": round(auc_opa, 4),
            "raw_heuristic": round(auc_raw, 4),
            "random": round(auc_rnd, 4),
            "oracle_ceiling": round(auc_oracle, 4),
        },
        "curves_at_budgets": {
            "budgets": [round(f, 2) for f in fracs],
            "calibrated_confidence": [round(x, 4) for x in cal_curve],
            "opacity": [round(x, 4) for x in opa_curve],
        },
    }
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
