"""P4 — certified LOD/streaming evaluation across all scenes.

For each of {truck, garden, kitchen, room} this builds a Bonferroni-corrected
certified LOD plan (:func:`aura.lod.certified_lod_plan`) from the committed
``outputs/reliability_<scene>.npz`` using the **same seed-0 50/50
calibration/eval carrier split as ``experiments/calibrate_confidence.py``** — the
plan is constructed on the calibration half ONLY, then the evaluation half (which
never touches plan construction) is used to measure the empirical lost-reliability
mass at each level and check it against the certified bound ``epsilon_k``.

The lost-reliability mass at level ``k`` is the per-carrier reliability discarded by
pruning to keep-fraction ``f_k`` (``mean(reliability * 1[conf < tau_k])``); the
certificate bounds it distribution-free with family-wise confidence ``1 - alpha``.
Every bound is expected to hold empirically; any that does not is reported (not
tuned away).

Pure CPU / numpy over the committed ``outputs/reliability_*.npz``; seed-0
deterministic; no GPU, no retraining, no dataset access. Writes
``outputs/lod_certified.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCENES = ["truck", "garden", "kitchen", "room"]


def calibration_eval_split(m, seed=0):
    """The seed-0 50/50 carrier split used by ``calibrate_confidence.py``.

    Returns ``(cal_idx, eval_idx)`` — disjoint by construction (two halves of one
    permutation). The eval half must never inform plan construction.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    half = m // 2
    cal_idx, eval_idx = perm[:half], perm[half:]
    assert set(cal_idx.tolist()).isdisjoint(eval_idx.tolist()), "cal/eval overlap"
    return cal_idx, eval_idx


def build_lod_eval_report(outdir, *, feature="train_agree", alpha=0.1,
                          levels=(0.10, 0.25, 0.50, 1.00), seed=0) -> dict:
    """Compute the certified-LOD evaluation for all scenes; returns a JSON dict."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from aura.calibration import IsotonicConfidenceCalibrator
    from aura.lod import certified_lod_plan, evaluate_lod_plan

    outdir = Path(outdir)
    levels = [float(f) for f in levels]

    report = {
        "experiment": "P4_certified_lod_streaming",
        "scenes": SCENES,
        "feature": feature,
        "alpha": alpha,
        "alpha_per_level": alpha / len(levels),
        "correction": "bonferroni",
        "family_wise_confidence": round(1.0 - alpha, 6),
        "levels": levels,
        "seed": seed,
        "loss": "discarded_reliability_mass = mean(reliability * 1[conf < tau])",
        "protocol": "seed-0 50/50 carrier split (matches calibrate_confidence.py): "
                    "calibrator fit + LOD plan built on the CALIBRATION half only; "
                    "empirical lost-reliability mass measured on the disjoint EVAL half.",
        "scope": "epsilon bounds RELIABILITY mass (colour-agreement / occlusion-aware "
                 "proxy), NOT rendered PSNR; opacity remains the PSNR-preserving prune "
                 "signal (see docs/P0_CALIBRATED_CONFIDENCE.md, docs/P2_FULLRES_RENDERLOSS.md).",
        "by_scene": {},
        "all_bounds_hold": True,
        "violations": [],
    }

    for scene in SCENES:
        d = np.load(outdir / f"reliability_{scene}.npz")
        labeled = d["labeled"]
        feat = d[feature][labeled].astype("float64")
        rel = d["reliability"][labeled].astype("float64")
        m = feat.shape[0]

        cal_idx, eval_idx = calibration_eval_split(m, seed=seed)

        calibrator = IsotonicConfidenceCalibrator().fit(feat[cal_idx], rel[cal_idx])
        conf_cal = calibrator.predict(feat[cal_idx])
        conf_eval = calibrator.predict(feat[eval_idx])
        rel_cal, rel_eval = rel[cal_idx], rel[eval_idx]

        plan = certified_lod_plan(conf_cal, rel_cal, levels=levels, alpha=alpha, scene=scene)
        evals = evaluate_lod_plan(plan, conf_eval, rel_eval)

        scene_levels = []
        for plvl, elvl in zip(plan["levels"], evals):
            # holds is decided on full precision inside evaluate_lod_plan; round only
            # for the human-readable table / doc.
            row = {
                "keep_fraction": plvl["keep_fraction"],
                "tau": round(float(plvl["tau"]), 6),
                "epsilon_certified": round(float(plvl["epsilon_certified"]), 6),
                "n_kept_cal": plvl["n_kept_cal"],
                "empirical_loss_eval": round(float(elvl["empirical_loss_eval"]), 6),
                "holds": elvl["holds"],
                "trivial": plvl["trivial"],
            }
            scene_levels.append(row)
            if not elvl["holds"]:
                report["all_bounds_hold"] = False
                report["violations"].append({
                    "scene": scene, "keep_fraction": plvl["keep_fraction"],
                    "epsilon_certified": plvl["epsilon_certified"],
                    "empirical_loss_eval": elvl["empirical_loss_eval"],
                })

        report["by_scene"][scene] = {
            "n_labeled": int(m),
            "n_cal": int(cal_idx.shape[0]),
            "n_eval": int(eval_idx.shape[0]),
            "mean_reliability_eval": round(float(rel_eval.mean()), 4),
            "plan": plan,
            "levels": scene_levels,
        }

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--report", default="outputs/lod_certified.json")
    ap.add_argument("--feature", default="train_agree")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.00])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    report = build_lod_eval_report(
        a.outdir, feature=a.feature, alpha=a.alpha, levels=a.levels, seed=a.seed)

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"certified LOD eval — alpha={a.alpha} (family-wise), "
          f"alpha'={report['alpha_per_level']:.4f} Bonferroni over K={len(a.levels)} levels")
    for scene in SCENES:
        sc = report["by_scene"][scene]
        print(f"\n  {scene}  (n_cal={sc['n_cal']} n_eval={sc['n_eval']} "
              f"mean_rel_eval={sc['mean_reliability_eval']})")
        print(f"    {'keep':>5} {'tau':>9} {'eps_cert':>9} {'emp_loss':>9} {'holds':>6}")
        for row in sc["levels"]:
            tag = " (trivial)" if row["trivial"] else ""
            print(f"    {row['keep_fraction']:>5.2f} {row['tau']:>9.4f} "
                  f"{row['epsilon_certified']:>9.4f} {row['empirical_loss_eval']:>9.4f} "
                  f"{str(row['holds']):>6}{tag}")
    if report["all_bounds_hold"]:
        print("\nAll certified bounds hold on the eval half.")
    else:
        print(f"\nVIOLATIONS ({len(report['violations'])}):")
        for v in report["violations"]:
            print(f"  {v['scene']} keep={v['keep_fraction']}: "
                  f"empirical {v['empirical_loss_eval']} > epsilon {v['epsilon_certified']}")
    print(f"\nwrote {a.report}")


if __name__ == "__main__":
    main()
