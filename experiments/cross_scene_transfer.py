"""P1a — cross-scene calibrator transfer.

The P0 paper deliberately declined the transfer question ("does one calibrator
work on another scene?") for lack of measurement. This script measures it.

For every ordered scene pair (A -> B) over {truck, garden, kitchen, room} and each
reliability label {color, depth}, we:

  * fit an ``IsotonicConfidenceCalibrator`` on the export-time ``train_agree``
    feature of ALL labeled carriers of the SOURCE scene A;
  * evaluate the resulting (transferred) confidence on the TARGET scene B, on B's
    held-out eval half (the ``ev_idx`` of B's 50/50 seed-0 split, the exact split
    convention of ``calibrate_confidence.py``), so its ECE and selection AUC are
    directly comparable to B's own in-scene diagonal;
  * report the conformal pruning certificate (epsilon=0.6, alpha=0.1) computed with
    the TRANSFERRED confidence but on B's OWN local calibration split (B's
    ``cal_idx``) for the conformal threshold. This is the honest deployment story:
    a transferred calibrator is only a monotone re-scoring, so distribution-free
    conformal validity is restored by keeping a small LOCAL conformal set on B.

Diagonal cells (A == B) reproduce ``calibrate_confidence.py`` exactly (calibrator
fit on B's cal half, evaluated on B's eval half) and serve as the in-scene
reference against which every off-diagonal transfer delta is measured.

Two subtleties worth stating up front, both surfaced by the numbers:
  * Selection AUC is a *ranking* metric and an isotonic calibrator is a monotone
    map, so the carrier ordering B is scored by is the same whether the calibrator
    came from A or B. Selection quality therefore transfers essentially for free;
    the interesting transfer question is about ECE (absolute calibration).
  * A transferred calibrator can shift the absolute probability scale (that is what
    ECE measures), but never the ordering — which is why the local conformal set
    still yields a valid certificate.

Pure CPU / numpy over the committed ``outputs/reliability_*.npz``; seed-0
deterministic; no GPU, no retraining, no dataset access.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCENES = ["truck", "garden", "kitchen", "room"]
# label name -> reliability npz suffix ("" = colour-agreement, "_depth" = occlusion-aware)
LABELS = {"color": "", "depth": "_depth"}


def _load(scene: str, suffix: str, outdir: Path):
    import numpy as np

    d = np.load(outdir / f"reliability_{scene}{suffix}.npz")
    labeled = d["labeled"]
    return {
        "feat": d["train_agree"][labeled].astype("float64"),
        "raw_conf": d["raw_conf"][labeled].astype("float64"),
        "reliability": d["reliability"][labeled].astype("float64"),
        "opacity": d["opacity"][labeled].astype("float64"),
        "n": int(labeled.sum()),
        "label": str(d["label"]),
    }


def _split(m: int, seed: int):
    """Reproduce calibrate_confidence.py's 50/50 seed-0 split for m labeled carriers."""
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    half = m // 2
    return perm[:half], perm[half:]


def build_transfer_report(outdir, *, alpha=0.1, epsilon=0.6, seed=0) -> dict:
    """Compute the full 4x4x2 cross-scene transfer report from the
    ``reliability_*.npz`` files under ``outdir``. Pure CPU/numpy; returns the
    JSON-serialisable report dict (see module docstring for the protocol)."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from aura.calibration import (
        IsotonicConfidenceCalibrator,
        conformal_prune_certificate,
        expected_calibration_error,
        selection_quality_curve,
    )

    outdir = Path(outdir)
    fracs = np.linspace(0.1, 1.0, 10)

    report = {
        "experiment": "P1a_cross_scene_calibrator_transfer",
        "scenes": SCENES,
        "labels": list(LABELS),
        "seed": seed,
        "alpha": alpha,
        "epsilon": epsilon,
        "protocol": {
            "diagonal": "in-scene: calibrator fit on B's cal half, evaluated on B's "
                        "eval half (reproduces calibrate_confidence.py / calib_<scene>.json)",
            "off_diagonal": "transferred: calibrator fit on ALL labeled carriers of "
                            "source A, evaluated on B's eval half (same ev_idx as the diagonal)",
            "certificate": "epsilon=0.6, alpha=0.1, computed with the transferred "
                           "confidence on B's OWN calibration split (cal_idx) — the local "
                           "conformal set that restores distribution-free validity",
            "auc_note": "selection AUC is rank-based and isotonic calibration is monotone, "
                        "so transferred and in-scene AUC coincide up to isotonic tie-pooling",
        },
        "by_label": {},
    }

    for label_name, suffix in LABELS.items():
        data = {s: _load(s, suffix, outdir) for s in SCENES}
        splits = {s: _split(data[s]["n"], seed) for s in SCENES}

        # --- source calibrators fit on ALL labeled carriers of A ---
        cal_all = {}
        for s in SCENES:
            cal_all[s] = IsotonicConfidenceCalibrator().fit(
                data[s]["feat"], data[s]["reliability"])

        # --- per-target in-scene reference (diagonal, held-out) ---
        in_scene = {}
        cal_cal = {}  # calibrator fit on B's cal half (in-scene)
        for b in SCENES:
            cal_idx, ev_idx = splits[b]
            g = IsotonicConfidenceCalibrator().fit(
                data[b]["feat"][cal_idx], data[b]["reliability"][cal_idx])
            cal_cal[b] = g
            feat_ev = data[b]["feat"][ev_idx]
            rel_ev = data[b]["reliability"][ev_idx]
            conf_ev = g.predict(feat_ev)
            ece_raw = expected_calibration_error(data[b]["raw_conf"][ev_idx], rel_ev)
            ece_inscene = expected_calibration_error(conf_ev, rel_ev)
            _, _, auc_inscene = selection_quality_curve(conf_ev, rel_ev, fracs)
            _, _, auc_opacity = selection_quality_curve(data[b]["opacity"][ev_idx], rel_ev, fracs)
            _, _, auc_oracle = selection_quality_curve(rel_ev, rel_ev, fracs)
            # in-scene certificate (P0 protocol: calibrator on cal half, cert on eval half)
            cert_in = conformal_prune_certificate(conf_ev, rel_ev, epsilon=epsilon, alpha=alpha)
            in_scene[b] = {
                "n_labeled": data[b]["n"],
                "n_eval": int(ev_idx.shape[0]),
                "ece_raw": round(ece_raw, 4),
                "ece_inscene": round(ece_inscene, 4),
                "auc_inscene": round(auc_inscene, 4),
                "auc_opacity": round(auc_opacity, 4),
                "auc_oracle": round(auc_oracle, 4),
                "cert_inscene": {
                    "tau": round(cert_in.tau, 4),
                    "certified": bool(cert_in.certified),
                    "kept_fraction": round(cert_in.kept_fraction, 4),
                    "empirical_risk_kept": round(cert_in.empirical_risk, 4),
                },
            }

        # --- full A x B transfer matrix ---
        matrix = []
        for source in SCENES:
            for target in SCENES:
                cal_idx, ev_idx = splits[target]
                feat_ev = data[target]["feat"][ev_idx]
                rel_ev = data[target]["reliability"][ev_idx]
                if source == target:
                    # diagonal = in-scene held-out reference (calibrator on B's cal half)
                    g = cal_cal[target]
                    kind = "in_scene"
                else:
                    # off-diagonal = transferred calibrator (fit on ALL of A)
                    g = cal_all[source]
                    kind = "transferred"
                conf_ev = g.predict(feat_ev)
                ece_t = expected_calibration_error(conf_ev, rel_ev)
                _, _, auc_t = selection_quality_curve(conf_ev, rel_ev, fracs)

                # certificate with the (transferred) confidence on B's LOCAL cal split
                feat_cal = data[target]["feat"][cal_idx]
                rel_cal = data[target]["reliability"][cal_idx]
                conf_cal = g.predict(feat_cal)
                cert = conformal_prune_certificate(
                    conf_cal, rel_cal, epsilon=epsilon, alpha=alpha)

                ref = in_scene[target]
                matrix.append({
                    "source": source,
                    "target": target,
                    "kind": kind,
                    "ece_raw": ref["ece_raw"],
                    "ece": round(ece_t, 4),
                    "ece_inscene": ref["ece_inscene"],
                    "ece_delta_vs_inscene": round(ece_t - ref["ece_inscene"], 4),
                    "auc": round(auc_t, 4),
                    "auc_inscene": ref["auc_inscene"],
                    "auc_opacity": ref["auc_opacity"],
                    "auc_oracle": ref["auc_oracle"],
                    "auc_delta_vs_inscene": round(auc_t - ref["auc_inscene"], 4),
                    "cert_transferred_local_split": {
                        "tau": round(cert.tau, 4),
                        "certified": bool(cert.certified),
                        "kept_fraction": round(cert.kept_fraction, 4),
                        "empirical_risk_kept": round(cert.empirical_risk, 4),
                        "n_conformal": int(cal_idx.shape[0]),
                    },
                })

        # --- off-diagonal summary ---
        off = [c for c in matrix if c["source"] != c["target"]]
        ece_deltas = np.array([c["ece_delta_vs_inscene"] for c in off])
        auc_deltas = np.array([c["auc_delta_vs_inscene"] for c in off])
        ece_abs = np.array([c["ece"] for c in off])
        mean_raw_ece = float(np.mean([in_scene[b]["ece_raw"] for b in SCENES]))
        mean_inscene_ece = float(np.mean([in_scene[b]["ece_inscene"] for b in SCENES]))
        all_cert = all(c["cert_transferred_local_split"]["certified"] for c in off)

        worst_ece = max(off, key=lambda c: c["ece_delta_vs_inscene"])
        best_ece = min(off, key=lambda c: c["ece_delta_vs_inscene"])
        worst_auc = min(off, key=lambda c: c["auc_delta_vs_inscene"])
        best_auc = max(off, key=lambda c: c["auc_delta_vs_inscene"])

        report["by_label"][label_name] = {
            "in_scene_reference": in_scene,
            "matrix": matrix,
            "summary": {
                "n_off_diagonal_pairs": len(off),
                "mean_raw_ece": round(mean_raw_ece, 4),
                "mean_inscene_ece": round(mean_inscene_ece, 4),
                "offdiag_transferred_ece_mean": round(float(ece_abs.mean()), 4),
                "offdiag_transferred_ece_median": round(float(np.median(ece_abs)), 4),
                "offdiag_transferred_ece_max": round(float(ece_abs.max()), 4),
                "offdiag_ece_delta_mean": round(float(ece_deltas.mean()), 4),
                "offdiag_ece_delta_max": round(float(ece_deltas.max()), 4),
                "offdiag_auc_delta_mean": round(float(auc_deltas.mean()), 4),
                "offdiag_auc_delta_min": round(float(auc_deltas.min()), 4),
                "offdiag_auc_delta_max": round(float(auc_deltas.max()), 4),
                "all_transferred_certificates_valid": bool(all_cert),
                "best_transfer_pair_by_ece": {
                    "pair": f"{best_ece['source']}->{best_ece['target']}",
                    "ece": best_ece["ece"], "ece_inscene": best_ece["ece_inscene"],
                    "ece_delta": best_ece["ece_delta_vs_inscene"],
                },
                "worst_transfer_pair_by_ece": {
                    "pair": f"{worst_ece['source']}->{worst_ece['target']}",
                    "ece": worst_ece["ece"], "ece_inscene": worst_ece["ece_inscene"],
                    "ece_delta": worst_ece["ece_delta_vs_inscene"],
                },
                "best_transfer_pair_by_auc": {
                    "pair": f"{best_auc['source']}->{best_auc['target']}",
                    "auc": best_auc["auc"], "auc_inscene": best_auc["auc_inscene"],
                    "auc_delta": best_auc["auc_delta_vs_inscene"],
                },
                "worst_transfer_pair_by_auc": {
                    "pair": f"{worst_auc['source']}->{worst_auc['target']}",
                    "auc": worst_auc["auc"], "auc_inscene": worst_auc["auc_inscene"],
                    "auc_delta": worst_auc["auc_delta_vs_inscene"],
                },
            },
        }

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs",
                    help="directory holding reliability_*.npz and the report")
    ap.add_argument("--report", default="outputs/cross_scene_transfer.json")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--epsilon", type=float, default=0.6,
                    help="pruning risk budget for the reported certificate")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    report = build_transfer_report(a.outdir, alpha=a.alpha, epsilon=a.epsilon, seed=a.seed)

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")

    # console digest
    for label_name in LABELS:
        s = report["by_label"][label_name]["summary"]
        print(f"[{label_name}] raw ECE {s['mean_raw_ece']:.4f} | in-scene ECE "
              f"{s['mean_inscene_ece']:.4f} | transferred off-diag ECE mean "
              f"{s['offdiag_transferred_ece_mean']:.4f} (max {s['offdiag_transferred_ece_max']:.4f})")
        print(f"           ECE delta vs in-scene: mean {s['offdiag_ece_delta_mean']:+.4f} "
              f"max {s['offdiag_ece_delta_max']:+.4f} | AUC delta: mean "
              f"{s['offdiag_auc_delta_mean']:+.4f} [{s['offdiag_auc_delta_min']:+.4f},"
              f"{s['offdiag_auc_delta_max']:+.4f}]")
        print(f"           worst pair (ECE): {s['worst_transfer_pair_by_ece']['pair']} "
              f"-> {s['worst_transfer_pair_by_ece']['ece']:.4f} "
              f"(in-scene {s['worst_transfer_pair_by_ece']['ece_inscene']:.4f}); "
              f"all transferred certs valid: {s['all_transferred_certificates_valid']}")
    print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
