"""Tests for the P1 experiment drivers (CPU / numpy, self-contained).

These exercise ``experiments/cross_scene_transfer.py`` (P1a) and
``experiments/cert_sweep.py`` (P1b) on tiny SYNTHETIC ``reliability_*.npz`` written
to a temp dir, so they do not depend on the (gitignored) real reliability npz. They
guard the two structural invariants the P1 findings rest on:

  * P1a: selection AUC is rank-invariant under a monotone (isotonic) calibrator, so
    transferring a calibrator across scenes barely moves AUC, while it still
    calibrates far better than the raw heuristic (ECE); every transferred
    certificate is a valid, distribution-free object on the target's local split.
  * P1b: the certified kept-fraction is monotone non-decreasing in epsilon, and the
    reported selectivity onset is the largest epsilon whose kept-fraction < 1.

The scene-specific reliability curves differ (different exponents), so cross-scene
transfer is a non-trivial map, not an identity.
"""
import sys
from pathlib import Path

import numpy as np

from experiments.cross_scene_transfer import SCENES, build_transfer_report
from experiments.cert_sweep import build_cert_sweep_report

# gamma per scene -> distinct reliability(feature) curves, so transfer is non-trivial
_GAMMA = {"truck": 2.2, "garden": 1.5, "kitchen": 1.1, "room": 0.7}


def _write_synthetic(outdir: Path, n: int = 400) -> None:
    for si, scene in enumerate(SCENES):
        for label_name, suffix in (("color", ""), ("depth", "_depth")):
            rng = np.random.default_rng(1000 * si + (0 if suffix == "" else 1))
            feat = rng.uniform(0.0, 1.0, n)
            gamma = _GAMMA[scene]
            reliability = np.clip(feat**gamma + rng.normal(0, 0.05, n), 0.0, 1.0)
            # saturated view-count heuristic: near-constant high -> badly miscalibrated
            raw_conf = np.clip(0.85 + rng.normal(0, 0.04, n), 0.0, 1.0)
            opacity = rng.uniform(0.0, 1.0, n)
            labeled = np.ones(n, dtype=bool)
            np.savez(
                outdir / f"reliability_{scene}{suffix}.npz",
                train_agree=feat.astype("float32"),
                raw_conf=raw_conf.astype("float32"),
                reliability=reliability.astype("float32"),
                opacity=opacity.astype("float32"),
                labeled=labeled,
                label=np.array("color" if suffix == "" else "depth_aware"),
            )


def test_transfer_report_structure_and_rank_invariance(tmp_path):
    _write_synthetic(tmp_path)
    rep = build_transfer_report(tmp_path, alpha=0.1, epsilon=0.6, seed=0)

    assert rep["experiment"] == "P1a_cross_scene_calibrator_transfer"
    for label in ("color", "depth"):
        bl = rep["by_label"][label]
        matrix = bl["matrix"]
        assert len(matrix) == len(SCENES) ** 2  # full 4x4 per label

        diag = [c for c in matrix if c["source"] == c["target"]]
        off = [c for c in matrix if c["source"] != c["target"]]
        assert len(diag) == len(SCENES) and all(c["kind"] == "in_scene" for c in diag)
        assert all(c["kind"] == "transferred" for c in off)

        for c in off:
            # Rank-invariance: a monotone calibrator does not change the target's
            # carrier ordering, so transferred selection AUC ~ in-scene AUC.
            assert abs(c["auc_delta_vs_inscene"]) < 0.05
            # Every transferred certificate is a well-formed object on B's local split.
            cert = c["cert_transferred_local_split"]
            assert 0.0 <= cert["kept_fraction"] <= 1.0
            assert isinstance(cert["certified"], bool)

        s = bl["summary"]
        assert s["n_off_diagonal_pairs"] == len(SCENES) * (len(SCENES) - 1)
        # Transfer still calibrates far better than the raw (saturated) heuristic.
        assert s["offdiag_transferred_ece_mean"] < 0.5 * s["mean_raw_ece"]
        # In-scene calibration is at least as good as any transfer, on average.
        assert s["mean_inscene_ece"] <= s["offdiag_transferred_ece_mean"] + 1e-9


def test_cert_sweep_monotone_and_selectivity_onset(tmp_path):
    _write_synthetic(tmp_path)
    rep = build_cert_sweep_report(
        tmp_path, alpha=0.1, eps_lo=0.30, eps_hi=0.65, eps_step=0.05, seed=0)

    grid = rep["epsilon_grid"]
    assert grid[0] == 0.3 and grid[-1] == 0.65
    for scene in SCENES:
        for label in ("color", "depth"):
            r = rep["by_scene"][scene][label]
            kept = [row["kept_fraction"] for row in r["sweep"]]
            eps = [row["epsilon"] for row in r["sweep"]]
            # kept fraction is monotone non-decreasing in epsilon.
            assert all(b >= a - 1e-9 for a, b in zip(kept, kept[1:]))
            onset = r["selectivity_onset_epsilon"]
            if onset is None:
                assert all(k >= 1.0 - 1e-9 for k in kept)  # never selective
            else:
                # onset is the largest epsilon with kept < 1; strictly above it kept == 1.
                for e, k in zip(eps, kept):
                    if e > onset + 1e-9:
                        assert k >= 1.0 - 1e-9
                assert any(
                    abs(e - onset) < 1e-9 and k < 1.0 - 1e-9
                    for e, k in zip(eps, kept)
                )
