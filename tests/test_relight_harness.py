"""CI-safe tests for the relighting benchmark harness.

Covers: metric correctness on known inputs, the --smoke full-pipeline run
(JSON + Markdown writers), and missing-dataset exit(2) behavior. No dataset,
no GPU, no network (LPIPS is disabled via env for hermeticity).
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the harness module by path (it lives under experiments/, not a package).
_HARNESS_PATH = Path(__file__).resolve().parent.parent / "experiments" / "relight_benchmark_harness.py"
_spec = importlib.util.spec_from_file_location("relight_benchmark_harness", _HARNESS_PATH)
H = importlib.util.module_from_spec(_spec)
sys.modules["relight_benchmark_harness"] = H
_spec.loader.exec_module(H)


@pytest.fixture(autouse=True)
def _disable_lpips(monkeypatch):
    """Keep every test hermetic: never touch the network for LPIPS weights."""
    monkeypatch.setenv(H.DISABLE_LPIPS_ENV, "1")


# --------------------------------------------------------------------------- #
# Metric correctness on known inputs
# --------------------------------------------------------------------------- #
def test_psnr_identical_is_inf():
    img = np.random.default_rng(0).random((16, 16, 3))
    assert H.psnr(img, img) == float("inf")


def test_psnr_known_mse():
    # constant offset of 0.1 over the whole image -> MSE = 0.01 -> PSNR = 20 dB.
    gt = np.full((8, 8, 3), 0.5)
    pred = gt + 0.1
    assert H.psnr(pred, gt) == pytest.approx(20.0, abs=1e-9)


def test_psnr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        H.psnr(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


def test_ssim_identical_is_one():
    img = np.random.default_rng(1).random((32, 32, 3))
    assert H.ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssim_degrades_for_noisy():
    rng = np.random.default_rng(2)
    gt = rng.random((32, 32, 3))
    noisy = np.clip(gt + rng.normal(0, 0.3, gt.shape), 0, 1)
    assert H.ssim(noisy, gt) < 0.99
    assert -1.0 <= H.ssim(noisy, gt) <= 1.0


def test_normal_mae_identical_is_zero():
    rng = np.random.default_rng(3)
    n = rng.normal(0, 1, (10, 3))
    assert H.normal_mean_angular_error(n, n) == pytest.approx(0.0, abs=1e-6)


def test_normal_mae_known_ninety_degrees():
    pred = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert H.normal_mean_angular_error(pred, gt) == pytest.approx(90.0, abs=1e-6)


def test_normal_mae_known_angle():
    # 45 degrees between the two unit vectors.
    pred = np.array([[1.0, 0.0, 0.0]])
    gt = np.array([[1.0, 1.0, 0.0]])
    assert H.normal_mean_angular_error(pred, gt) == pytest.approx(45.0, abs=1e-6)


def test_normal_mae_signed_vs_unsigned():
    # Opposite vectors: signed error 180 deg, unsigned error 0 (sign folded away).
    pred = np.array([[0.0, 0.0, 1.0]])
    gt = np.array([[0.0, 0.0, -1.0]])
    assert H.normal_mean_angular_error(pred, gt, signed=True) == pytest.approx(180.0, abs=1e-6)
    assert H.normal_mean_angular_error(pred, gt, signed=False) == pytest.approx(0.0, abs=1e-6)


def test_normal_mae_mask_selects_subset():
    pred = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2nd is 90 deg off
    mask = np.array([True, False])                     # keep only the aligned one
    assert H.normal_mean_angular_error(pred, gt, mask=mask) == pytest.approx(0.0, abs=1e-6)


def test_lpips_disabled_is_honest():
    a = np.random.default_rng(4).random((8, 8, 3))
    val, backend = H.lpips_distance(a, a)
    assert val is None and backend == "disabled"


# --------------------------------------------------------------------------- #
# Per-sample evaluation + aggregation
# --------------------------------------------------------------------------- #
def test_evaluate_sample_emits_all_branches():
    s = H.make_smoke_samples(n=1)[0]
    m = H.evaluate_sample(s)
    for key in ("relit_psnr", "relit_ssim", "relit_lpips", "relit_lpips_backend",
                "albedo_psnr", "normal_mae_deg", "normal_mae_deg_unsigned"):
        assert key in m
    assert math.isfinite(m["relit_psnr"])
    assert m["relit_lpips_backend"] == "disabled"
    # unsigned MAE cannot exceed signed MAE for the same field.
    assert m["normal_mae_deg_unsigned"] <= m["normal_mae_deg"] + 1e-9


def test_aggregate_means_finite_only():
    rows = [
        {"relit_psnr": 20.0, "relit_lpips": None, "relit_lpips_backend": "disabled"},
        {"relit_psnr": float("inf"), "relit_lpips": None, "relit_lpips_backend": "disabled"},
        {"relit_psnr": 30.0},
    ]
    agg = H.aggregate(rows)
    assert agg["relit_psnr"] == pytest.approx(25.0)  # inf and None dropped
    assert "relit_lpips" not in agg  # all None -> absent


# --------------------------------------------------------------------------- #
# --smoke full pipeline (writers)
# --------------------------------------------------------------------------- #
def test_smoke_runs_full_pipeline_and_writes(tmp_path):
    jp = tmp_path / "out" / "report.json"
    mp = tmp_path / "out" / "report.md"
    rc = H.main(["--smoke", "--output-json", str(jp), "--output-md", str(mp)])
    assert rc == H.EXIT_OK
    assert jp.is_file() and mp.is_file()

    report = json.loads(jp.read_text())
    assert report["format"] == "AURA_RELIGHT_BENCHMARK_REPORT"
    assert report["dataset"] == "smoke"
    assert report["nSamples"] == 3
    agg = report["aggregate"]
    for key in ("relit_psnr", "relit_ssim", "albedo_psnr",
                "normal_mae_deg", "normal_mae_deg_unsigned"):
        assert key in agg and math.isfinite(agg[key])
    assert report["lpipsBackends"] == ["disabled"]

    md = mp.read_text()
    assert "AURA relighting benchmark" in md
    assert "Relit PSNR" in md


# --------------------------------------------------------------------------- #
# Missing-dataset exit(2) + usage
# --------------------------------------------------------------------------- #
def test_missing_dataset_exits_2(monkeypatch):
    monkeypatch.delenv(H.TENSOIR_ENV, raising=False)
    with pytest.raises(SystemExit) as ei:
        H.resolve_dataset_root("tensoir", env={})
    assert ei.value.code == H.EXIT_DATASET_MISSING


def test_dataset_root_pointing_nowhere_exits_2():
    with pytest.raises(SystemExit) as ei:
        H.resolve_dataset_root("stanford_orb", env={H.STANFORD_ORB_ENV: "/no/such/dir/xyz"})
    assert ei.value.code == H.EXIT_DATASET_MISSING


def test_main_requires_dataset_without_smoke():
    assert H.main([]) == H.EXIT_USAGE


def test_main_missing_dataset_exit_2(monkeypatch):
    monkeypatch.delenv(H.TENSOIR_ENV, raising=False)
    with pytest.raises(SystemExit) as ei:
        H.main(["--dataset", "tensoir"])
    assert ei.value.code == H.EXIT_DATASET_MISSING


# --------------------------------------------------------------------------- #
# End-to-end manifest loading against a tiny on-disk fixture (npy, hermetic)
# --------------------------------------------------------------------------- #
def test_manifest_loading_and_scoring(tmp_path):
    root = tmp_path / "ds"
    (root / "lego").mkdir(parents=True)
    (root / "preds").mkdir()
    rng = np.random.default_rng(7)
    gt = rng.random((16, 16, 3))
    pred = np.clip(gt + rng.normal(0, 0.02, gt.shape), 0, 1)
    np.save(root / "lego" / "gt.npy", gt)
    np.save(root / "preds" / "pred.npy", pred)
    manifest = {
        "dataset": "tensoir",
        "samples": [
            {"scene": "lego", "view": "r0",
             "relit_gt": "lego/gt.npy", "relit_pred": "preds/pred.npy"},
        ],
    }
    (root / "aura_relight_eval.json").write_text(json.dumps(manifest))

    samples = H.load_manifest_samples("tensoir", root)
    assert len(samples) == 1
    row = H.evaluate_sample(samples[0])
    assert math.isfinite(row["relit_psnr"]) and row["relit_psnr"] > 20
    # no albedo/normal GT in this manifest -> those metrics absent
    assert "albedo_psnr" not in row and "normal_mae_deg" not in row


def test_manifest_absent_returns_empty(tmp_path):
    root = tmp_path / "ds_empty"
    root.mkdir()
    assert H.load_manifest_samples("tensoir", root) == []
