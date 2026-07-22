"""Tests for the real-scene-bound publication gates (aura.publication).

Each rebound gate is checked two ways: it PASSES on the committed outputs/ artifacts
(or the real trained asset) and FAILS on tampered / missing / malformed data. The
real-asset relight and secondary-ray gates run on outputs/truck-sidecar.aura on CPU.
"""
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura import publication as P  # noqa: E402
from aura.carriers import CarrierKind, CarrierSpec, default_registry  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"


@pytest.fixture
def artifacts_dir(tmp_path):
    """A writable copy of the committed calibration artifacts to tamper with."""
    for pattern in ("calib_*.json", "rl_*_fr.json"):
        for src in OUTPUTS.glob(pattern):
            shutil.copy(src, tmp_path / src.name)
    for name in ("cert_sweep.json", "cross_scene_transfer.json", "p2_summary.json",
                 "lod_certified.json"):
        shutil.copy(OUTPUTS / name, tmp_path / name)
    return tmp_path


def _rewrite(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


# --- calibration_ece ------------------------------------------------------------

def test_calibration_ece_gate_passes_on_committed(artifacts_dir):
    gate = P._calibration_ece_gate(artifacts_dir, {})
    assert gate.passed is True
    assert gate.id == "calibration_ece"


def test_calibration_ece_gate_fails_on_regressed_ece(artifacts_dir):
    _rewrite(artifacts_dir / "calib_truck.json",
             lambda d: d["calibration"].__setitem__("ece_calibrated", 0.05))
    gate = P._calibration_ece_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("ECE" in g for g in gate.gaps)


def test_calibration_ece_gate_fails_on_missing_artifact(artifacts_dir):
    (artifacts_dir / "calib_room_fr.json").unlink()
    gate = P._calibration_ece_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("calib_room_fr.json" in g for g in gate.gaps)


def test_calibration_ece_gate_fails_on_leaked_split(artifacts_dir):
    # Inject a leaked split: carriers trained on all frames (train == total).
    def leak(d):
        rs = d["scenes"]["garden"]["fullres_color"]["reliability_summary"]
        rs["train_views"] = rs["train_views"] + rs["test_views"]  # trained on all
    _rewrite(artifacts_dir / "p2_summary.json", leak)
    gate = P._calibration_ece_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("split guard" in g and "leak" in g.lower() for g in gate.gaps)


def test_calibration_ece_gate_fails_on_malformed_json(artifacts_dir):
    (artifacts_dir / "calib_kitchen.json").write_text("{ not json")
    gate = P._calibration_ece_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("malformed" in g for g in gate.gaps)


# --- pruning_certificate --------------------------------------------------------

def test_pruning_certificate_gate_passes_on_committed(artifacts_dir):
    assert P._pruning_certificate_gate(artifacts_dir, {}).passed is True


def test_pruning_certificate_gate_fails_when_not_certified(artifacts_dir):
    def uncertify(d):
        d["by_scene"]["truck"]["color"]["sweep"] = [
            {**r, "certified": False} if abs(r["epsilon"] - 0.6) < 1e-9 else r
            for r in d["by_scene"]["truck"]["color"]["sweep"]
        ]
    _rewrite(artifacts_dir / "cert_sweep.json", uncertify)
    gate = P._pruning_certificate_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("not certified" in g for g in gate.gaps)


def test_pruning_certificate_gate_fails_on_wrong_epsilon(artifacts_dir):
    _rewrite(artifacts_dir / "calib_truck.json",
             lambda d: d["pruning_certificate"].__setitem__("epsilon", 0.9))
    gate = P._pruning_certificate_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("epsilon" in g for g in gate.gaps)


def test_pruning_certificate_gate_fails_on_missing_sweep(artifacts_dir):
    (artifacts_dir / "cert_sweep.json").unlink()
    gate = P._pruning_certificate_gate(artifacts_dir, {})
    assert gate.passed is False


# --- cross_scene_transfer -------------------------------------------------------

def test_cross_scene_transfer_gate_passes_on_committed(artifacts_dir):
    assert P._cross_scene_transfer_gate(artifacts_dir, {}).passed is True


def test_cross_scene_transfer_gate_fails_on_auc_drift(artifacts_dir):
    _rewrite(artifacts_dir / "cross_scene_transfer.json",
             lambda d: d["by_label"]["color"]["matrix"][1].__setitem__("auc_delta_vs_inscene", 0.05))
    gate = P._cross_scene_transfer_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("ΔAUC" in g for g in gate.gaps)


def test_cross_scene_transfer_gate_fails_on_uncertified_transfer(artifacts_dir):
    def kill(d):
        for row in d["by_label"]["color"]["matrix"]:
            if row["source"] != row["target"]:
                row["cert_transferred_local_split"]["certified"] = False
                break
    _rewrite(artifacts_dir / "cross_scene_transfer.json", kill)
    gate = P._cross_scene_transfer_gate(artifacts_dir, {})
    assert gate.passed is False


def test_cross_scene_transfer_gate_fails_on_missing(tmp_path):
    gate = P._cross_scene_transfer_gate(tmp_path, {})
    assert gate.passed is False


# --- fullres_renderloss ---------------------------------------------------------

def test_fullres_renderloss_gate_passes_on_committed(artifacts_dir):
    assert P._fullres_renderloss_gate(artifacts_dir, {}).passed is True


def test_fullres_renderloss_gate_fails_on_decorrelated_fullres(artifacts_dir):
    _rewrite(artifacts_dir / "p2_summary.json",
             lambda d: d["scenes"]["kitchen"]["fullres_color"]["reliability_summary"]
             .__setitem__("corr_trainagree_reliability", 0.2))
    gate = P._fullres_renderloss_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("corr" in g for g in gate.gaps)


def test_fullres_renderloss_gate_flags_truck_leak_fingerprint(artifacts_dir):
    # Truck full-res colour certificate back at 1.00 == the P0 leak returned.
    _rewrite(artifacts_dir / "p2_summary.json",
             lambda d: d["scenes"]["truck"]["fullres_color"]["pruning_certificate"]
             .__setitem__("kept_fraction", 1.0))
    gate = P._fullres_renderloss_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("leak fingerprint" in g for g in gate.gaps)


def test_fullres_renderloss_gate_fails_on_renderloss_collapse(artifacts_dir):
    _rewrite(artifacts_dir / "rl_truck_fr.json",
             lambda d: d.__setitem__("corr_trainagree_reliability", 0.1))
    gate = P._fullres_renderloss_gate(artifacts_dir, {})
    assert gate.passed is False


# --- certified_lod ----------------------------------------------------------------

def test_certified_lod_gate_passes_on_committed(artifacts_dir):
    gate = P._certified_lod_gate(artifacts_dir, {})
    assert gate.passed is True
    assert gate.id == "certified_lod"


def test_certified_lod_gate_fails_when_a_bound_breaks(artifacts_dir):
    def break_bound(d):
        d["by_scene"]["truck"]["levels"][0]["holds"] = False
    _rewrite(artifacts_dir / "lod_certified.json", break_bound)
    gate = P._certified_lod_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("does not hold" in g for g in gate.gaps)


def test_certified_lod_gate_fails_on_broken_bonferroni_accounting(artifacts_dir):
    # alpha_per_level must equal alpha / R over the non-trivial levels — quoting the
    # uncorrected family-wise alpha is optimistic.
    _rewrite(artifacts_dir / "lod_certified.json",
             lambda d: d.__setitem__("alpha_per_level", d["alpha"]))
    gate = P._certified_lod_gate(artifacts_dir, {})
    assert gate.passed is False
    assert any("family-wise accounting" in g for g in gate.gaps)


def test_certified_lod_gate_fails_on_missing(tmp_path):
    gate = P._certified_lod_gate(tmp_path, {})
    assert gate.passed is False


# --- secondary_ray_reflection (real asset) --------------------------------------

@pytest.fixture
def fake_asset_dir(tmp_path):
    """An outputs dir with a placeholder truck-sidecar.aura/carriers.npz.

    The prober-injection tests never load the file — they only need the gate's
    asset-exists pre-check to pass — so these tests stay CI-runnable without the
    gitignored 164 MB real asset (which the *_on_real_asset tests, marked
    local_data, exercise).
    """
    asset = tmp_path / "truck-sidecar.aura"
    asset.mkdir()
    (asset / "carriers.npz").write_bytes(b"")
    return tmp_path


@pytest.mark.local_data
def test_secondary_ray_gate_passes_on_real_asset():
    gate = P._secondary_reflection_gate(OUTPUTS, {})
    assert gate.passed is True
    assert gate.status == "passed"


def test_secondary_ray_gate_requires_gpu_when_probe_cannot_run(fake_asset_dir):
    def broken(_aura):
        raise RuntimeError("torch unavailable")
    gate = P._secondary_reflection_gate(fake_asset_dir, {}, prober=broken)
    assert gate.passed is False
    assert gate.status == "requires_gpu"


def test_secondary_ray_gate_unverified_when_asset_missing(tmp_path):
    gate = P._secondary_reflection_gate(tmp_path, {})
    assert gate.passed is False
    assert gate.status == "unverified"


def test_secondary_ray_gate_fails_on_bad_metrics(fake_asset_dir):
    metrics = {"carriers": 10, "primary_hits": 0,
               "shadow_bounded_rate": 0.0, "reflection_finite_rate": 0.0}
    gate = P._secondary_reflection_gate(fake_asset_dir, {}, prober=lambda _a: metrics)
    assert gate.passed is False
    assert gate.status == "failed"


# --- inverse_materials (real asset relight) -------------------------------------

@pytest.mark.local_data
def test_inverse_materials_gate_passes_on_real_asset():
    gate = P._inverse_materials_gate(OUTPUTS, {})
    assert gate.passed is True
    assert gate.status == "passed"


def test_inverse_materials_gate_requires_gpu_when_probe_cannot_run(fake_asset_dir):
    def broken(_aura):
        raise RuntimeError("torch unavailable")
    gate = P._inverse_materials_gate(fake_asset_dir, {}, prober=broken)
    assert gate.passed is False
    assert gate.status == "requires_gpu"


def test_inverse_materials_gate_fails_when_light_does_not_change_output(fake_asset_dir):
    metrics = {"carriers": 10, "light_delta": 0.0, "albedo_delta": 0.0,
               "finite": True, "nonnegative": True}
    gate = P._inverse_materials_gate(fake_asset_dir, {}, prober=lambda _a: metrics)
    assert gate.passed is False
    assert gate.status == "failed"


def test_inverse_materials_gate_unverified_when_asset_missing(tmp_path):
    gate = P._inverse_materials_gate(tmp_path, {})
    assert gate.passed is False
    assert gate.status == "unverified"


# --- carrier_registry_honesty ---------------------------------------------------

def test_carrier_registry_honesty_gate_passes_on_committed(artifacts_dir):
    gate = P._carrier_registry_honesty_gate(artifacts_dir, {})
    assert gate.passed is True
    assert gate.id == "carrier_registry_honesty"


def test_carrier_registry_honesty_gate_fails_when_demo_carrier_claims_trained(artifacts_dir):
    reg = dict(default_registry())
    reg["gabor"] = dataclasses.replace(reg["gabor"], maturity="trained")  # over-claim
    gate = P._carrier_registry_honesty_gate(artifacts_dir, {}, registry=reg)
    assert gate.passed is False
    assert any("gabor" in g and "trained" in g for g in gate.gaps)


def test_carrier_registry_honesty_gate_fails_on_type_absent_from_contract(artifacts_dir):
    reg = dict(default_registry())
    reg["myst"] = CarrierSpec(id="myst", kind=CarrierKind.GAUSSIAN_FALLBACK,
                              description="unlisted", primary_render=True, ray_query=True,
                              maturity="trained")
    gate = P._carrier_registry_honesty_gate(artifacts_dir, {}, registry=reg)
    assert gate.passed is False
    assert any("myst" in g and "contract" in g for g in gate.gaps)


def test_carrier_registry_honesty_gate_fails_when_trained_evidence_missing(tmp_path):
    # Default registry (gaussian/beta = trained) but no committed calib_*.json here.
    gate = P._carrier_registry_honesty_gate(tmp_path, {})
    assert gate.passed is False
    assert any("evidence missing" in g for g in gate.gaps)


def test_carrier_registry_honesty_gate_is_registered_in_report():
    report = P.publication_validation_report()
    assert "carrier_registry_honesty" in {g.id for g in report.gates}
