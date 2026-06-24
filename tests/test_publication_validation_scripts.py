import json
import subprocess
import sys


def test_secondary_reflection_validation_script_writes_passing_artifact(tmp_path):
    out = tmp_path / "secondary.json"
    subprocess.run(
        [sys.executable, "experiments/secondary_reflection_validation.py", "--out", str(out)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["format"] == "AURA_SECONDARY_RAY_REFLECTION_VALIDATION"
    assert payload["passed"] is True
    assert payload["shadowTransmittanceReadyRate"] == 1.0
    assert payload["reflectionVectorReadyRate"] > 0.0


def test_inverse_material_validation_script_writes_passing_artifact(tmp_path):
    out = tmp_path / "materials.json"
    subprocess.run(
        [sys.executable, "experiments/inverse_material_validation.py", "--out", str(out), "--device", "cpu"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["format"] == "AURA_INVERSE_MATERIAL_VALIDATION"
    assert payload["passed"] is True
    assert payload["albedoSource"] == "explicit_payload"
    assert payload["roughnessSource"] == "explicit_payload"
    assert payload["metallicSource"] == "explicit_payload"
    assert payload["differentLightingChangesOutput"] is True


def test_external_baseline_smokes_merge_keeps_missing_official_methods(tmp_path):
    sys.path.insert(0, "experiments")
    from external_baseline_smokes import merge_smoke_baselines

    payload = {
        "baselines": {
            "3dgs": {"label": "existing 3DGS"},
        },
        "missingBaselines": ["colmap", "nerf", "2dgs", "ray_traced_gs"],
    }
    merged = merge_smoke_baselines(
        payload,
        {
            "colmap": {"label": "COLMAP smoke"},
            "nerf": {"label": "compact NeRF smoke"},
        },
    )

    assert merged["baselines"]["colmap"]["label"] == "COLMAP smoke"
    assert merged["baselines"]["nerf"]["label"] == "compact NeRF smoke"
    assert merged["missingBaselines"] == ["2dgs", "ray_traced_gs"]


def test_external_baseline_entries_include_2dgs_and_ray_traced_gs(tmp_path):
    sys.path.insert(0, "experiments")
    from external_baseline_smokes import baseline_entries

    smoke = {
        "scene": "truck",
        "device": "cuda",
        "colmap": {"frames": 1, "scale": 0.25, "psnr": 9.0, "ssim": 0.1, "lpips": 0.7},
        "nerf": {"frames": 1, "scale": 0.25, "psnr": 8.5, "ssim": 0.1, "lpips": 0.9, "iterations": 1},
        "two_dgs": {"frames": 1, "scale": 0.25, "psnr": 10.0, "ssim": 0.2, "lpips": 0.6},
        "ray_traced_gs": {"frames": 1, "scale": 0.25, "psnr": 11.0, "ssim": 0.3, "lpips": 0.5},
    }
    entries = baseline_entries(smoke, tmp_path / "smoke.json")

    assert {"colmap", "nerf", "2dgs", "ray_traced_gs"}.issubset(entries)
    assert entries["2dgs"]["sourceType"] == "same_split_cuda_2dgs_style_surfel_smoke"
    assert entries["ray_traced_gs"]["sourceType"] == "same_split_cuda_ray_traced_gs_style_smoke"
    assert entries["2dgs"]["device"] == "cuda"
    assert entries["ray_traced_gs"]["device"] == "cuda"
