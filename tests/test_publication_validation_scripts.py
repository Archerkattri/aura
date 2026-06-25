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


def test_cuda_production_validation_rejects_fallback_payload():
    sys.path.insert(0, "experiments")
    from cuda_production_backend_validation import summarize_cuda_gate

    report = summarize_cuda_gate(
        compiled_cuda_dispatch=False,
        fallback_used=True,
        device="cpu",
        max_abs_error=0.0,
        parity_threshold=0.001,
        rays_per_second=10.0,
        min_rays_per_second=1.0,
    )

    assert report["passed"] is False
    assert "compiled CUDA dispatch was not used" in report["failures"]
    assert "fallback backend was used" in report["failures"]


def test_cuda_production_validation_accepts_compiled_payload():
    sys.path.insert(0, "experiments")
    from cuda_production_backend_validation import summarize_cuda_gate

    report = summarize_cuda_gate(
        compiled_cuda_dispatch=True,
        fallback_used=False,
        device="cuda",
        max_abs_error=0.0001,
        parity_threshold=0.001,
        rays_per_second=1000.0,
        min_rays_per_second=1.0,
    )

    assert report["passed"] is True
    assert report["failures"] == []


def test_engine_integration_validation_writes_engine_artifacts(tmp_path):
    sys.path.insert(0, "experiments")
    from engine_integration_validation import validate_engine_exports

    out = tmp_path / "engine.json"
    artifact_dir = tmp_path / "exports"
    payload = validate_engine_exports(out, artifact_dir)

    assert payload["format"] == "AURA_ENGINE_INTEGRATION_VALIDATION"
    assert payload["passed"] is True
    assert payload["gltf"]["glbMagicValid"] is True
    assert payload["gltf"]["usesKHRGaussianSplatting"] is True
    assert payload["usd"]["hasGaussianPointsPrim"] is True
    assert payload["runtime"]["engineWorkflow"]["nativeRuntimeReady"] is True


def test_viewer_compatibility_validation_accepts_engine_exports(tmp_path):
    sys.path.insert(0, "experiments")
    from engine_integration_validation import validate_engine_exports
    from viewer_compatibility_validation import validate_viewer_compatibility

    validate_engine_exports(tmp_path / "engine.json", tmp_path / "exports")
    payload = validate_viewer_compatibility(
        tmp_path / "exports/aura_splat.glb",
        tmp_path / "exports/aura_scene.usda",
        tmp_path / "viewer.json",
    )

    assert payload["format"] == "AURA_VIEWER_COMPATIBILITY_VALIDATION"
    assert payload["passed"] is True
    assert payload["gltf"]["usesKHRGaussianSplatting"] is True
    assert payload["gltf"]["requiredAttributesPresent"] is True
    assert payload["usd"]["balancedBraces"] is True


def test_real_scene_fps_sweep_fps_helper():
    sys.path.insert(0, "experiments")
    from real_scene_fps_sweep import _fps

    assert _fps(2.0) == 500.0
    assert _fps(0.0) == 0.0


def test_collect_official_multiscene_baselines_writes_missing_rows(tmp_path):
    sys.path.insert(0, "experiments")
    from collect_official_multiscene_baselines import collect_official_multiscene_baselines

    payload = collect_official_multiscene_baselines(tmp_path / "official.json")

    assert payload["format"] == "AURA_OFFICIAL_MULTISCENE_BASELINES"
    assert "official_2dgs" in payload["completedSceneCounts"]
    assert "official_3dgut" in payload["missing"]


def test_native_real_capture_validation_rejects_incomplete_audit():
    sys.path.insert(0, "experiments")
    from native_real_capture_validation import summarize_native_gate

    report = summarize_native_gate(
        audit={
            "local_scene_count": 2,
            "complete": False,
            "missing": ["room"],
            "scenes": [
                {"scene": "truck", "has_beta": True, "has_gaussian": True, "delta_psnr": 0.1},
                {"scene": "room", "has_beta": True, "has_gaussian": False, "delta_psnr": None},
            ],
        },
        multiscene={"scenes": [{"scene": "truck", "delta_psnr": 0.1}], "mean_delta_psnr": 0.1},
        min_scene_count=2,
    )

    assert report["passed"] is False
    assert "not all locally downloaded scenes have complete Beta/Gaussian metrics" in report["failures"]


def test_native_real_capture_validation_accepts_complete_audit():
    sys.path.insert(0, "experiments")
    from native_real_capture_validation import summarize_native_gate

    report = summarize_native_gate(
        audit={
            "local_scene_count": 2,
            "complete": True,
            "missing": [],
            "scenes": [
                {"scene": "truck", "has_beta": True, "has_gaussian": True, "delta_psnr": 0.1},
                {"scene": "room", "has_beta": True, "has_gaussian": True, "delta_psnr": 0.2},
            ],
        },
        multiscene={
            "scenes": [
                {"scene": "truck", "delta_psnr": 0.1},
                {"scene": "room", "delta_psnr": 0.2},
            ],
            "mean_delta_psnr": 0.15,
        },
        min_scene_count=2,
    )

    assert report["passed"] is True
    assert report["allLocalScenesComplete"] is True
    assert report["sceneCount"] == 2


def test_torch_backend_validation_rejects_cpu_or_empty_payload():
    sys.path.insert(0, "experiments")
    from torch_backend_validation import summarize_torch_backend_gate

    report = summarize_torch_backend_gate(
        device="cpu",
        manifest_frame_count=251,
        manifest_region_count=129531,
        loaded_frame_count=1,
        scene_element_count=2048,
        packed_batch_count=1,
        packed_target_count=64,
        max_batch_target_count=64,
        finite_losses=True,
        render_seconds=0.1,
        max_allowed_batch_targets=256,
        min_manifest_regions=1000,
        min_packed_targets=32,
    )

    assert report["passed"] is False
    assert "torch backend did not run on cuda" in report["failures"]


def test_torch_backend_validation_accepts_cuda_real_capture_payload():
    sys.path.insert(0, "experiments")
    from torch_backend_validation import summarize_torch_backend_gate

    report = summarize_torch_backend_gate(
        device="cuda",
        manifest_frame_count=251,
        manifest_region_count=129531,
        loaded_frame_count=1,
        scene_element_count=2048,
        packed_batch_count=1,
        packed_target_count=64,
        max_batch_target_count=64,
        finite_losses=True,
        render_seconds=0.1,
        max_allowed_batch_targets=256,
        min_manifest_regions=1000,
        min_packed_targets=32,
    )

    assert report["passed"] is True
    assert report["failures"] == []
