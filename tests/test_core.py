import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from aura import (
    Bounds,
    Ray,
    ReconstructionConfig,
    RegionEvidence,
    RenderTarget,
    TrainingFrame,
    TrainingRegion,
    load_package,
    load_training_dataset,
    load_training_frames,
    reconstruct_demo_scene,
    synthetic_training_dataset,
    synthetic_training_frames,
    validate_training_dataset_document,
    write_synthetic_training_frames,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_synthetic_training_frames_are_posed_native_inputs():
    frames = synthetic_training_frames()

    assert len(frames) == 4
    assert all(frame.target_depth > 0.0 for frame in frames)
    assert {frame.semantic_label for frame in frames} >= {"wall", "fixture_object"}


def test_training_frames_round_trip_through_json(tmp_path):
    output = write_synthetic_training_frames(tmp_path / "frames.json")
    loaded = load_training_frames(output)
    dataset = load_training_dataset(output)
    fixture = load_training_dataset(FIXTURE_DIR / "training_frames.json")

    assert loaded == synthetic_training_frames()
    assert dataset == synthetic_training_dataset()
    assert fixture == synthetic_training_dataset()


def test_training_dataset_schema_accepts_native_fixture_contract():
    validate_training_dataset_document(synthetic_training_dataset().to_dict())
    validate_training_dataset_document(json.loads((FIXTURE_DIR / "training_frames.json").read_text(encoding="utf-8")))


def test_training_dataset_schema_rejects_missing_regions(tmp_path):
    path = tmp_path / "missing_regions.json"
    payload = synthetic_training_dataset().to_dict()
    payload.pop("regions")
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_training_dataset(path)
    except ValueError as exc:
        assert "training_dataset.schema.json validation failed" in str(exc)
        assert "'regions' is a required property" in str(exc)
    else:
        raise AssertionError("missing regions should fail schema validation")


def test_training_dataset_schema_rejects_invalid_region_opacity(tmp_path):
    path = tmp_path / "invalid_opacity.json"
    payload = synthetic_training_dataset().to_dict()
    payload["regions"][0]["opacity"] = 1.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_training_dataset(path)
    except ValueError as exc:
        assert "training_dataset.schema.json validation failed at regions.0.opacity" in str(exc)
        assert "greater than the maximum of 1" in str(exc)
    else:
        raise AssertionError("invalid opacity should fail schema validation")


def test_training_dataset_rejects_regions_with_unknown_frame_ids(tmp_path):
    path = tmp_path / "unknown_frame.json"
    payload = synthetic_training_dataset().to_dict()
    payload["regions"][0]["frame_id"] = "not_a_frame"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_training_dataset(path)
    except ValueError as exc:
        assert "training regions reference unknown frame ids: not_a_frame" in str(exc)
    else:
        raise AssertionError("unknown frame references should fail dataset validation")


def test_reconstruct_demo_accepts_data_driven_regions_without_fixture_ids():
    frames = (
        TrainingFrame(
            id="capture_a",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.7, 0.6, 0.5),
            target_depth=2.0,
            semantic_label="panel",
        ),
    )
    regions = (
        TrainingRegion(
            id="custom_surface",
            frame_id="capture_a",
            bounds=Bounds((-0.25, -0.25, 0.0), (0.25, 0.25, 0.1)),
            evidence=RegionEvidence(geometry_confidence=0.9, edit_need=0.6),
            opacity=0.8,
            normal=(0.0, 0.0, -1.0),
        ),
    )

    result = reconstruct_demo_scene(
        ReconstructionConfig(iterations=2, enable_adaptive_evolution=False),
        frames=frames,
        regions=regions,
        name="custom_reconstruct",
    )

    assert result.scene.name == "custom_reconstruct"
    assert result.scene.elements[0].id == "custom_surface"
    assert result.scene.elements[0].carrier_id == "surface"
    assert result.report.frames == frames


def test_reconstruct_demo_uses_explicit_capture_pixel_targets():
    frames = (
        TrainingFrame(
            id="capture_a",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.2, 0.2, 0.2),
            target_depth=2.0,
        ),
    )
    regions = (
        TrainingRegion(
            id="custom_surface",
            frame_id="capture_a",
            bounds=Bounds((-0.25, -0.25, 0.0), (0.25, 0.25, 0.1)),
            evidence=RegionEvidence(geometry_confidence=0.9, edit_need=0.6),
            opacity=1.0,
            normal=(0.0, 0.0, -1.0),
        ),
    )
    render_targets = (
        RenderTarget(
            frame_id="capture_a",
            ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(1.0, 0.0, 0.0),
            target_depth=2.0,
            target_normal=(0.0, 0.0, -1.0),
        ),
        RenderTarget(
            frame_id="capture_a",
            ray=Ray(origin=(0.1, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(0.0, 1.0, 0.0),
            target_depth=2.0,
        ),
    )

    result = reconstruct_demo_scene(
        ReconstructionConfig(iterations=1, enable_adaptive_evolution=False),
        frames=frames,
        regions=regions,
        render_targets=render_targets,
        name="capture_pixels",
    )
    report = result.report.to_dict()

    assert "capture_tensor_pixel_targets" in report["stages"]
    assert "capture_tensor_pixels" in report["sources"]
    assert len(report["iterations"][0]["predictions"]) == 2
    assert report["iterations"][0]["normal_loss"] == 0.0
    assert [item["target_color"] for item in report["iterations"][0]["predictions"]] == [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    assert report["iterations"][0]["predictions"][0]["target_normal"] == (0.0, 0.0, -1.0)
    assert report["iterations"][0]["predictions"][0]["normal_loss"] == 0.0


def test_reconstruct_demo_rejects_render_targets_for_unknown_frames():
    with pytest.raises(ValueError, match="render targets reference unknown frame ids"):
        reconstruct_demo_scene(
            ReconstructionConfig(iterations=1),
            render_targets=(
                RenderTarget(
                    frame_id="missing",
                    ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                    target_color=(1.0, 1.0, 1.0),
                    target_depth=2.0,
                ),
            ),
        )


def test_reconstruct_demo_builds_native_aura_core_scene_without_3dgs():
    frames = load_training_frames(FIXTURE_DIR / "training_frames.json")
    result = reconstruct_demo_scene(ReconstructionConfig(iterations=3, render_width=8, render_height=8), frames=frames)
    report = result.report.to_dict()

    assert result.scene.name == "reconstruct_demo"
    assert result.scene.carrier_ids() == ["beta", "gabor", "gaussian", "neural", "semantic", "surface", "volume"]
    by_id = {element.id: element for element in result.scene.elements}
    assert by_id["surface_wall"].metadata["source"] == "aura-core-training-region"
    assert by_id["surface_wall"].metadata["confidence_updated_by"] == "aura-core-residual-confidence"
    assert by_id["surface_wall"].confidence <= 1.0
    assert by_id["surface_wall"].confidence_map["optimization_image_loss"] >= 0.0
    assert by_id["surface_wall"].confidence_map["optimization_depth_loss"] >= 0.0
    assert by_id["surface_wall"].confidence_map["optimization_query_loss"] == 0.0
    assert 0.0 <= by_id["surface_wall"].confidence_map["optimization_residual"] <= 1.0
    assert by_id["soft_volume_beta_detail"].metadata["source"] == "aura-core-adaptive-evolution"
    assert by_id["soft_volume_beta_detail"].carrier_id == "beta"
    assert by_id["soft_volume_beta_detail"].payload["type"] == "beta_kernel"
    assert by_id["soft_volume_beta_detail"].confidence_map["query"] == 0.0
    assert "optimization_residual" in by_id["soft_volume_beta_detail"].confidence_map
    assert by_id["semantic_object_neural_residual"].metadata["source"] == "aura-core-adaptive-evolution"
    assert by_id["semantic_object_neural_residual"].carrier_id == "neural"
    assert by_id["semantic_object_neural_residual"].payload["type"] == "neural_residual"
    assert by_id["semantic_object_neural_residual"].confidence_map["query"] == 0.0
    chunk_members = tuple(element_id for chunk in result.scene.chunks for element_id in chunk.element_ids)
    assert set(chunk_members) == set(by_id)
    assert len(chunk_members) == len(by_id)
    chunk_by_id = {chunk.id: chunk for chunk in result.scene.chunks}
    assert "soft_volume_beta_detail" in chunk_by_id["detail_beta_lod1"].element_ids
    assert "semantic_object_neural_residual" in chunk_by_id["residual_neural_lod1"].element_ids
    assert report["format"] == "AURA_CORE_RECONSTRUCTION_REPORT"
    assert report["evolutionPolicy"]["enabled"] is True
    assert report["renderingPolicy"] == {
        "requestedBackend": "cpu",
        "requestedDevice": None,
        "requireCuda": False,
    }
    assert report["renderBackend"] == "cpu"
    assert report["renderDevice"] is None
    assert report["iterations"][0]["render_backend"] == "cpu"
    assert report["iterations"][0]["render_device"] is None
    assert report["evolutionPolicy"]["splitImageLossThreshold"] == 0.03
    assert report["evolutionPolicy"]["demoteAfterIteration"] == 3
    assert report["sources"] == ["posed_training_frames", "training_regions", "depth_targets", "semantic_labels"]
    assert "native_evidence_initialization" in report["stages"]
    assert "cpu_differentiable_reference_render" in report["stages"]
    assert report["nativeCarrierFraction"] > 0.8
    assert len(report["iterations"]) == 3
    assert report["iterations"][-1]["image_loss"] < report["iterations"][0]["image_loss"]
    assert report["iterations"][-1]["depth_loss"] < report["iterations"][0]["depth_loss"]
    assert report["iterations"][-1]["total_loss"] < report["iterations"][0]["total_loss"]
    assert report["iterations"][0]["query_loss"] == 0.0
    assert report["iterations"][-1]["query_loss"] == 0.0
    assert len(report["iterations"][0]["predictions"]) == len(report["frames"])
    assert {item["carrier_id"] for item in report["iterations"][0]["predictions"]} >= {"surface", "volume", "gabor", "semantic"}
    assert {item["action"] for item in report["iterations"][0]["carrier_evolution"]} >= {
        "split_beta_detail",
        "promote_neural_residual",
    }
    assert {item["created_element_id"] for item in report["iterations"][0]["carrier_evolution"]} >= {
        "soft_volume_beta_detail",
        "semantic_object_neural_residual",
    }
    assert report["iterations"][0]["carrier_evolution_report"]["actionCounts"]["split_beta_detail"] >= 2
    assert report["iterations"][0]["carrier_evolution_report"]["actionCounts"]["promote_neural_residual"] == 1
    assert set(report["iterations"][0]["carrier_evolution_report"]["createdElementIds"]) >= {
        "soft_volume_beta_detail",
        "semantic_object_neural_residual",
    }
    split_decision = next(
        item for item in report["iterations"][0]["carrier_evolution"] if item["action"] == "split_beta_detail"
    )
    assert split_decision["metrics"]["imageLoss"] > split_decision["thresholds"]["splitImageLossThreshold"]
    assert all("ray_direction" in item for item in report["iterations"][0]["predictions"])
    assert all("color_gradient" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_transmittance" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_opacity" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_confidence" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_normal" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_material_id" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_semantic_id" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_residual" in item for item in report["iterations"][0]["predictions"])
    assert all("predicted_provenance" in item for item in report["iterations"][0]["predictions"])
    assert all("target_semantic_id" in item for item in report["iterations"][0]["predictions"])
    assert all("target_material_id" in item for item in report["iterations"][0]["predictions"])
    assert all("query_loss" in item for item in report["iterations"][0]["predictions"])
    assert all("target_normal" in item for item in report["iterations"][0]["predictions"])
    assert all("normal_loss" in item for item in report["iterations"][0]["predictions"])
    assert all(0.0 <= item["predicted_transmittance"] <= 1.0 for item in report["iterations"][0]["predictions"])
    assert all(0.0 <= item["predicted_opacity"] <= 1.0 for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_material_id"] == "mat_wall_plaster" for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_semantic_id"] == "wall" for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_semantic_id"] == "fixture_object" for item in report["iterations"][0]["predictions"])
    assert any(item["target_semantic_id"] == "wall" for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_provenance"] == item["element_id"] for item in report["iterations"][0]["predictions"])
    assert any(item["gradient_norm"] > 0.0 for item in report["iterations"][0]["predictions"])


def test_reconstruct_demo_rejects_cpu_backend_when_cuda_is_required():
    with pytest.raises(ValueError, match="require_cuda"):
        ReconstructionConfig(render_backend="cpu", require_cuda=True)


def test_reconstruct_demo_reports_torch_backend_when_requested_if_available(monkeypatch):
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(RuntimeError, match="torch"):
            reconstruct_demo_scene(ReconstructionConfig(iterations=1, render_backend="torch", torch_device="cpu"))
        return

    import aura.torch_renderer as torch_renderer_module

    def fail_render_target_wrapper(*_args, **_kwargs):
        raise AssertionError("torch reconstruction should use tensor targets directly")

    monkeypatch.setattr(torch_renderer_module, "torch_render_targets", fail_render_target_wrapper)

    result = reconstruct_demo_scene(ReconstructionConfig(iterations=1, render_backend="torch", torch_device="cpu"))
    report = result.report.to_dict()

    assert report["renderBackend"] == "torch"
    assert report["renderDevice"] == "cpu"
    assert report["renderingPolicy"]["requestedBackend"] == "torch"
    assert "torch_native_tensor_render" in report["stages"]
    assert report["iterations"][0]["render_backend"] == "torch"
    assert report["iterations"][0]["render_device"] == "cpu"
    assert report["iterations"][0]["predictions"]


def test_reconstruct_demo_exposes_configurable_evolution_thresholds():
    config = ReconstructionConfig(iterations=2, split_image_loss_threshold=1.0)
    result = reconstruct_demo_scene(config)
    report = result.report.to_dict()
    created = {
        item["created_element_id"]
        for step in report["iterations"]
        for item in step["carrier_evolution"]
        if item["created_element_id"] is not None
    }

    assert report["evolutionPolicy"]["splitImageLossThreshold"] == 1.0
    assert "soft_volume_beta_detail" not in created
    assert "semantic_object_neural_residual" not in created


def test_reconstruction_config_rejects_invalid_evolution_thresholds():
    with pytest.raises(ValueError, match="split_image_loss_threshold"):
        ReconstructionConfig(split_image_loss_threshold=-0.1)
    with pytest.raises(ValueError, match="demote_after_iteration"):
        ReconstructionConfig(demote_after_iteration=-1)


def test_reconstruct_demo_merges_and_demotes_converged_adaptive_children():
    result = reconstruct_demo_scene(ReconstructionConfig(iterations=6, render_width=8, render_height=8))
    by_id = {element.id: element for element in result.scene.elements}
    actions = [item["action"] for step in result.report.to_dict()["iterations"] for item in step["carrier_evolution"]]

    assert "soft_volume_beta_detail" not in by_id
    assert by_id["soft_volume"].metadata["simplification"] == "merge_beta_detail"
    assert by_id["soft_volume"].metadata["simplified_child"] == "soft_volume_beta_detail"
    assert "semantic_object_neural_residual" not in by_id
    assert by_id["semantic_object"].metadata["simplification"] == "demote_neural_residual"
    assert by_id["semantic_object"].metadata["simplified_child"] == "semantic_object_neural_residual"
    assert "merge_beta_detail" in actions
    assert "demote_neural_residual" in actions
    reports = [step["carrier_evolution_report"] for step in result.report.to_dict()["iterations"]]
    assert any("soft_volume_beta_detail" in report["removedElementIds"] for report in reports)
    assert any("semantic_object_neural_residual" in report["removedElementIds"] for report in reports)


def test_reconstruct_demo_cli_writes_package_and_training_report(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "reconstruct-demo",
            "--output-dir",
            str(tmp_path),
            "--frames",
            str(FIXTURE_DIR / "training_frames.json"),
            "--iterations",
            "3",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    package = load_package(tmp_path)
    report = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))

    assert str(tmp_path) in result.stdout
    assert package.asset.name == "reconstruct_demo"
    assert package.scene.carrier_ids() == ["beta", "gabor", "gaussian", "neural", "semantic", "surface", "volume"]
    assert {element.id for element in package.scene.elements} >= {"soft_volume_beta_detail", "semantic_object_neural_residual"}
    assert report["name"] == "reconstruct_demo"
    assert report["format"] == "AURA_CORE_RECONSTRUCTION_REPORT"
    assert report["iterations"][0]["predictions"]
    assert report["iterations"][0]["carrier_evolution"]


def test_reconstruct_demo_cli_accepts_adaptive_policy_flags(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "reconstruct-demo",
            "--output-dir",
            str(tmp_path),
            "--frames",
            str(FIXTURE_DIR / "training_frames.json"),
            "--iterations",
            "2",
            "--split-image-loss-threshold",
            "1.0",
            "--demote-after-iteration",
            "5",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    report = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))
    created = {
        item["created_element_id"]
        for step in report["iterations"]
        for item in step["carrier_evolution"]
        if item["created_element_id"] is not None
    }

    assert report["evolutionPolicy"]["splitImageLossThreshold"] == 1.0
    assert report["evolutionPolicy"]["demoteAfterIteration"] == 5
    assert "soft_volume_beta_detail" not in created
    assert "semantic_object_neural_residual" not in created


def test_reconstruct_demo_cli_records_render_backend_policy(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "reconstruct-demo",
            "--output-dir",
            str(tmp_path),
            "--frames",
            str(FIXTURE_DIR / "training_frames.json"),
            "--iterations",
            "1",
            "--render-backend",
            "auto",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    report = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))

    assert report["renderingPolicy"]["requestedBackend"] == "auto"
    assert report["renderBackend"] in {"cpu", "torch"}
    assert report["iterations"][0]["render_backend"] == report["renderBackend"]


def test_reconstruct_demo_cli_can_disable_adaptive_evolution(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "reconstruct-demo",
            "--output-dir",
            str(tmp_path),
            "--frames",
            str(FIXTURE_DIR / "training_frames.json"),
            "--iterations",
            "2",
            "--disable-adaptive-evolution",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    report = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))

    assert report["evolutionPolicy"]["enabled"] is False
    assert all(not step["carrier_evolution"] for step in report["iterations"])


def test_write_training_frames_demo_cli_outputs_reconstructable_frames(tmp_path):
    frames_path = tmp_path / "frames.json"
    package_dir = tmp_path / "package.aura"

    subprocess.run(
        [sys.executable, "-m", "aura.cli", "write-training-frames-demo", "--output", str(frames_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "reconstruct-demo",
            "--frames",
            str(frames_path),
            "--output-dir",
            str(package_dir),
            "--iterations",
            "3",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    package = load_package(package_dir)
    assert package.asset.name == "reconstruct_demo"


# --- Additional coverage tests ---

def test_training_frame_rejects_empty_id():
    """Cover TrainingFrame.__post_init__ empty id check (line 52)."""
    with pytest.raises(ValueError, match="id is required"):
        TrainingFrame(
            id="",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.5, 0.5, 0.5),
            target_depth=1.0,
        )


def test_training_frame_rejects_non_positive_depth():
    """Cover TrainingFrame.__post_init__ target_depth check (line 54)."""
    with pytest.raises(ValueError, match="target_depth must be positive"):
        TrainingFrame(
            id="f1",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.5, 0.5, 0.5),
            target_depth=0.0,
        )
    with pytest.raises(ValueError, match="target_depth must be positive"):
        TrainingFrame(
            id="f1",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.5, 0.5, 0.5),
            target_depth=-1.0,
        )


def test_training_frame_rejects_incomplete_intrinsics():
    """Cover TrainingFrame.__post_init__ intrinsics missing keys (line 59)."""
    with pytest.raises(ValueError, match="intrinsics missing keys"):
        TrainingFrame(
            id="f1",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.5, 0.5, 0.5),
            target_depth=1.0,
            intrinsics={"fx": 1.0, "fy": 1.0},  # missing cx, cy, width, height
        )


def test_training_frame_from_dict_rejects_non_dict():
    """Cover TrainingFrame.from_dict non-dict input (line 80)."""
    with pytest.raises(ValueError, match="must be an object"):
        TrainingFrame.from_dict("not_a_dict")  # type: ignore


def test_training_region_rejects_empty_id():
    """Cover TrainingRegion.__post_init__ empty id check (line 117)."""
    with pytest.raises(ValueError, match="region id is required"):
        TrainingRegion(
            id="",
            frame_id="f1",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            evidence=RegionEvidence(),
        )


def test_training_region_rejects_empty_frame_id():
    """Cover TrainingRegion.__post_init__ empty frame_id check (line 119)."""
    with pytest.raises(ValueError, match="frame_id is required"):
        TrainingRegion(
            id="r1",
            frame_id="",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            evidence=RegionEvidence(),
        )


def test_training_region_rejects_opacity_out_of_range():
    """Cover TrainingRegion.__post_init__ opacity range check (line 121)."""
    with pytest.raises(ValueError, match="opacity must be in"):
        TrainingRegion(
            id="r1",
            frame_id="f1",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            evidence=RegionEvidence(),
            opacity=1.5,
        )
    with pytest.raises(ValueError, match="opacity must be in"):
        TrainingRegion(
            id="r1",
            frame_id="f1",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            evidence=RegionEvidence(),
            opacity=-0.1,
        )


def test_training_region_rejects_confidence_out_of_range():
    """Cover TrainingRegion.__post_init__ confidence range check (line 123)."""
    with pytest.raises(ValueError, match="confidence must be in"):
        TrainingRegion(
            id="r1",
            frame_id="f1",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            evidence=RegionEvidence(),
            confidence=1.5,
        )


def test_training_region_from_dict_rejects_non_dict():
    """Cover TrainingRegion.from_dict non-dict input (line 143)."""
    with pytest.raises(ValueError, match="must be an object"):
        TrainingRegion.from_dict("not_a_dict")  # type: ignore


def test_training_region_from_dict_rejects_non_dict_bounds():
    """Cover TrainingRegion.from_dict non-dict bounds (line 146)."""
    with pytest.raises(ValueError, match="bounds must be an object"):
        TrainingRegion.from_dict({
            "id": "r1",
            "frame_id": "f1",
            "bounds": "not_a_dict",
            "evidence": {},
        })


def test_training_region_from_dict_rejects_non_dict_evidence():
    """Cover TrainingRegion.from_dict non-dict evidence (line 149)."""
    with pytest.raises(ValueError, match="evidence must be an object"):
        TrainingRegion.from_dict({
            "id": "r1",
            "frame_id": "f1",
            "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
            "evidence": "not_a_dict",
        })


def test_reconstruction_config_rejects_invalid_iterations():
    """Cover ReconstructionConfig.__post_init__ validation (lines 220, 222, 224, 226)."""
    with pytest.raises(ValueError, match="iterations must be positive"):
        ReconstructionConfig(iterations=0)
    with pytest.raises(ValueError, match="render dimensions must be positive"):
        ReconstructionConfig(render_width=0)
    with pytest.raises(ValueError, match="render dimensions must be positive"):
        ReconstructionConfig(render_height=0)
    with pytest.raises(ValueError, match="color_learning_rate"):
        ReconstructionConfig(color_learning_rate=0.0)
    with pytest.raises(ValueError, match="render_backend"):
        ReconstructionConfig(render_backend="invalid_backend")


def test_load_training_dataset_rejects_non_dict(tmp_path):
    """Cover load_training_dataset non-dict check (line 476)."""
    import json
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_training_dataset(bad_path)


def test_load_training_dataset_rejects_duplicate_frame_ids(tmp_path):
    """Cover _validate_training_dataset_links duplicate frame ids (line 533)."""
    import json
    dataset = synthetic_training_dataset()
    payload = dataset.to_dict()
    # Duplicate the first frame
    payload["frames"].append(payload["frames"][0])
    bad_path = tmp_path / "dup_frames.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate frame ids"):
        load_training_dataset(bad_path)


def test_load_training_dataset_rejects_duplicate_region_ids(tmp_path):
    """Cover _validate_training_dataset_links duplicate region ids (line 536)."""
    import json
    dataset = synthetic_training_dataset()
    payload = dataset.to_dict()
    # Duplicate the first region
    payload["regions"].append(payload["regions"][0])
    bad_path = tmp_path / "dup_regions.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate region ids"):
        load_training_dataset(bad_path)


def test_initial_evidence_from_regions_rejects_no_frames():
    """Cover _initial_evidence_from_regions empty frames check (line 615)."""
    from aura.core import _initial_evidence_from_regions
    region = TrainingRegion(
        id="r1",
        frame_id="f1",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        evidence=RegionEvidence(),
    )
    with pytest.raises(ValueError, match="at least one posed training frame"):
        _initial_evidence_from_regions([], [region])


def test_initial_evidence_from_regions_rejects_no_regions():
    """Cover _initial_evidence_from_regions empty regions check (line 617)."""
    from aura.core import _initial_evidence_from_regions
    frame = TrainingFrame(
        id="f1",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.5, 0.5, 0.5),
        target_depth=1.0,
    )
    with pytest.raises(ValueError, match="at least one training region"):
        _initial_evidence_from_regions([frame], [])


def test_initial_evidence_from_regions_rejects_unknown_frame_ids():
    """Cover _initial_evidence_from_regions unknown frame ids (line 621)."""
    from aura.core import _initial_evidence_from_regions
    frame = TrainingFrame(
        id="f1",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.5, 0.5, 0.5),
        target_depth=1.0,
    )
    region = TrainingRegion(
        id="r1",
        frame_id="missing_frame",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        evidence=RegionEvidence(),
    )
    with pytest.raises(ValueError, match="unknown frame ids"):
        _initial_evidence_from_regions([frame], [region])


def test_resolve_render_backend_auto_falls_back_to_cpu_when_torch_unavailable(monkeypatch):
    """Cover _resolve_reconstruction_render_backend auto+no-torch path (lines 805-808)."""
    from types import SimpleNamespace
    import aura.torch_renderer as torch_renderer_module
    from aura.core import _resolve_reconstruction_render_backend

    monkeypatch.setattr(
        torch_renderer_module,
        "torch_renderer_status",
        lambda: SimpleNamespace(available=False, cuda_available=False, default_device=None, reason="no torch"),
    )
    config = ReconstructionConfig(render_backend="auto")
    backend, device = _resolve_reconstruction_render_backend(config)
    assert backend == "cpu"
    assert device is None


def test_resolve_render_backend_auto_raises_when_cuda_required_but_torch_unavailable(monkeypatch):
    """Cover _resolve_reconstruction_render_backend auto+cuda_required+no-torch (line 807)."""
    from types import SimpleNamespace
    import aura.torch_renderer as torch_renderer_module
    from aura.core import _resolve_reconstruction_render_backend

    monkeypatch.setattr(
        torch_renderer_module,
        "torch_renderer_status",
        lambda: SimpleNamespace(available=False, cuda_available=False, default_device=None, reason="no torch"),
    )
    # render_backend="auto", require_cuda=True, torch unavailable => hits line 807
    config = ReconstructionConfig(render_backend="auto", require_cuda=True)
    with pytest.raises(RuntimeError, match="CUDA reconstruction was required"):
        _resolve_reconstruction_render_backend(config)


def test_resolve_render_backend_torch_raises_when_torch_unavailable(monkeypatch):
    """Cover _resolve_reconstruction_render_backend torch+unavailable (line 810)."""
    from types import SimpleNamespace
    import aura.torch_renderer as torch_renderer_module
    from aura.core import _resolve_reconstruction_render_backend

    monkeypatch.setattr(
        torch_renderer_module,
        "torch_renderer_status",
        lambda: SimpleNamespace(available=False, cuda_available=False, default_device=None, reason="PyTorch not installed"),
    )
    config = ReconstructionConfig(render_backend="torch")
    with pytest.raises(RuntimeError, match="PyTorch"):
        _resolve_reconstruction_render_backend(config)


def test_resolve_render_backend_raises_when_cuda_required_but_cuda_unavailable(monkeypatch):
    """Cover _resolve_reconstruction_render_backend cuda required but cuda not available (line 812)."""
    from types import SimpleNamespace
    import aura.torch_renderer as torch_renderer_module
    from aura.core import _resolve_reconstruction_render_backend

    monkeypatch.setattr(
        torch_renderer_module,
        "torch_renderer_status",
        lambda: SimpleNamespace(available=True, cuda_available=False, default_device="cpu", reason=None),
    )
    config = ReconstructionConfig(render_backend="torch", require_cuda=True)
    with pytest.raises(RuntimeError, match="torch.cuda is unavailable"):
        _resolve_reconstruction_render_backend(config)


def test_resolve_render_backend_raises_when_cuda_required_but_device_is_cpu(monkeypatch):
    """Cover _resolve_reconstruction_render_backend cuda required but device is cpu (line 815)."""
    from types import SimpleNamespace
    import aura.torch_renderer as torch_renderer_module
    from aura.core import _resolve_reconstruction_render_backend

    monkeypatch.setattr(
        torch_renderer_module,
        "torch_renderer_status",
        lambda: SimpleNamespace(available=True, cuda_available=True, default_device="cpu", reason=None),
    )
    config = ReconstructionConfig(render_backend="torch", require_cuda=True, torch_device="cpu")
    with pytest.raises(RuntimeError, match="requested device"):
        _resolve_reconstruction_render_backend(config)


def test_refine_scene_returns_unchanged_when_no_element_targets():
    """Cover _refine_scene_from_predictions early return when no targets (line 842)."""
    from aura.core import _refine_scene_from_predictions, ReconstructionConfig
    from aura import AuraScene, AuraElement, Bounds

    scene = AuraScene(
        name="t",
        elements=(AuraElement(id="e1", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))),),
    )
    # Predictions with element_id=None means no targets_by_element entries
    from aura.core import FramePrediction
    pred = FramePrediction(
        frame_id="f1",
        element_id=None,  # key: no element_id
        carrier_id=None,
        ray_direction=(0.0, 0.0, 1.0),
        predicted_color=(0.5, 0.5, 0.5),
        target_color=(0.5, 0.5, 0.5),
        predicted_depth=None,
        target_depth=1.0,
        target_semantic_id=None,
        target_material_id=None,
        target_normal=None,
        predicted_transmittance=1.0,
        predicted_opacity=0.0,
        predicted_confidence=0.0,
        predicted_normal=None,
        predicted_material_id=None,
        predicted_semantic_id=None,
        predicted_residual=False,
        predicted_provenance=None,
        image_loss=0.0,
        depth_loss=0.0,
        query_loss=0.0,
        normal_loss=0.0,
    )
    config = ReconstructionConfig(iterations=1)
    result_scene = _refine_scene_from_predictions(scene, (pred,), config, ())
    assert result_scene is scene


def test_normalized_direction_raises_for_identical_points():
    """Cover _normalized_direction zero-length vector (line 899)."""
    from aura.core import _normalized_direction
    with pytest.raises(ValueError, match="must differ"):
        _normalized_direction((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))


def test_vec3_from_payload_rejects_non_list():
    """Cover _vec3_from_payload non-list input (line 905)."""
    from aura.core import _vec3_from_payload
    with pytest.raises(ValueError, match="must be a 3-vector"):
        _vec3_from_payload("not_a_vector", "test_field")
    with pytest.raises(ValueError, match="must be a 3-vector"):
        _vec3_from_payload([1.0, 2.0], "test_field")  # wrong length


def test_clamp_unit_clips_values():
    """Cover _clamp_unit function (line 926)."""
    from aura.core import _clamp_unit
    assert _clamp_unit(-1.0) == 0.0
    assert _clamp_unit(2.0) == 1.0
    assert _clamp_unit(0.5) == pytest.approx(0.5)
