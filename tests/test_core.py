import json
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
        ),
    )
    render_targets = (
        RenderTarget(
            frame_id="capture_a",
            ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(1.0, 0.0, 0.0),
            target_depth=2.0,
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
    assert [item["target_color"] for item in report["iterations"][0]["predictions"]] == [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]


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
    assert set(result.scene.chunks[0].element_ids) == set(by_id)
    assert report["format"] == "AURA_CORE_RECONSTRUCTION_REPORT"
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
        "refine_radiance",
        "split_beta_detail",
        "promote_neural_residual",
    }
    assert {item["created_element_id"] for item in report["iterations"][0]["carrier_evolution"]} >= {
        "soft_volume_beta_detail",
        "semantic_object_neural_residual",
    }
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
    assert all(0.0 <= item["predicted_transmittance"] <= 1.0 for item in report["iterations"][0]["predictions"])
    assert all(0.0 <= item["predicted_opacity"] <= 1.0 for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_material_id"] == "mat_wall_plaster" for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_semantic_id"] == "wall" for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_semantic_id"] == "fixture_object" for item in report["iterations"][0]["predictions"])
    assert any(item["target_semantic_id"] == "wall" for item in report["iterations"][0]["predictions"])
    assert any(item["predicted_provenance"] == item["element_id"] for item in report["iterations"][0]["predictions"])
    assert any(item["gradient_norm"] > 0.0 for item in report["iterations"][0]["predictions"])


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
