import json
import subprocess
import sys
from pathlib import Path

from aura import (
    ReconstructionConfig,
    load_package,
    load_training_frames,
    reconstruct_demo_scene,
    synthetic_training_frames,
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
    fixture = load_training_frames(FIXTURE_DIR / "training_frames.json")

    assert loaded == synthetic_training_frames()
    assert fixture == synthetic_training_frames()


def test_reconstruct_demo_builds_native_aura_core_scene_without_3dgs():
    frames = load_training_frames(FIXTURE_DIR / "training_frames.json")
    result = reconstruct_demo_scene(ReconstructionConfig(iterations=3, render_width=8, render_height=8), frames=frames)
    report = result.report.to_dict()

    assert result.scene.name == "reconstruct_demo"
    assert result.scene.carrier_ids() == ["beta", "gabor", "gaussian", "neural", "semantic", "surface", "volume"]
    by_id = {element.id: element for element in result.scene.elements}
    assert by_id["surface_wall"].metadata["source"] == "aura-core-synthetic"
    assert by_id["soft_volume_beta_detail"].metadata["source"] == "aura-core-adaptive-evolution"
    assert by_id["soft_volume_beta_detail"].carrier_id == "beta"
    assert by_id["soft_volume_beta_detail"].payload["type"] == "beta_kernel"
    assert by_id["semantic_object_neural_residual"].metadata["source"] == "aura-core-adaptive-evolution"
    assert by_id["semantic_object_neural_residual"].carrier_id == "neural"
    assert by_id["semantic_object_neural_residual"].payload["type"] == "neural_residual"
    assert set(result.scene.chunks[0].element_ids) == set(by_id)
    assert report["format"] == "AURA_CORE_RECONSTRUCTION_REPORT"
    assert report["sources"] == ["posed_training_frames", "depth_targets", "semantic_labels"]
    assert "native_evidence_initialization" in report["stages"]
    assert "cpu_reference_render_loss" in report["stages"]
    assert report["nativeCarrierFraction"] > 0.8
    assert len(report["iterations"]) == 3
    assert report["iterations"][-1]["image_loss"] < report["iterations"][0]["image_loss"]
    assert report["iterations"][-1]["depth_loss"] < report["iterations"][0]["depth_loss"]
    assert report["iterations"][-1]["total_loss"] < report["iterations"][0]["total_loss"]
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
