import json
import subprocess
import sys

from aura import ReconstructionConfig, load_package, reconstruct_demo_scene, synthetic_training_frames


def test_synthetic_training_frames_are_posed_native_inputs():
    frames = synthetic_training_frames()

    assert len(frames) == 4
    assert all(frame.target_depth > 0.0 for frame in frames)
    assert {frame.semantic_label for frame in frames} >= {"wall", "fixture_object"}


def test_reconstruct_demo_builds_native_aura_core_scene_without_3dgs():
    result = reconstruct_demo_scene(ReconstructionConfig(iterations=3, render_width=8, render_height=8))
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
    assert report["sources"] == ["synthetic_posed_images", "synthetic_depth", "semantic_masks"]
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


def test_reconstruct_demo_cli_writes_package_and_training_report(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "reconstruct-demo",
            "--output-dir",
            str(tmp_path),
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
