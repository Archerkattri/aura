import importlib.util
import json
import struct
import subprocess
import sys

import pytest

from aura import AuraElement, AuraScene, Bounds, Ray, RenderTarget, load_package, torch_render_targets


def _cuda_available():
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="CUDA torch device is unavailable")


def test_torch_render_targets_runs_on_cuda_device():
    scene = AuraScene(
        name="cuda_render_smoke",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
                target_normal=(0.0, 0.0, -1.0),
            ),
        ),
        device="cuda",
    )

    assert batch.device.startswith("cuda")
    assert batch.element_ids == ("surface",)
    assert batch.predicted_color[0] == pytest.approx((1.0, 0.0, 0.0))
    assert batch.predicted_depth == pytest.approx((2.0,))


def test_train_cli_uses_cuda_and_render_cli_renders_trained_package(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)
    package_dir = tmp_path / "cuda-trained.aura"
    render_path = tmp_path / "cuda-trained.ppm"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "train",
            str(manifest_path),
            "--output",
            str(package_dir),
            "--iterations",
            "1",
            "--max-targets-per-frame",
            "2",
            "--max-targets-per-batch",
            "1",
            "--tile-size",
            "1",
            "--device",
            "cuda",
            "--disable-evolution",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    render_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "render",
            str(package_dir),
            "--output",
            str(render_path),
            "--width",
            "4",
            "--height",
            "4",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    package = load_package(package_dir)
    report = json.loads((package_dir / "training_report.json").read_text(encoding="utf-8"))

    assert package.asset.name == "aura_train"
    assert report["device"] == "cuda"
    assert report["adaptiveEvolutionEnabled"] is False
    assert report["torch"]["cudaAvailable"] is True
    assert report["packedBatchCount"] == 2
    assert report["packedTargetCount"] == 2
    assert all(step["device"].startswith("cuda") for step in report["steps"])
    assert all(step["sample_count"] == 1 for step in report["steps"])
    assert render_result.stdout.strip() == str(render_path)
    assert render_path.read_text(encoding="ascii").startswith("P3\n4 4\n255\n")


def _write_asset_manifest(tmp_path):
    root = tmp_path / "capture"
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "masks").mkdir()
    (root / "normal").mkdir()
    (root / "images" / "frame_000001.ppm").write_text(
        "P3\n2 1\n4\n4 0 0 0 2 2\n",
        encoding="ascii",
    )
    (root / "depth" / "frame_000001.pgm").write_text(
        "P2\n2 1\n4\n2 4\n",
        encoding="ascii",
    )
    (root / "masks" / "frame_000001.pgm").write_text(
        "P2\n2 1\n2\n2 2\n",
        encoding="ascii",
    )
    _write_colmap_normal_map(root / "normal" / "frame_000001.bin", 2, 1, ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)))
    manifest_path = tmp_path / "cuda_asset_capture.json"
    manifest_path.write_text(json.dumps(_capture_asset_manifest_payload(root)), encoding="utf-8")
    return manifest_path


def _capture_asset_manifest_payload(root):
    return {
        "format": "AURA_CAPTURE_MANIFEST",
        "root": str(root),
        "frames": [
            {
                "id": "frame_000001",
                "image_path": "images/frame_000001.ppm",
                "depth_path": "depth/frame_000001.pgm",
                "mask_path": "masks/frame_000001.pgm",
                "normal_path": "normal/frame_000001.bin",
                "camera_origin": [0.0, 0.0, -2.0],
                "look_at": [0.0, 0.0, 0.0],
                "target_color": [0.1, 0.1, 0.1],
                "target_depth": 2.0,
                "semantic_label": "fixture",
            }
        ],
        "regions": [
            {
                "id": "surface_000001",
                "frame_id": "frame_000001",
                "bounds": {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.1]},
                "evidence": {"geometry_confidence": 0.9, "edit_need": 0.5},
                "opacity": 0.9,
                "confidence": 0.8,
                "normal": [0.0, 0.0, -1.0],
                "fallback_source": "capture-manifest",
            }
        ],
    }


def _write_colmap_normal_map(path, width, height, values):
    flat = [component for normal in values for component in normal]
    path.write_bytes(f"{width}&{height}&3&".encode("ascii") + struct.pack("<" + "f" * len(flat), *flat))
