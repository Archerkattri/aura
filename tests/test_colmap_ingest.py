import json
import subprocess
import sys

from aura import colmap_text_to_capture_manifest, load_capture_manifest, load_colmap_text_model


def test_load_colmap_text_model_parses_cameras_images_and_points(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)

    cameras, images, points = load_colmap_text_model(colmap_dir)

    assert set(cameras) == {"1"}
    assert cameras["1"].intrinsics() == {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "width": 640.0, "height": 480.0}
    assert len(images) == 2
    assert images[0].camera_origin == (-0.0, -0.0, -0.0)
    assert images[1].camera_origin == (1.0, -0.0, -0.0)
    assert len(points) == 2
    assert points[0].xyz == (-0.5, 0.0, 2.0)


def test_colmap_text_model_converts_to_capture_manifest_contract(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)

    manifest = colmap_text_to_capture_manifest(
        colmap_dir,
        root="data/custom-captures/colmap-fixture",
        image_dir="images",
    )
    dataset = manifest.to_training_dataset()

    assert manifest.root == "data/custom-captures/colmap-fixture"
    assert [frame.id for frame in dataset.frames] == ["colmap_image_1", "colmap_image_2"]
    assert dataset.frames[0].image_path == "images/frame_000001.png"
    assert dataset.frames[0].camera_model == "PINHOLE"
    assert dataset.frames[0].intrinsics["fx"] == 800.0
    assert dataset.frames[0].look_at == (0.0, 0.0, 2.0)
    assert dataset.frames[0].target_depth == 2.0
    assert dataset.frames[1].target_depth > 2.0
    assert len(dataset.regions) == 1
    assert dataset.regions[0].id == "colmap_sparse_prior"
    assert dataset.regions[0].frame_id == "colmap_image_1"
    assert dataset.regions[0].fallback_source == "colmap-text"
    assert dataset.regions[0].semantic_label == "colmap_sparse_prior"


def test_colmap_images_parser_accepts_blank_observation_lines(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)
    (colmap_dir / "images.txt").write_text(
        "\n".join(
            [
                "# Image list with two lines of data per image:",
                "1 1 0 0 0 0 0 0 1 frame_000001.png",
                "",
                "2 1 0 0 0 -1 0 0 1 frame_000002.png",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _cameras, images, _points = load_colmap_text_model(colmap_dir)

    assert [image.id for image in images] == ["1", "2"]


def test_colmap_to_capture_manifest_cli_writes_reconstructable_manifest(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)
    manifest_path = tmp_path / "capture-from-colmap.json"
    package_dir = tmp_path / "reconstruct-colmap.aura"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "colmap-to-capture-manifest",
            str(colmap_dir),
            "--output",
            str(manifest_path),
            "--root",
            "data/custom-captures/colmap-fixture",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    manifest = load_capture_manifest(manifest_path)

    assert manifest.frames[0].image_path == "images/frame_000001.png"
    assert manifest.regions[0].fallback_source == "colmap-text"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "reconstruct-capture-manifest",
            str(manifest_path),
            "--output-dir",
            str(package_dir),
            "--iterations",
            "2",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    report = json.loads((package_dir / "training_report.json").read_text(encoding="utf-8"))

    assert report["name"] == "reconstruct_capture"
    assert report["frames"][0]["image_path"] == "images/frame_000001.png"


def _write_colmap_text_model(tmp_path):
    colmap_dir = tmp_path / "colmap"
    colmap_dir.mkdir()
    (colmap_dir / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "1 PINHOLE 640 480 800 800 320 240",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (colmap_dir / "images.txt").write_text(
        "\n".join(
            [
                "# Image list with two lines of data per image:",
                "1 1 0 0 0 0 0 0 1 frame_000001.png",
                "0 0 -1",
                "2 1 0 0 0 -1 0 0 1 frame_000002.png",
                "0 0 -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (colmap_dir / "points3D.txt").write_text(
        "\n".join(
            [
                "# 3D point list with one line of data per point:",
                "1 -0.5 0.0 2.0 255 0 0 0.1 1 0",
                "2 0.5 0.0 2.0 0 0 255 0.1 2 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return colmap_dir
