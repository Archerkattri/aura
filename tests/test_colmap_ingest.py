import json
import struct
import subprocess
import sys
import zlib

from aura import (
    colmap_binary_to_capture_manifest,
    colmap_text_to_capture_manifest,
    colmap_to_capture_manifest,
    load_capture_assets,
    load_capture_manifest,
    load_colmap_binary_model,
    load_colmap_model,
    load_colmap_text_model,
)


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


def test_load_colmap_binary_model_parses_cameras_images_and_points(tmp_path):
    colmap_dir = _write_colmap_binary_model(tmp_path)

    cameras, images, points = load_colmap_binary_model(colmap_dir)

    assert set(cameras) == {"1"}
    assert cameras["1"].intrinsics() == {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "width": 640.0, "height": 480.0}
    assert len(images) == 2
    assert images[0].camera_origin == (-0.0, -0.0, -0.0)
    assert images[1].camera_origin == (1.0, -0.0, -0.0)
    assert len(points) == 2
    assert points[0].xyz == (-0.5, 0.0, 2.0)


def test_load_colmap_model_prefers_binary_when_both_formats_exist(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)
    _write_colmap_binary_files(colmap_dir)

    _cameras, images, _points, source = load_colmap_model(colmap_dir)

    assert source == "colmap-binary"
    assert [image.name for image in images] == ["frame_000001.png", "frame_000002.png"]


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


def test_colmap_text_model_splits_sparse_points_into_depth_layers(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)
    (colmap_dir / "points3D.txt").write_text(
        "\n".join(
            [
                "# 3D point list with one line of data per point:",
                "1 -0.5 0.0 1.0 255 0 0 0.1 1 0",
                "2 0.5 0.0 3.0 0 0 255 0.1 2 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = colmap_text_to_capture_manifest(
        colmap_dir,
        root="data/custom-captures/colmap-fixture",
        image_dir="images",
    )
    dataset = manifest.to_training_dataset()

    assert [region.id for region in dataset.regions] == ["colmap_sparse_prior_near", "colmap_sparse_prior_far"]
    assert dataset.regions[0].bounds.max_corner[2] < dataset.regions[1].bounds.min_corner[2]
    assert all(region.fallback_source == "colmap-text" for region in dataset.regions)
    assert all(region.semantic_label == "colmap_sparse_prior" for region in dataset.regions)


def test_colmap_binary_model_converts_to_capture_manifest_contract(tmp_path):
    colmap_dir = _write_colmap_binary_model(tmp_path)

    manifest = colmap_binary_to_capture_manifest(
        colmap_dir,
        root="data/custom-captures/colmap-fixture",
        image_dir="images",
    )
    dataset = manifest.to_training_dataset()

    assert [frame.id for frame in dataset.frames] == ["colmap_image_1", "colmap_image_2"]
    assert dataset.frames[0].image_path == "images/frame_000001.png"
    assert dataset.frames[0].camera_model == "PINHOLE"
    assert dataset.frames[0].look_at == (0.0, 0.0, 2.0)
    assert dataset.regions[0].fallback_source == "colmap-binary"


def test_colmap_auto_model_converts_binary_to_capture_manifest_contract(tmp_path):
    colmap_dir = _write_colmap_binary_model(tmp_path)

    manifest = colmap_to_capture_manifest(
        colmap_dir,
        root="data/custom-captures/colmap-fixture",
        image_dir="images",
    )

    assert manifest.regions[0].fallback_source == "colmap-binary"


def test_colmap_manifest_links_standard_depth_maps(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)
    _write_colmap_depth_map(
        tmp_path / "stereo" / "depth_maps" / "frame_000001.png.photometric.bin",
        2,
        1,
        (1.0, 3.0),
    )

    manifest = colmap_text_to_capture_manifest(
        colmap_dir,
        root=str(tmp_path),
        image_dir="images",
    )

    assert manifest.frames[0].depth_path == "stereo/depth_maps/frame_000001.png.photometric.bin"


def test_colmap_manifest_links_standard_normal_maps(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)
    _write_colmap_normal_map(
        tmp_path / "stereo" / "normal_maps" / "frame_000001.png.photometric.bin",
        2,
        1,
        ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)),
    )

    manifest = colmap_text_to_capture_manifest(
        colmap_dir,
        root=str(tmp_path),
        image_dir="images",
    )

    assert manifest.frames[0].normal_path == "stereo/normal_maps/frame_000001.png.photometric.bin"


def test_colmap_manifest_depth_maps_can_materialize_training_depth(tmp_path):
    colmap_dir = _write_colmap_text_model(tmp_path)
    (tmp_path / "images").mkdir()
    _write_png(tmp_path / "images" / "frame_000001.png", width=2, height=1, channels=3, values=(255, 0, 0, 0, 128, 128))
    _write_png(tmp_path / "images" / "frame_000002.png", width=2, height=1, channels=3, values=(255, 0, 0, 0, 128, 128))
    _write_colmap_depth_map(
        tmp_path / "stereo" / "depth_maps" / "frame_000001.png.photometric.bin",
        2,
        1,
        (1.0, 3.0),
    )
    _write_colmap_normal_map(
        tmp_path / "stereo" / "normal_maps" / "frame_000001.png.photometric.bin",
        2,
        1,
        ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)),
    )

    manifest = colmap_text_to_capture_manifest(
        colmap_dir,
        root=str(tmp_path),
        image_dir="images",
    )
    assets = load_capture_assets(manifest)
    dataset = manifest.to_training_dataset(load_assets=True)

    assert assets[0].average_depth == 2.0
    assert assets[0].min_depth == 1.0
    assert assets[0].max_depth == 3.0
    assert assets[0].depth_coverage == 1.0
    assert assets[0].average_normal == (0.0, 0.0, -1.0)
    assert [item["average"] for item in assets[0].depth_bins] == [1.0, 3.0]
    assert dataset.frames[0].target_depth == 2.0
    assert [region.id for region in dataset.regions[-2:]] == ["colmap_image_1_depth_prior_0", "colmap_image_1_depth_prior_1"]
    assert all(region.fallback_source == "capture-depth-prior" for region in dataset.regions[-2:])
    assert all(region.evidence.geometry_confidence == 0.75 for region in dataset.regions[-2:])
    assert all(region.normal == (0.0, 0.0, -1.0) for region in dataset.regions[-2:])
    assert dataset.frames[1].target_depth > 2.0


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


def _write_colmap_binary_model(tmp_path):
    colmap_dir = tmp_path / "colmap"
    colmap_dir.mkdir()
    _write_colmap_binary_files(colmap_dir)
    return colmap_dir


def _write_colmap_binary_files(colmap_dir):
    (colmap_dir / "cameras.bin").write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<iiQQ", 1, 1, 640, 480)
        + struct.pack("<dddd", 800.0, 800.0, 320.0, 240.0)
    )
    (colmap_dir / "images.bin").write_bytes(
        struct.pack("<Q", 2)
        + _colmap_binary_image(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "frame_000001.png")
        + _colmap_binary_image(2, (1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 1, "frame_000002.png")
    )
    (colmap_dir / "points3D.bin").write_bytes(
        struct.pack("<Q", 2)
        + _colmap_binary_point(1, (-0.5, 0.0, 2.0), (255, 0, 0), 0.1, ((1, 0),))
        + _colmap_binary_point(2, (0.5, 0.0, 2.0), (0, 0, 255), 0.1, ((2, 0),))
    )


def _colmap_binary_image(image_id, qvec, tvec, camera_id, name):
    return (
        struct.pack("<idddddddi", image_id, *qvec, *tvec, camera_id)
        + name.encode("utf-8")
        + b"\x00"
        + struct.pack("<Q", 1)
        + struct.pack("<ddq", 0.0, 0.0, -1)
    )


def _colmap_binary_point(point_id, xyz, rgb, error, track):
    payload = struct.pack("<QdddBBBdQ", point_id, *xyz, *rgb, error, len(track))
    for image_id, point2d_idx in track:
        payload += struct.pack("<ii", image_id, point2d_idx)
    return payload


def _write_colmap_depth_map(path, width, height, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{width}&{height}&1&".encode("ascii") + struct.pack("<" + "f" * len(values), *values))


def _write_colmap_normal_map(path, width, height, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [component for normal in values for component in normal]
    path.write_bytes(f"{width}&{height}&3&".encode("ascii") + struct.pack("<" + "f" * len(flat), *flat))


def _write_png(path, *, width, height, channels, values):
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    scanline_width = width * channels
    rows = []
    for row in range(height):
        start = row * scanline_width
        rows.append(b"\x00" + bytes(values[start : start + scanline_width]))
    raw = b"".join(rows)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _png_chunk(kind, data):
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def test_intrinsics_scaled_to_actual_image_resolution(tmp_path):
    """COLMAP cameras are often at a higher resolution than the shipped images
    (e.g. Tanks and Temples: 1957x1091 model vs 979x546 images). Intrinsics must
    be rescaled to the real image or every ray is wrong."""
    import importlib.util
    if importlib.util.find_spec("imageio") is None:
        import pytest
        pytest.skip("imageio not installed")
    import numpy as np
    import imageio.v3 as iio
    from aura.ingest.colmap import _intrinsics_for_image, _probe_image_size

    img = tmp_path / "frame.jpg"
    iio.imwrite(img, np.zeros((50, 100, 3), dtype=np.uint8))  # H=50, W=100

    assert _probe_image_size(img) == (100, 50)

    # Camera modelled at twice the resolution of the actual image.
    cam = {"fx": 200.0, "fy": 200.0, "cx": 100.0, "cy": 50.0, "width": 200.0, "height": 100.0}
    scaled = _intrinsics_for_image(cam, img)
    assert scaled["width"] == 100.0 and scaled["height"] == 50.0
    assert scaled["fx"] == 100.0 and scaled["fy"] == 100.0
    assert scaled["cx"] == 50.0 and scaled["cy"] == 25.0

    # Matching resolution is a no-op; missing file leaves intrinsics unchanged.
    assert _intrinsics_for_image(dict(scaled), img) == scaled
    assert _intrinsics_for_image(cam, tmp_path / "missing.jpg") == cam


def test_sparse_prior_regions_dense_voxel_seeding():
    """Real point clouds seed one carrier-region per occupied voxel (not 2 boxes)."""
    from aura.ingest.colmap import _sparse_prior_regions, ColmapPoint3D
    pts = [
        ColmapPoint3D(id=str(i), xyz=(float(i % 10), float((i // 10) % 10), float((i // 100) % 10)), rgb=(0.5, 0.5, 0.5))
        for i in range(1000)
    ]
    regions = _sparse_prior_regions("f0", pts, None, 2.0, "colmap-binary", max_seed_regions=512)
    assert len(regions) > 2  # dense seeding, not the legacy near/far split
    assert len(regions) <= 512  # respects the budget cap
    assert all(r["id"].startswith("colmap_sparse_voxel_") for r in regions)
    # Each region has real local extent (min < max on at least one axis).
    r = regions[0]
    assert any(lo < hi for lo, hi in zip(r["bounds"]["min"], r["bounds"]["max"]))


def test_sparse_prior_regions_small_model_keeps_legacy_path():
    """Small synthetic models keep the legacy near/far seeding (unchanged)."""
    from aura.ingest.colmap import _sparse_prior_regions, ColmapPoint3D
    pts = [ColmapPoint3D(id=str(i), xyz=(0.0, 0.0, float(i)), rgb=(0.5, 0.5, 0.5)) for i in range(4)]
    regions = _sparse_prior_regions("f0", pts, None, 2.0, "colmap-text")
    assert all(not r["id"].startswith("colmap_sparse_voxel_") for r in regions)
