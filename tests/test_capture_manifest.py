import json
import struct
import subprocess
import sys
import zlib

from aura import (
    load_capture_assets,
    load_capture_manifest,
    validate_capture_manifest_document,
    write_capture_manifest_template,
)


def test_capture_manifest_template_loads_as_training_dataset(tmp_path):
    path = write_capture_manifest_template(tmp_path / "capture.json")
    manifest = load_capture_manifest(path)
    dataset = manifest.to_training_dataset()

    assert manifest.root == "data/custom-captures/example-scene"
    assert dataset.frames[0].image_path == "images/frame_000001.png"
    assert dataset.frames[0].depth_path == "depth/frame_000001.exr"
    assert dataset.frames[0].mask_path == "masks/frame_000001.png"
    assert dataset.frames[0].intrinsics["fx"] == 1200.0
    assert dataset.regions[0].fallback_source == "capture-manifest"


def test_capture_manifest_schema_rejects_unknown_evidence_field(tmp_path):
    path = write_capture_manifest_template(tmp_path / "capture.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["regions"][0]["evidence"]["made_up_axis"] = 1.0

    try:
        validate_capture_manifest_document(payload)
    except ValueError as exc:
        assert "capture_manifest.schema.json validation failed" in str(exc)
        assert "made_up_axis" in str(exc)
    else:
        raise AssertionError("unknown evidence axis should fail schema validation")


def test_capture_manifest_rejects_unknown_region_frame(tmp_path):
    path = write_capture_manifest_template(tmp_path / "capture.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["regions"][0]["frame_id"] = "missing_frame"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_capture_manifest(path)
    except ValueError as exc:
        assert "capture regions reference unknown frame ids: missing_frame" in str(exc)
    else:
        raise AssertionError("unknown frame reference should fail")


def test_capture_manifest_cli_converts_to_training_dataset(tmp_path):
    manifest_path = tmp_path / "capture.json"
    training_path = tmp_path / "training.json"

    subprocess.run(
        [sys.executable, "-m", "aura.cli", "write-capture-manifest-template", "--output", str(manifest_path)],
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
            "capture-manifest-to-training",
            str(manifest_path),
            "--output",
            str(training_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(training_path.read_text(encoding="utf-8"))
    assert payload["format"] == "AURA_TRAINING_FRAMES"
    assert payload["frames"][0]["image_path"] == "images/frame_000001.png"


def test_capture_manifest_loads_ppm_pgm_asset_summaries(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)
    manifest = load_capture_manifest(manifest_path)

    assets = load_capture_assets(manifest)
    dataset = manifest.to_training_dataset(load_assets=True)

    assert len(assets) == 1
    assert assets[0].frame_id == "frame_000001"
    assert assets[0].width == 2
    assert assets[0].height == 1
    assert assets[0].average_color == (0.5, 0.25, 0.25)
    assert assets[0].average_depth == 0.75
    assert assets[0].mask_coverage == 0.5
    assert dataset.frames[0].target_color == (0.5, 0.25, 0.25)
    assert dataset.frames[0].target_depth == 0.75


def test_capture_manifest_loads_png_asset_summaries(tmp_path):
    manifest_path = _write_png_asset_manifest(tmp_path)
    manifest = load_capture_manifest(manifest_path)

    assets = load_capture_assets(manifest)
    dataset = manifest.to_training_dataset(load_assets=True)

    assert len(assets) == 1
    assert assets[0].width == 2
    assert assets[0].height == 1
    assert assets[0].average_color == (0.5, 128 / 510, 128 / 510)
    assert assets[0].average_depth == 383 / 510
    assert assets[0].mask_coverage == 0.5
    assert dataset.frames[0].target_color == (0.5, 128 / 510, 128 / 510)
    assert dataset.frames[0].target_depth == 383 / 510


def test_capture_manifest_loads_colmap_depth_map_summary(tmp_path):
    manifest_path = _write_colmap_depth_asset_manifest(tmp_path)
    manifest = load_capture_manifest(manifest_path)

    assets = load_capture_assets(manifest)
    dataset = manifest.to_training_dataset(load_assets=True)

    assert assets[0].average_color == (0.5, 0.25, 0.25)
    assert assets[0].average_depth == 1.5
    assert dataset.frames[0].target_depth == 1.5


def test_capture_manifest_asset_loader_marks_exr_as_future_tensor_backend(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["frames"][0]["depth_path"] = "depth/frame_000001.exr"
    (tmp_path / "capture" / "depth" / "frame_000001.exr").write_bytes(b"not-real-exr")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_capture_manifest(manifest_path)

    try:
        load_capture_assets(manifest)
    except ValueError as exc:
        assert "future GPU tensor asset backend" in str(exc)
    else:
        raise AssertionError("EXR assets should require the tensor backend")


def test_capture_manifest_asset_loader_rejects_missing_image(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["frames"][0]["image_path"] = "images/missing.ppm"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_capture_manifest(manifest_path)

    try:
        load_capture_assets(manifest)
    except FileNotFoundError as exc:
        assert "missing.ppm" in str(exc)
    else:
        raise AssertionError("missing capture image should fail")


def test_capture_manifest_cli_can_materialize_training_targets_from_assets(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)
    training_path = tmp_path / "training.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "capture-manifest-to-training",
            str(manifest_path),
            "--output",
            str(training_path),
            "--load-assets",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(training_path.read_text(encoding="utf-8"))

    assert payload["frames"][0]["target_color"] == [0.5, 0.25, 0.25]
    assert payload["frames"][0]["target_depth"] == 0.75


def test_inspect_capture_assets_cli_reports_asset_summaries(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "inspect-capture-assets", str(manifest_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload[0]["frameId"] == "frame_000001"
    assert payload[0]["averageColor"] == [0.5, 0.25, 0.25]
    assert payload[0]["averageDepth"] == 0.75
    assert payload[0]["maskCoverage"] == 0.5


def _write_asset_manifest(tmp_path):
    root = tmp_path / "capture"
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "masks").mkdir()
    (root / "images" / "frame_000001.ppm").write_text(
        "P3\n2 1\n4\n4 0 0 0 2 2\n",
        encoding="ascii",
    )
    (root / "depth" / "frame_000001.pgm").write_text(
        "P2\n2 1\n4\n2 4\n",
        encoding="ascii",
    )
    (root / "masks" / "frame_000001.pgm").write_text(
        "P2\n2 1\n2\n2 0\n",
        encoding="ascii",
    )
    payload = capture_asset_manifest_payload(root)
    manifest_path = tmp_path / "asset_capture.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _write_png_asset_manifest(tmp_path):
    root = tmp_path / "capture"
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "masks").mkdir()
    _write_png(root / "images" / "frame_000001.png", width=2, height=1, channels=3, values=(255, 0, 0, 0, 128, 128))
    _write_png(root / "depth" / "frame_000001.png", width=2, height=1, channels=1, values=(128, 255))
    _write_png(root / "masks" / "frame_000001.png", width=2, height=1, channels=1, values=(255, 0))
    payload = capture_asset_manifest_payload(root)
    payload["frames"][0]["image_path"] = "images/frame_000001.png"
    payload["frames"][0]["depth_path"] = "depth/frame_000001.png"
    payload["frames"][0]["mask_path"] = "masks/frame_000001.png"
    manifest_path = tmp_path / "png_asset_capture.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _write_colmap_depth_asset_manifest(tmp_path):
    root = tmp_path / "capture"
    (root / "images").mkdir(parents=True)
    (root / "stereo" / "depth_maps").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "images" / "frame_000001.ppm").write_text(
        "P3\n2 1\n4\n4 0 0 0 2 2\n",
        encoding="ascii",
    )
    _write_colmap_depth_map(root / "stereo" / "depth_maps" / "frame_000001.png.photometric.bin", 2, 1, (1.0, 2.0))
    (root / "masks" / "frame_000001.pgm").write_text(
        "P2\n2 1\n2\n2 0\n",
        encoding="ascii",
    )
    payload = capture_asset_manifest_payload(root)
    payload["frames"][0]["depth_path"] = "stereo/depth_maps/frame_000001.png.photometric.bin"
    manifest_path = tmp_path / "colmap_depth_asset_capture.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def capture_asset_manifest_payload(root):
    return {
        "format": "AURA_CAPTURE_MANIFEST",
        "root": str(root),
        "frames": [
            {
                "id": "frame_000001",
                "image_path": "images/frame_000001.ppm",
                "depth_path": "depth/frame_000001.pgm",
                "mask_path": "masks/frame_000001.pgm",
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


def _write_colmap_depth_map(path, width, height, values):
    path.write_bytes(f"{width}&{height}&1&".encode("ascii") + struct.pack("<" + "f" * len(values), *values))
