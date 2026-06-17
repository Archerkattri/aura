import json
import subprocess
import sys

from aura import (
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
