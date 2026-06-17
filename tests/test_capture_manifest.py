import importlib.util
import json
import struct
import subprocess
import sys
import zlib

import pytest

from aura import (
    CaptureTensor,
    CaptureFrameTensors,
    TrainingFrame,
    capture_tensors_to_render_targets,
    capture_tensors_to_training_dataset,
    decompose_evidence,
    load_capture_asset_tensors,
    load_capture_assets,
    load_capture_manifest,
    load_package,
    plan_capture_tensor_sampling,
    validate_capture_manifest_document,
    write_capture_manifest_template,
)
from aura.ingest.capture import PackedFloatBuffer


def test_capture_manifest_template_loads_as_training_dataset(tmp_path):
    path = write_capture_manifest_template(tmp_path / "capture.json")
    manifest = load_capture_manifest(path)
    dataset = manifest.to_training_dataset()

    assert manifest.root == "data/custom-captures/example-scene"
    assert dataset.frames[0].image_path == "images/frame_000001.png"
    assert dataset.frames[0].depth_path == "depth/frame_000001.exr"
    assert dataset.frames[0].mask_path == "masks/frame_000001.png"
    assert dataset.frames[0].normal_path == "normal/frame_000001.bin"
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
    assert assets[0].min_depth == 0.5
    assert assets[0].max_depth == 1.0
    assert assets[0].depth_coverage == 1.0
    assert [item["average"] for item in assets[0].depth_bins] == [0.5, 1.0]
    assert assets[0].mask_coverage == 0.5
    assert assets[0].average_normal == (0.0, 0.0, -1.0)
    assert dataset.frames[0].target_color == (0.5, 0.25, 0.25)
    assert dataset.frames[0].target_depth == 0.75
    assert [region.id for region in dataset.regions[-3:]] == [
        "frame_000001_depth_prior_0",
        "frame_000001_depth_prior_1",
        "frame_000001_mask_semantic",
    ]
    assert all(region.fallback_source == "capture-depth-prior" for region in dataset.regions[-3:-1])
    assert all(region.evidence.geometry_confidence == 0.75 for region in dataset.regions[-3:-1])
    assert all(region.normal == (0.0, 0.0, -1.0) for region in dataset.regions[-3:-1])
    assert dataset.regions[-1].fallback_source == "capture-mask-prior"
    assert dataset.regions[-1].normal == (0.0, 0.0, -1.0)
    assert dataset.regions[-1].evidence.semantic_confidence == 0.825


def test_capture_manifest_asset_tensors_seed_feature_proposal_regions(tmp_path):
    manifest = load_capture_manifest(_write_asset_manifest(tmp_path))
    dataset = manifest.to_training_dataset(load_assets=True)
    by_id = {region.id: region for region in dataset.regions}

    assert by_id["frame_000001_image_detail_proposal"].fallback_source == "capture-feature-proposal"
    assert by_id["frame_000001_image_detail_proposal"].evidence.high_frequency >= 0.8
    assert by_id["frame_000001_image_detail_proposal"].semantic_label is None
    assert by_id["frame_000001_depth_edge_proposal"].fallback_source == "capture-feature-proposal"
    assert by_id["frame_000001_depth_edge_proposal"].evidence.compact_detail >= 0.8
    assert by_id["frame_000001_depth_edge_proposal"].semantic_label is None


def test_capture_manifest_to_training_dataset_loads_asset_tensors_once(tmp_path, monkeypatch):
    import aura.ingest.capture as capture_module

    manifest = load_capture_manifest(_write_asset_manifest(tmp_path))
    original_loader = capture_module.load_capture_asset_tensors
    calls = []

    def counted_loader(candidate):
        calls.append(candidate)
        return original_loader(candidate)

    monkeypatch.setattr(capture_module, "load_capture_asset_tensors", counted_loader)

    dataset = manifest.to_training_dataset(load_assets=True)

    assert calls == [manifest]
    assert dataset.frames[0].target_color == (0.5, 0.25, 0.25)
    assert "frame_000001_image_detail_proposal" in {region.id for region in dataset.regions}


def test_capture_manifest_feature_proposals_decompose_to_native_detail_carriers(tmp_path):
    manifest = load_capture_manifest(_write_asset_manifest(tmp_path))
    dataset = manifest.to_training_dataset(load_assets=True)
    frame_by_id = {frame.id: frame for frame in dataset.frames}
    samples = tuple(region.to_evidence_sample(frame_by_id[region.frame_id]) for region in dataset.regions)

    scene = decompose_evidence(samples)
    by_id = {element.id: element for element in scene.elements}

    assert by_id["frame_000001_image_detail_proposal"].carrier_id == "gabor"
    assert by_id["frame_000001_image_detail_proposal"].payload["type"] == "gabor_frequency"
    assert by_id["frame_000001_depth_edge_proposal"].carrier_id == "beta"
    assert by_id["frame_000001_depth_edge_proposal"].payload["type"] == "beta_kernel"
    assert scene.semantic_graph.nodes[0].element_ids == (
        "frame_000001_depth_prior_0",
        "frame_000001_depth_prior_1",
        "frame_000001_mask_semantic",
    )


def test_capture_manifest_loads_per_pixel_asset_tensors(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)
    manifest = load_capture_manifest(manifest_path)

    tensors = load_capture_asset_tensors(manifest)

    assert len(tensors) == 1
    assert tensors[0].frame_id == "frame_000001"
    assert tensors[0].image.shape == (1, 2, 3)
    assert tensors[0].image.backend == "stdlib"
    assert isinstance(tensors[0].image.values, PackedFloatBuffer)
    assert tensors[0].image.sample_values() == (1.0, 0.0, 0.0, 0.0, 0.5, 0.5)
    assert tensors[0].depth.shape == (1, 2, 1)
    assert tensors[0].depth.values == (0.5, 1.0)
    assert tensors[0].mask.values == (1.0, 0.0)
    assert tensors[0].normal.shape == (1, 2, 3)
    assert tensors[0].normal.values == (0.0, 0.0, -1.0, 0.0, 0.0, -1.0)


def test_capture_tensors_create_masked_per_pixel_render_targets():
    frame = TrainingFrame(
        id="frame_000001",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
        semantic_label="fixture",
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    tensors = CaptureFrameTensors(
        frame_id="frame_000001",
        image=CaptureTensor("image.ppm", "Netpbm", "stdlib", 2, 1, 3, (1.0, 0.0, 0.0, 0.0, 0.5, 0.5)),
        depth=CaptureTensor("depth.pgm", "Netpbm", "stdlib", 2, 1, 1, (0.25, 0.75)),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 2, 1, 1, (1.0, 0.0)),
        normal=CaptureTensor("normal.bin", "COLMAP_DENSE_MAP", "stdlib", 2, 1, 3, (0.0, 0.0, -1.0, 0.0, 1.0, 0.0)),
    )

    targets = capture_tensors_to_render_targets((frame,), (tensors,))

    assert len(targets) == 1
    assert targets[0].pixel == (0, 0)
    assert targets[0].render_target.target_color == (1.0, 0.0, 0.0)
    assert targets[0].render_target.target_depth == 0.25
    assert targets[0].render_target.target_semantic_id == "fixture"
    assert targets[0].render_target.target_normal == (0.0, 0.0, -1.0)
    assert targets[0].mask_value == 1.0
    assert targets[0].target_normal == (0.0, 0.0, -1.0)
    assert targets[0].render_target.ray.origin == (0.0, 0.0, -2.0)
    assert targets[0].render_target.ray.direction == (0.0, 0.0, 1.0)


def test_capture_tensor_sampling_plan_counts_tiles_and_masks():
    frame = TrainingFrame(
        id="frame_000001",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
    )
    tensors = CaptureFrameTensors(
        frame_id="frame_000001",
        image=CaptureTensor("image.ppm", "Netpbm", "stdlib", 2, 1, 3, (1.0, 0.0, 0.0, 0.0, 0.5, 0.5)),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 2, 1, 1, (1.0, 0.0)),
    )

    plan = plan_capture_tensor_sampling((frame,), (tensors,), tile_size=1)
    payload = plan.to_dict()

    assert plan.total_sampled_pixel_count == 1
    assert plan.total_masked_pixel_count == 1
    assert payload["format"] == "AURA_CAPTURE_SAMPLING_PLAN"
    assert payload["tileCount"] == 2
    assert payload["tiles"][0]["sampledPixelCount"] == 1
    assert payload["tiles"][1]["maskedPixelCount"] == 1


def test_capture_tensors_to_training_dataset_reuses_loaded_tensor_batch(tmp_path):
    manifest = load_capture_manifest(_write_asset_manifest(tmp_path))
    tensors = load_capture_asset_tensors(manifest)

    dataset = capture_tensors_to_training_dataset(manifest, tensors)

    assert dataset.frames[0].target_color == (0.5, 0.25, 0.25)
    assert dataset.frames[0].target_depth == 0.75
    assert {region.fallback_source for region in dataset.regions}.issuperset(
        {"capture-manifest", "capture-feature-proposal", "capture-depth-prior", "capture-mask-prior"}
    )


def test_capture_tensors_to_training_dataset_rejects_mismatched_tensor_batch(tmp_path):
    manifest = load_capture_manifest(_write_asset_manifest(tmp_path))
    tensors = load_capture_asset_tensors(manifest)
    mismatched = (CaptureFrameTensors(frame_id="unknown_frame", image=tensors[0].image),)

    with pytest.raises(ValueError, match="missing manifest frame ids"):
        capture_tensors_to_training_dataset(manifest, mismatched)


def test_capture_tensors_use_frame_depth_when_pixel_depth_is_missing():
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("image.ppm", "Netpbm", "stdlib", 1, 1, 3, (0.25, 0.5, 0.75)),
        depth=CaptureTensor("depth.pgm", "Netpbm", "stdlib", 1, 1, 1, (0.0,)),
    )

    targets = capture_tensors_to_render_targets((frame,), (tensors,))

    assert targets[0].render_target.target_depth == 2.0


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
    assert assets[0].min_depth == 128 / 255
    assert assets[0].max_depth == 1.0
    assert assets[0].depth_coverage == 1.0
    assert len(assets[0].depth_bins) == 2
    assert assets[0].mask_coverage == 0.5
    assert dataset.frames[0].target_color == (0.5, 128 / 510, 128 / 510)
    assert dataset.frames[0].target_depth == 383 / 510


def test_capture_manifest_depth_asset_regions_become_native_surface_evidence(tmp_path):
    manifest = load_capture_manifest(_write_asset_manifest(tmp_path))
    dataset = manifest.to_training_dataset(load_assets=True)
    frame_by_id = {frame.id: frame for frame in dataset.frames}
    samples = tuple(region.to_evidence_sample(frame_by_id[region.frame_id]) for region in dataset.regions)

    scene = decompose_evidence(samples)
    by_id = {element.id: element for element in scene.elements}

    assert by_id["frame_000001_depth_prior_0"].carrier_id == "surface"
    assert by_id["frame_000001_depth_prior_1"].carrier_id == "surface"
    assert by_id["frame_000001_depth_prior_0"].metadata["frame_id"] == "frame_000001"
    assert by_id["frame_000001_depth_prior_0"].payload["type"] == "surface_cell"
    assert scene.semantic_graph.nodes[0].element_ids == (
        "frame_000001_depth_prior_0",
        "frame_000001_depth_prior_1",
        "frame_000001_mask_semantic",
    )


def test_capture_manifest_mask_asset_regions_become_native_semantic_evidence(tmp_path):
    manifest = load_capture_manifest(_write_asset_manifest(tmp_path))
    dataset = manifest.to_training_dataset(load_assets=True)
    frame_by_id = {frame.id: frame for frame in dataset.frames}
    samples = tuple(region.to_evidence_sample(frame_by_id[region.frame_id]) for region in dataset.regions)

    scene = decompose_evidence(samples)
    by_id = {element.id: element for element in scene.elements}

    assert by_id["frame_000001_mask_semantic"].carrier_id == "semantic"
    assert by_id["frame_000001_mask_semantic"].semantic_id == "fixture"
    assert by_id["frame_000001_mask_semantic"].payload["type"] == "semantic_feature"
    assert scene.semantic_graph.nodes[0].label == "fixture"


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
        assert "optional tensor asset backend" in str(exc)
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


def test_reconstruct_capture_manifest_cli_uses_per_pixel_asset_targets(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)
    package_dir = tmp_path / "reconstruct-capture.aura"

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
            "1",
            "--load-assets",
            "--max-targets-per-frame",
            "2",
            "--tile-size",
            "1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    report = json.loads((package_dir / "training_report.json").read_text(encoding="utf-8"))

    assert "capture_tensor_pixel_targets" in report["stages"]
    assert "capture_tensor_pixels" in report["sources"]
    assert len(report["iterations"][0]["predictions"]) == 1
    assert report["iterations"][0]["predictions"][0]["target_color"] == [1.0, 0.0, 0.0]
    assert report["iterations"][0]["predictions"][0]["target_depth"] == 0.5
    assert report["captureSamplingPlan"]["tileSize"] == 1
    assert report["captureSamplingPlan"]["tileCount"] == 2
    assert report["captureSamplingPlan"]["totalSampledPixelCount"] == 1
    assert report["captureSamplingPlan"]["totalMaskedPixelCount"] == 1


def test_torch_optimize_capture_manifest_cli_reports_install_hint_when_unavailable(tmp_path):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")
    manifest_path = _write_asset_manifest(tmp_path)
    package_dir = tmp_path / "torch-optimize-capture.aura"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "torch-optimize-capture-manifest",
            str(manifest_path),
            "--output-dir",
            str(package_dir),
            "--iterations",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode != 0
    assert "torch" in result.stderr.lower()


def test_torch_optimize_capture_manifest_cli_writes_package_and_report(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    manifest_path = _write_asset_manifest(tmp_path)
    package_dir = tmp_path / "torch-optimize-capture.aura"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "torch-optimize-capture-manifest",
            str(manifest_path),
            "--output-dir",
            str(package_dir),
            "--iterations",
            "2",
            "--max-targets-per-frame",
            "2",
            "--tile-size",
            "1",
            "--device",
            "cpu",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    package = load_package(package_dir)
    report = json.loads((package_dir / "torch_training_report.json").read_text(encoding="utf-8"))

    assert package.asset.name == "torch_optimize_capture"
    assert report["format"] == "AURA_CORE_TORCH_OPTIMIZATION_REPORT"
    assert "torch_reference_optimization" in report["stages"]
    assert report["steps"][0]["sample_count"] == 1
    assert report["steps"][0]["normal_loss"] == 0.0
    assert report["finalLoss"] == report["steps"][-1]["total_loss"]
    assert report["captureSamplingPlan"]["tileSize"] == 1
    assert report["captureSamplingPlan"]["tileCount"] == 2


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


def test_inspect_capture_tensors_cli_reports_tensor_shapes(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "inspect-capture-tensors", str(manifest_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload[0]["frameId"] == "frame_000001"
    assert payload[0]["image"]["shape"] == [1, 2, 3]
    assert payload[0]["image"]["valueCount"] == 6
    assert payload[0]["depth"]["sampleValues"] == [0.5, 1.0]


def test_plan_capture_sampling_cli_reports_tile_schedule(tmp_path):
    manifest_path = _write_asset_manifest(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "plan-capture-sampling",
            str(manifest_path),
            "--tile-size",
            "1",
            "--pixel-stride",
            "1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_CAPTURE_SAMPLING_PLAN"
    assert payload["tileSize"] == 1
    assert payload["tileCount"] == 2
    assert payload["totalSampledPixelCount"] == 1
    assert payload["totalMaskedPixelCount"] == 1


def test_reconstruct_capture_manifest_cli_reuses_single_loaded_tensor_batch(tmp_path, monkeypatch):
    import aura.cli as cli

    manifest_path = _write_asset_manifest(tmp_path)
    original_loader = cli.load_capture_asset_tensors
    calls = 0

    def counted_loader(manifest):
        nonlocal calls
        calls += 1
        return original_loader(manifest)

    monkeypatch.setattr(cli, "load_capture_asset_tensors", counted_loader)

    exit_code = cli.main(
        [
            "reconstruct-capture-manifest",
            str(manifest_path),
            "--load-assets",
            "--output-dir",
            str(tmp_path / "reconstruct.aura"),
            "--iterations",
            "1",
            "--max-targets-per-frame",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == 1
    assert (tmp_path / "reconstruct.aura" / "training_report.json").exists()


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
        "P2\n2 1\n2\n2 0\n",
        encoding="ascii",
    )
    _write_colmap_normal_map(root / "normal" / "frame_000001.bin", 2, 1, ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)))
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
    payload["frames"][0]["normal_path"] = None
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
    payload["frames"][0]["normal_path"] = None
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


def _write_colmap_normal_map(path, width, height, values):
    flat = [component for normal in values for component in normal]
    path.write_bytes(f"{width}&{height}&3&".encode("ascii") + struct.pack("<" + "f" * len(flat), *flat))
