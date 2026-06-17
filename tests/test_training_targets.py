import importlib.util

import pytest

from aura import CaptureFrameTensors, CaptureTensor, TrainingFrame
from aura.training_targets import (
    capture_tensors_to_packed_render_batches,
    capture_tensors_to_render_targets,
    plan_capture_tensor_sampling,
)


def test_capture_sampling_plan_records_deterministic_masked_tile_batches():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            "image.ppm",
            "Netpbm",
            "stdlib",
            4,
            2,
            3,
            (
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
            ),
        ),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 4, 2, 1, (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)),
    )

    plan = plan_capture_tensor_sampling((frame,), (tensors,), tile_size=2, max_targets_per_batch=3)
    payload = plan.to_dict()

    assert payload["deterministic"] is True
    assert payload["sampleOrder"] == "row-major tiles, row-major pixels"
    assert payload["maskRule"] == "sample mask values greater than 0; skip zero or negative mask values"
    assert payload["maxTargetsPerBatch"] == 3
    assert payload["batchCount"] == 2
    assert payload["maxBatchTargetCount"] == 3
    assert payload["totalCandidatePixelCount"] == 8
    assert payload["totalSampledPixelCount"] == 6
    assert payload["totalMaskedPixelCount"] == 2
    assert payload["tiles"] == [
        {
            "frameId": "frame",
            "origin": [0, 0],
            "size": [2, 2],
            "tileIndex": 0,
            "targetOffset": 0,
            "candidatePixelCount": 4,
            "sampledPixelCount": 3,
            "maskedPixelCount": 1,
            "firstSampledPixel": [0, 0],
            "lastSampledPixel": [1, 1],
        },
        {
            "frameId": "frame",
            "origin": [2, 0],
            "size": [2, 2],
            "tileIndex": 1,
            "targetOffset": 3,
            "candidatePixelCount": 4,
            "sampledPixelCount": 3,
            "maskedPixelCount": 1,
            "firstSampledPixel": [2, 0],
            "lastSampledPixel": [3, 1],
        },
    ]
    assert payload["batches"] == [
        {"batchIndex": 0, "tileIndices": [0], "targetOffset": 0, "targetCount": 3, "maxTargetCount": 3},
        {"batchIndex": 1, "tileIndices": [1], "targetOffset": 3, "targetCount": 3, "maxTargetCount": 3},
    ]


def test_capture_sampling_plan_matches_frame_limited_render_targets():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            "image.ppm",
            "Netpbm",
            "stdlib",
            3,
            1,
            3,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 3, 1, 1, (1.0, 0.0, 1.0)),
    )

    plan = plan_capture_tensor_sampling(
        (frame,),
        (tensors,),
        tile_size=3,
        max_targets_per_frame=2,
        max_targets_per_batch=2,
    )
    targets = capture_tensors_to_render_targets((frame,), (tensors,), max_targets_per_frame=2)

    assert [target.pixel for target in targets] == [(0, 0), (2, 0)]
    assert plan.total_sampled_pixel_count == len(targets)
    assert plan.total_masked_pixel_count == 1
    assert plan.tiles[0].candidate_pixel_count == 3
    assert plan.tiles[0].first_sampled_pixel == targets[0].pixel
    assert plan.tiles[0].last_sampled_pixel == targets[-1].pixel
    assert plan.batches[0].target_count == 2


def test_capture_sampling_plan_splits_oversized_tiles_into_bounded_batches():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            "image.ppm",
            "Netpbm",
            "stdlib",
            3,
            1,
            3,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
    )

    plan = plan_capture_tensor_sampling(
        (frame,),
        (tensors,),
        tile_size=3,
        max_targets_per_batch=2,
    )

    assert [batch.to_dict() for batch in plan.batches] == [
        {"batchIndex": 0, "tileIndices": [0], "targetOffset": 0, "targetCount": 2, "maxTargetCount": 2},
        {"batchIndex": 1, "tileIndices": [0], "targetOffset": 2, "targetCount": 1, "maxTargetCount": 2},
    ]
    assert plan.max_batch_target_count == 2


def test_capture_tensors_to_packed_render_batches_match_legacy_target_order():
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
        semantic_label="tooth",
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 3.0, "height": 2.0},
    )
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            "image.ppm",
            "Netpbm",
            "stdlib",
            3,
            2,
            3,
            (
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.25,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.75,
                0.75,
                0.75,
            ),
        ),
        depth=CaptureTensor("depth.pgm", "Netpbm", "stdlib", 3, 2, 1, (0.5, 0.0, 1.5, 2.0, 2.5, 3.0)),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 3, 2, 1, (1.0, 0.0, 0.5, 1.0, 1.0, 1.0)),
        normal=CaptureTensor(
            "normal.ppm",
            "Netpbm",
            "stdlib",
            3,
            2,
            3,
            (
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                -2.0,
            ),
        ),
    )

    legacy_targets = capture_tensors_to_render_targets((frame,), (tensors,))
    batches = capture_tensors_to_packed_render_batches(
        (frame,),
        (tensors,),
        tile_size=3,
        max_targets_per_batch=2,
    )

    assert [batch.target_count for batch in batches] == [2, 2, 1]
    assert all(batch.target_count <= batch.max_target_count for batch in batches)
    assert batches[0].frame_ids == ("frame",)
    assert batches[0].frame_semantic_ids == ("tooth",)
    assert batches[0].to_dict()["bounded"] is True
    assert batches[0].to_dict()["sourceWindows"] == [
        {
            "frameId": "frame",
            "tileIndex": 0,
            "tileOrigin": [0, 0],
            "tileSize": [3, 2],
            "batchTargetOffset": 0,
            "targetOffset": 0,
            "targetCount": 2,
        }
    ]
    assert batches[1].source_windows[0].target_offset == 2
    assert batches[1].source_windows[0].target_count == 2
    assert batches[2].source_windows[0].target_offset == 4
    assert batches[2].source_windows[0].target_count == 1
    packed_pixels = [
        tuple(batch.pixel_xy[index : index + 2])
        for batch in batches
        for index in range(0, len(batch.pixel_xy), 2)
    ]
    packed_depths = [depth for batch in batches for depth in batch.target_depth]
    packed_colors = [
        tuple(batch.target_color[index : index + 3])
        for batch in batches
        for index in range(0, len(batch.target_color), 3)
    ]
    packed_normal_present = [present for batch in batches for present in batch.target_normal_present]

    assert packed_pixels == [target.pixel for target in legacy_targets]
    assert packed_depths == [target.render_target.target_depth for target in legacy_targets]
    assert packed_colors == [target.render_target.target_color for target in legacy_targets]
    assert packed_normal_present == [1, 1, 1, 1, 1]
    assert all(type(batch.pixel_xy).__name__ == "array" for batch in batches)


def test_packed_render_batches_record_multi_tile_source_windows():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            "image.ppm",
            "Netpbm",
            "stdlib",
            3,
            1,
            3,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 3, 1, 1, (1.0, 1.0, 1.0)),
    )

    batches = capture_tensors_to_packed_render_batches(
        (frame,),
        (tensors,),
        tile_size=1,
        max_targets_per_batch=3,
    )

    assert [batch.target_count for batch in batches] == [3]
    assert [window.to_dict() for window in batches[0].source_windows] == [
        {
            "frameId": "frame",
            "tileIndex": 0,
            "tileOrigin": [0, 0],
            "tileSize": [1, 1],
            "batchTargetOffset": 0,
            "targetOffset": 0,
            "targetCount": 1,
        },
        {
            "frameId": "frame",
            "tileIndex": 1,
            "tileOrigin": [1, 0],
            "tileSize": [1, 1],
            "batchTargetOffset": 1,
            "targetOffset": 1,
            "targetCount": 1,
        },
        {
            "frameId": "frame",
            "tileIndex": 2,
            "tileOrigin": [2, 0],
            "tileSize": [1, 1],
            "batchTargetOffset": 2,
            "targetOffset": 2,
            "targetCount": 1,
        },
    ]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_packed_render_batch_converts_to_torch_capture_training_batch():
    from aura.torch_renderer import torch_capture_training_batch_from_packed

    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            "image.ppm",
            "Netpbm",
            "stdlib",
            2,
            1,
            3,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        ),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 2, 1, 1, (1.0, 0.0)),
    )
    packed = capture_tensors_to_packed_render_batches(
        (frame,),
        (tensors,),
        tile_size=2,
        max_targets_per_batch=1,
    )[0]

    batch = torch_capture_training_batch_from_packed(packed, device="cpu")

    assert batch.frame_ids == ("frame",)
    assert tuple(batch.frame_indices.tolist()) == (0,)
    assert batch.pixel_xy.tolist() == [[0, 0]]
    assert batch.target_color.tolist() == [[1.0, 0.0, 0.0]]
    assert batch.target_mask.tolist() == [1.0]


def test_capture_sampling_rejects_non_positive_limits():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("image.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
    )

    with pytest.raises(ValueError, match="max_targets_per_frame must be positive"):
        capture_tensors_to_render_targets((frame,), (tensors,), max_targets_per_frame=0)
    with pytest.raises(ValueError, match="max_targets_per_frame must be positive"):
        plan_capture_tensor_sampling((frame,), (tensors,), max_targets_per_frame=0)
    with pytest.raises(ValueError, match="max_targets_per_batch must be positive"):
        plan_capture_tensor_sampling((frame,), (tensors,), max_targets_per_batch=0)


def _training_frame() -> TrainingFrame:
    return TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
    )
