import pytest

from aura import CaptureFrameTensors, CaptureTensor, TrainingFrame
from aura.training_targets import capture_tensors_to_render_targets, plan_capture_tensor_sampling


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
