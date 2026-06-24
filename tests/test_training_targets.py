import importlib.util
from array import array

import pytest

from aura import CaptureFrameTensors, CaptureTensor, TrainingFrame
from aura.optimize import RenderTarget
from aura.ray import Ray
from aura.training_targets import (
    CapturePackedRenderBatch,
    CapturePackedRenderSourceWindow,
    CapturePixelTarget,
    CaptureSamplingBatch,
    CaptureSamplingPlan,
    CaptureSamplingTile,
    _packed_buffer_metadata,
    _pixel_ray_direction,
    _normalize,
    _require_buffer_length,
    _scalar_at,
    _rgb_at,
    _normal_at,
    capture_tensors_to_packed_render_batches,
    capture_tensors_to_render_targets,
    plan_capture_tensor_sampling,
    sampling_coverage_report,
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
    plan = plan_capture_tensor_sampling(
        (frame,),
        (tensors,),
        tile_size=3,
        max_targets_per_batch=2,
    )
    batches = capture_tensors_to_packed_render_batches(
        (frame,),
        (tensors,),
        tile_size=3,
        max_targets_per_batch=2,
        sampling_plan=plan,
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


def test_packed_render_batches_reject_mismatched_sampling_plan():
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
    )
    plan = plan_capture_tensor_sampling(
        (frame,),
        (tensors,),
        tile_size=1,
        max_targets_per_batch=1,
    )

    with pytest.raises(ValueError, match="sampling_plan tile_size"):
        capture_tensors_to_packed_render_batches(
            (frame,),
            (tensors,),
            tile_size=2,
            max_targets_per_batch=1,
            sampling_plan=plan,
        )


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


def test_packed_render_batches_max_targets_per_frame_does_not_spill_into_next_tile():
    """Regression: tile-sample spill when max_targets_per_frame truncates a tile.

    When max_targets_per_frame < tile pixel count, the sampling plan marks the
    tile as having only N sampled pixels, but _append_tile_samples_to_packed_batch
    used to keep counting into the next frame's target slots, producing too many
    samples for the batch and triggering CapturePackedRenderBatch validation.
    """
    frames = []
    tensors_list = []
    for i in range(3):
        frame = TrainingFrame(
            id=f"frame{i}",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.1, 0.1, 0.1),
            target_depth=2.0,
        )
        # 4×4 image so tile has 16 pixels, but max_targets_per_frame=2
        data = tuple([1.0, 0.0, 0.0] * 16)
        tensors = CaptureFrameTensors(
            frame_id=f"frame{i}",
            image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 4, 4, 3, data),
        )
        frames.append(frame)
        tensors_list.append(tensors)

    # 3 frames × 2 targets = 6 total; batch_size=4 → 1 full batch + 1 partial
    batches = capture_tensors_to_packed_render_batches(
        frames,
        tensors_list,
        tile_size=4,
        max_targets_per_frame=2,
        max_targets_per_batch=4,
    )
    total = sum(b.target_count for b in batches)
    assert total == 6, f"expected 6 targets total, got {total}"
    for b in batches:
        assert len(b.frame_indices) == b.target_count, (
            f"batch target_count={b.target_count} but frame_indices has {len(b.frame_indices)} entries"
        )


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


# ---------------------------------------------------------------------------
# CapturePixelTarget.to_dict — line 27 (target_normal is not None branch)
# ---------------------------------------------------------------------------


def test_capture_pixel_target_to_dict_includes_normal_when_present():
    rt = RenderTarget(
        frame_id="f",
        ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        target_color=(1.0, 0.0, 0.0),
        target_depth=1.0,
        target_semantic_id=None,
    )
    cpt = CapturePixelTarget(
        frame_id="f",
        pixel=(0, 0),
        render_target=rt,
        mask_value=0.5,
        target_normal=(0.0, 1.0, 0.0),
    )
    d = cpt.to_dict()
    assert d["targetNormal"] == [0.0, 1.0, 0.0]
    assert d["frameId"] == "f"
    assert d["maskValue"] == 0.5


def test_capture_pixel_target_to_dict_none_normal():
    rt = RenderTarget(
        frame_id="f",
        ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        target_color=(1.0, 0.0, 0.0),
        target_depth=1.0,
        target_semantic_id=None,
    )
    cpt = CapturePixelTarget(frame_id="f", pixel=(0, 0), render_target=rt)
    d = cpt.to_dict()
    assert d["targetNormal"] is None


# ---------------------------------------------------------------------------
# CapturePackedRenderSourceWindow.__post_init__ validation — lines 108-116
# ---------------------------------------------------------------------------


def _make_source_window(**overrides):
    defaults = dict(
        frame_id="f",
        tile_index=0,
        tile_origin=(0, 0),
        tile_size=(4, 4),
        batch_target_offset=0,
        target_offset=0,
        target_count=1,
    )
    defaults.update(overrides)
    return CapturePackedRenderSourceWindow(**defaults)


def test_packed_render_source_window_rejects_negative_tile_index():
    with pytest.raises(ValueError, match="tile_index cannot be negative"):
        _make_source_window(tile_index=-1)


def test_packed_render_source_window_rejects_negative_batch_target_offset():
    with pytest.raises(ValueError, match="batch_target_offset cannot be negative"):
        _make_source_window(batch_target_offset=-1)


def test_packed_render_source_window_rejects_negative_target_offset():
    with pytest.raises(ValueError, match="target_offset cannot be negative"):
        _make_source_window(target_offset=-1)


def test_packed_render_source_window_rejects_zero_target_count():
    with pytest.raises(ValueError, match="target_count must be positive"):
        _make_source_window(target_count=0)


def test_packed_render_source_window_rejects_zero_tile_size():
    with pytest.raises(ValueError, match="tile_size must be positive"):
        _make_source_window(tile_size=(0, 4))


# ---------------------------------------------------------------------------
# CapturePackedRenderBatch.__post_init__ validation — lines 154-189
# ---------------------------------------------------------------------------


def _make_packed_batch(**overrides):
    """Create a minimal valid CapturePackedRenderBatch, then apply overrides."""
    defaults = dict(
        batch_index=0,
        frame_ids=("f",),
        frame_semantic_ids=(None,),
        target_offset=0,
        target_count=1,
        max_target_count=2,
        frame_indices=array("q", [0]),
        pixel_xy=array("q", [0, 0]),
        ray_origins=array("d", [0.0, 0.0, 0.0]),
        ray_directions=array("d", [0.0, 0.0, 1.0]),
        target_color=array("d", [1.0, 0.0, 0.0]),
        target_depth=array("d", [1.0]),
    )
    defaults.update(overrides)
    return CapturePackedRenderBatch(**defaults)


def test_packed_render_batch_rejects_negative_target_count():
    with pytest.raises(ValueError, match="target_count cannot be negative"):
        _make_packed_batch(target_count=-1, frame_indices=array("q"), pixel_xy=array("q"), ray_origins=array("d"), ray_directions=array("d"), target_color=array("d"), target_depth=array("d"))


def test_packed_render_batch_rejects_zero_max_target_count():
    with pytest.raises(ValueError, match="max_target_count must be positive"):
        _make_packed_batch(max_target_count=0, target_count=0, frame_indices=array("q"), pixel_xy=array("q"), ray_origins=array("d"), ray_directions=array("d"), target_color=array("d"), target_depth=array("d"))


def test_packed_render_batch_rejects_count_exceeding_max():
    with pytest.raises(ValueError, match="exceeds max_target_count"):
        _make_packed_batch(
            target_count=3,
            max_target_count=2,
            frame_indices=array("q", [0, 0, 0]),
            pixel_xy=array("q", [0] * 6),
            ray_origins=array("d", [0.0] * 9),
            ray_directions=array("d", [0.0, 0.0, 1.0] * 3),
            target_color=array("d", [1.0, 0.0, 0.0] * 3),
            target_depth=array("d", [1.0, 1.0, 1.0]),
        )


def test_packed_render_batch_rejects_mismatched_semantic_ids():
    with pytest.raises(ValueError, match="frame semantic ids must match"):
        _make_packed_batch(frame_semantic_ids=())


def test_packed_render_batch_rejects_normal_without_normal_present():
    with pytest.raises(ValueError, match="target_normal_present is required"):
        _make_packed_batch(
            target_normal=array("d", [0.0, 0.0, 1.0]),
        )


def test_packed_render_batch_rejects_normal_present_without_normal():
    with pytest.raises(ValueError, match="target_normal is required with target_normal_present"):
        _make_packed_batch(
            target_normal_present=array("B", [1]),
        )


def test_packed_render_batch_rejects_out_of_range_frame_index():
    with pytest.raises(ValueError, match="frame index is out of range"):
        _make_packed_batch(frame_indices=array("q", [5]))


def test_packed_render_batch_rejects_source_window_unknown_frame():
    sw = _make_source_window(frame_id="unknown", target_count=1)
    with pytest.raises(ValueError, match="references unknown frame id"):
        _make_packed_batch(source_windows=(sw,))


def test_packed_render_batch_rejects_non_contiguous_source_windows():
    sw1 = _make_source_window(frame_id="f", batch_target_offset=0, target_count=1)
    sw2 = _make_source_window(frame_id="f", batch_target_offset=5, target_count=1)  # gap
    with pytest.raises(ValueError, match="source windows must be contiguous"):
        _make_packed_batch(
            target_count=2,
            frame_indices=array("q", [0, 0]),
            pixel_xy=array("q", [0] * 4),
            ray_origins=array("d", [0.0] * 6),
            ray_directions=array("d", [0.0, 0.0, 1.0] * 2),
            target_color=array("d", [1.0, 0.0, 0.0] * 2),
            target_depth=array("d", [1.0, 1.0]),
            source_windows=(sw1, sw2),
        )


def test_packed_render_batch_rejects_source_windows_not_covering_target_count():
    sw = _make_source_window(frame_id="f", batch_target_offset=0, target_count=1)
    with pytest.raises(ValueError, match="source windows must cover target_count"):
        _make_packed_batch(
            target_count=2,
            frame_indices=array("q", [0, 0]),
            pixel_xy=array("q", [0] * 4),
            ray_origins=array("d", [0.0] * 6),
            ray_directions=array("d", [0.0, 0.0, 1.0] * 2),
            target_color=array("d", [1.0, 0.0, 0.0] * 2),
            target_depth=array("d", [1.0, 1.0]),
            source_windows=(sw,),
        )


# ---------------------------------------------------------------------------
# CaptureSamplingPlan.__post_init__ validation — lines 240-265
# ---------------------------------------------------------------------------


def _make_tile(tile_index: int, target_offset: int, sampled: int, candidate: int = None) -> CaptureSamplingTile:
    if candidate is None:
        candidate = sampled
    return CaptureSamplingTile(
        frame_id="frame",
        origin=(0, 0),
        size=(4, 4),
        sampled_pixel_count=sampled,
        tile_index=tile_index,
        candidate_pixel_count=candidate,
        target_offset=target_offset,
    )


def _make_batch(batch_index: int, tile_indices: tuple, target_offset: int, target_count: int, max_target_count: int) -> CaptureSamplingBatch:
    return CaptureSamplingBatch(
        batch_index=batch_index,
        tile_indices=tile_indices,
        target_offset=target_offset,
        target_count=target_count,
        max_target_count=max_target_count,
    )


def test_capture_sampling_plan_rejects_non_positive_pixel_stride():
    with pytest.raises(ValueError, match="pixel_stride must be positive"):
        CaptureSamplingPlan(
            pixel_stride=0,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(),
            max_targets_per_batch=1,
        )


def test_capture_sampling_plan_rejects_non_positive_tile_size():
    with pytest.raises(ValueError, match="tile_size must be positive"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=0,
            max_targets_per_frame=None,
            tiles=(),
            max_targets_per_batch=1,
        )


def test_capture_sampling_plan_rejects_non_positive_max_targets_per_frame():
    with pytest.raises(ValueError, match="max_targets_per_frame must be positive"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=0,
            tiles=(),
            max_targets_per_batch=1,
        )


def test_capture_sampling_plan_rejects_non_positive_max_targets_per_batch():
    with pytest.raises(ValueError, match="max_targets_per_batch must be positive"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(),
            max_targets_per_batch=0,
        )


def test_capture_sampling_plan_rejects_non_contiguous_tile_indices():
    t0 = _make_tile(0, 0, 1)
    t2 = _make_tile(2, 1, 1)  # index 2 instead of 1
    with pytest.raises(ValueError, match="sampling tile indices must be contiguous"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(t0, t2),
            max_targets_per_batch=4,
        )


def test_capture_sampling_plan_rejects_over_candidate_count():
    # sampled + masked > candidate
    t = CaptureSamplingTile(
        frame_id="frame",
        origin=(0, 0),
        size=(4, 4),
        sampled_pixel_count=3,
        masked_pixel_count=2,
        tile_index=0,
        candidate_pixel_count=4,  # 3 + 2 > 4
        target_offset=0,
    )
    with pytest.raises(ValueError, match="candidate counts cannot be smaller"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(t,),
            max_targets_per_batch=4,
        )


def test_capture_sampling_plan_rejects_non_zero_first_tile_offset():
    t = _make_tile(0, 5, 1)  # target_offset=5 but should be 0
    with pytest.raises(ValueError, match="first sampling tile target offset must be zero"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(t,),
            max_targets_per_batch=4,
        )


def test_capture_sampling_plan_rejects_non_contiguous_tile_offsets():
    t0 = _make_tile(0, 0, 2)
    t1 = _make_tile(1, 5, 2)  # should be 2 (0 + 2)
    with pytest.raises(ValueError, match="tile target offsets must be contiguous"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(t0, t1),
            max_targets_per_batch=4,
        )


def test_capture_sampling_plan_rejects_batch_exceeding_max_target_count():
    t = _make_tile(0, 0, 2)
    b = _make_batch(0, (0,), 0, 3, 2)  # target_count=3 > max_target_count=2
    with pytest.raises(ValueError, match="sampling batch exceeds max_target_count"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(t,),
            max_targets_per_batch=2,
            batches=(b,),
        )


def test_capture_sampling_plan_rejects_batch_with_mismatched_max_target_count():
    t = _make_tile(0, 0, 2)
    b = _make_batch(0, (0,), 0, 2, 4)  # batch max=4 but plan max=2
    with pytest.raises(ValueError, match="max_target_count must match plan max_targets_per_batch"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(t,),
            max_targets_per_batch=2,
            batches=(b,),
        )


def test_capture_sampling_plan_rejects_empty_batch_tile_indices():
    t = _make_tile(0, 0, 2)
    b = CaptureSamplingBatch(
        batch_index=0,
        tile_indices=(),  # empty
        target_offset=0,
        target_count=2,
        max_target_count=2,
    )
    with pytest.raises(ValueError, match="sampling batches must reference at least one tile"):
        CaptureSamplingPlan(
            pixel_stride=1,
            tile_size=4,
            max_targets_per_frame=None,
            tiles=(t,),
            max_targets_per_batch=2,
            batches=(b,),
        )


def test_capture_sampling_plan_max_batch_target_count_zero_when_no_batches():
    plan = CaptureSamplingPlan(
        pixel_stride=1,
        tile_size=4,
        max_targets_per_frame=None,
        tiles=(),
        max_targets_per_batch=4,
        batches=(),
    )
    assert plan.max_batch_target_count == 0


# ---------------------------------------------------------------------------
# capture_tensors_to_render_targets: unknown frame (line 324), pixel_stride validation (316)
# ---------------------------------------------------------------------------


def test_capture_tensors_to_render_targets_rejects_unknown_frame():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="different_frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="unknown training frame"):
        capture_tensors_to_render_targets((frame,), (tensors,))


def test_capture_tensors_to_render_targets_rejects_non_positive_pixel_stride():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="pixel_stride must be positive"):
        capture_tensors_to_render_targets((frame,), (tensors,), pixel_stride=0)


# ---------------------------------------------------------------------------
# plan_capture_tensor_sampling: unknown frame (line 495), pixel_stride/tile_size (481-484)
# ---------------------------------------------------------------------------


def test_plan_capture_tensor_sampling_rejects_unknown_frame():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="other",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="unknown training frame"):
        plan_capture_tensor_sampling((frame,), (tensors,))


def test_plan_capture_tensor_sampling_rejects_non_positive_pixel_stride():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="pixel_stride must be positive"):
        plan_capture_tensor_sampling((frame,), (tensors,), pixel_stride=0)


def test_plan_capture_tensor_sampling_rejects_non_positive_tile_size():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="tile_size must be positive"):
        plan_capture_tensor_sampling((frame,), (tensors,), tile_size=0)


# ---------------------------------------------------------------------------
# _validate_sampling_plan_matches_request — lines 460, 462, 467
# ---------------------------------------------------------------------------


def test_packed_render_batches_reject_mismatched_pixel_stride():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 2, 1, 3, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    )
    plan = plan_capture_tensor_sampling((frame,), (tensors,), pixel_stride=1, tile_size=2, max_targets_per_batch=1)
    with pytest.raises(ValueError, match="sampling_plan pixel_stride"):
        capture_tensors_to_packed_render_batches(
            (frame,), (tensors,), pixel_stride=2, tile_size=2, max_targets_per_batch=1, sampling_plan=plan
        )


def test_packed_render_batches_reject_mismatched_max_targets_per_frame():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 2, 1, 3, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    )
    plan = plan_capture_tensor_sampling((frame,), (tensors,), tile_size=2, max_targets_per_batch=2)
    with pytest.raises(ValueError, match="sampling_plan max_targets_per_frame"):
        capture_tensors_to_packed_render_batches(
            (frame,), (tensors,), tile_size=2, max_targets_per_frame=1, max_targets_per_batch=2, sampling_plan=plan
        )


def test_packed_render_batches_reject_mismatched_max_targets_per_batch():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 2, 1, 3, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    )
    plan = plan_capture_tensor_sampling((frame,), (tensors,), tile_size=2, max_targets_per_batch=1)
    with pytest.raises(ValueError, match="sampling_plan max_targets_per_batch"):
        capture_tensors_to_packed_render_batches(
            (frame,), (tensors,), tile_size=2, max_targets_per_batch=2, sampling_plan=plan
        )


# ---------------------------------------------------------------------------
# _validate_tensor_dimensions — line 721 (mismatched dimensions)
# ---------------------------------------------------------------------------


def test_validate_tensor_dimensions_rejects_mismatched_depth():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 2, 2, 3, (1.0, 0.0, 0.0) * 4),
        depth=CaptureTensor("depth.pgm", "Netpbm", "stdlib", 1, 1, 1, (1.0,)),  # wrong size
    )
    with pytest.raises(ValueError, match="depth tensor dimensions must match"):
        capture_tensors_to_render_targets((frame,), (tensors,))


def test_validate_tensor_dimensions_rejects_mismatched_mask():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 2, 2, 3, (1.0, 0.0, 0.0) * 4),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 1, 1, 1, (1.0,)),  # wrong size
    )
    with pytest.raises(ValueError, match="mask tensor dimensions must match"):
        capture_tensors_to_render_targets((frame,), (tensors,))


def test_validate_tensor_dimensions_rejects_mismatched_normal():
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 2, 2, 3, (1.0, 0.0, 0.0) * 4),
        normal=CaptureTensor("normal.ppm", "Netpbm", "stdlib", 1, 1, 3, (0.0, 0.0, 1.0)),  # wrong size
    )
    with pytest.raises(ValueError, match="normal tensor dimensions must match"):
        capture_tensors_to_render_targets((frame,), (tensors,))


# ---------------------------------------------------------------------------
# _rgb_at — line 726 (fewer than 3 channels)
# ---------------------------------------------------------------------------


def test_rgb_at_rejects_fewer_than_three_channels():
    tensor = CaptureTensor("img.pgm", "Netpbm", "stdlib", 1, 1, 1, (0.5,))
    with pytest.raises(ValueError, match="image tensor must have at least three channels"):
        _rgb_at(tensor, 0, 0)


# ---------------------------------------------------------------------------
# _scalar_at — line 735 (not 1 channel)
# ---------------------------------------------------------------------------


def test_scalar_at_rejects_multi_channel_tensor():
    tensor = CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="scalar tensor must have one channel"):
        _scalar_at(tensor, 0, 0)


# ---------------------------------------------------------------------------
# _normal_at — line 743 (not 3 channels)
# ---------------------------------------------------------------------------


def test_normal_at_rejects_non_three_channel_tensor():
    tensor = CaptureTensor("img.pgm", "Netpbm", "stdlib", 1, 1, 1, (0.5,))
    with pytest.raises(ValueError, match="normal tensor must have three channels"):
        _normal_at(tensor, 0, 0)


# ---------------------------------------------------------------------------
# _pixel_ray_direction — line 758 (forward nearly parallel to up → use right fallback)
# ---------------------------------------------------------------------------


def test_pixel_ray_direction_uses_fallback_when_forward_parallel_to_world_up():
    # Camera looking straight up: forward = (0, 1, 0), parallel to world up (0,1,0)
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, -2.0, 0.0),
        look_at=(0.0, 0.0, 0.0),  # forward = (0,1,0)
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    # Should not raise and should return a normalised direction
    direction = _pixel_ray_direction(frame, 0, 0)
    mag = sum(v * v for v in direction) ** 0.5
    assert abs(mag - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _normalize — line 783 (zero vector)
# ---------------------------------------------------------------------------


def test_normalize_raises_on_zero_vector():
    with pytest.raises(ValueError, match="cannot normalize zero vector"):
        _normalize((0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# _packed_buffer_metadata — line 798 (None input returns None)
# ---------------------------------------------------------------------------


def test_packed_render_batch_to_dict_none_mask_and_normal():
    """to_dict with no mask/normal hits the None branch in _packed_buffer_metadata."""
    batch = _make_packed_batch()
    d = batch.to_dict()
    assert d["targetMask"] is None
    assert d["targetNormal"] is None
    assert d["targetNormalPresent"] is None


# ---------------------------------------------------------------------------
# _append_tile_samples_to_packed_batch: null normal branch (lines 682-683)
# Covered by providing normal tensor with a zero-normal pixel that normalises to None path
# We just need a packed batch render with a normal tensor present but pixel normal absent
# ---------------------------------------------------------------------------


def test_packed_render_batch_normal_none_fills_zeros():
    """Normal tensor present but pixel normal is a zero vector: target_normal_present=0."""
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    # Normal tensor with a zero vector — _normal_at will raise on normalize, so use
    # a non-zero but unrelated pixel. We test by having normal=None in frame tensors
    # while another frame has normal, triggering include_normal=True but giving None normal.
    frame2 = TrainingFrame(
        id="frame2",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.1, 0.1),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    tensors1 = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
        # no normal tensor → normal=None for this frame's pixels
    )
    tensors2 = CaptureFrameTensors(
        frame_id="frame2",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (0.0, 1.0, 0.0)),
        normal=CaptureTensor("normal.ppm", "Netpbm", "stdlib", 1, 1, 3, (0.0, 0.0, 1.0)),
    )
    batches = capture_tensors_to_packed_render_batches(
        (frame, frame2),
        (tensors1, tensors2),
        tile_size=1,
        max_targets_per_batch=4,
    )
    all_normal_present = [v for b in batches for v in b.target_normal_present]
    # frame has no normal → present=0; frame2 has normal → present=1
    assert 0 in all_normal_present
    assert 1 in all_normal_present


# ---------------------------------------------------------------------------
# capture_tensors_to_packed_render_batches: batch.target_count == 0 (line 404)
# ---------------------------------------------------------------------------


def test_packed_render_batches_skips_zero_count_batches():
    """A sampling plan with a zero-count batch should not produce a packed batch for it."""
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 1, 3, (1.0, 0.0, 0.0)),
    )
    plan = plan_capture_tensor_sampling((frame,), (tensors,), tile_size=1, max_targets_per_batch=1)
    # inject a batch with zero target_count manually via a modified plan
    zero_batch = CaptureSamplingBatch(
        batch_index=99,
        tile_indices=(0,),
        target_offset=0,
        target_count=0,
        max_target_count=1,
    )
    real_batch = plan.batches[0]
    new_plan = CaptureSamplingPlan(
        pixel_stride=plan.pixel_stride,
        tile_size=plan.tile_size,
        max_targets_per_frame=plan.max_targets_per_frame,
        tiles=plan.tiles,
        max_targets_per_batch=plan.max_targets_per_batch,
        batches=(zero_batch, real_batch),
    )
    batches = capture_tensors_to_packed_render_batches(
        (frame,), (tensors,), tile_size=1, max_targets_per_batch=1, sampling_plan=new_plan
    )
    # Only the real batch should be present (zero-count one skipped)
    assert len(batches) == 1
    assert batches[0].target_count == 1


# ---------------------------------------------------------------------------
# capture_tensors_to_packed_render_batches: batch with tile that has 0 samples (line 413)
# ---------------------------------------------------------------------------


def test_packed_render_batch_skips_tiles_with_zero_sampled_pixels():
    """Tile with sampled_pixel_count=0 should be skipped inside a batch."""
    frame = _training_frame()
    # 2-pixel image; mask zeroes out second pixel → tile 0 has 1 sample, tile 1 has 0
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 2, 1, 3, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 2, 1, 1, (1.0, 0.0)),
    )
    batches = capture_tensors_to_packed_render_batches(
        (frame,), (tensors,), tile_size=1, max_targets_per_batch=2
    )
    assert sum(b.target_count for b in batches) == 1


# ---------------------------------------------------------------------------
# plan_capture_tensor_sampling: stop_frame outer break (line 501)
# Need multi-row image where max_targets_per_frame is hit in row 0 tile scan
# so that the outer tile_y loop hits the stop_frame break on next iteration.
# ---------------------------------------------------------------------------


def test_plan_capture_tensor_sampling_stop_frame_breaks_outer_tile_y_loop():
    """max_targets_per_frame=1 on a 2-row image stops tile_y outer loop via stop_frame."""
    frame = _training_frame()
    # 1×2 image (width=1, height=2): 2 rows of 1 pixel each with tile_size=1
    # After sampling pixel (0,0) produced=1 == max_targets_per_frame=1, stop_frame=True
    # Next tile_y iteration should hit the `if stop_frame: break` at line 501
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor("img.ppm", "Netpbm", "stdlib", 1, 2, 3, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    )
    plan = plan_capture_tensor_sampling(
        (frame,), (tensors,), tile_size=1, max_targets_per_frame=1, max_targets_per_batch=2
    )
    # Only 1 pixel sampled (not 2)
    assert plan.total_sampled_pixel_count == 1


# ---------------------------------------------------------------------------
# _require_buffer_length — line 793 (raises ValueError)
# ---------------------------------------------------------------------------


def test_require_buffer_length_raises_on_wrong_length():
    with pytest.raises(ValueError, match="packed render batch foo length must be 3"):
        _require_buffer_length([1, 2], 3, "foo")


# ---------------------------------------------------------------------------
# _packed_buffer_metadata — line 798 (values is None → return None)
# ---------------------------------------------------------------------------


def test_packed_buffer_metadata_returns_none_for_none_input():
    result = _packed_buffer_metadata(None, "float64", (4,))
    assert result is None


def test_packed_buffer_metadata_returns_dict_for_values():
    result = _packed_buffer_metadata([1.0, 2.0, 3.0], "float64", (3,))
    assert result is not None
    assert result["dtype"] == "float64"
    assert result["shape"] == [3]
    assert result["valueCount"] == 3


# ---------------------------------------------------------------------------
# sampling_coverage_report — convergence diagnostic (the README (carrier-gradient/convergence notes))
# ---------------------------------------------------------------------------


def _full_coverage_tensors(frame_id: str = "frame") -> CaptureFrameTensors:
    """A 4x2 all-visible RGB capture (8 candidate pixels, none masked)."""
    width, height = 4, 2
    image_values = tuple(float((x + y) % 2) for y in range(height) for x in range(width) for _ in range(3))
    return CaptureFrameTensors(
        frame_id=frame_id,
        image=CaptureTensor("image.ppm", "Netpbm", "stdlib", width, height, 3, image_values),
    )


def test_sampling_coverage_report_full_when_uncapped():
    """With no per-frame cap every candidate pixel is supervised → coverage 1.0."""
    frame = _training_frame()
    tensors = _full_coverage_tensors()

    plan = plan_capture_tensor_sampling((frame,), (tensors,), tile_size=4, max_targets_per_batch=8)
    report = sampling_coverage_report(plan)

    assert report["format"] == "AURA_SAMPLING_COVERAGE_REPORT"
    assert report["capacityPixelCount"] == 8
    assert report["sampledPixelCount"] == 8
    assert report["maskedPixelCount"] == 0
    assert report["coverageFraction"] == 1.0
    assert report["frameCount"] == 1
    assert report["perFrame"]["frame"]["coverageFraction"] == 1.0


def test_sampling_coverage_report_quantifies_per_frame_cap_starvation():
    """The convergence root cause: a tight max_targets_per_frame cap supervises
    only a small top-left slice of the frame, leaving most pixels — and thus
    most carriers — unsupervised. The diagnostic must surface that as a small
    coverage fraction (< 1.0)."""
    frame = _training_frame()
    tensors = _full_coverage_tensors()

    capped = plan_capture_tensor_sampling(
        (frame,), (tensors,), tile_size=4, max_targets_per_frame=2, max_targets_per_batch=2
    )
    report = sampling_coverage_report(capped)

    # 2 of the tile's 8-pixel capacity supervised → coverage = 0.25, far from full.
    assert report["sampledPixelCount"] == 2
    assert report["capacityPixelCount"] == 8
    assert report["coverageFraction"] == 0.25
    assert report["coverageFraction"] < 1.0

    # And it is strictly worse than the uncapped plan on the same frame —
    # the property the fix in the README (carrier-gradient/convergence notes) must improve.
    uncapped = plan_capture_tensor_sampling(
        (frame,), (tensors,), tile_size=4, max_targets_per_batch=8
    )
    assert report["coverageFraction"] < sampling_coverage_report(uncapped)["coverageFraction"]


def test_sampling_coverage_report_accounts_for_masked_pixels():
    """Masked pixels are candidates that are never sampled; coverage uses the
    full candidate count as the denominator so masking lowers coverage."""
    frame = _training_frame()
    tensors = CaptureFrameTensors(
        frame_id="frame",
        image=CaptureTensor(
            "image.ppm", "Netpbm", "stdlib", 4, 2, 3,
            tuple(0.5 for _ in range(4 * 2 * 3)),
        ),
        # Mask out half the pixels (every other one is 0).
        mask=CaptureTensor("mask.pgm", "Netpbm", "stdlib", 4, 2, 1, (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)),
    )

    plan = plan_capture_tensor_sampling((frame,), (tensors,), tile_size=4, max_targets_per_batch=8)
    report = sampling_coverage_report(plan)

    assert report["capacityPixelCount"] == 8
    assert report["maskedPixelCount"] == 4
    assert report["sampledPixelCount"] == 4
    assert report["coverageFraction"] == 0.5
