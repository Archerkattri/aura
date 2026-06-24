"""Capture tensor sampling, tiled sampling plans, and packed render-target batches."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from math import sqrt
from typing import Sequence

from aura.core import TrainingFrame
from aura.ingest.capture import CaptureFrameTensors, CaptureTensor
from aura.optimize import RenderTarget
from aura.ray import Ray, Vec3


@dataclass(frozen=True)
class CapturePixelTarget:
    """A single per-pixel render target derived from capture image/depth/mask tensors."""

    frame_id: str
    pixel: tuple[int, int]
    render_target: RenderTarget
    mask_value: float | None = None
    target_normal: Vec3 | None = None

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "pixel": list(self.pixel),
            "rayOrigin": list(self.render_target.ray.origin),
            "rayDirection": list(self.render_target.ray.direction),
            "targetColor": list(self.render_target.target_color),
            "targetDepth": self.render_target.target_depth,
            "targetSemanticId": self.render_target.target_semantic_id,
            "maskValue": self.mask_value,
            "targetNormal": list(self.target_normal) if self.target_normal is not None else None,
        }


@dataclass(frozen=True)
class CaptureSamplingTile:
    """A single tile's sampling metadata within a capture frame."""

    frame_id: str
    origin: tuple[int, int]
    size: tuple[int, int]
    sampled_pixel_count: int
    masked_pixel_count: int = 0
    tile_index: int = 0
    candidate_pixel_count: int = 0
    target_offset: int = 0
    first_sampled_pixel: tuple[int, int] | None = None
    last_sampled_pixel: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "origin": list(self.origin),
            "size": list(self.size),
            "tileIndex": self.tile_index,
            "targetOffset": self.target_offset,
            "candidatePixelCount": self.candidate_pixel_count,
            "sampledPixelCount": self.sampled_pixel_count,
            "maskedPixelCount": self.masked_pixel_count,
            "firstSampledPixel": (
                list(self.first_sampled_pixel) if self.first_sampled_pixel is not None else None
            ),
            "lastSampledPixel": (
                list(self.last_sampled_pixel) if self.last_sampled_pixel is not None else None
            ),
        }


@dataclass(frozen=True)
class CaptureSamplingBatch:
    """A bounded batch of tile indices referencing a contiguous target range."""

    batch_index: int
    tile_indices: tuple[int, ...]
    target_offset: int
    target_count: int
    max_target_count: int

    def to_dict(self) -> dict:
        return {
            "batchIndex": self.batch_index,
            "tileIndices": list(self.tile_indices),
            "targetOffset": self.target_offset,
            "targetCount": self.target_count,
            "maxTargetCount": self.max_target_count,
        }


@dataclass(frozen=True)
class CapturePackedRenderSourceWindow:
    """Tile target range used to build a bounded packed render batch."""

    frame_id: str
    tile_index: int
    tile_origin: tuple[int, int]
    tile_size: tuple[int, int]
    batch_target_offset: int
    target_offset: int
    target_count: int

    def __post_init__(self) -> None:
        if self.tile_index < 0:
            raise ValueError("packed render source window tile_index cannot be negative")
        if self.batch_target_offset < 0:
            raise ValueError("packed render source window batch_target_offset cannot be negative")
        if self.target_offset < 0:
            raise ValueError("packed render source window target_offset cannot be negative")
        if self.target_count <= 0:
            raise ValueError("packed render source window target_count must be positive")
        if self.tile_size[0] <= 0 or self.tile_size[1] <= 0:
            raise ValueError("packed render source window tile_size must be positive")

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "tileIndex": self.tile_index,
            "tileOrigin": list(self.tile_origin),
            "tileSize": list(self.tile_size),
            "batchTargetOffset": self.batch_target_offset,
            "targetOffset": self.target_offset,
            "targetCount": self.target_count,
        }


@dataclass(frozen=True)
class CapturePackedRenderBatch:
    """Bounded packed capture targets for tensor/CUDA ingestion."""

    batch_index: int
    frame_ids: tuple[str, ...]
    frame_semantic_ids: tuple[str | None, ...]
    target_offset: int
    target_count: int
    max_target_count: int
    frame_indices: Sequence[int]
    pixel_xy: Sequence[int]
    ray_origins: Sequence[float]
    ray_directions: Sequence[float]
    target_color: Sequence[float]
    target_depth: Sequence[float]
    target_mask: Sequence[float] | None = None
    target_normal: Sequence[float] | None = None
    target_normal_present: Sequence[int] | None = None
    sample_order: str = "row-major tiles, row-major pixels"
    source_windows: tuple[CapturePackedRenderSourceWindow, ...] = ()

    def __post_init__(self) -> None:
        if self.target_count < 0:
            raise ValueError("packed render batch target_count cannot be negative")
        if self.max_target_count <= 0:
            raise ValueError("packed render batch max_target_count must be positive")
        if self.target_count > self.max_target_count:
            raise ValueError("packed render batch exceeds max_target_count")
        if len(self.frame_semantic_ids) != len(self.frame_ids):
            raise ValueError("packed render batch frame semantic ids must match frame ids")
        _require_buffer_length(self.frame_indices, self.target_count, "frame_indices")
        _require_buffer_length(self.pixel_xy, self.target_count * 2, "pixel_xy")
        _require_buffer_length(self.ray_origins, self.target_count * 3, "ray_origins")
        _require_buffer_length(self.ray_directions, self.target_count * 3, "ray_directions")
        _require_buffer_length(self.target_color, self.target_count * 3, "target_color")
        _require_buffer_length(self.target_depth, self.target_count, "target_depth")
        if self.target_mask is not None:
            _require_buffer_length(self.target_mask, self.target_count, "target_mask")
        if self.target_normal is not None:
            _require_buffer_length(self.target_normal, self.target_count * 3, "target_normal")
            if self.target_normal_present is None:
                raise ValueError("packed render batch target_normal_present is required with target_normal")
        if self.target_normal_present is not None:
            _require_buffer_length(self.target_normal_present, self.target_count, "target_normal_present")
            if self.target_normal is None:
                raise ValueError("packed render batch target_normal is required with target_normal_present")
        for frame_index in self.frame_indices:
            if frame_index < 0 or frame_index >= len(self.frame_ids):
                raise ValueError("packed render batch frame index is out of range")
        if self.source_windows:
            expected_offset = 0
            for window in self.source_windows:
                if window.frame_id not in self.frame_ids:
                    raise ValueError("packed render batch source window references unknown frame id")
                if window.batch_target_offset != expected_offset:
                    raise ValueError("packed render batch source windows must be contiguous")
                expected_offset += window.target_count
            if expected_offset != self.target_count:
                raise ValueError("packed render batch source windows must cover target_count")

    def to_dict(self) -> dict:
        return {
            "format": "AURA_CAPTURE_PACKED_RENDER_BATCH",
            "batchIndex": self.batch_index,
            "frameIds": list(self.frame_ids),
            "frameSemanticIds": list(self.frame_semantic_ids),
            "targetOffset": self.target_offset,
            "targetCount": self.target_count,
            "maxTargetCount": self.max_target_count,
            "sampleOrder": self.sample_order,
            "bounded": self.target_count <= self.max_target_count,
            "frameIndices": _packed_buffer_metadata(self.frame_indices, "int64", (self.target_count,)),
            "pixelXY": _packed_buffer_metadata(self.pixel_xy, "int64", (self.target_count, 2)),
            "rayOrigins": _packed_buffer_metadata(self.ray_origins, "float64", (self.target_count, 3)),
            "rayDirections": _packed_buffer_metadata(self.ray_directions, "float64", (self.target_count, 3)),
            "targetColor": _packed_buffer_metadata(self.target_color, "float64", (self.target_count, 3)),
            "targetDepth": _packed_buffer_metadata(self.target_depth, "float64", (self.target_count,)),
            "targetMask": _packed_buffer_metadata(self.target_mask, "float64", (self.target_count,))
            if self.target_mask is not None
            else None,
            "targetNormal": _packed_buffer_metadata(self.target_normal, "float64", (self.target_count, 3))
            if self.target_normal is not None
            else None,
            "targetNormalPresent": _packed_buffer_metadata(
                self.target_normal_present,
                "uint8_bool",
                (self.target_count,),
            )
            if self.target_normal_present is not None
            else None,
            "sourceWindows": [window.to_dict() for window in self.source_windows],
        }


@dataclass(frozen=True)
class CaptureSamplingPlan:
    """Deterministic tiled sampling plan computed before render-target materialization."""

    pixel_stride: int
    tile_size: int
    max_targets_per_frame: int | None
    tiles: tuple[CaptureSamplingTile, ...]
    max_targets_per_batch: int
    batches: tuple[CaptureSamplingBatch, ...] = ()
    sample_order: str = "row-major tiles, row-major pixels"
    mask_rule: str = "sample mask values greater than 0; skip zero or negative mask values"

    def __post_init__(self) -> None:
        if self.pixel_stride <= 0:
            raise ValueError("pixel_stride must be positive")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.max_targets_per_frame is not None and self.max_targets_per_frame <= 0:
            raise ValueError("max_targets_per_frame must be positive when provided")
        if self.max_targets_per_batch <= 0:
            raise ValueError("max_targets_per_batch must be positive")
        if tuple(tile.tile_index for tile in self.tiles) != tuple(range(len(self.tiles))):
            raise ValueError("sampling tile indices must be contiguous and deterministic")
        if any(
            tile.sampled_pixel_count + tile.masked_pixel_count > tile.candidate_pixel_count for tile in self.tiles
        ):
            raise ValueError("sampling tile candidate counts cannot be smaller than sampled plus masked counts")
        if self.tiles and self.tiles[0].target_offset != 0:
            raise ValueError("first sampling tile target offset must be zero")
        for previous, current in zip(self.tiles, self.tiles[1:]):
            expected = previous.target_offset + previous.sampled_pixel_count
            if current.target_offset != expected:
                raise ValueError("sampling tile target offsets must be contiguous")
        for batch in self.batches:
            if batch.target_count > batch.max_target_count:
                raise ValueError("sampling batch exceeds max_target_count")
            if batch.max_target_count != self.max_targets_per_batch:
                raise ValueError("sampling batch max_target_count must match plan max_targets_per_batch")
            if not batch.tile_indices:
                raise ValueError("sampling batches must reference at least one tile")

    @property
    def total_sampled_pixel_count(self) -> int:
        return sum(tile.sampled_pixel_count for tile in self.tiles)

    @property
    def total_masked_pixel_count(self) -> int:
        return sum(tile.masked_pixel_count for tile in self.tiles)

    @property
    def total_candidate_pixel_count(self) -> int:
        return sum(tile.candidate_pixel_count for tile in self.tiles)

    @property
    def max_batch_target_count(self) -> int:
        if not self.batches:
            return 0
        return max(batch.target_count for batch in self.batches)

    def to_dict(self) -> dict:
        return {
            "format": "AURA_CAPTURE_SAMPLING_PLAN",
            "pixelStride": self.pixel_stride,
            "tileSize": self.tile_size,
            "maxTargetsPerFrame": self.max_targets_per_frame,
            "maxTargetsPerBatch": self.max_targets_per_batch,
            "sampleOrder": self.sample_order,
            "maskRule": self.mask_rule,
            "deterministic": True,
            "tileCount": len(self.tiles),
            "batchCount": len(self.batches),
            "maxBatchTargetCount": self.max_batch_target_count,
            "totalCandidatePixelCount": self.total_candidate_pixel_count,
            "totalSampledPixelCount": self.total_sampled_pixel_count,
            "totalMaskedPixelCount": self.total_masked_pixel_count,
            "tiles": [tile.to_dict() for tile in self.tiles],
            "batches": [batch.to_dict() for batch in self.batches],
        }


def capture_tensors_to_render_targets(
    frames: Sequence[TrainingFrame],
    tensors: Sequence[CaptureFrameTensors],
    *,
    pixel_stride: int = 1,
    max_targets_per_frame: int | None = None,
) -> tuple[CapturePixelTarget, ...]:
    """Convert capture image/depth/mask tensors into per-pixel render targets."""

    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if max_targets_per_frame is not None and max_targets_per_frame <= 0:
        raise ValueError("max_targets_per_frame must be positive when provided")
    by_frame = {frame.id: frame for frame in frames}
    targets: list[CapturePixelTarget] = []
    for frame_tensors in tensors:
        frame = by_frame.get(frame_tensors.frame_id)
        if frame is None:
            raise ValueError(f"capture tensors reference unknown training frame: {frame_tensors.frame_id}")
        _validate_tensor_dimensions(frame_tensors)
        produced = 0
        for y in range(0, frame_tensors.image.height, pixel_stride):
            for x in range(0, frame_tensors.image.width, pixel_stride):
                mask_value = _scalar_at(frame_tensors.mask, x, y)
                if mask_value is not None and mask_value <= 0.0:
                    continue
                color = _rgb_at(frame_tensors.image, x, y)
                depth = _scalar_at(frame_tensors.depth, x, y)
                normal = _normal_at(frame_tensors.normal, x, y)
                target = RenderTarget(
                    frame_id=frame.id,
                    ray=Ray(origin=frame.camera_origin, direction=_pixel_ray_direction(frame, x, y)),
                    target_color=color,
                    target_depth=depth if depth is not None and depth > 0.0 else frame.target_depth,
                    target_semantic_id=frame.semantic_label,
                    target_normal=normal,
                )
                targets.append(
                    CapturePixelTarget(
                        frame_id=frame.id,
                        pixel=(x, y),
                        render_target=target,
                        mask_value=mask_value,
                        target_normal=normal,
                    )
                )
                produced += 1
                if max_targets_per_frame is not None and produced >= max_targets_per_frame:
                    break
            if max_targets_per_frame is not None and produced >= max_targets_per_frame:
                break
    return tuple(targets)


def capture_tensors_to_packed_render_batches(
    frames: Sequence[TrainingFrame],
    tensors: Sequence[CaptureFrameTensors],
    *,
    pixel_stride: int = 1,
    max_targets_per_frame: int | None = None,
    tile_size: int = 256,
    max_targets_per_batch: int | None = None,
    sampling_plan: CaptureSamplingPlan | None = None,
) -> tuple[CapturePackedRenderBatch, ...]:
    """Convert capture tensors into bounded packed render-target batches.

    This is the GPU-ready companion to ``capture_tensors_to_render_targets``.
    It preserves the same deterministic mask-aware sampling order, but stores
    each bounded batch as flat packed arrays instead of per-pixel Python
    ``CapturePixelTarget`` objects.
    """

    plan = sampling_plan or plan_capture_tensor_sampling(
        frames,
        tensors,
        pixel_stride=pixel_stride,
        max_targets_per_frame=max_targets_per_frame,
        tile_size=tile_size,
        max_targets_per_batch=max_targets_per_batch,
    )
    _validate_sampling_plan_matches_request(
        plan,
        pixel_stride=pixel_stride,
        max_targets_per_frame=max_targets_per_frame,
        tile_size=tile_size,
        max_targets_per_batch=max_targets_per_batch,
    )
    by_frame = {frame.id: frame for frame in frames}
    tensor_by_frame = {frame_tensors.frame_id: frame_tensors for frame_tensors in tensors}
    frame_ids = tuple(frame_tensors.frame_id for frame_tensors in tensors)
    frame_index_by_id = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    frame_semantic_ids = tuple(by_frame[frame_id].semantic_label for frame_id in frame_ids)
    tiles_by_index = {tile.tile_index: tile for tile in plan.tiles}
    include_mask = any(frame_tensors.mask is not None for frame_tensors in tensors)
    include_normal = any(frame_tensors.normal is not None for frame_tensors in tensors)
    packed_batches: list[CapturePackedRenderBatch] = []
    for batch in plan.batches:
        if batch.target_count == 0:
            continue
        builders = _PackedRenderBatchBuilder(include_mask=include_mask, include_normal=include_normal)
        source_windows: list[CapturePackedRenderSourceWindow] = []
        batch_start = batch.target_offset
        batch_stop = batch.target_offset + batch.target_count
        for tile_index in batch.tile_indices:
            tile = tiles_by_index[tile_index]
            tile_stop = tile.target_offset + tile.sampled_pixel_count
            if tile.sampled_pixel_count == 0 or batch_start >= tile_stop or batch_stop <= tile.target_offset:
                continue
            source_windows.append(_source_window_for_batch_tile(batch_start, batch_stop, tile))
            frame_tensors = tensor_by_frame[tile.frame_id]
            frame = by_frame[tile.frame_id]
            frame_index = frame_index_by_id[tile.frame_id]
            _append_tile_samples_to_packed_batch(
                builders,
                frame,
                frame_tensors,
                frame_index=frame_index,
                tile=tile,
                pixel_stride=pixel_stride,
                target_start=batch_start,
                target_stop=batch_stop,
            )
        packed_batches.append(
            CapturePackedRenderBatch(
                batch_index=batch.batch_index,
                frame_ids=frame_ids,
                frame_semantic_ids=frame_semantic_ids,
                target_offset=batch.target_offset,
                target_count=batch.target_count,
                max_target_count=batch.max_target_count,
                frame_indices=builders.frame_indices,
                pixel_xy=builders.pixel_xy,
                ray_origins=builders.ray_origins,
                ray_directions=builders.ray_directions,
                target_color=builders.target_color,
                target_depth=builders.target_depth,
                target_mask=builders.target_mask if include_mask else None,
                target_normal=builders.target_normal if include_normal else None,
                target_normal_present=builders.target_normal_present if include_normal else None,
                source_windows=tuple(source_windows),
            )
        )
    return tuple(packed_batches)


def _validate_sampling_plan_matches_request(
    plan: CaptureSamplingPlan,
    *,
    pixel_stride: int,
    max_targets_per_frame: int | None,
    tile_size: int,
    max_targets_per_batch: int | None,
) -> None:
    if plan.pixel_stride != pixel_stride:
        raise ValueError("sampling_plan pixel_stride does not match packed batch request")
    if plan.max_targets_per_frame != max_targets_per_frame:
        raise ValueError("sampling_plan max_targets_per_frame does not match packed batch request")
    if plan.tile_size != tile_size:
        raise ValueError("sampling_plan tile_size does not match packed batch request")
    expected_max_targets_per_batch = max_targets_per_batch or _max_sampled_pixels_per_tile(tile_size, pixel_stride)
    if plan.max_targets_per_batch != expected_max_targets_per_batch:
        raise ValueError("sampling_plan max_targets_per_batch does not match packed batch request")


def plan_capture_tensor_sampling(
    frames: Sequence[TrainingFrame],
    tensors: Sequence[CaptureFrameTensors],
    *,
    pixel_stride: int = 1,
    max_targets_per_frame: int | None = None,
    tile_size: int = 256,
    max_targets_per_batch: int | None = None,
) -> CaptureSamplingPlan:
    """Plan tiled pixel sampling before materializing render targets."""

    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if max_targets_per_frame is not None and max_targets_per_frame <= 0:
        raise ValueError("max_targets_per_frame must be positive when provided")
    if max_targets_per_batch is not None and max_targets_per_batch <= 0:
        raise ValueError("max_targets_per_batch must be positive when provided")
    resolved_max_targets_per_batch = max_targets_per_batch or _max_sampled_pixels_per_tile(tile_size, pixel_stride)
    by_frame = {frame.id: frame for frame in frames}
    tiles: list[CaptureSamplingTile] = []
    target_offset = 0
    for frame_tensors in tensors:
        if frame_tensors.frame_id not in by_frame:
            raise ValueError(f"capture tensors reference unknown training frame: {frame_tensors.frame_id}")
        _validate_tensor_dimensions(frame_tensors)
        produced = 0
        stop_frame = False
        for tile_y in range(0, frame_tensors.image.height, tile_size):
            if stop_frame:
                break
            for tile_x in range(0, frame_tensors.image.width, tile_size):
                width = min(tile_size, frame_tensors.image.width - tile_x)
                height = min(tile_size, frame_tensors.image.height - tile_y)
                sampled = 0
                masked = 0
                candidate = 0
                first_sampled_pixel = None
                last_sampled_pixel = None
                for y in range(tile_y, tile_y + height, pixel_stride):
                    for x in range(tile_x, tile_x + width, pixel_stride):
                        candidate += 1
                        mask_value = _scalar_at(frame_tensors.mask, x, y)
                        if mask_value is not None and mask_value <= 0.0:
                            masked += 1
                            continue
                        if first_sampled_pixel is None:
                            first_sampled_pixel = (x, y)
                        last_sampled_pixel = (x, y)
                        sampled += 1
                        produced += 1
                        if max_targets_per_frame is not None and produced >= max_targets_per_frame:
                            stop_frame = True
                            break
                    if stop_frame:
                        break
                tiles.append(
                    CaptureSamplingTile(
                        frame_id=frame_tensors.frame_id,
                        origin=(tile_x, tile_y),
                        size=(width, height),
                        sampled_pixel_count=sampled,
                        masked_pixel_count=masked,
                        tile_index=len(tiles),
                        candidate_pixel_count=candidate,
                        target_offset=target_offset,
                        first_sampled_pixel=first_sampled_pixel,
                        last_sampled_pixel=last_sampled_pixel,
                    )
                )
                target_offset += sampled
                if stop_frame:
                    break
    batches = _sampling_batches(tiles, resolved_max_targets_per_batch)
    return CaptureSamplingPlan(
        pixel_stride=pixel_stride,
        tile_size=tile_size,
        max_targets_per_frame=max_targets_per_frame,
        tiles=tuple(tiles),
        max_targets_per_batch=resolved_max_targets_per_batch,
        batches=batches,
    )


def sampling_coverage_report(plan: CaptureSamplingPlan) -> dict:
    """Quantify how much of each frame a sampling plan actually supervises.

    Pure, side-effect-free diagnostic computed from the already-materialized
    ``CaptureSamplingPlan`` tile fields. It does NOT re-sample, does NOT touch
    the plan↔packed-batch invariant, and changes no training target.

    Motivation: with a tight ``max_targets_per_frame`` cap the planner stops
    after the first N valid pixels of each frame (raster order, top-left), so
    most of the frame — and most carriers — are never supervised, and the same
    subset is reused every iteration (see the README (carrier-gradient/convergence notes)). This report
    surfaces that starvation as a number so a GPU run can measure it before and
    after applying a coverage fix.

    Coverage denominator note: when ``max_targets_per_frame`` truncates the
    per-frame scan early, the planner stops visiting later pixels/tiles, so the
    tile's scan-truncated ``candidate_pixel_count`` would hide the starvation.
    To surface it honestly, ``capacityPixelCount`` here is each present tile's
    FULL stride-decimated grid capacity (derived from its ``size`` and the
    plan's ``pixel_stride``) — i.e. how many pixels the tile *could* supervise.
    ``coverageFraction = sampled / capacity`` therefore drops below 1.0 exactly
    when the cap leaves pixels in a present tile unsupervised. (Tiles dropped
    entirely by an early frame-level stop are not in the plan and cannot be
    counted from the plan alone; ``tileCount`` is reported so a shrinking
    tile count also signals dropped coverage.)

    Returns a dict with overall and per-frame coverage:

    - ``sampledPixelCount`` / ``capacityPixelCount`` / ``maskedPixelCount``
    - ``coverageFraction``: sampled / capacity (1.0 = every pixel in every
      present tile supervised; small values indicate the cap is starving
      coverage)
    - ``tileCount``: number of tiles present in the plan
    - ``perFrame``: the same fields keyed by frame id

    ``coverageFraction`` is in ``[0, 1]``; a capacity of 0 reports 0.0.
    """

    def _fraction(sampled: int, capacity: int) -> float:
        return (sampled / capacity) if capacity > 0 else 0.0

    def _tile_capacity(tile: CaptureSamplingTile) -> int:
        width, height = tile.size
        cols = (width + plan.pixel_stride - 1) // plan.pixel_stride
        rows = (height + plan.pixel_stride - 1) // plan.pixel_stride
        return cols * rows

    per_frame: dict[str, dict] = {}
    for tile in plan.tiles:
        bucket = per_frame.setdefault(
            tile.frame_id,
            {
                "sampledPixelCount": 0,
                "capacityPixelCount": 0,
                "maskedPixelCount": 0,
                "tileCount": 0,
            },
        )
        bucket["sampledPixelCount"] += tile.sampled_pixel_count
        bucket["capacityPixelCount"] += _tile_capacity(tile)
        bucket["maskedPixelCount"] += tile.masked_pixel_count
        bucket["tileCount"] += 1
    for bucket in per_frame.values():
        bucket["coverageFraction"] = _fraction(
            bucket["sampledPixelCount"], bucket["capacityPixelCount"]
        )

    total_sampled = sum(b["sampledPixelCount"] for b in per_frame.values())
    total_capacity = sum(b["capacityPixelCount"] for b in per_frame.values())
    total_masked = sum(b["maskedPixelCount"] for b in per_frame.values())
    return {
        "format": "AURA_SAMPLING_COVERAGE_REPORT",
        "pixelStride": plan.pixel_stride,
        "tileSize": plan.tile_size,
        "maxTargetsPerFrame": plan.max_targets_per_frame,
        "frameCount": len(per_frame),
        "tileCount": len(plan.tiles),
        "sampledPixelCount": total_sampled,
        "capacityPixelCount": total_capacity,
        "maskedPixelCount": total_masked,
        "coverageFraction": _fraction(total_sampled, total_capacity),
        "minFrameCoverageFraction": min(
            (b["coverageFraction"] for b in per_frame.values()), default=0.0
        ),
        "maxFrameCoverageFraction": max(
            (b["coverageFraction"] for b in per_frame.values()), default=0.0
        ),
        "perFrame": per_frame,
    }


def _max_sampled_pixels_per_tile(tile_size: int, pixel_stride: int) -> int:
    sampled_axis = (tile_size + pixel_stride - 1) // pixel_stride
    return max(1, sampled_axis * sampled_axis)


def _sampling_batches(
    tiles: Sequence[CaptureSamplingTile],
    max_targets_per_batch: int,
) -> tuple[CaptureSamplingBatch, ...]:
    batches: list[CaptureSamplingBatch] = []
    pending_tile_indices: list[int] = []
    pending_target_count = 0
    pending_target_offset = 0
    def flush_pending() -> None:
        nonlocal pending_tile_indices, pending_target_count, pending_target_offset
        if not pending_tile_indices:
            return
        batches.append(
            CaptureSamplingBatch(
                batch_index=len(batches),
                tile_indices=tuple(pending_tile_indices),
                target_offset=pending_target_offset,
                target_count=pending_target_count,
                max_target_count=max_targets_per_batch,
            )
        )
        pending_tile_indices = []
        pending_target_count = 0

    for tile in tiles:
        if tile.sampled_pixel_count > max_targets_per_batch:
            flush_pending()
            consumed = 0
            while consumed < tile.sampled_pixel_count:
                target_count = min(max_targets_per_batch, tile.sampled_pixel_count - consumed)
                batches.append(
                    CaptureSamplingBatch(
                        batch_index=len(batches),
                        tile_indices=(tile.tile_index,),
                        target_offset=tile.target_offset + consumed,
                        target_count=target_count,
                        max_target_count=max_targets_per_batch,
                    )
                )
                consumed += target_count
            continue
        if not pending_tile_indices:
            pending_target_offset = tile.target_offset
        would_exceed = (
            pending_tile_indices
            and tile.sampled_pixel_count > 0
            and pending_target_count + tile.sampled_pixel_count > max_targets_per_batch
        )
        if would_exceed:
            flush_pending()
            pending_target_offset = tile.target_offset
        pending_tile_indices.append(tile.tile_index)
        pending_target_count += tile.sampled_pixel_count
    flush_pending()
    return tuple(batches)


@dataclass
class _PackedRenderBatchBuilder:
    include_mask: bool
    include_normal: bool
    frame_indices: array = field(init=False)
    pixel_xy: array = field(init=False)
    ray_origins: array = field(init=False)
    ray_directions: array = field(init=False)
    target_color: array = field(init=False)
    target_depth: array = field(init=False)
    target_mask: array | None = field(init=False)
    target_normal: array | None = field(init=False)
    target_normal_present: array | None = field(init=False)

    def __post_init__(self) -> None:
        self.frame_indices = array("q")
        self.pixel_xy = array("q")
        self.ray_origins = array("d")
        self.ray_directions = array("d")
        self.target_color = array("d")
        self.target_depth = array("d")
        self.target_mask = array("d") if self.include_mask else None
        self.target_normal = array("d") if self.include_normal else None
        self.target_normal_present = array("B") if self.include_normal else None


def _append_tile_samples_to_packed_batch(
    builders: _PackedRenderBatchBuilder,
    frame: TrainingFrame,
    frame_tensors: CaptureFrameTensors,
    *,
    frame_index: int,
    tile: CaptureSamplingTile,
    pixel_stride: int,
    target_start: int,
    target_stop: int,
) -> None:
    sampled_offset = tile.target_offset
    # Stop at whichever is smaller: the batch end or the tile's own allocation
    # (tile.sampled_pixel_count may be less than tile capacity when max_targets_per_frame
    # truncates the scan early; without this bound the function would spill into
    # the next tile's target slots and produce too many samples for the batch).
    tile_sample_stop = tile.target_offset + tile.sampled_pixel_count
    effective_stop = min(target_stop, tile_sample_stop)
    tile_x, tile_y = tile.origin
    width, height = tile.size
    for y in range(tile_y, tile_y + height, pixel_stride):
        for x in range(tile_x, tile_x + width, pixel_stride):
            mask_value = _scalar_at(frame_tensors.mask, x, y)
            if mask_value is not None and mask_value <= 0.0:
                continue
            if target_start <= sampled_offset < effective_stop:
                color = _rgb_at(frame_tensors.image, x, y)
                depth = _scalar_at(frame_tensors.depth, x, y)
                normal = _normal_at(frame_tensors.normal, x, y)
                builders.frame_indices.append(frame_index)
                builders.pixel_xy.extend((x, y))
                builders.ray_origins.extend(frame.camera_origin)
                builders.ray_directions.extend(_pixel_ray_direction(frame, x, y))
                builders.target_color.extend(color)
                builders.target_depth.append(depth if depth is not None and depth > 0.0 else frame.target_depth)
                if builders.target_mask is not None:
                    builders.target_mask.append(mask_value if mask_value is not None else 1.0)
                if builders.target_normal is not None and builders.target_normal_present is not None:
                    if normal is None:
                        builders.target_normal.extend((0.0, 0.0, 0.0))
                        builders.target_normal_present.append(0)
                    else:
                        builders.target_normal.extend(normal)
                        builders.target_normal_present.append(1)
            sampled_offset += 1
            if sampled_offset >= effective_stop:
                return


def _source_window_for_batch_tile(
    batch_start: int,
    batch_stop: int,
    tile: CaptureSamplingTile,
) -> CapturePackedRenderSourceWindow:
    tile_stop = tile.target_offset + tile.sampled_pixel_count
    overlap_start = max(batch_start, tile.target_offset)
    overlap_stop = min(batch_stop, tile_stop)
    return CapturePackedRenderSourceWindow(
        frame_id=tile.frame_id,
        tile_index=tile.tile_index,
        tile_origin=tile.origin,
        tile_size=tile.size,
        batch_target_offset=overlap_start - batch_start,
        target_offset=overlap_start,
        target_count=overlap_stop - overlap_start,
    )


def _validate_tensor_dimensions(frame_tensors: CaptureFrameTensors) -> None:
    image_shape = (frame_tensors.image.width, frame_tensors.image.height)
    for name, tensor in (
        ("depth", frame_tensors.depth),
        ("mask", frame_tensors.mask),
        ("normal", frame_tensors.normal),
    ):
        if tensor is None:
            continue
        if (tensor.width, tensor.height) != image_shape:
            raise ValueError(f"{name} tensor dimensions must match image tensor dimensions")


def _rgb_at(tensor: CaptureTensor, x: int, y: int) -> Vec3:
    if tensor.channels < 3:
        raise ValueError("image tensor must have at least three channels")
    offset = (y * tensor.width + x) * tensor.channels
    return (tensor.values[offset], tensor.values[offset + 1], tensor.values[offset + 2])


def _scalar_at(tensor: CaptureTensor | None, x: int, y: int) -> float | None:
    if tensor is None:
        return None
    if tensor.channels != 1:
        raise ValueError("scalar tensor must have one channel")
    return tensor.values[y * tensor.width + x]


def _normal_at(tensor: CaptureTensor | None, x: int, y: int) -> Vec3 | None:
    if tensor is None:
        return None
    if tensor.channels != 3:
        raise ValueError("normal tensor must have three channels")
    offset = (y * tensor.width + x) * tensor.channels
    return _normalize((tensor.values[offset], tensor.values[offset + 1], tensor.values[offset + 2]))


def _pixel_ray_direction(frame: TrainingFrame, x: int, y: int) -> Vec3:
    forward = _normalize(tuple(frame.look_at[index] - frame.camera_origin[index] for index in range(3)))
    if frame.intrinsics is None:
        return forward
    fx = max(frame.intrinsics["fx"], 1e-6)
    fy = max(frame.intrinsics["fy"], 1e-6)
    cx = frame.intrinsics["cx"]
    cy = frame.intrinsics["cy"]
    right_raw = _cross(forward, (0.0, 1.0, 0.0))
    if _norm(right_raw) <= 1e-12:
        right_raw = _cross(forward, (1.0, 0.0, 0.0))
    right = _normalize(right_raw)
    up = _normalize(_cross(right, forward))
    px = ((x + 0.5) - cx) / fx
    py = ((y + 0.5) - cy) / fy
    return _normalize(
        (
            forward[0] + right[0] * px - up[0] * py,
            forward[1] + right[1] * px - up[1] * py,
            forward[2] + right[2] * px - up[2] * py,
        )
    )


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: Vec3) -> Vec3:
    norm = _norm(vector)
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _norm(vector: Vec3) -> float:
    return sqrt(sum(axis * axis for axis in vector))


def _require_buffer_length(values: Sequence[object], expected: int, name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"packed render batch {name} length must be {expected}")


def _packed_buffer_metadata(values: Sequence[object] | None, dtype: str, shape: tuple[int, ...]) -> dict | None:
    if values is None:
        return None
    return {
        "dtype": dtype,
        "shape": list(shape),
        "valueCount": len(values),
        "sampleValues": list(values[: min(len(values), 12)]),
    }
