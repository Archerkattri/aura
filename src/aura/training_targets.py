from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from aura.core import TrainingFrame
from aura.ingest.capture import CaptureFrameTensors, CaptureTensor
from aura.optimize import RenderTarget
from aura.ray import Ray, Vec3


@dataclass(frozen=True)
class CapturePixelTarget:
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
    frame_id: str
    origin: tuple[int, int]
    size: tuple[int, int]
    sampled_pixel_count: int
    masked_pixel_count: int = 0

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "origin": list(self.origin),
            "size": list(self.size),
            "sampledPixelCount": self.sampled_pixel_count,
            "maskedPixelCount": self.masked_pixel_count,
        }


@dataclass(frozen=True)
class CaptureSamplingPlan:
    pixel_stride: int
    tile_size: int
    max_targets_per_frame: int | None
    tiles: tuple[CaptureSamplingTile, ...]

    @property
    def total_sampled_pixel_count(self) -> int:
        return sum(tile.sampled_pixel_count for tile in self.tiles)

    @property
    def total_masked_pixel_count(self) -> int:
        return sum(tile.masked_pixel_count for tile in self.tiles)

    def to_dict(self) -> dict:
        return {
            "format": "AURA_CAPTURE_SAMPLING_PLAN",
            "pixelStride": self.pixel_stride,
            "tileSize": self.tile_size,
            "maxTargetsPerFrame": self.max_targets_per_frame,
            "tileCount": len(self.tiles),
            "totalSampledPixelCount": self.total_sampled_pixel_count,
            "totalMaskedPixelCount": self.total_masked_pixel_count,
            "tiles": [tile.to_dict() for tile in self.tiles],
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


def plan_capture_tensor_sampling(
    frames: Sequence[TrainingFrame],
    tensors: Sequence[CaptureFrameTensors],
    *,
    pixel_stride: int = 1,
    max_targets_per_frame: int | None = None,
    tile_size: int = 256,
) -> CaptureSamplingPlan:
    """Plan tiled pixel sampling before materializing render targets."""

    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    by_frame = {frame.id: frame for frame in frames}
    tiles: list[CaptureSamplingTile] = []
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
                for y in range(tile_y, tile_y + height, pixel_stride):
                    for x in range(tile_x, tile_x + width, pixel_stride):
                        mask_value = _scalar_at(frame_tensors.mask, x, y)
                        if mask_value is not None and mask_value <= 0.0:
                            masked += 1
                            continue
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
                    )
                )
                if stop_frame:
                    break
    return CaptureSamplingPlan(
        pixel_stride=pixel_stride,
        tile_size=tile_size,
        max_targets_per_frame=max_targets_per_frame,
        tiles=tuple(tiles),
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
