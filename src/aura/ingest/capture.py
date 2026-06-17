from __future__ import annotations

from array import array
from collections.abc import Iterable as IterableABC
from collections.abc import Sequence as SequenceABC
import json
import struct
import zlib
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aura.assignment import RegionEvidence
from aura.core import TrainingDataset, TrainingFrame, TrainingRegion
from aura.elements import Bounds
from aura.proposals import propose_training_regions_from_tensors
from aura.ray import Vec3


class PackedFloatBuffer(Sequence[float]):
    """Packed float sequence for dense capture tensors without Python-float tuples."""

    def __init__(self, values: IterableABC[object] = ()) -> None:
        self._values = array("d", (float(value) for value in values))

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self._values[index])
        return self._values[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PackedFloatBuffer):
            return self._values == other._values
        if isinstance(other, SequenceABC):
            return tuple(self._values) == tuple(other)
        return False

    def sample(self, limit: int) -> tuple[float, ...]:
        return tuple(self._values[: max(0, limit)])


@dataclass(frozen=True)
class CaptureManifest:
    """A real-capture ingest contract before images are loaded or optimized."""

    root: str
    frames: tuple[TrainingFrame, ...]
    regions: tuple[TrainingRegion, ...]

    def to_training_dataset(self, *, load_assets: bool = False) -> TrainingDataset:
        if not load_assets:
            return TrainingDataset(frames=self.frames, regions=self.regions)
        return capture_tensors_to_training_dataset(self, load_capture_asset_tensors(self))

    def to_dict(self) -> dict:
        return {
            "format": "AURA_CAPTURE_MANIFEST",
            "root": self.root,
            "frames": [frame.to_dict() for frame in self.frames],
            "regions": [region.to_dict() for region in self.regions],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CaptureManifest":
        if not isinstance(payload, dict):
            raise ValueError("capture manifest payload must be an object")
        validate_capture_manifest_document(payload)
        root = str(payload.get("root") or ".")
        frames = tuple(_frame_from_capture_payload(item) for item in payload["frames"])
        regions = tuple(_region_from_capture_payload(item) for item in payload["regions"])
        dataset = TrainingDataset(frames=frames, regions=regions)
        _validate_manifest_links(dataset)
        return cls(root=root, frames=frames, regions=regions)


@dataclass(frozen=True)
class CaptureFrameAssets:
    """Loaded image/depth/mask summaries for one capture-manifest frame."""

    frame_id: str
    image_path: str
    width: int
    height: int
    average_color: Vec3
    depth_path: str | None = None
    average_depth: float | None = None
    min_depth: float | None = None
    max_depth: float | None = None
    depth_coverage: float | None = None
    depth_bins: tuple[dict[str, float], ...] = ()
    mask_path: str | None = None
    mask_coverage: float | None = None
    normal_path: str | None = None
    average_normal: Vec3 | None = None

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "imagePath": self.image_path,
            "width": self.width,
            "height": self.height,
            "averageColor": list(self.average_color),
            "depthPath": self.depth_path,
            "averageDepth": self.average_depth,
            "minDepth": self.min_depth,
            "maxDepth": self.max_depth,
            "depthCoverage": self.depth_coverage,
            "depthBins": [dict(item) for item in self.depth_bins],
            "maskPath": self.mask_path,
            "maskCoverage": self.mask_coverage,
            "normalPath": self.normal_path,
            "averageNormal": list(self.average_normal) if self.average_normal is not None else None,
        }


@dataclass(frozen=True)
class CaptureTensor:
    """Per-pixel capture asset values ready for CPU reference or GPU upload."""

    path: str
    format: str
    backend: str
    width: int
    height: int
    channels: int
    values: Sequence[float]

    def __post_init__(self) -> None:
        if not isinstance(self.values, PackedFloatBuffer):
            object.__setattr__(self, "values", PackedFloatBuffer(self.values))
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.path} tensor dimensions must be positive")
        if self.channels <= 0:
            raise ValueError(f"{self.path} tensor channels must be positive")
        if len(self.values) != self.width * self.height * self.channels:
            raise ValueError(f"{self.path} tensor payload does not match dimensions")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def sample_values(self, limit: int = 12) -> tuple[float, ...]:
        if isinstance(self.values, PackedFloatBuffer):
            return self.values.sample(limit)
        return tuple(self.values[: max(0, limit)])

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "backend": self.backend,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "shape": list(self.shape),
            "valueCount": len(self.values),
            "sampleValues": list(self.sample_values()),
        }


@dataclass(frozen=True)
class CaptureFrameTensors:
    """Loaded tensor assets for one capture-manifest frame."""

    frame_id: str
    image: CaptureTensor
    depth: CaptureTensor | None = None
    mask: CaptureTensor | None = None
    normal: CaptureTensor | None = None

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "image": self.image.to_dict(),
            "depth": self.depth.to_dict() if self.depth is not None else None,
            "mask": self.mask.to_dict() if self.mask is not None else None,
            "normal": self.normal.to_dict() if self.normal is not None else None,
        }


def load_capture_manifest(path: Path | str) -> CaptureManifest:
    """Load an AURA capture manifest and convert it to training contracts.

    This intentionally does not read image pixels. The GPU-side implementation
    can replace target_color/depth summaries with real differentiable image
    sampling while keeping the same manifest and frame identifiers.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capture manifest JSON must be an object")
    validate_capture_manifest_document(payload)

    if "root" not in payload:
        payload = {**payload, "root": str(manifest_path.parent)}
    return CaptureManifest.from_dict(payload)


def load_capture_assets(manifest: CaptureManifest) -> tuple[CaptureFrameAssets, ...]:
    """Load manifest image/depth/mask assets and return deterministic summaries.

    PNG and Netpbm PPM/PGM are loaded without optional dependencies so capture
    manifests can be validated in CI before the GPU tensor backend is installed.
    """

    return _capture_assets_from_tensors(load_capture_asset_tensors(manifest))


def capture_tensors_to_training_dataset(
    manifest: CaptureManifest,
    tensor_frames: Sequence[CaptureFrameTensors],
) -> TrainingDataset:
    """Derive training frames/regions from an already loaded capture tensor batch."""

    _validate_capture_tensor_frame_set(manifest, tensor_frames)
    tensors = {item.frame_id: item for item in tensor_frames}
    assets = {item.frame_id: item for item in _capture_assets_from_tensors(tensor_frames)}
    frames = tuple(_frame_with_asset_summaries(frame, assets.get(frame.id)) for frame in manifest.frames)
    regions = (
        *manifest.regions,
        *_feature_regions_from_tensors(frames, tensors, assets),
        *_depth_regions_from_assets(frames, assets),
        *_mask_regions_from_assets(frames, assets),
    )
    return TrainingDataset(frames=frames, regions=regions)


def _capture_assets_from_tensors(tensor_frames: Sequence[CaptureFrameTensors]) -> tuple[CaptureFrameAssets, ...]:
    assets = []
    for tensors in tensor_frames:
        image = _tensor_to_raster(tensors.image)
        depth = _tensor_to_raster(tensors.depth) if tensors.depth is not None else None
        depth_summary = _depth_summary(depth)
        mask = _tensor_to_raster(tensors.mask) if tensors.mask is not None else None
        normal = _tensor_to_raster(tensors.normal) if tensors.normal is not None else None
        assets.append(
            CaptureFrameAssets(
                frame_id=tensors.frame_id,
                image_path=tensors.image.path,
                width=image.width,
                height=image.height,
                average_color=_average_rgb(image),
                depth_path=tensors.depth.path if tensors.depth is not None else None,
                average_depth=depth_summary.average if depth_summary is not None else None,
                min_depth=depth_summary.minimum if depth_summary is not None else None,
                max_depth=depth_summary.maximum if depth_summary is not None else None,
                depth_coverage=depth_summary.coverage if depth_summary is not None else None,
                depth_bins=depth_summary.bins if depth_summary is not None else (),
                mask_path=tensors.mask.path if tensors.mask is not None else None,
                mask_coverage=_average_scalar(mask) if mask is not None else None,
                normal_path=tensors.normal.path if tensors.normal is not None else None,
                average_normal=_average_normal(normal),
            )
        )
    return tuple(assets)


def _validate_capture_tensor_frame_set(manifest: CaptureManifest, tensor_frames: Sequence[CaptureFrameTensors]) -> None:
    expected = tuple(frame.id for frame in manifest.frames)
    actual = tuple(item.frame_id for item in tensor_frames)
    duplicates = sorted({frame_id for frame_id in actual if actual.count(frame_id) > 1})
    if duplicates:
        raise ValueError(f"capture tensor batch contains duplicate frame ids: {', '.join(duplicates)}")
    missing = sorted(set(expected).difference(actual))
    if missing:
        raise ValueError(f"capture tensor batch is missing manifest frame ids: {', '.join(missing)}")
    unknown = sorted(set(actual).difference(expected))
    if unknown:
        raise ValueError(f"capture tensor batch references unknown manifest frame ids: {', '.join(unknown)}")


def load_capture_asset_tensors(manifest: CaptureManifest) -> tuple[CaptureFrameTensors, ...]:
    """Load manifest image/depth/mask/normal assets as per-pixel tensors."""

    root = Path(manifest.root)
    frames = []
    for frame in manifest.frames:
        if frame.image_path is None:
            raise ValueError(f"capture frame {frame.id} is missing image_path")
        image_path = _resolve_capture_path(root, frame.image_path)
        image = _read_capture_tensor(image_path)
        if image.channels < 3:
            raise ValueError(f"capture frame {frame.id} image_path must reference an RGB/RGBA image")
        depth_path = _resolve_capture_path(root, frame.depth_path) if frame.depth_path is not None else None
        depth = _read_capture_tensor(depth_path) if depth_path is not None else None
        if depth is not None and depth.channels != 1:
            raise ValueError(f"capture frame {frame.id} depth_path must reference a single-channel image")
        mask_path = _resolve_capture_path(root, frame.mask_path) if frame.mask_path is not None else None
        mask = _read_capture_tensor(mask_path) if mask_path is not None else None
        if mask is not None and mask.channels != 1:
            raise ValueError(f"capture frame {frame.id} mask_path must reference a single-channel image")
        normal_path = _resolve_capture_path(root, frame.normal_path) if frame.normal_path is not None else None
        normal = _read_capture_tensor(normal_path) if normal_path is not None else None
        if normal is not None and normal.channels != 3:
            raise ValueError(f"capture frame {frame.id} normal_path must reference a 3-channel normal map")
        frames.append(CaptureFrameTensors(frame_id=frame.id, image=image, depth=depth, mask=mask, normal=normal))
    return tuple(frames)


def write_capture_manifest_template(path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(capture_manifest_template(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_capture_manifest(manifest: CaptureManifest, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def capture_manifest_template() -> dict:
    return {
        "format": "AURA_CAPTURE_MANIFEST",
        "root": "data/custom-captures/example-scene",
        "frames": [
            {
                "id": "frame_000001",
                "image_path": "images/frame_000001.png",
                "depth_path": "depth/frame_000001.exr",
                "mask_path": "masks/frame_000001.png",
                "normal_path": "normal/frame_000001.bin",
                "camera_model": "pinhole",
                "intrinsics": {"fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0, "width": 1920.0, "height": 1080.0},
                "camera_origin": [0.0, 0.0, -2.0],
                "look_at": [0.0, 0.0, 0.0],
                "target_color": [0.72, 0.68, 0.61],
                "target_depth": 2.0,
                "semantic_label": "room",
            }
        ],
        "regions": [
            {
                "id": "wall_surface_000001",
                "frame_id": "frame_000001",
                "bounds": {"min": [-0.6, -0.4, 0.0], "max": [0.6, 0.4, 0.1]},
                "evidence": {"geometry_confidence": 0.9, "material_confidence": 0.7, "ray_need": 0.8, "edit_need": 0.5},
                "color": [0.72, 0.68, 0.61],
                "opacity": 0.9,
                "confidence": 0.85,
                "normal": [0.0, 0.0, -1.0],
                "material_id": "mat_wall",
                "semantic_label": "wall",
                "fallback_source": "capture-manifest",
            }
        ],
    }


def validate_capture_manifest_document(payload: dict) -> None:
    schema_path = resources.files("aura.schemas").joinpath("capture_manifest.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    try:
        validator.validate(payload)
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.absolute_path)
        location = f" at {path}" if path else ""
        raise ValueError(f"capture_manifest.schema.json validation failed{location}: {exc.message}") from exc


def _frame_from_capture_payload(payload: dict[str, Any]) -> TrainingFrame:
    return TrainingFrame(
        id=str(payload["id"]),
        camera_origin=_vec3(payload["camera_origin"], "camera_origin"),
        look_at=_vec3(payload["look_at"], "look_at"),
        target_color=_vec3(payload["target_color"], "target_color"),
        target_depth=float(payload["target_depth"]),
        semantic_label=str(payload["semantic_label"]) if payload.get("semantic_label") is not None else None,
        image_path=str(payload["image_path"]),
        depth_path=str(payload["depth_path"]) if payload.get("depth_path") is not None else None,
        mask_path=str(payload["mask_path"]) if payload.get("mask_path") is not None else None,
        normal_path=str(payload["normal_path"]) if payload.get("normal_path") is not None else None,
        camera_model=str(payload["camera_model"]) if payload.get("camera_model") is not None else None,
        intrinsics={key: float(value) for key, value in payload["intrinsics"].items()}
        if payload.get("intrinsics") is not None
        else None,
    )


def _region_from_capture_payload(payload: dict[str, Any]) -> TrainingRegion:
    bounds = payload["bounds"]
    evidence = payload.get("evidence", {})
    return TrainingRegion(
        id=str(payload["id"]),
        frame_id=str(payload["frame_id"]),
        bounds=Bounds(min_corner=_vec3(bounds["min"], "bounds.min"), max_corner=_vec3(bounds["max"], "bounds.max")),
        evidence=RegionEvidence(**{key: float(value) for key, value in evidence.items()}),
        color=_vec3(payload["color"], "color") if payload.get("color") is not None else None,
        opacity=float(payload.get("opacity", 1.0)),
        confidence=float(payload.get("confidence", 1.0)),
        normal=_vec3(payload["normal"], "normal") if payload.get("normal") is not None else None,
        material_id=str(payload["material_id"]) if payload.get("material_id") is not None else None,
        semantic_label=str(payload["semantic_label"]) if payload.get("semantic_label") is not None else None,
        fallback_source=str(payload.get("fallback_source", "capture-manifest")),
    )


def _validate_manifest_links(dataset: TrainingDataset) -> None:
    frame_ids = {frame.id for frame in dataset.frames}
    if len(frame_ids) != len(dataset.frames):
        raise ValueError("capture manifest contains duplicate frame ids")
    region_ids = {region.id for region in dataset.regions}
    if len(region_ids) != len(dataset.regions):
        raise ValueError("capture manifest contains duplicate region ids")
    missing = sorted({region.frame_id for region in dataset.regions}.difference(frame_ids))
    if missing:
        raise ValueError(f"capture regions reference unknown frame ids: {', '.join(missing)}")


def _vec3(payload: object, name: str) -> Vec3:
    if not isinstance(payload, list | tuple) or len(payload) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return (float(payload[0]), float(payload[1]), float(payload[2]))


@dataclass(frozen=True)
class _RasterImage:
    format: str
    width: int
    height: int
    channels: int
    values: Sequence[float]

    def __post_init__(self) -> None:
        if not isinstance(self.values, PackedFloatBuffer):
            object.__setattr__(self, "values", PackedFloatBuffer(self.values))
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.format} dimensions must be positive")
        if len(self.values) != self.width * self.height * self.channels:
            raise ValueError(f"{self.format} payload does not match dimensions")


@dataclass(frozen=True)
class _ScalarSummary:
    average: float
    minimum: float
    maximum: float
    coverage: float
    bins: tuple[dict[str, float], ...] = ()


def _frame_with_asset_summaries(frame: TrainingFrame, assets: CaptureFrameAssets | None) -> TrainingFrame:
    if assets is None:
        return frame
    return replace(
        frame,
        target_color=assets.average_color,
        target_depth=assets.average_depth if assets.average_depth is not None else frame.target_depth,
    )


def _depth_regions_from_assets(
    frames: tuple[TrainingFrame, ...],
    assets: dict[str, CaptureFrameAssets],
) -> tuple[TrainingRegion, ...]:
    regions = []
    for frame in frames:
        asset = assets.get(frame.id)
        if asset is None or asset.average_depth is None:
            continue
        bins = asset.depth_bins or (
            {
                "id": 0.0,
                "average": asset.average_depth,
                "minimum": asset.min_depth if asset.min_depth is not None else asset.average_depth,
                "maximum": asset.max_depth if asset.max_depth is not None else asset.average_depth,
                "coverage": asset.depth_coverage or 1.0,
            },
        )
        for bin_index, depth_bin in enumerate(bins):
            suffix = "depth_prior" if len(bins) == 1 else f"depth_prior_{int(depth_bin['id'])}"
            regions.append(_depth_region_for_bin(frame, asset, depth_bin, suffix, bin_index))
    return tuple(regions)


def _feature_regions_from_tensors(
    frames: tuple[TrainingFrame, ...],
    tensors: dict[str, CaptureFrameTensors],
    assets: dict[str, CaptureFrameAssets],
) -> tuple[TrainingRegion, ...]:
    return propose_training_regions_from_tensors(frames, tensors, assets)


def _depth_region_for_bin(
    frame: TrainingFrame,
    asset: CaptureFrameAssets,
    depth_bin: dict[str, float],
    suffix: str,
    bin_index: int,
) -> TrainingRegion:
    center_depth = depth_bin["average"]
    min_depth = depth_bin["minimum"]
    max_depth = depth_bin["maximum"]
    coverage = depth_bin["coverage"]
    half_width = _depth_region_half_extent(frame, center_depth) * max(0.35, min(1.0, coverage * 2.0))
    thickness = max(max_depth - min_depth, center_depth * 0.01, 1e-3)
    geometry_confidence = min(1.0, 0.55 + 0.4 * coverage)
    return TrainingRegion(
        id=f"{frame.id}_{suffix}",
        frame_id=frame.id,
        bounds=Bounds(
            min_corner=(-half_width, -half_width, max(center_depth - thickness / 2.0, 1e-6)),
            max_corner=(half_width, half_width, center_depth + thickness / 2.0),
        ),
        evidence=RegionEvidence(
            geometry_confidence=geometry_confidence,
            ray_need=0.75,
            edit_need=0.45,
            compact_detail=0.8 if coverage < 0.35 else 0.0,
            fuzzy_confidence=0.1 * (1.0 - coverage),
        ),
        color=frame.target_color,
        opacity=min(0.95, max(0.35, coverage)),
        confidence=geometry_confidence,
        normal=asset.average_normal or (0.0, 0.0, -1.0),
        material_id=f"mat_depth_prior_{bin_index}",
        semantic_label=frame.semantic_label,
        fallback_source="capture-depth-prior",
    )


def _depth_region_half_extent(frame: TrainingFrame, depth: float) -> float:
    if frame.intrinsics is None:
        return max(0.05, depth * 0.05)
    width = frame.intrinsics.get("width", 1.0)
    height = frame.intrinsics.get("height", 1.0)
    fx = max(frame.intrinsics.get("fx", 1.0), 1e-6)
    fy = max(frame.intrinsics.get("fy", 1.0), 1e-6)
    half_x = depth * width / (2.0 * fx)
    half_y = depth * height / (2.0 * fy)
    return max(0.05, min(half_x, half_y))


def _mask_regions_from_assets(
    frames: tuple[TrainingFrame, ...],
    assets: dict[str, CaptureFrameAssets],
) -> tuple[TrainingRegion, ...]:
    regions = []
    for frame in frames:
        asset = assets.get(frame.id)
        if asset is None or asset.mask_coverage is None or asset.mask_coverage <= 0.0:
            continue
        depth = asset.average_depth if asset.average_depth is not None else frame.target_depth
        half_width = _depth_region_half_extent(frame, depth) * max(0.25, min(1.0, asset.mask_coverage * 2.0))
        label = frame.semantic_label or f"{frame.id}_mask"
        semantic_confidence = min(1.0, 0.65 + 0.35 * asset.mask_coverage)
        regions.append(
            TrainingRegion(
                id=f"{frame.id}_mask_semantic",
                frame_id=frame.id,
                bounds=Bounds(
                    min_corner=(-half_width, -half_width, max(depth - max(depth * 0.02, 1e-3), 1e-6)),
                    max_corner=(half_width, half_width, depth + max(depth * 0.02, 1e-3)),
                ),
                evidence=RegionEvidence(
                    semantic_confidence=semantic_confidence,
                    geometry_confidence=min(0.75, 0.35 + 0.4 * asset.mask_coverage),
                    ray_need=0.65,
                    edit_need=0.6,
                ),
                color=frame.target_color,
                opacity=min(0.9, max(0.25, asset.mask_coverage)),
                confidence=semantic_confidence,
                normal=asset.average_normal,
                material_id="mat_mask_semantic",
                semantic_label=label,
                fallback_source="capture-mask-prior",
            )
        )
    return tuple(regions)


def _resolve_capture_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _average_rgb(image: _RasterImage) -> Vec3:
    if image.channels < 3:
        raise ValueError("average RGB requires at least a 3-channel image")
    totals = [0.0, 0.0, 0.0]
    for pixel_start in range(0, len(image.values), image.channels):
        for channel in range(3):
            totals[channel] += image.values[pixel_start + channel]
    pixels = image.width * image.height
    return (totals[0] / pixels, totals[1] / pixels, totals[2] / pixels)


def _average_scalar(image: _RasterImage | None) -> float | None:
    if image is None:
        return None
    if image.channels != 1:
        raise ValueError("average scalar requires a 1-channel image")
    return sum(image.values) / len(image.values)


def _average_normal(image: _RasterImage | None) -> Vec3 | None:
    if image is None:
        return None
    if image.channels != 3:
        raise ValueError("average normal requires a 3-channel image")
    totals = [0.0, 0.0, 0.0]
    for pixel_start in range(0, len(image.values), image.channels):
        for channel in range(3):
            totals[channel] += image.values[pixel_start + channel]
    pixels = image.width * image.height
    vector = tuple(total / pixels for total in totals)
    norm = sum(channel * channel for channel in vector) ** 0.5
    if norm <= 1e-12:
        return None
    return tuple(channel / norm for channel in vector)  # type: ignore[return-value]


def _depth_summary(image: _RasterImage | None) -> _ScalarSummary | None:
    if image is None:
        return None
    if image.channels != 1:
        raise ValueError("depth summary requires a 1-channel image")
    valid = tuple(value for value in image.values if value > 0.0)
    if not valid:
        raise ValueError("depth asset contains no positive samples")
    return _ScalarSummary(
        average=sum(valid) / len(valid),
        minimum=min(valid),
        maximum=max(valid),
        coverage=len(valid) / len(image.values),
        bins=_depth_bins(valid, len(image.values)),
    )


def _depth_bins(valid: tuple[float, ...], total_count: int) -> tuple[dict[str, float], ...]:
    if not valid:
        return tuple()
    minimum = min(valid)
    maximum = max(valid)
    if len(valid) < 2 or maximum - minimum <= max(maximum * 0.05, 1e-4):
        return (
            {
                "id": 0.0,
                "average": sum(valid) / len(valid),
                "minimum": minimum,
                "maximum": maximum,
                "coverage": len(valid) / total_count,
            },
        )
    midpoint = (minimum + maximum) / 2.0
    bins = []
    for bin_id, values in ((0.0, tuple(value for value in valid if value <= midpoint)), (1.0, tuple(value for value in valid if value > midpoint))):
        if not values:
            continue
        bins.append(
            {
                "id": bin_id,
                "average": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
                "coverage": len(values) / total_count,
            }
        )
    return tuple(bins)


def _read_capture_raster(path: Path) -> _RasterImage:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".ppm", ".pgm", ".pnm"}:
        return _read_netpbm(path)
    if suffix == ".png":
        return _read_png(path)
    if suffix == ".bin":
        return _read_colmap_dense_map(path)
    if suffix in {".exr", ".hdr", ".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError(
            f"{path} requires the future GPU tensor asset backend; current stdlib loader supports PNG, PPM/PGM, and COLMAP depth maps"
        )
    raise ValueError(f"unsupported capture asset extension {suffix!r}; expected PNG, PPM, PGM, or COLMAP depth .bin")


def _read_capture_tensor(path: Path) -> CaptureTensor:
    suffix = path.suffix.lower()
    if suffix in {".ppm", ".pgm", ".pnm", ".png", ".bin"}:
        return _raster_to_tensor(path, _read_capture_raster(path), backend="stdlib")
    if suffix in {".exr", ".hdr", ".mp4", ".mov", ".mkv", ".avi"}:
        return _read_imageio_tensor(path)
    raise ValueError(
        f"unsupported capture asset extension {suffix!r}; expected PNG, PPM, PGM, COLMAP depth .bin, EXR/HDR, or video"
    )


def _raster_to_tensor(path: Path, raster: _RasterImage, *, backend: str) -> CaptureTensor:
    return CaptureTensor(
        path=str(path),
        format=raster.format,
        backend=backend,
        width=raster.width,
        height=raster.height,
        channels=raster.channels,
        values=raster.values,
    )


def _tensor_to_raster(tensor: CaptureTensor) -> _RasterImage:
    return _RasterImage(
        format=tensor.format,
        width=tensor.width,
        height=tensor.height,
        channels=tensor.channels,
        values=tensor.values,
    )


def _read_imageio_tensor(path: Path) -> CaptureTensor:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import imageio.v3 as imageio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            f"{path} requires the optional tensor asset backend; install aura-core[assets] to load EXR/HDR/video assets"
        ) from exc
    array = imageio.imread(path)
    shape = tuple(int(item) for item in getattr(array, "shape", ()))
    if len(shape) == 2:
        height, width = shape
        channels = 1
        flat = array.reshape(-1)
    elif len(shape) == 3:
        height, width, channels = shape
        flat = array.reshape(-1)
    elif len(shape) == 4:
        _frames, height, width, channels = shape
        first_frame = array[0]
        flat = first_frame.reshape(-1)
    else:
        raise ValueError(f"{path} tensor backend returned unsupported shape {shape!r}")
    values = _normalize_tensor_values(flat, dtype=str(getattr(array, "dtype", "")))
    return CaptureTensor(
        path=str(path),
        format=path.suffix.lower().lstrip(".").upper(),
        backend="imageio",
        width=width,
        height=height,
        channels=channels,
        values=values,
    )


def _normalize_tensor_values(values: IterableABC[object], *, dtype: str) -> PackedFloatBuffer:
    if dtype.startswith("uint8"):
        return PackedFloatBuffer(float(value) / 255.0 for value in values)
    if dtype.startswith("uint16"):
        return PackedFloatBuffer(float(value) / 65535.0 for value in values)
    return PackedFloatBuffer(values)


def _read_netpbm(path: Path) -> _RasterImage:
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    offset = 0
    magic, offset = _netpbm_token(data, offset)
    width_token, offset = _netpbm_token(data, offset)
    height_token, offset = _netpbm_token(data, offset)
    max_token, offset = _netpbm_token(data, offset)
    width = int(width_token)
    height = int(height_token)
    max_value = int(max_token)
    if magic not in {"P2", "P3", "P5", "P6"}:
        raise ValueError(f"unsupported capture asset format {magic!r}; expected PPM/PGM Netpbm")
    channels = 3 if magic in {"P3", "P6"} else 1
    expected = width * height * channels
    if magic in {"P2", "P3"}:
        values = [int(item) for item in data[offset:].decode("ascii").split()]
    else:
        if max_value > 255:
            raise ValueError("binary Netpbm capture fixtures only support max_value <= 255")
        offset = _skip_netpbm_space_and_comments(data, offset)
        raw = data[offset : offset + expected]
        if len(raw) != expected:
            raise ValueError(f"{path} expected {expected} binary channel values but found {len(raw)}")
        values = list(raw)
    if len(values) != expected:
        raise ValueError(f"{path} expected {expected} channel values but found {len(values)}")
    if any(value < 0 or value > max_value for value in values):
        raise ValueError(f"{path} contains channel values outside [0, {max_value}]")
    return _RasterImage(
        format="Netpbm",
        width=width,
        height=height,
        channels=channels,
        values=tuple(value / max_value for value in values),
    )


def _read_png(path: Path) -> _RasterImage:
    data = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError(f"{path} is not a PNG file")
    offset = len(signature)
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError(f"{path} has a truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        if len(chunk_data) != length:
            raise ValueError(f"{path} has a truncated PNG chunk payload")
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if bit_depth != 8:
                raise ValueError(f"{path} PNG loader supports only 8-bit channels")
            if color_type not in {0, 2, 4, 6}:
                raise ValueError(f"{path} PNG loader supports grayscale, RGB, grayscale-alpha, and RGBA")
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError(f"{path} PNG loader supports only non-interlaced deflate PNG images")
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"{path} is missing a PNG IHDR chunk")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = zlib.decompress(bytes(idat))
    row_bytes = width * channels
    expected = height * (1 + row_bytes)
    if len(raw) != expected:
        raise ValueError(f"{path} expected {expected} inflated PNG bytes but found {len(raw)}")
    rows: list[bytes] = []
    previous = bytes(row_bytes)
    offset = 0
    for _row in range(height):
        filter_type = raw[offset]
        scanline = raw[offset + 1 : offset + 1 + row_bytes]
        if len(scanline) != row_bytes:
            raise ValueError(f"{path} has a truncated PNG scanline")
        reconstructed = _png_unfilter(filter_type, scanline, previous, channels)
        rows.append(reconstructed)
        previous = reconstructed
        offset += 1 + row_bytes
    values = tuple(channel / 255.0 for row in rows for channel in row)
    return _RasterImage(format="PNG", width=width, height=height, channels=channels, values=values)


def _read_colmap_dense_map(path: Path) -> _RasterImage:
    data = path.read_bytes()
    header_parts: list[bytes] = []
    offset = 0
    for _index in range(3):
        end = data.find(b"&", offset)
        if end < 0:
            raise ValueError(f"{path} is missing a COLMAP dense-map header")
        header_parts.append(data[offset:end])
        offset = end + 1
    try:
        width, height, channels = (int(part.decode("ascii")) for part in header_parts)
    except ValueError as exc:
        raise ValueError(f"{path} has an invalid COLMAP dense-map header") from exc
    if channels not in {1, 3}:
        raise ValueError(f"{path} must be a single-channel COLMAP depth map or 3-channel normal map")
    expected_values = width * height * channels
    expected_bytes = expected_values * 4
    payload = data[offset:]
    if len(payload) != expected_bytes:
        raise ValueError(f"{path} expected {expected_bytes} float32 depth bytes but found {len(payload)}")
    values = struct.unpack("<" + "f" * expected_values, payload)
    if channels == 1 and any(value < 0.0 for value in values):
        raise ValueError(f"{path} contains negative depth values")
    return _RasterImage(format="COLMAP_DENSE_MAP", width=width, height=height, channels=channels, values=tuple(float(value) for value in values))


def _png_unfilter(filter_type: int, scanline: bytes, previous: bytes, bytes_per_pixel: int) -> bytes:
    out = bytearray(len(scanline))
    for index, value in enumerate(scanline):
        left = out[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index]
        up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) // 2
        elif filter_type == 4:
            predictor = _paeth_predictor(left, up, up_left)
        else:
            raise ValueError(f"unsupported PNG filter type {filter_type}")
        out[index] = (value + predictor) & 0xFF
    return bytes(out)


def _paeth_predictor(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _netpbm_token(data: bytes, offset: int) -> tuple[str, int]:
    offset = _skip_netpbm_space_and_comments(data, offset)
    start = offset
    while offset < len(data) and data[offset] not in b" \t\r\n":
        offset += 1
    if start == offset:
        raise ValueError("unexpected end of Netpbm header")
    return data[start:offset].decode("ascii"), offset


def _skip_netpbm_space_and_comments(data: bytes, offset: int) -> int:
    while offset < len(data):
        if data[offset] in b" \t\r\n":
            offset += 1
            continue
        if data[offset] == ord("#"):
            while offset < len(data) and data[offset] not in b"\r\n":
                offset += 1
            continue
        break
    return offset
