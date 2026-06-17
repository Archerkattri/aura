from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aura.assignment import RegionEvidence
from aura.core import TrainingDataset, TrainingFrame, TrainingRegion
from aura.elements import Bounds
from aura.ray import Vec3


@dataclass(frozen=True)
class CaptureManifest:
    """A real-capture ingest contract before images are loaded or optimized."""

    root: str
    frames: tuple[TrainingFrame, ...]
    regions: tuple[TrainingRegion, ...]

    def to_training_dataset(self, *, load_assets: bool = False) -> TrainingDataset:
        if not load_assets:
            return TrainingDataset(frames=self.frames, regions=self.regions)
        assets = {item.frame_id: item for item in load_capture_assets(self)}
        frames = tuple(_frame_with_asset_summaries(frame, assets.get(frame.id)) for frame in self.frames)
        return TrainingDataset(frames=frames, regions=self.regions)

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
    mask_path: str | None = None
    mask_coverage: float | None = None

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "imagePath": self.image_path,
            "width": self.width,
            "height": self.height,
            "averageColor": list(self.average_color),
            "depthPath": self.depth_path,
            "averageDepth": self.average_depth,
            "maskPath": self.mask_path,
            "maskCoverage": self.mask_coverage,
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

    root = Path(manifest.root)
    assets = []
    for frame in manifest.frames:
        if frame.image_path is None:
            raise ValueError(f"capture frame {frame.id} is missing image_path")
        image_path = _resolve_capture_path(root, frame.image_path)
        image = _read_capture_raster(image_path)
        if image.channels < 3:
            raise ValueError(f"capture frame {frame.id} image_path must reference an RGB/RGBA image")
        depth_path = _resolve_capture_path(root, frame.depth_path) if frame.depth_path is not None else None
        depth = _read_capture_raster(depth_path) if depth_path is not None else None
        if depth is not None and depth.channels != 1:
            raise ValueError(f"capture frame {frame.id} depth_path must reference a single-channel image")
        mask_path = _resolve_capture_path(root, frame.mask_path) if frame.mask_path is not None else None
        mask = _read_capture_raster(mask_path) if mask_path is not None else None
        if mask is not None and mask.channels != 1:
            raise ValueError(f"capture frame {frame.id} mask_path must reference a single-channel image")
        assets.append(
            CaptureFrameAssets(
                frame_id=frame.id,
                image_path=str(image_path),
                width=image.width,
                height=image.height,
                average_color=_average_rgb(image),
                depth_path=str(depth_path) if depth_path is not None else None,
                average_depth=_average_scalar(depth) if depth is not None else None,
                mask_path=str(mask_path) if mask_path is not None else None,
                mask_coverage=_average_scalar(mask) if mask is not None else None,
            )
        )
    return tuple(assets)


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
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.format} dimensions must be positive")
        if len(self.values) != self.width * self.height * self.channels:
            raise ValueError(f"{self.format} payload does not match dimensions")


def _frame_with_asset_summaries(frame: TrainingFrame, assets: CaptureFrameAssets | None) -> TrainingFrame:
    if assets is None:
        return frame
    return replace(
        frame,
        target_color=assets.average_color,
        target_depth=assets.average_depth if assets.average_depth is not None else frame.target_depth,
    )


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


def _read_capture_raster(path: Path) -> _RasterImage:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".ppm", ".pgm", ".pnm"}:
        return _read_netpbm(path)
    if suffix == ".png":
        return _read_png(path)
    if suffix in {".exr", ".hdr", ".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError(
            f"{path} requires the future GPU tensor asset backend; current stdlib loader supports PNG and PPM/PGM"
        )
    raise ValueError(f"unsupported capture asset extension {suffix!r}; expected PNG, PPM, or PGM")


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
