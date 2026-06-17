from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from math import exp, sqrt
from pathlib import Path
from typing import Sequence

from aura.assignment import RegionEvidence
from aura.decomposition import EvidenceSample, decompose_evidence
from aura.elements import AuraElement, Bounds
from aura.carrier_payloads import GaussianFallbackPayload
from aura.ray import Vec3
from aura.scene import AuraScene

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

PLY_SCALAR_TYPES = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


@dataclass(frozen=True)
class PlyProperty:
    name: str
    type_name: str

    @property
    def struct_format(self) -> str:
        if self.type_name not in PLY_SCALAR_TYPES:
            raise ValueError(f"unsupported PLY property type: {self.type_name}")
        return PLY_SCALAR_TYPES[self.type_name][0]

    @property
    def byte_size(self) -> int:
        if self.type_name not in PLY_SCALAR_TYPES:
            raise ValueError(f"unsupported PLY property type: {self.type_name}")
        return PLY_SCALAR_TYPES[self.type_name][1]


@dataclass(frozen=True)
class PlyHeader:
    format_name: str
    properties: tuple[PlyProperty, ...]
    vertex_count: int
    data_start: int


def _vec3(name: str, value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly three values")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _unit(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _matrix3(name: str, value: Sequence[Sequence[float]]) -> Matrix3:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError(f"{name} must be a 3x3 matrix")
    matrix = tuple(tuple(float(item) for item in row) for row in value)
    diag = (matrix[0][0], matrix[1][1], matrix[2][2])
    if any(item <= 0.0 for item in diag):
        raise ValueError(f"{name} diagonal entries must be positive")
    return matrix  # type: ignore[return-value]


def _logistic(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class GaussianSplatSample:
    id: str
    mean: Vec3
    covariance: Matrix3
    opacity: float
    color: Vec3
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict, index: int) -> "GaussianSplatSample":
        if "covariance" in payload:
            covariance = _matrix3("covariance", payload["covariance"])
        elif "covariance_diag" in payload:
            diag = _vec3("covariance_diag", payload["covariance_diag"])
            covariance = ((diag[0], 0.0, 0.0), (0.0, diag[1], 0.0), (0.0, 0.0, diag[2]))
        else:
            raise ValueError("splat must define covariance or covariance_diag")

        return cls(
            id=str(payload.get("id", f"splat_{index:04d}")),
            mean=_vec3("mean", payload["mean"]),
            covariance=covariance,
            opacity=_unit("opacity", payload["opacity"]),
            color=_vec3("color", payload.get("color", (1.0, 1.0, 1.0))),
            confidence=_unit("confidence", payload.get("confidence", 1.0)),
            metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
        )

    def to_element(self, *, radius_sigma: float = 2.0, chunk_id: str = "root") -> AuraElement:
        """Build a compatibility Gaussian element directly from a splat sample.

        New scene-level ingest should prefer ``to_evidence_sample`` followed by
        adaptive decomposition.
        """

        evidence = self.to_evidence_sample(radius_sigma=radius_sigma)
        bounds = evidence.bounds
        payload = GaussianFallbackPayload(mean=self.mean, covariance=self.covariance, source="3dgs-ingest").to_dict()
        return AuraElement(
            id=self.id,
            carrier_id="gaussian",
            bounds=bounds,
            color=self.color,
            opacity=self.opacity,
            confidence=self.confidence,
            chunk_id=chunk_id,
            metadata=dict(evidence.metadata),
            confidence_map=dict(evidence.confidence_map),
            edit=dict(evidence.edit),
            payload=payload,
        )

    def to_evidence_sample(self, *, radius_sigma: float = 2.0) -> EvidenceSample:
        if radius_sigma <= 0.0:
            raise ValueError("radius_sigma must be positive")
        sigma = (
            sqrt(self.covariance[0][0]),
            sqrt(self.covariance[1][1]),
            sqrt(self.covariance[2][2]),
        )
        radius = tuple(radius_sigma * value for value in sigma)
        bounds = Bounds(
            min_corner=(
                self.mean[0] - radius[0],
                self.mean[1] - radius[1],
                self.mean[2] - radius[2],
            ),
            max_corner=(
                self.mean[0] + radius[0],
                self.mean[1] + radius[1],
                self.mean[2] + radius[2],
            ),
        )
        return EvidenceSample(
            id=self.id,
            bounds=bounds,
            evidence=RegionEvidence(
                image_error=0.05,
                geometry_confidence=0.35,
                material_confidence=0.35,
                ray_need=0.2,
                edit_need=0.1,
            ),
            color=self.color,
            opacity=self.opacity,
            confidence=self.confidence,
            metadata={
                "source": "3dgs-export",
                "mean": json.dumps(list(self.mean)),
                "covariance": json.dumps([list(row) for row in self.covariance]),
                **self.metadata,
            },
            confidence_map={"splat": self.confidence},
            edit={"source": "aura-ingest:3dgs"},
            gaussian_mean=self.mean,
            gaussian_covariance=self.covariance,
            fallback_source="3dgs-ingest",
        )


def load_3dgs_export(path: Path | str) -> list[GaussianSplatSample]:
    source = Path(path)
    if source.suffix.lower() == ".ply":
        return load_3dgs_ply(source)
    payload = _read_3dgs_payload(source)
    return _samples_from_payload(payload)


def _read_3dgs_payload(path: Path | str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("3DGS export must be a JSON object")
    return payload


def _samples_from_payload(payload: dict) -> list[GaussianSplatSample]:
    splats = payload.get("splats")
    if not isinstance(splats, list) or not splats:
        raise ValueError("3DGS export must contain a non-empty splats list")
    return [GaussianSplatSample.from_dict(item, index) for index, item in enumerate(splats)]


def load_3dgs_ply(path: Path | str) -> list[GaussianSplatSample]:
    """Read an ASCII or binary little-endian Gaussian Splat PLY export.

    This supports the common 3DGS vertex schema with x/y/z, opacity, scale_0..2,
    rot_0..3, and either RGB or DC spherical harmonic color properties. Rotation
    is preserved as metadata and applied to the world covariance used for
    axis-aligned AURA bounds.
    """

    source = Path(path)
    data = source.read_bytes()
    header = _parse_ply_header(data)
    if header.format_name == "ascii":
        return _load_ascii_ply_samples(data, header)
    if header.format_name == "binary_little_endian":
        return _load_binary_little_endian_ply_samples(data, header)
    raise ValueError(f"unsupported PLY format: {header.format_name}")


def _load_ascii_ply_samples(data: bytes, header: PlyHeader) -> list[GaussianSplatSample]:
    body = data[header.data_start :].decode("utf-8")
    lines = body.splitlines()
    samples = []
    properties = [item.name for item in header.properties]
    for index, line in enumerate(lines[: header.vertex_count]):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < len(properties):
            raise ValueError(f"PLY vertex {index} has {len(values)} values for {len(properties)} properties")
        row = {name: float(values[offset]) for offset, name in enumerate(properties)}
        samples.append(_sample_from_ply_row(row, index))
    if len(samples) != header.vertex_count:
        raise ValueError(f"PLY expected {header.vertex_count} vertices but parsed {len(samples)}")
    return samples


def _load_binary_little_endian_ply_samples(data: bytes, header: PlyHeader) -> list[GaussianSplatSample]:
    row_format = "<" + "".join(item.struct_format for item in header.properties)
    row_size = struct.calcsize(row_format)
    expected_size = header.data_start + row_size * header.vertex_count
    if len(data) < expected_size:
        raise ValueError(f"PLY binary body is too short for {header.vertex_count} vertices")
    samples = []
    property_names = [item.name for item in header.properties]
    offset = header.data_start
    for index in range(header.vertex_count):
        values = struct.unpack_from(row_format, data, offset)
        row = {name: float(value) for name, value in zip(property_names, values)}
        samples.append(_sample_from_ply_row(row, index))
        offset += row_size
    return samples


def _parse_ply_header(data: bytes) -> PlyHeader:
    marker = b"end_header"
    marker_offset = data.find(marker)
    if marker_offset < 0:
        raise ValueError("PLY export missing end_header")
    line_end = data.find(b"\n", marker_offset)
    if line_end < 0:
        data_start = marker_offset + len(marker)
        header_bytes = data[:data_start]
    else:
        data_start = line_end + 1
        header_bytes = data[:data_start]
    header_text = header_bytes.decode("ascii")
    lines = header_text.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("PLY export must start with a ply header")

    format_name: str | None = None
    properties: list[PlyProperty] = []
    vertex_count: int | None = None
    in_vertex = False
    for index, line in enumerate(lines[1:], start=1):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            if len(parts) < 3 or parts[2] != "1.0":
                raise ValueError("PLY export must use format version 1.0")
            format_name = parts[1]
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("PLY vertex list properties are not supported")
            properties.append(PlyProperty(name=parts[-1], type_name=parts[1]))
            continue
        if parts[0] == "end_header":
            if vertex_count is None:
                raise ValueError("PLY export must define an element vertex count")
            if format_name is None:
                raise ValueError("PLY export must define a format")
            if format_name not in {"ascii", "binary_little_endian"}:
                raise ValueError(f"unsupported PLY format: {format_name}")
            property_names = [item.name for item in properties]
            required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
            missing = sorted(required.difference(property_names))
            if missing:
                raise ValueError(f"PLY export missing required properties: {', '.join(missing)}")
            # Validate scalar types before binary body parsing starts.
            for prop in properties:
                _ = prop.struct_format
            return PlyHeader(
                format_name=format_name,
                properties=tuple(properties),
                vertex_count=vertex_count,
                data_start=data_start,
            )
    raise ValueError("PLY export missing end_header")


def _sample_from_ply_row(row: dict[str, float], index: int) -> GaussianSplatSample:
    mean = (row["x"], row["y"], row["z"])
    covariance = _covariance_from_ply_row(row)
    opacity_value = row["opacity"]
    opacity = opacity_value if 0.0 <= opacity_value <= 1.0 else _logistic(opacity_value)
    return GaussianSplatSample(
        id=f"ply_splat_{index:04d}",
        mean=mean,
        covariance=covariance,
        opacity=_unit("opacity", opacity),
        color=_color_from_ply_row(row),
        confidence=1.0,
        metadata=_ply_metadata(row),
    )


def _covariance_from_ply_row(row: dict[str, float]) -> Matrix3:
    scale = (exp(row["scale_0"]), exp(row["scale_1"]), exp(row["scale_2"]))
    local_variance = (scale[0] * scale[0], scale[1] * scale[1], scale[2] * scale[2])
    rotation = _rotation_matrix_from_ply_row(row)
    return _rotate_diagonal_covariance(rotation, local_variance)


def _rotation_matrix_from_ply_row(row: dict[str, float]) -> Matrix3:
    rotation_keys = ("rot_0", "rot_1", "rot_2", "rot_3")
    if not set(rotation_keys).issubset(row):
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    w, x, y, z = (row[key] for key in rotation_keys)
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("PLY rotation quaternion must be non-zero")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _rotate_diagonal_covariance(rotation: Matrix3, variance: Vec3) -> Matrix3:
    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(sum(rotation[i][axis] * variance[axis] * rotation[j][axis] for axis in range(3)))
        rows.append(tuple(row))
    return tuple(rows)  # type: ignore[return-value]


def _color_from_ply_row(row: dict[str, float]) -> Vec3:
    if {"red", "green", "blue"}.issubset(row):
        return (_color_channel(row["red"]), _color_channel(row["green"]), _color_channel(row["blue"]))
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(row):
        sh_c0 = 0.28209479177387814
        return (
            _clamp01(0.5 + sh_c0 * row["f_dc_0"]),
            _clamp01(0.5 + sh_c0 * row["f_dc_1"]),
            _clamp01(0.5 + sh_c0 * row["f_dc_2"]),
        )
    return (1.0, 1.0, 1.0)


def _ply_metadata(row: dict[str, float]) -> dict[str, str]:
    metadata = {"source_format": "ply"}
    metadata["scale_encoding"] = "log"
    rotation_keys = ("rot_0", "rot_1", "rot_2", "rot_3")
    if set(rotation_keys).issubset(row):
        metadata["rotation_quaternion"] = json.dumps([row[key] for key in rotation_keys])
    return metadata


def _color_channel(value: float) -> float:
    return _clamp01(value / 255.0 if value > 1.0 else value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def splats_to_scene(
    samples: Sequence[GaussianSplatSample],
    *,
    name: str = "3dgs_fixture",
    radius_sigma: float = 2.0,
) -> AuraScene:
    if not samples:
        raise ValueError("samples must be non-empty")
    evidence = tuple(sample.to_evidence_sample(radius_sigma=radius_sigma) for sample in samples)
    return decompose_evidence(evidence, name=name)


def load_3dgs_scene(path: Path | str, *, name: str | None = None, radius_sigma: float = 2.0) -> AuraScene:
    source = Path(path)
    if source.suffix.lower() == ".ply":
        samples = load_3dgs_ply(source)
        scene_name = name or source.stem
        return splats_to_scene(samples, name=scene_name, radius_sigma=radius_sigma)
    payload = _read_3dgs_payload(source)
    samples = _samples_from_payload(payload)
    scene_name = name or str(payload.get("scene") or source.stem)
    return splats_to_scene(samples, name=scene_name, radius_sigma=radius_sigma)


def _union_bounds(bounds: Sequence[Bounds]) -> Bounds:
    return Bounds(
        min_corner=(
            min(item.min_corner[0] for item in bounds),
            min(item.min_corner[1] for item in bounds),
            min(item.min_corner[2] for item in bounds),
        ),
        max_corner=(
            max(item.max_corner[0] for item in bounds),
            max(item.max_corner[1] for item in bounds),
            max(item.max_corner[2] for item in bounds),
        ),
    )
