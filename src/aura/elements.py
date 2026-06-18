from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, pi, sin
from typing import Any, Dict, Tuple

from aura.ray import Ray, RayQueryResult, Vec3


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned bounded local support."""

    min_corner: Vec3
    max_corner: Vec3

    def __post_init__(self) -> None:
        for lo, hi in zip(self.min_corner, self.max_corner):
            if float(lo) > float(hi):
                raise ValueError("bounds must satisfy min <= max per axis")

    def intersect_ray(self, ray: Ray) -> tuple[float, float] | None:
        """Return ``(entry_depth, exit_depth)`` for the ray/AABB intersection, or ``None`` on miss."""
        t_min = 0.0
        t_max = float("inf")
        for origin, direction, lower, upper in zip(ray.origin, ray.direction, self.min_corner, self.max_corner):
            if abs(direction) < 1e-12:
                if origin < lower or origin > upper:
                    return None
                continue
            inv = 1.0 / direction
            t0 = (lower - origin) * inv
            t1 = (upper - origin) * inv
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_max < t_min:
                return None
        return (t_min, t_max)


@dataclass(frozen=True)
class AuraElement:
    """One primitive in an AURA scene.

    Each element occupies an axis-aligned bounding box, is assigned to a
    single carrier type (``carrier_id``), and carries optional per-element
    attributes such as color, opacity, normals, and a typed payload dict.
    """

    id: str
    carrier_id: str
    bounds: Bounds
    color: Vec3 = (1.0, 1.0, 1.0)
    opacity: float = 1.0
    confidence: float = 1.0
    normal: Vec3 | None = None
    material_id: str | None = None
    semantic_id: str | None = None
    residual: bool = False
    lod: int = 0
    chunk_id: str = "root"
    metadata: Dict[str, str] = field(default_factory=dict)
    confidence_map: Dict[str, float] = field(default_factory=dict)
    edit: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("element id is required")
        if not self.carrier_id:
            raise ValueError("carrier_id is required")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise ValueError("opacity must be in [0, 1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        for name, value in self.confidence_map.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"confidence_map value for {name!r} must be in [0, 1]")

    def ray_query(self, ray: Ray) -> RayQueryResult | None:
        """Intersect a ray with this element and return the compositing result.

        Returns ``None`` when the ray misses the element's bounding box.
        """
        hit = self.bounds.intersect_ray(ray)
        if hit is None:
            return None
        depth, exit_depth = hit
        color = self.color
        transmittance = 1.0 - self.opacity
        confidence = self.confidence
        normal = self.normal
        semantic_id = self.semantic_id
        residual = self.residual
        payload_type = self.payload.get("type")

        if payload_type == "surface_cell" and normal is None:
            normal = _payload_vec3(self.payload.get("normal"))
        elif payload_type == "volume_cell":
            density = _clamp_unit(float(self.payload.get("density", self.opacity)))
            volume_opacity = _clamp_unit(float(self.payload.get("opacity", 1.0)))
            path_length = max(0.0, exit_depth - depth)
            alpha = volume_opacity * (1.0 - exp(-density * path_length))
            transmittance = _clamp_unit(1.0 - alpha)
        elif payload_type == "beta_kernel":
            hit_point = _ray_point(ray, depth)
            weight = _beta_weight(self.bounds, hit_point, self.payload)
            transmittance = _clamp_unit(1.0 - self.opacity * weight)
        elif payload_type == "gabor_frequency":
            hit_point = _ray_point(ray, depth)
            color = _gabor_color(self.color, hit_point, self.payload)
            bandwidth = max(0.0, float(self.payload.get("bandwidth", 1.0)))
            confidence = _clamp_unit(self.confidence * min(1.0, bandwidth))
        elif payload_type == "neural_residual":
            residual = True
            confidence = _clamp_unit(self.confidence * (1.0 - float(self.payload.get("residual_scale", 0.0)) * 0.25))
        elif payload_type == "semantic_feature":
            semantic_id = semantic_id or str(self.payload.get("label", ""))
            confidence = _clamp_unit(float(self.payload.get("confidence", self.confidence)))
        elif payload_type == "gaussian_fallback":
            weight = _gaussian_weight(ray, self.payload, depth, exit_depth)
            transmittance = _clamp_unit(1.0 - self.opacity * weight)
            confidence = _clamp_unit(self.confidence * weight)

        return RayQueryResult(
            color=color,
            transmittance=transmittance,
            confidence=confidence,
            depth=depth,
            normal=normal,
            material_id=self.material_id,
            semantic_id=semantic_id,
            residual=residual,
            provenance=self.id,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bounds"] = {
            "min": list(self.bounds.min_corner),
            "max": list(self.bounds.max_corner),
        }
        return payload


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _payload_vec3(value: Any) -> Vec3 | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _ray_point(ray: Ray, depth: float) -> Vec3:
    return tuple(ray.origin[index] + ray.direction[index] * depth for index in range(3))  # type: ignore[return-value]


def _beta_weight(bounds: Bounds, point: Vec3, payload: Dict[str, Any]) -> float:
    coordinates = []
    for value, lower, upper in zip(point, bounds.min_corner, bounds.max_corner):
        extent = upper - lower
        if extent > 1e-12:
            coordinates.append(_clamp_unit((value - lower) / extent))
    u = sum(coordinates) / len(coordinates) if coordinates else 0.5
    alpha = max(1e-6, float(payload.get("alpha", 1.0)))
    beta = max(1e-6, float(payload.get("beta", 1.0)))
    raw = (u ** (alpha - 1.0)) * ((1.0 - u) ** (beta - 1.0))
    if alpha > 1.0 and beta > 1.0:
        mode = (alpha - 1.0) / (alpha + beta - 2.0)
        peak = (mode ** (alpha - 1.0)) * ((1.0 - mode) ** (beta - 1.0))
        if peak > 0.0:
            raw /= peak
    return _clamp_unit(raw)


def _gabor_color(color: Vec3, point: Vec3, payload: Dict[str, Any]) -> Vec3:
    frequency = _payload_vec3(payload.get("frequency")) or (0.0, 0.0, 0.0)
    phase = float(payload.get("phase", 0.0))
    bandwidth = min(1.0, max(0.0, float(payload.get("bandwidth", 1.0))))
    dot = sum(axis * position for axis, position in zip(frequency, point))
    wave = 0.5 + 0.5 * sin(2.0 * pi * dot + phase)
    modulation = 1.0 - bandwidth + bandwidth * wave
    return tuple(_clamp_unit(channel * modulation) for channel in color)  # type: ignore[return-value]


def _gaussian_weight(ray: Ray, payload: Dict[str, Any], entry_depth: float, exit_depth: float) -> float:
    mean = _payload_vec3(payload.get("mean"))
    covariance = payload.get("covariance")
    if mean is None or not _is_matrix3(covariance):
        return 1.0
    direction_norm = sum(axis * axis for axis in ray.direction)
    if direction_norm <= 1e-12:
        sample_depth = entry_depth
    else:
        projected = sum((mean[index] - ray.origin[index]) * ray.direction[index] for index in range(3)) / direction_norm
        sample_depth = max(entry_depth, min(exit_depth, projected))
    point = _ray_point(ray, sample_depth)
    delta = tuple(point[index] - mean[index] for index in range(3))
    inverse = _invert_matrix3(covariance)
    if inverse is None:
        return 1.0
    mahalanobis = sum(delta[row] * inverse[row][column] * delta[column] for row in range(3) for column in range(3))
    return _clamp_unit(exp(-0.5 * max(0.0, mahalanobis)))


def _is_matrix3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value)


def _invert_matrix3(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    matrix = tuple(tuple(float(item) for item in row) for row in value)
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) <= 1e-12:
        return None
    inv_det = 1.0 / determinant
    return (
        ((e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det),
        ((f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det),
        ((d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det),
    )


@dataclass(frozen=True)
class AuraChunk:
    id: str
    bounds: Bounds
    element_ids: Tuple[str, ...]
    lod: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bounds": {"min": list(self.bounds.min_corner), "max": list(self.bounds.max_corner)},
            "element_ids": list(self.element_ids),
            "lod": self.lod,
        }
