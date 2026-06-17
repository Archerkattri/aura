from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

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

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("element id is required")
        if not self.carrier_id:
            raise ValueError("carrier_id is required")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise ValueError("opacity must be in [0, 1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def ray_query(self, ray: Ray) -> RayQueryResult | None:
        hit = self.bounds.intersect_ray(ray)
        if hit is None:
            return None
        depth = hit[0]
        return RayQueryResult(
            color=self.color,
            transmittance=1.0 - self.opacity,
            confidence=self.confidence,
            depth=depth,
            normal=self.normal,
            material_id=self.material_id,
            semantic_id=self.semantic_id,
            residual=self.residual,
            provenance=self.id,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bounds"] = {
            "min": list(self.bounds.min_corner),
            "max": list(self.bounds.max_corner),
        }
        return payload


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

