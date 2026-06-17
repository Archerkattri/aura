from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple

Vec3 = Tuple[float, float, float]


def _check_vec3(name: str, value: Vec3) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly three values")
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _check_unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class Ray:
    """Normalized ray used by AURA query contracts."""

    origin: Vec3
    direction: Vec3

    def __post_init__(self) -> None:
        origin = _check_vec3("origin", self.origin)
        direction = _check_vec3("direction", self.direction)
        norm = sqrt(sum(v * v for v in direction))
        if norm == 0.0:
            raise ValueError("direction must be non-zero")
        normalized = tuple(v / norm for v in direction)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", normalized)


@dataclass(frozen=True)
class RayQueryResult:
    """Common response every AURA carrier must eventually provide."""

    color: Vec3
    transmittance: float
    confidence: float
    depth: Optional[float] = None
    normal: Optional[Vec3] = None
    material_id: Optional[str] = None
    semantic_id: Optional[str] = None
    residual: bool = False
    provenance: Optional[str] = None

    def __post_init__(self) -> None:
        color = _check_vec3("color", self.color)
        transmittance = _check_unit_interval("transmittance", self.transmittance)
        confidence = _check_unit_interval("confidence", self.confidence)
        if self.depth is not None and self.depth < 0.0:
            raise ValueError("depth must be non-negative")
        if self.normal is not None:
            _check_vec3("normal", self.normal)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "transmittance", transmittance)
        object.__setattr__(self, "confidence", confidence)

    @property
    def opacity(self) -> float:
        return 1.0 - self.transmittance

