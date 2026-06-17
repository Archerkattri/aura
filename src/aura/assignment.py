from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Optional

from aura.carriers import CarrierSpec, default_registry


def _unit(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class RegionEvidence:
    """Normalized evidence used to choose an AURA carrier for one region."""

    image_error: float = 0.0
    geometry_confidence: float = 0.0
    material_confidence: float = 0.0
    ray_need: float = 0.0
    edit_need: float = 0.0
    high_frequency: float = 0.0
    view_dependent: float = 0.0
    semantic_confidence: float = 0.0
    fuzzy_confidence: float = 0.0
    compact_detail: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _unit(field.name, getattr(self, field.name)))


def choose_carrier(
    evidence: RegionEvidence,
    registry: Optional[Mapping[str, CarrierSpec]] = None,
) -> CarrierSpec:
    """Choose the simplest carrier supported by the current evidence.

    This first implementation intentionally uses explicit priority rules rather
    than learned scoring. It is the CPU-only contract baseline.
    """

    registry = registry or default_registry()

    if evidence.semantic_confidence >= 0.8:
        return registry["semantic"]
    if evidence.fuzzy_confidence >= 0.7 and evidence.geometry_confidence < 0.6:
        return registry["volume"]
    if evidence.high_frequency >= 0.8:
        return registry["gabor"]
    if evidence.view_dependent >= 0.75 and evidence.material_confidence < 0.5:
        return registry["neural"]
    if evidence.geometry_confidence >= 0.75 and evidence.edit_need >= 0.4:
        return registry["surface"]
    if evidence.compact_detail >= 0.75:
        return registry["beta"]
    return registry["gaussian"]

