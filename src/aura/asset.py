from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

from aura.carriers import CarrierSpec
from aura.schema import AURA_SCHEMA_VERSION


@dataclass(frozen=True)
class AuraAsset:
    """Minimal AURA asset manifest for carrier capability reporting."""

    name: str
    carrier_ids: Sequence[str]
    version: str = AURA_SCHEMA_VERSION
    units: str = "meters"
    coordinate_system: str = "right-handed-y-up"
    fallbacks: Dict[str, str] = field(default_factory=dict)

    def capabilities(self, registry: Mapping[str, CarrierSpec]) -> Dict[str, bool]:
        specs = []
        for carrier_id in self.carrier_ids:
            if carrier_id not in registry:
                raise KeyError(f"unknown carrier id: {carrier_id}")
            specs.append(registry[carrier_id])

        return {
            "primaryRender": any(spec.primary_render for spec in specs),
            "rayQuery": all(spec.ray_query for spec in specs),
            "collisionProxy": any(spec.collision_proxy for spec in specs),
            "directRelighting": any(spec.direct_relighting for spec in specs),
            "semanticQuery": any(spec.semantic_query for spec in specs),
            "neuralResidual": any(spec.neural_residual for spec in specs),
        }
