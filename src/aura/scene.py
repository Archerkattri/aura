from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from aura.elements import AuraChunk, AuraElement
from aura.ray import Ray, RayQueryResult, Vec3
from aura.semantic import SemanticGraph


@dataclass(frozen=True)
class AuraScene:
    """Reference scene for the AURA ray-query contract."""

    name: str
    elements: Sequence[AuraElement]
    chunks: Sequence[AuraChunk] = field(default_factory=tuple)
    semantic_graph: SemanticGraph = field(default_factory=SemanticGraph)

    def ray_query(self, ray: Ray) -> RayQueryResult:
        hits = [hit for element in self.elements if (hit := element.ray_query(ray)) is not None]
        if not hits:
            return RayQueryResult(color=(0.0, 0.0, 0.0), transmittance=1.0, confidence=0.0, provenance="miss")
        hits.sort(key=lambda item: item.depth if item.depth is not None else float("inf"))
        return composite_front_to_back(hits)

    def carrier_ids(self) -> list[str]:
        return sorted({element.carrier_id for element in self.elements})

    def chunk_ids(self) -> list[str]:
        return sorted({element.chunk_id for element in self.elements})


def composite_front_to_back(hits: Iterable[RayQueryResult]) -> RayQueryResult:
    color: Vec3 = (0.0, 0.0, 0.0)
    transmittance = 1.0
    confidence_num = 0.0
    confidence_den = 0.0
    first = None
    provenance: list[str] = []
    residual = False
    for hit in hits:
        if first is None:
            first = hit
        alpha = 1.0 - hit.transmittance
        weight = transmittance * alpha
        color = (
            color[0] + weight * hit.color[0],
            color[1] + weight * hit.color[1],
            color[2] + weight * hit.color[2],
        )
        confidence_num += weight * hit.confidence
        confidence_den += weight
        transmittance *= hit.transmittance
        provenance.append(hit.provenance or "unknown")
        residual = residual or hit.residual
    confidence = 0.0 if confidence_den == 0.0 else confidence_num / confidence_den
    return RayQueryResult(
        color=color,
        transmittance=transmittance,
        confidence=confidence,
        depth=first.depth if first else None,
        normal=first.normal if first else None,
        material_id=first.material_id if first else None,
        semantic_id=first.semantic_id if first else None,
        residual=residual,
        provenance=",".join(provenance),
    )
