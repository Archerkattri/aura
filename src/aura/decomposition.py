from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from aura.assignment import RegionEvidence, choose_carrier
from aura.carrier_payloads import (
    BetaKernelPayload,
    GaborFrequencyPayload,
    GaussianFallbackPayload,
    NeuralResidualPayload,
    SemanticFeaturePayload,
    SurfaceCellPayload,
    VolumeCellPayload,
)
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.ray import Vec3
from aura.scene import AuraScene
from aura.semantic import SemanticGraph, SemanticNode

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class EvidenceSample:
    """Scene-local evidence sample used by adaptive AURA decomposition."""

    id: str
    bounds: Bounds
    evidence: RegionEvidence
    color: Vec3 = (1.0, 1.0, 1.0)
    opacity: float = 1.0
    confidence: float = 1.0
    normal: Vec3 | None = None
    material_id: str | None = None
    semantic_label: str | None = None
    gaussian_mean: Vec3 | None = None
    gaussian_covariance: Matrix3 | None = None
    fallback_source: str = "adaptive-decomposition"
    confidence_map: Mapping[str, float] = field(default_factory=dict)
    edit: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)


def decompose_evidence(samples: Sequence[EvidenceSample], name: str = "aura_decomposition") -> AuraScene:
    """Convert evidence samples into typed native AURA carriers."""

    if not samples:
        return AuraScene(name=name, elements=tuple(), chunks=tuple())

    elements = tuple(_sample_to_element(sample) for sample in samples)
    chunk = AuraChunk(id="root", bounds=_union_bounds([sample.bounds for sample in samples]), element_ids=tuple(item.id for item in elements))
    semantic_graph = _semantic_graph_for(samples, elements)
    return AuraScene(name=name, elements=elements, chunks=(chunk,), semantic_graph=semantic_graph)


def _sample_to_element(sample: EvidenceSample) -> AuraElement:
    carrier = choose_carrier(sample.evidence)
    payload = _payload_for(sample, carrier.id)
    semantic_id = sample.semantic_label if carrier.id == "semantic" else None
    return AuraElement(
        id=sample.id,
        carrier_id=carrier.id,
        bounds=sample.bounds,
        color=sample.color,
        opacity=sample.opacity,
        confidence=sample.confidence,
        normal=sample.normal,
        material_id=sample.material_id,
        semantic_id=semantic_id,
        residual=carrier.id == "neural",
        metadata={"decomposition": "evidence-v0", **dict(sample.metadata)},
        confidence_map={"assignment": sample.confidence, **dict(sample.confidence_map)},
        edit={"source": "adaptive-decomposition", **dict(sample.edit)},
        payload=payload,
    )


def _payload_for(sample: EvidenceSample, carrier_id: str) -> dict:
    evidence = sample.evidence
    size = _extent(sample.bounds)
    half_extent = tuple(max(axis / 2.0, 1e-4) for axis in size)
    normal = sample.normal or (0.0, 0.0, 1.0)

    if carrier_id == "surface":
        thickness = max(min(size), 1e-4)
        roughness = 1.0 - evidence.material_confidence
        return SurfaceCellPayload(normal=normal, thickness=thickness, roughness=roughness).to_dict()
    if carrier_id == "volume":
        density = max(0.05, evidence.fuzzy_confidence)
        return VolumeCellPayload(density=density).to_dict()
    if carrier_id == "beta":
        return BetaKernelPayload(
            alpha=1.0 + evidence.compact_detail * 4.0,
            beta=1.0 + (1.0 - evidence.image_error) * 4.0,
            support_radius=half_extent,
        ).to_dict()
    if carrier_id == "gabor":
        frequency = (max(evidence.high_frequency, 0.05), 0.0, 0.0)
        bandwidth = max(0.05, 1.0 - evidence.high_frequency)
        return GaborFrequencyPayload(frequency=frequency, bandwidth=bandwidth).to_dict()
    if carrier_id == "neural":
        residual_scale = max(evidence.view_dependent, evidence.image_error)
        return NeuralResidualPayload(latent_dim=16, residual_scale=residual_scale).to_dict()
    if carrier_id == "semantic":
        label = sample.semantic_label or sample.id
        return SemanticFeaturePayload(label=label, confidence=evidence.semantic_confidence).to_dict()

    covariance = sample.gaussian_covariance or (
        (max(half_extent[0] ** 2, 1e-6), 0.0, 0.0),
        (0.0, max(half_extent[1] ** 2, 1e-6), 0.0),
        (0.0, 0.0, max(half_extent[2] ** 2, 1e-6)),
    )
    return GaussianFallbackPayload(
        mean=sample.gaussian_mean or _center(sample.bounds),
        covariance=covariance,
        source=sample.fallback_source,
    ).to_dict()


def _extent(bounds: Bounds) -> Vec3:
    return tuple(hi - lo for lo, hi in zip(bounds.min_corner, bounds.max_corner))  # type: ignore[return-value]


def _center(bounds: Bounds) -> Vec3:
    return tuple((lo + hi) / 2.0 for lo, hi in zip(bounds.min_corner, bounds.max_corner))  # type: ignore[return-value]


def _union_bounds(bounds: Sequence[Bounds]) -> Bounds:
    min_corner = tuple(min(item.min_corner[index] for item in bounds) for index in range(3))
    max_corner = tuple(max(item.max_corner[index] for item in bounds) for index in range(3))
    return Bounds(min_corner=min_corner, max_corner=max_corner)  # type: ignore[arg-type]


def _semantic_graph_for(samples: Sequence[EvidenceSample], elements: Sequence[AuraElement]) -> SemanticGraph:
    nodes = []
    element_by_id = {element.id: element for element in elements}
    for sample in samples:
        label = sample.semantic_label
        element = element_by_id[sample.id]
        if element.carrier_id == "semantic" and label is None:
            label = sample.id
        if label is None:
            continue
        nodes.append(
            SemanticNode(
                id=f"object:{label}",
                label=label,
                element_ids=(sample.id,),
                confidence=sample.evidence.semantic_confidence or sample.confidence,
                attributes={"source": "evidence"},
            )
        )
    return SemanticGraph(nodes=tuple(nodes))
