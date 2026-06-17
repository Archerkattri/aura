from __future__ import annotations

from dataclasses import dataclass, field, fields
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

    elements, chunks = carrier_lod_elements_and_chunks(tuple(_sample_to_element(sample) for sample in samples))
    semantic_graph = _semantic_graph_for(samples, elements)
    return AuraScene(name=name, elements=elements, chunks=chunks, semantic_graph=semantic_graph)


def _sample_to_element(sample: EvidenceSample) -> AuraElement:
    carrier = choose_carrier(sample.evidence)
    payload = _payload_for(sample, carrier.id)
    semantic_id = sample.semantic_label
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
        metadata={**dict(sample.metadata), **_decomposition_metadata(sample, carrier.id)},
        confidence_map={"assignment": sample.confidence, **dict(sample.confidence_map)},
        edit={"source": "adaptive-decomposition", **dict(sample.edit)},
        payload=payload,
    )


def _decomposition_metadata(sample: EvidenceSample, carrier_id: str) -> dict[str, str]:
    reason_metric, reason_value, reason_rule = _selection_reason(sample.evidence)
    role = "fallback" if carrier_id == "gaussian" else "native"
    metadata = {
        "decomposition": "evidence-v1",
        "decomposition_role": role,
        "selected_carrier": carrier_id,
        "selection_reason": reason_rule,
        "selection_evidence": f"{reason_metric}={reason_value:.3f}",
        "evidence_summary": _evidence_summary(sample.evidence),
    }
    if carrier_id == "gaussian":
        metadata["fallback_label"] = "gaussian_fallback"
        metadata["fallback_reason"] = "no_structured_native_evidence"
    return metadata


def _selection_reason(evidence: RegionEvidence) -> tuple[str, float, str]:
    if evidence.semantic_confidence >= 0.8:
        return ("semantic_confidence", evidence.semantic_confidence, "semantic_confidence>=0.80")
    if evidence.fuzzy_confidence >= 0.7 and evidence.geometry_confidence < 0.6:
        return ("fuzzy_confidence", evidence.fuzzy_confidence, "fuzzy_confidence>=0.70 and geometry_confidence<0.60")
    if evidence.high_frequency >= 0.8:
        return ("high_frequency", evidence.high_frequency, "high_frequency>=0.80")
    if evidence.view_dependent >= 0.75 and evidence.material_confidence < 0.5:
        return ("view_dependent", evidence.view_dependent, "view_dependent>=0.75 and material_confidence<0.50")
    if evidence.geometry_confidence >= 0.75 and evidence.edit_need >= 0.4:
        return ("geometry_confidence", evidence.geometry_confidence, "geometry_confidence>=0.75 and edit_need>=0.40")
    if evidence.compact_detail >= 0.75:
        return ("compact_detail", evidence.compact_detail, "compact_detail>=0.75")
    metric, value = max(
        ((field.name, float(getattr(evidence, field.name))) for field in fields(evidence)),
        key=lambda item: item[1],
    )
    return (metric, value, "no native carrier threshold met")


def _evidence_summary(evidence: RegionEvidence) -> str:
    nonzero = [
        f"{field.name}={float(getattr(evidence, field.name)):.3f}"
        for field in fields(evidence)
        if float(getattr(evidence, field.name)) > 0.0
    ]
    return ";".join(nonzero) or "none"


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


def carrier_lod_elements_and_chunks(elements: Sequence[AuraElement]) -> tuple[tuple[AuraElement, ...], tuple[AuraChunk, ...]]:
    """Assign deterministic carrier/LOD chunks to native AURA elements."""

    chunked_elements = tuple(
        AuraElement(
            id=element.id,
            carrier_id=element.carrier_id,
            bounds=element.bounds,
            color=element.color,
            opacity=element.opacity,
            confidence=element.confidence,
            normal=element.normal,
            material_id=element.material_id,
            semantic_id=element.semantic_id,
            residual=element.residual,
            lod=_lod_for_carrier(element.carrier_id),
            chunk_id=_chunk_id_for_carrier(element.carrier_id),
            metadata=element.metadata,
            confidence_map=element.confidence_map,
            edit=element.edit,
            payload=element.payload,
        )
        for element in elements
    )
    return chunked_elements, _chunks_for_elements(chunked_elements)


def _chunks_for_elements(elements: Sequence[AuraElement]) -> tuple[AuraChunk, ...]:
    grouped: dict[str, list[AuraElement]] = {}
    for element in elements:
        grouped.setdefault(element.chunk_id, []).append(element)
    chunks = []
    for chunk_id, chunk_elements in grouped.items():
        chunks.append(
            AuraChunk(
                id=chunk_id,
                bounds=_union_bounds([element.bounds for element in chunk_elements]),
                element_ids=tuple(element.id for element in chunk_elements),
                lod=min(element.lod for element in chunk_elements),
            )
        )
    return tuple(chunks)


def _chunk_id_for_carrier(carrier_id: str) -> str:
    return {
        "surface": "base_surface_lod0",
        "volume": "base_volume_lod0",
        "semantic": "semantic_object_lod0",
        "beta": "detail_beta_lod1",
        "gabor": "detail_gabor_lod1",
        "neural": "residual_neural_lod1",
        "gaussian": "fallback_gaussian_lod2",
    }.get(carrier_id, f"{carrier_id}_lod2")


def _lod_for_carrier(carrier_id: str) -> int:
    return {
        "surface": 0,
        "volume": 0,
        "semantic": 0,
        "beta": 1,
        "gabor": 1,
        "neural": 1,
        "gaussian": 2,
    }.get(carrier_id, 2)


def _semantic_graph_for(samples: Sequence[EvidenceSample], elements: Sequence[AuraElement]) -> SemanticGraph:
    grouped: dict[str, list[str]] = {}
    confidences: dict[str, list[float]] = {}
    element_by_id = {element.id: element for element in elements}
    for sample in samples:
        label = sample.semantic_label
        element = element_by_id[sample.id]
        if element.carrier_id == "semantic" and label is None:
            label = sample.id
        if label is None:
            continue
        grouped.setdefault(label, []).append(sample.id)
        confidences.setdefault(label, []).append(sample.evidence.semantic_confidence or sample.confidence)
    nodes = tuple(
        SemanticNode(
            id=f"object:{label}",
            label=label,
            element_ids=tuple(element_ids),
            confidence=sum(confidences[label]) / len(confidences[label]),
            attributes={"source": "evidence"},
        )
        for label, element_ids in grouped.items()
    )
    return SemanticGraph(nodes=nodes)
