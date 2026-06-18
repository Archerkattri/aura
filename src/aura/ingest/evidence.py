"""Ingest adapter registry and depth-evidence helpers for bridging raw observations to AURA evidence samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from aura.assignment import RegionEvidence
from aura.decomposition import EvidenceSample
from aura.elements import Bounds
from aura.ray import Vec3


@dataclass(frozen=True)
class IngestAdapterSpec:
    """Descriptor for a supported ingest adapter and its current implementation status."""

    id: str
    status: str
    output: str = "EvidenceSample"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DepthEvidencePoint:
    """A single observed depth point convertible to an AURA surface evidence sample."""

    id: str
    position: Vec3
    normal: Vec3
    confidence: float
    radius: float = 0.02

    def to_evidence_sample(self) -> EvidenceSample:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        bounds = _bounds_around(self.position, self.radius)
        return EvidenceSample(
            id=self.id,
            bounds=bounds,
            evidence=RegionEvidence(geometry_confidence=self.confidence, material_confidence=0.5, edit_need=0.6),
            normal=self.normal,
            confidence=self.confidence,
            confidence_map={"depth": self.confidence},
            metadata={"source": "depth-prior"},
        )


@dataclass(frozen=True)
class SemanticMaskRegion:
    """A bounded semantic segmentation region convertible to an AURA semantic evidence sample."""

    id: str
    label: str
    bounds: Bounds
    confidence: float

    def to_evidence_sample(self) -> EvidenceSample:
        return EvidenceSample(
            id=self.id,
            bounds=self.bounds,
            evidence=RegionEvidence(semantic_confidence=self.confidence),
            semantic_label=self.label,
            confidence=self.confidence,
            confidence_map={"semantic": self.confidence},
            metadata={"source": "semantic-mask"},
        )


@dataclass(frozen=True)
class SparsePointPrior:
    """A COLMAP or SLAM sparse point convertible to an AURA geometry evidence sample."""

    id: str
    position: Vec3
    confidence: float
    radius: float = 0.03

    def to_evidence_sample(self) -> EvidenceSample:
        bounds = _bounds_around(self.position, self.radius)
        return EvidenceSample(
            id=self.id,
            bounds=bounds,
            evidence=RegionEvidence(geometry_confidence=self.confidence, compact_detail=0.8),
            confidence=self.confidence,
            confidence_map={"sparse_point": self.confidence},
            metadata={"source": "colmap-sparse-prior"},
        )


def supported_ingest_adapters() -> tuple[IngestAdapterSpec, ...]:
    """Return descriptors for all supported and planned AURA ingest adapters."""
    return (
        IngestAdapterSpec(id="3dgs", status="implemented", notes="PLY/JSON splats are converted to EvidenceSample before AURA decomposition."),
        IngestAdapterSpec(id="depth-prior", status="contract", notes="Depth points become surface evidence samples."),
        IngestAdapterSpec(id="semantic-mask", status="contract", notes="Mask regions become semantic/object evidence samples."),
        IngestAdapterSpec(
            id="colmap-sparse",
            status="implemented",
            notes="COLMAP binary/text sparse models become AURA_CAPTURE_MANIFEST frames plus sparse prior regions.",
        ),
        IngestAdapterSpec(id="pixelsplat", status="future", notes="Future feed-forward splats must enter as evidence samples."),
        IngestAdapterSpec(id="idesplat", status="future", notes="Future depth/semantic splat variants must enter as evidence samples."),
    )


def depth_points_to_evidence(points: Sequence[DepthEvidencePoint]) -> tuple[EvidenceSample, ...]:
    """Convert a sequence of depth evidence points to AURA evidence samples."""
    return tuple(point.to_evidence_sample() for point in points)


def semantic_masks_to_evidence(regions: Sequence[SemanticMaskRegion]) -> tuple[EvidenceSample, ...]:
    """Convert a sequence of semantic mask regions to AURA evidence samples."""
    return tuple(region.to_evidence_sample() for region in regions)


def sparse_points_to_evidence(points: Sequence[SparsePointPrior]) -> tuple[EvidenceSample, ...]:
    """Convert a sequence of sparse point priors to AURA evidence samples."""
    return tuple(point.to_evidence_sample() for point in points)


def _bounds_around(position: Vec3, radius: float) -> Bounds:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return Bounds(
        min_corner=(position[0] - radius, position[1] - radius, position[2] - radius),
        max_corner=(position[0] + radius, position[1] + radius, position[2] + radius),
    )
