"""GPU-ready AURA representation contract scaffold."""

from aura.asset import AuraAsset
from aura.assignment import RegionEvidence, choose_carrier
from aura.benchmark import AblationConfig, BenchmarkCase, BenchmarkSuite, default_benchmark_suite
from aura.carriers import CarrierKind, CarrierSpec, default_registry
from aura.carrier_payloads import (
    BetaKernelPayload,
    GaborFrequencyPayload,
    GaussianFallbackPayload,
    NeuralResidualPayload,
    SemanticFeaturePayload,
    SurfaceCellPayload,
    VolumeCellPayload,
)
from aura.decomposition import EvidenceSample, decompose_evidence
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.ingest import (
    BaselineExport,
    DepthEvidencePoint,
    GaussianSplatSample,
    IngestAdapterSpec,
    SemanticMaskRegion,
    SparsePointPrior,
    depth_points_to_evidence,
    discover_3dgs_export,
    load_3dgs_export,
    load_3dgs_ply,
    load_3dgs_scene,
    package_3dgs_export,
    semantic_masks_to_evidence,
    splats_to_scene,
    sparse_points_to_evidence,
    supported_ingest_adapters,
)
from aura.package import AuraPackage, load_package, package_scene, validate_package, validate_package_documents
from aura.ray import Ray, RayQueryResult
from aura.render import RenderImage, compare_images, image_mse, image_psnr, read_ppm, render_orthographic
from aura.schema import AURA_FORMAT, AURA_SCHEMA_VERSION, AURA_SUPPORTED_MAJOR_VERSIONS
from aura.scene import AuraScene
from aura.semantic import SemanticEdge, SemanticGraph, SemanticNode

__all__ = [
    "AuraChunk",
    "AuraElement",
    "AuraPackage",
    "AuraAsset",
    "AuraScene",
    "AblationConfig",
    "AURA_FORMAT",
    "AURA_SCHEMA_VERSION",
    "AURA_SUPPORTED_MAJOR_VERSIONS",
    "BaselineExport",
    "BenchmarkCase",
    "BenchmarkSuite",
    "BetaKernelPayload",
    "Bounds",
    "CarrierKind",
    "CarrierSpec",
    "DepthEvidencePoint",
    "EvidenceSample",
    "GaborFrequencyPayload",
    "GaussianFallbackPayload",
    "GaussianSplatSample",
    "IngestAdapterSpec",
    "NeuralResidualPayload",
    "Ray",
    "RayQueryResult",
    "RegionEvidence",
    "RenderImage",
    "SemanticFeaturePayload",
    "SemanticEdge",
    "SemanticGraph",
    "SemanticNode",
    "SemanticMaskRegion",
    "SparsePointPrior",
    "SurfaceCellPayload",
    "VolumeCellPayload",
    "choose_carrier",
    "compare_images",
    "depth_points_to_evidence",
    "decompose_evidence",
    "default_benchmark_suite",
    "default_registry",
    "discover_3dgs_export",
    "image_mse",
    "image_psnr",
    "load_3dgs_export",
    "load_3dgs_ply",
    "load_3dgs_scene",
    "load_package",
    "package_scene",
    "package_3dgs_export",
    "read_ppm",
    "render_orthographic",
    "semantic_masks_to_evidence",
    "splats_to_scene",
    "sparse_points_to_evidence",
    "supported_ingest_adapters",
    "validate_package",
    "validate_package_documents",
]
