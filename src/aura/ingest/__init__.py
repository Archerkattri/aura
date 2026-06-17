"""Input adapters that turn external scene evidence into AURA elements."""

from aura.ingest.baselines import BaselineExport, discover_3dgs_export, package_3dgs_export
from aura.ingest.capture import (
    CaptureManifest,
    capture_manifest_template,
    load_capture_manifest,
    validate_capture_manifest_document,
    write_capture_manifest_template,
)
from aura.ingest.evidence import (
    DepthEvidencePoint,
    IngestAdapterSpec,
    SemanticMaskRegion,
    SparsePointPrior,
    depth_points_to_evidence,
    semantic_masks_to_evidence,
    sparse_points_to_evidence,
    supported_ingest_adapters,
)
from aura.ingest.splats import GaussianSplatSample, load_3dgs_export, load_3dgs_ply, load_3dgs_scene, splats_to_scene

__all__ = [
    "BaselineExport",
    "CaptureManifest",
    "DepthEvidencePoint",
    "GaussianSplatSample",
    "IngestAdapterSpec",
    "SemanticMaskRegion",
    "SparsePointPrior",
    "capture_manifest_template",
    "depth_points_to_evidence",
    "discover_3dgs_export",
    "load_3dgs_export",
    "load_3dgs_ply",
    "load_3dgs_scene",
    "load_capture_manifest",
    "package_3dgs_export",
    "semantic_masks_to_evidence",
    "splats_to_scene",
    "sparse_points_to_evidence",
    "supported_ingest_adapters",
    "validate_capture_manifest_document",
    "write_capture_manifest_template",
]
