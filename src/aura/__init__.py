"""GPU-ready AURA representation contract scaffold."""

from aura.asset import AuraAsset
from aura.assignment import RegionEvidence, choose_carrier
from aura.baselines import BaselineExport, discover_3dgs_export, package_3dgs_export
from aura.carriers import CarrierKind, CarrierSpec, default_registry
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.package import AuraPackage, load_package, package_scene, validate_package, validate_package_documents
from aura.ray import Ray, RayQueryResult
from aura.render import RenderImage, compare_images, image_mse, image_psnr, read_ppm, render_orthographic
from aura.schema import AURA_FORMAT, AURA_SCHEMA_VERSION, AURA_SUPPORTED_MAJOR_VERSIONS
from aura.scene import AuraScene
from aura.splats import GaussianSplatSample, load_3dgs_export, load_3dgs_ply, load_3dgs_scene, splats_to_scene

__all__ = [
    "AuraChunk",
    "AuraElement",
    "AuraPackage",
    "AuraAsset",
    "AuraScene",
    "AURA_FORMAT",
    "AURA_SCHEMA_VERSION",
    "AURA_SUPPORTED_MAJOR_VERSIONS",
    "BaselineExport",
    "Bounds",
    "CarrierKind",
    "CarrierSpec",
    "GaussianSplatSample",
    "Ray",
    "RayQueryResult",
    "RegionEvidence",
    "RenderImage",
    "choose_carrier",
    "compare_images",
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
    "splats_to_scene",
    "validate_package",
    "validate_package_documents",
]
