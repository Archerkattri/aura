"""CPU-only AURA representation contract scaffold."""

from aura.asset import AuraAsset
from aura.assignment import RegionEvidence, choose_carrier
from aura.carriers import CarrierKind, CarrierSpec, default_registry
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.package import AuraPackage, package_scene
from aura.ray import Ray, RayQueryResult
from aura.scene import AuraScene

__all__ = [
    "AuraChunk",
    "AuraElement",
    "AuraPackage",
    "AuraAsset",
    "AuraScene",
    "Bounds",
    "CarrierKind",
    "CarrierSpec",
    "Ray",
    "RayQueryResult",
    "RegionEvidence",
    "choose_carrier",
    "default_registry",
    "package_scene",
]
