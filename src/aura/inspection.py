from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from aura.ray import Ray, Vec3
from aura.scene import AuraScene


@dataclass(frozen=True)
class RayInspection:
    label: str
    origin: Vec3
    direction: Vec3
    first_hit: bool
    depth: float | None
    opacity: float
    transmittance: float
    semantic_id: str | None
    material_id: str | None
    confidence: float
    residual: bool
    provenance: str | None
    occluded: bool
    shadow_ready: bool
    reflection_ready: bool
    collision_proxy_ready: bool

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "origin": list(self.origin),
            "direction": list(self.direction),
            "firstHit": self.first_hit,
            "depth": self.depth,
            "opacity": self.opacity,
            "transmittance": self.transmittance,
            "semanticId": self.semantic_id,
            "materialId": self.material_id,
            "confidence": self.confidence,
            "residual": self.residual,
            "provenance": self.provenance,
            "occluded": self.occluded,
            "shadowReady": self.shadow_ready,
            "reflectionReady": self.reflection_ready,
            "collisionProxyReady": self.collision_proxy_ready,
        }


def inspect_ray(scene: AuraScene, ray: Ray, *, label: str = "ray") -> RayInspection:
    result = scene.ray_query(ray)
    first_hit = result.provenance != "miss"
    opacity = result.opacity
    return RayInspection(
        label=label,
        origin=ray.origin,
        direction=ray.direction,
        first_hit=first_hit,
        depth=result.depth,
        opacity=opacity,
        transmittance=result.transmittance,
        semantic_id=result.semantic_id,
        material_id=result.material_id,
        confidence=result.confidence,
        residual=result.residual,
        provenance=result.provenance,
        occluded=first_hit and opacity > 0.0,
        shadow_ready=first_hit,
        reflection_ready=first_hit and result.normal is not None,
        collision_proxy_ready=first_hit and (result.normal is not None or result.semantic_id is not None),
    )


def inspect_scene_rays(scene: AuraScene, *, max_rays: int = 8) -> tuple[RayInspection, ...]:
    if max_rays <= 0:
        raise ValueError("max_rays must be positive")
    if not scene.elements:
        return tuple()
    camera_z = min(element.bounds.min_corner[2] for element in scene.elements) - 2.0
    inspections = []
    for element in scene.elements[:max_rays]:
        center = tuple((lo + hi) / 2.0 for lo, hi in zip(element.bounds.min_corner, element.bounds.max_corner))
        ray = Ray(origin=(center[0], center[1], camera_z), direction=(0.0, 0.0, 1.0))
        inspections.append(inspect_ray(scene, ray, label=element.id))
    return tuple(inspections)


def native_demo_interaction_probes(scene: AuraScene) -> tuple[RayInspection, ...]:
    probes = (
        ("inserted_object_occlusion", Ray(origin=(-0.5, -0.5, -2.0), direction=(0.0, 0.0, 1.0))),
        ("semantic_object_query", Ray(origin=(0.125, 0.275, -2.0), direction=(0.0, 0.0, 1.0))),
        ("reflection_ready_surface", Ray(origin=(-0.5, -0.5, -2.0), direction=(0.0, 0.0, 1.0))),
        ("empty_space_control", Ray(origin=(2.0, 2.0, -2.0), direction=(0.0, 0.0, 1.0))),
    )
    return tuple(inspect_ray(scene, ray, label=label) for label, ray in probes)
