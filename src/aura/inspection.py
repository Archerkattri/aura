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
    hit_point: Vec3 | None
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
    shadow_direction: Vec3 | None
    shadow_transmittance: float | None
    shadow_occluded: bool | None
    reflection_ready: bool
    reflection_direction: Vec3 | None
    reflection_hit: bool | None
    collision_proxy_ready: bool
    collision_distance: float | None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "origin": list(self.origin),
            "direction": list(self.direction),
            "firstHit": self.first_hit,
            "hitPoint": list(self.hit_point) if self.hit_point is not None else None,
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
            "shadowDirection": list(self.shadow_direction) if self.shadow_direction is not None else None,
            "shadowTransmittance": self.shadow_transmittance,
            "shadowOccluded": self.shadow_occluded,
            "reflectionReady": self.reflection_ready,
            "reflectionDirection": list(self.reflection_direction) if self.reflection_direction is not None else None,
            "reflectionHit": self.reflection_hit,
            "collisionProxyReady": self.collision_proxy_ready,
            "collisionDistance": self.collision_distance,
        }


def inspect_ray(scene: AuraScene, ray: Ray, *, label: str = "ray") -> RayInspection:
    result = scene.ray_query(ray)
    first_hit = result.provenance != "miss"
    opacity = result.opacity
    hit_point = _ray_point(ray, result.depth) if first_hit and result.depth is not None else None
    shadow_direction = _shadow_direction(ray, result.normal) if first_hit else None
    shadow_transmittance = None
    shadow_occluded = None
    if hit_point is not None and shadow_direction is not None:
        shadow_result = scene.ray_query(Ray(origin=_offset_point(hit_point, shadow_direction), direction=shadow_direction))
        shadow_transmittance = shadow_result.transmittance
        shadow_occluded = shadow_result.provenance != "miss" and shadow_result.opacity > 0.0
    reflection_direction = _reflection_direction(ray.direction, result.normal) if first_hit and result.normal is not None else None
    reflection_hit = None
    if hit_point is not None and reflection_direction is not None:
        reflection_result = scene.ray_query(Ray(origin=_offset_point(hit_point, reflection_direction), direction=reflection_direction))
        reflection_hit = reflection_result.provenance != "miss"
    return RayInspection(
        label=label,
        origin=ray.origin,
        direction=ray.direction,
        first_hit=first_hit,
        hit_point=hit_point,
        depth=result.depth,
        opacity=opacity,
        transmittance=result.transmittance,
        semantic_id=result.semantic_id,
        material_id=result.material_id,
        confidence=result.confidence,
        residual=result.residual,
        provenance=result.provenance,
        occluded=first_hit and opacity > 0.0,
        shadow_ready=shadow_transmittance is not None,
        shadow_direction=shadow_direction,
        shadow_transmittance=shadow_transmittance,
        shadow_occluded=shadow_occluded,
        reflection_ready=reflection_direction is not None,
        reflection_direction=reflection_direction,
        reflection_hit=reflection_hit,
        collision_proxy_ready=first_hit and (result.normal is not None or result.semantic_id is not None),
        collision_distance=result.depth if first_hit else None,
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


def _ray_point(ray: Ray, depth: float) -> Vec3:
    return _clean_vec3(tuple(ray.origin[index] + ray.direction[index] * depth for index in range(3)))  # type: ignore[arg-type]


def _offset_point(point: Vec3, direction: Vec3, *, epsilon: float = 1e-4) -> Vec3:
    return _clean_vec3(tuple(point[index] + direction[index] * epsilon for index in range(3)))  # type: ignore[arg-type]


def _shadow_direction(ray: Ray, normal: Vec3 | None) -> Vec3:
    if normal is not None:
        return _clean_vec3(normal)
    return _clean_vec3(tuple(-axis for axis in ray.direction))  # type: ignore[arg-type]


def _reflection_direction(direction: Vec3, normal: Vec3) -> Vec3:
    dot = sum(axis * normal_axis for axis, normal_axis in zip(direction, normal))
    return _clean_vec3(tuple(direction[index] - 2.0 * dot * normal[index] for index in range(3)))  # type: ignore[arg-type]


def _clean_vec3(value: tuple[float, float, float]) -> Vec3:
    return tuple(0.0 if abs(axis) < 1e-12 else float(axis) for axis in value)  # type: ignore[return-value]
