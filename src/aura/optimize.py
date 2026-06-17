from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Sequence

from aura.elements import AuraElement
from aura.ray import Ray, Vec3
from aura.scene import AuraScene


@dataclass(frozen=True)
class RenderTarget:
    frame_id: str
    ray: Ray
    target_color: Vec3
    target_depth: float
    target_semantic_id: str | None = None
    target_material_id: str | None = None

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("render target frame_id is required")
        if self.target_depth <= 0.0:
            raise ValueError("render target depth must be positive")


@dataclass(frozen=True)
class DifferentiableRaySample:
    frame_id: str
    element_id: str | None
    carrier_id: str | None
    ray_direction: Vec3
    predicted_color: Vec3
    target_color: Vec3
    predicted_depth: float | None
    target_depth: float
    target_semantic_id: str | None
    target_material_id: str | None
    predicted_transmittance: float
    predicted_opacity: float
    predicted_confidence: float
    predicted_normal: Vec3 | None
    predicted_material_id: str | None
    predicted_semantic_id: str | None
    predicted_residual: bool
    predicted_provenance: str | None
    image_loss: float
    depth_loss: float
    query_loss: float
    color_jacobian: float
    color_gradient: Vec3
    depth_gradient: float
    gradient_norm: float

    def to_dict(self) -> dict:
        return asdict(self)


def differentiate_scene_rays(scene: AuraScene, targets: Sequence[RenderTarget]) -> tuple[DifferentiableRaySample, ...]:
    """Evaluate differentiable CPU reference samples for posed training rays.

    This is intentionally small and deterministic. It exposes gradients through
    the same scene ray-query contract that package/runtime code uses, so a later
    PyTorch or CUDA renderer can replace the implementation without changing the
    reconstruction API.
    """

    if not targets:
        raise ValueError("differentiable renderer requires at least one target")
    element_by_id = {element.id: element for element in scene.elements}
    return tuple(_differentiate_target(scene, element_by_id, target) for target in targets)


def _differentiate_target(
    scene: AuraScene,
    element_by_id: dict[str, AuraElement],
    target: RenderTarget,
) -> DifferentiableRaySample:
    result = scene.ray_query(target.ray)
    element_id = result.provenance.split(",", 1)[0] if result.provenance and result.provenance != "miss" else None
    element = element_by_id.get(element_id or "")
    image_loss = _color_mse(result.color, target.target_color)
    depth_loss = abs((result.depth or 0.0) - target.target_depth) if result.depth is not None else target.target_depth
    query_loss = _query_contract_loss(
        predicted_semantic_id=result.semantic_id,
        target_semantic_id=target.target_semantic_id,
        predicted_material_id=result.material_id,
        target_material_id=target.target_material_id,
    )
    color_jacobian = _color_jacobian(element, result.color)
    color_gradient = tuple(
        (2.0 / 3.0) * color_jacobian * (predicted - expected)
        for predicted, expected in zip(result.color, target.target_color)
    )
    depth_gradient = 0.0
    if result.depth is not None:
        depth_gradient = 1.0 if result.depth > target.target_depth else -1.0 if result.depth < target.target_depth else 0.0
    gradient_norm = sqrt(sum(channel * channel for channel in color_gradient) + depth_gradient * depth_gradient)
    return DifferentiableRaySample(
        frame_id=target.frame_id,
        element_id=element.id if element is not None else None,
        carrier_id=element.carrier_id if element is not None else None,
        ray_direction=target.ray.direction,
        predicted_color=result.color,
        target_color=target.target_color,
        predicted_depth=result.depth,
        target_depth=target.target_depth,
        target_semantic_id=target.target_semantic_id,
        target_material_id=target.target_material_id,
        predicted_transmittance=result.transmittance,
        predicted_opacity=result.opacity,
        predicted_confidence=result.confidence,
        predicted_normal=result.normal,
        predicted_material_id=result.material_id,
        predicted_semantic_id=result.semantic_id,
        predicted_residual=result.residual,
        predicted_provenance=result.provenance,
        image_loss=image_loss,
        depth_loss=depth_loss,
        query_loss=query_loss,
        color_jacobian=color_jacobian,
        color_gradient=color_gradient,  # type: ignore[arg-type]
        depth_gradient=depth_gradient,
        gradient_norm=gradient_norm,
    )


def gradient_descent_color_step(color: Vec3, gradient: Vec3, *, learning_rate: float) -> Vec3:
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    return tuple(_clamp_unit(channel - learning_rate * delta) for channel, delta in zip(color, gradient))  # type: ignore[return-value]


def precondition_color_gradient(gradient: Vec3, *, color_jacobian: float) -> Vec3:
    """Scale color gradients by the inverse local render Jacobian.

    The CPU reference renderer observes attenuated carrier colors. This
    preconditioner maps the MSE gradient back toward raw carrier color space so
    fixture optimization converges in a small, deterministic number of steps.
    """

    jacobian = max(abs(color_jacobian), 0.05)
    scale = 1.5 / (jacobian * jacobian)
    return tuple(channel * scale for channel in gradient)  # type: ignore[return-value]


def _color_jacobian(element: AuraElement | None, predicted_color: Vec3) -> float:
    if element is None:
        return 0.0
    ratios = [
        predicted / raw
        for raw, predicted in zip(element.color, predicted_color)
        if abs(raw) > 1e-6
    ]
    if not ratios:
        return 1.0 - element.opacity
    return max(0.05, min(1.0, sum(ratios) / len(ratios)))


def _color_mse(left: Vec3, right: Vec3) -> float:
    return sum((left_channel - right_channel) ** 2 for left_channel, right_channel in zip(left, right)) / 3.0


def _query_contract_loss(
    *,
    predicted_semantic_id: str | None,
    target_semantic_id: str | None,
    predicted_material_id: str | None,
    target_material_id: str | None,
) -> float:
    losses = []
    if target_semantic_id is not None:
        losses.append(0.0 if predicted_semantic_id == target_semantic_id else 1.0)
    if target_material_id is not None:
        losses.append(0.0 if predicted_material_id == target_material_id else 1.0)
    return 0.0 if not losses else sum(losses) / len(losses)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
