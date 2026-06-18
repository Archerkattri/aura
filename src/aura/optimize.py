from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Sequence

from aura.elements import AuraElement
from aura.ray import Ray, Vec3
from aura.scene import AuraScene


@dataclass(frozen=True)
class RenderTarget:
    """One supervised training sample: a ray paired with ground-truth targets.

    Used as input to the torch and CUDA renderers for loss computation.
    """

    frame_id: str
    ray: Ray
    target_color: Vec3
    target_depth: float
    target_semantic_id: str | None = None
    target_material_id: str | None = None
    target_normal: Vec3 | None = None
    target_confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("render target frame_id is required")
        _validate_vec3(self.target_color, "render target color")
        if not isfinite(self.target_depth) or self.target_depth <= 0.0:
            raise ValueError("render target depth must be positive")
        if self.target_normal is not None:
            _validate_vec3(self.target_normal, "render target normal")
        if self.target_confidence is not None and not 0.0 <= self.target_confidence <= 1.0:
            raise ValueError("render target confidence must be in [0, 1]")


@dataclass(frozen=True)
class TrainingLossWeights:
    """Per-term scalar weights for the combined AURA training objective."""

    image: float = 1.0
    depth: float = 1.0
    query: float = 1.0
    normal: float = 1.0
    mask: float = 1.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} loss weight must be finite and non-negative")
        if self.image + self.depth + self.query + self.normal + self.mask + self.confidence <= 0.0:
            raise ValueError("at least one training loss weight must be positive")

    def total(
        self,
        *,
        image_loss: float,
        depth_loss: float,
        query_loss: float,
        normal_loss: float,
        mask_loss: float = 0.0,
        confidence_loss: float = 0.0,
    ) -> float:
        return (
            self.image * image_loss
            + self.depth * depth_loss
            + self.query * query_loss
            + self.normal * normal_loss
            + self.mask * mask_loss
            + self.confidence * confidence_loss
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DifferentiableRaySample:
    """Scalar record of predicted vs. target quantities for one rendered ray.

    Aggregated into :class:`DifferentiableRenderBatch` for batch-level
    loss inspection and gradient-descent updates.
    """

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
    target_normal: Vec3 | None
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
    normal_loss: float
    mask_loss: float
    confidence_loss: float
    total_loss: float
    loss_weights: dict[str, float]
    color_jacobian: float
    color_gradient: Vec3
    depth_gradient: float
    gradient_norm: float

    def to_dict(self) -> dict:
        return asdict(self)


def differentiate_scene_rays(
    scene: AuraScene,
    targets: Sequence[RenderTarget],
    *,
    loss_weights: TrainingLossWeights | None = None,
) -> tuple[DifferentiableRaySample, ...]:
    """Evaluate differentiable CPU reference samples for posed training rays.

    This is intentionally small and deterministic. It exposes gradients through
    the same scene ray-query contract that package/runtime code uses, so a later
    PyTorch or CUDA renderer can replace the implementation without changing the
    reconstruction API.
    """

    if not targets:
        raise ValueError("differentiable renderer requires at least one target")
    loss_weights = loss_weights or TrainingLossWeights()
    element_by_id = {element.id: element for element in scene.elements}
    return tuple(
        _differentiate_target(scene, element_by_id, target, loss_weights)
        for target in targets
    )


def _differentiate_target(
    scene: AuraScene,
    element_by_id: dict[str, AuraElement],
    target: RenderTarget,
    loss_weights: TrainingLossWeights,
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
    normal_loss = _normal_loss(result.normal, target.target_normal)
    mask_loss = 0.0
    confidence_loss = _confidence_loss(result.confidence, target.target_confidence)
    total_loss = loss_weights.total(
        image_loss=image_loss,
        depth_loss=depth_loss,
        query_loss=query_loss,
        normal_loss=normal_loss,
        mask_loss=mask_loss,
        confidence_loss=confidence_loss,
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
        target_normal=target.target_normal,
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
        normal_loss=normal_loss,
        mask_loss=mask_loss,
        confidence_loss=confidence_loss,
        total_loss=total_loss,
        loss_weights=loss_weights.to_dict(),
        color_jacobian=color_jacobian,
        color_gradient=color_gradient,  # type: ignore[arg-type]
        depth_gradient=depth_gradient,
        gradient_norm=gradient_norm,
    )


def gradient_descent_color_step(color: Vec3, gradient: Vec3, *, learning_rate: float) -> Vec3:
    """Apply one gradient-descent step to an RGB color and clamp the result to [0, 1]."""
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


def _normal_loss(predicted: Vec3 | None, target: Vec3 | None) -> float:
    if target is None:
        return 0.0
    if predicted is None:
        return 1.0
    predicted_norm = _normalize(predicted)
    target_norm = _normalize(target)
    if predicted_norm is None or target_norm is None:
        return 1.0
    cosine = sum(left * right for left, right in zip(predicted_norm, target_norm))
    return max(0.0, min(1.0, (1.0 - cosine) / 2.0))


def _confidence_loss(predicted: float, target: float | None) -> float:
    if target is None:
        return 0.0
    delta = predicted - target
    return delta * delta


def _normalize(vector: Vec3) -> Vec3 | None:
    norm = sqrt(sum(axis * axis for axis in vector))
    if norm <= 1e-12:
        return None
    return tuple(axis / norm for axis in vector)  # type: ignore[return-value]


def _validate_vec3(vector: Vec3, name: str) -> None:
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if any(not isfinite(axis) for axis in vector):
        raise ValueError(f"{name} values must be finite")


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
