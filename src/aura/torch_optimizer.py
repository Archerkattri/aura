from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from aura.elements import AuraElement
from aura.scene import AuraScene
from aura.torch_kernels import torch_carrier_parameter_tensors
from aura.torch_renderer import (
    TorchCaptureTrainingBatch,
    TorchRenderBatch,
    require_torch,
    torch_render_capture_training_batch,
    torch_render_capture_training_objective,
)


@dataclass(frozen=True)
class TorchOptimizationConfig:
    iterations: int = 1
    color_learning_rate: float = 0.25

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("torch optimization iterations must be positive")
        if not 0.0 < self.color_learning_rate <= 1.0:
            raise ValueError("torch color_learning_rate must be in (0, 1]")


@dataclass(frozen=True)
class TorchOptimizationStep:
    iteration: int
    device: str
    sample_count: int
    image_loss: float
    depth_loss: float
    query_loss: float
    normal_loss: float
    total_loss: float
    carrier_counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TorchOptimizationResult:
    scene: AuraScene
    steps: tuple[TorchOptimizationStep, ...]

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.name,
            "steps": [step.to_dict() for step in self.steps],
            "finalLoss": self.steps[-1].total_loss if self.steps else None,
        }


def torch_optimize_capture_batch(
    scene: AuraScene,
    batch: TorchCaptureTrainingBatch,
    config: TorchOptimizationConfig | None = None,
) -> TorchOptimizationResult:
    """Run the torch AURA forward contract in an iterative training loop.

    This is the first GPU-facing optimization scaffold: it uses
    `torch_render_capture_training_objective` for live losses and gradient
    updates over native carrier tensors. The carrier kernels remain the
    reference torch semantics until they are replaced by CUDA kernels.
    """

    require_torch()
    if not scene.elements:
        raise ValueError("torch optimization requires at least one scene element")
    config = config or TorchOptimizationConfig()
    torch = require_torch()
    carrier_parameters = torch_carrier_parameter_tensors(torch, tuple(scene.elements), device=str(batch.ray_origins.device))
    steps = []
    for iteration in range(config.iterations):
        _zero_carrier_parameter_grads(carrier_parameters)
        objective = torch_render_capture_training_objective(scene, batch, carrier_parameters=carrier_parameters)
        rendered = torch_render_capture_training_batch(scene, batch, carrier_parameters=carrier_parameters)
        step = _optimization_step_from_rendered(iteration, rendered, scene.elements)
        steps.append(step)
        objective.total_loss.backward()
        _gradient_step_carrier_parameters(torch, carrier_parameters, learning_rate=config.color_learning_rate)
    optimized_scene = _scene_from_carrier_parameters(scene, carrier_parameters, rendered if steps else None)
    return TorchOptimizationResult(scene=optimized_scene, steps=tuple(steps))


def _optimization_step_from_rendered(
    iteration: int,
    rendered: TorchRenderBatch,
    elements: Sequence[AuraElement],
) -> TorchOptimizationStep:
    return TorchOptimizationStep(
        iteration=iteration,
        device=rendered.device,
        sample_count=len(rendered.frame_ids),
        image_loss=_mean(rendered.image_loss),
        depth_loss=_mean(rendered.depth_loss),
        query_loss=_mean(rendered.query_loss),
        normal_loss=_mean(rendered.normal_loss),
        total_loss=_mean(
            tuple(
                image + depth + query + normal
                for image, depth, query, normal in zip(
                    rendered.image_loss,
                    rendered.depth_loss,
                    rendered.query_loss,
                    rendered.normal_loss,
                )
            )
        ),
        carrier_counts=_carrier_counts(elements),
    )


def _zero_carrier_parameter_grads(carrier_parameters: dict[str, dict[str, Any]]) -> None:
    for fields in carrier_parameters.values():
        for parameter in fields.values():
            if getattr(parameter, "grad", None) is not None:
                parameter.grad.zero_()


def _gradient_step_carrier_parameters(torch: Any, carrier_parameters: dict[str, dict[str, Any]], *, learning_rate: float) -> None:
    with torch.no_grad():
        for fields in carrier_parameters.values():
            for name, parameter in fields.items():
                if getattr(parameter, "grad", None) is None:
                    continue
                parameter.sub_(learning_rate * parameter.grad)
                if name in {"color", "opacity", "confidence", "density", "bandwidth", "residual_scale"}:
                    parameter.clamp_(0.0, 1.0)
                elif name in {"alpha", "beta"}:
                    parameter.clamp_(1e-4)


def _scene_from_carrier_parameters(
    scene: AuraScene,
    carrier_parameters: dict[str, dict[str, Any]],
    rendered: TorchRenderBatch | None,
) -> AuraScene:
    elements = []
    loss_by_element = _loss_by_element(rendered) if rendered is not None else {}
    for element in scene.elements:
        fields = carrier_parameters.get(element.id, {})
        color = element.color
        opacity = element.opacity
        confidence = element.confidence
        payload = dict(element.payload)
        if "color" in fields:
            color = _tensor_vec3(fields["color"])
        if "opacity" in fields:
            opacity = _clamp_unit(_tensor_scalar(fields["opacity"]))
        if "confidence" in fields:
            confidence = _clamp_unit(_tensor_scalar(fields["confidence"]))
            if payload.get("type") == "semantic_feature":
                payload["confidence"] = confidence
        for name in ("density", "alpha", "beta", "phase", "bandwidth", "residual_scale"):
            if name in fields:
                payload[name] = _tensor_scalar(fields[name])
        if "frequency" in fields:
            payload["frequency"] = list(_tensor_vec3(fields["frequency"]))
        losses = loss_by_element.get(element.id)
        confidence_map = element.confidence_map
        metadata = element.metadata
        if losses is not None:
            confidence_map = {
                **element.confidence_map,
                "torch_image_loss": _clamp_unit(losses["image"]),
                "torch_depth_loss": _clamp_unit(losses["depth"]),
                "torch_query_loss": _clamp_unit(losses["query"]),
                "torch_normal_loss": _clamp_unit(losses["normal"]),
            }
            metadata = {
                **element.metadata,
                "optimized_by": "aura-core-torch-autograd-reference",
                "torch_device": rendered.device,
            }
        elements.append(
            replace(
                element,
                color=color,
                opacity=opacity,
                confidence=confidence,
                payload=payload,
                confidence_map=confidence_map,
                metadata=metadata,
            )
        )
    return AuraScene(name=scene.name, elements=tuple(elements), chunks=scene.chunks, semantic_graph=scene.semantic_graph)


def _loss_by_element(rendered: TorchRenderBatch) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for element_id, image, depth, query, normal in zip(
        rendered.element_ids,
        rendered.image_loss,
        rendered.depth_loss,
        rendered.query_loss,
        rendered.normal_loss,
    ):
        if element_id is None:
            continue
        totals.setdefault(element_id, {"image": 0.0, "depth": 0.0, "query": 0.0, "normal": 0.0})
        counts[element_id] = counts.get(element_id, 0) + 1
        totals[element_id]["image"] += image
        totals[element_id]["depth"] += depth
        totals[element_id]["query"] += query
        totals[element_id]["normal"] += normal
    return {
        element_id: {name: value / counts[element_id] for name, value in losses.items()}
        for element_id, losses in totals.items()
    }


def _carrier_counts(elements: Sequence[AuraElement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        counts[element.carrier_id] = counts.get(element.carrier_id, 0) + 1
    return counts


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(float(value) for value in values) / len(values)


def _tensor_scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _tensor_vec3(value: Any) -> tuple[float, float, float]:
    items = value.detach().cpu().tolist()
    return (float(items[0]), float(items[1]), float(items[2]))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
