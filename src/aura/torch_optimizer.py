from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Sequence

from aura.elements import AuraElement
from aura.scene import AuraScene
from aura.torch_renderer import TorchCaptureTrainingBatch, TorchRenderBatch, require_torch, torch_render_capture_training_batch


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
    `torch_render_capture_training_batch` for every forward pass and updates
    native carrier colors from batched residuals. The carrier kernels remain the
    reference torch semantics until they are replaced by CUDA/autograd kernels.
    """

    require_torch()
    if not scene.elements:
        raise ValueError("torch optimization requires at least one scene element")
    config = config or TorchOptimizationConfig()
    current = scene
    steps = []
    for iteration in range(config.iterations):
        rendered = torch_render_capture_training_batch(current, batch)
        step = _optimization_step_from_rendered(iteration, rendered, current.elements)
        steps.append(step)
        current = _refine_scene_from_torch_batch(current, rendered, learning_rate=config.color_learning_rate)
    return TorchOptimizationResult(scene=current, steps=tuple(steps))


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


def _refine_scene_from_torch_batch(scene: AuraScene, rendered: TorchRenderBatch, *, learning_rate: float) -> AuraScene:
    updates = _element_color_updates(rendered, learning_rate=learning_rate)
    if not updates:
        return scene
    elements = []
    loss_by_element = _loss_by_element(rendered)
    for element in scene.elements:
        delta = updates.get(element.id)
        color = element.color
        if delta is not None:
            color = tuple(_clamp_unit(element.color[index] + delta[index]) for index in range(3))
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
                "optimized_by": "aura-core-torch-reference",
                "torch_device": rendered.device,
            }
        elements.append(replace(element, color=color, confidence_map=confidence_map, metadata=metadata))
    return AuraScene(name=scene.name, elements=tuple(elements), chunks=scene.chunks, semantic_graph=scene.semantic_graph)


def _element_color_updates(rendered: TorchRenderBatch, *, learning_rate: float) -> dict[str, tuple[float, float, float]]:
    totals: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for element_id, predicted, target, transmittance in zip(
        rendered.element_ids,
        rendered.predicted_color,
        rendered.target_color,
        rendered.transmittance,
    ):
        if element_id is None:
            continue
        observed_alpha = max(1.0 - transmittance, 0.05)
        correction = [(target[index] - predicted[index]) / observed_alpha for index in range(3)]
        totals.setdefault(element_id, [0.0, 0.0, 0.0])
        counts[element_id] = counts.get(element_id, 0) + 1
        for index in range(3):
            totals[element_id][index] += correction[index]
    return {
        element_id: tuple(channel * learning_rate / counts[element_id] for channel in correction)
        for element_id, correction in totals.items()
    }


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


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
