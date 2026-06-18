from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite, sqrt
from typing import Any, Sequence

from aura.decomposition import carrier_lod_elements_and_chunks
from aura.elements import AuraElement
from aura.evolution import (
    CarrierEvolutionDecision,
    CarrierEvolutionPolicy,
    carrier_evolution_decisions,
    evolved_element_for,
    refined_confidence,
    simplification_metadata,
    updated_confidence_map,
)
from aura.optimize import TrainingLossWeights
from aura.scene import AuraScene
from aura.training_targets import CapturePackedRenderBatch
from aura.torch_renderer import (
    TorchCaptureTrainingBatch,
    TorchRenderBatch,
    require_torch,
    torch_renderer_status,
    torch_capture_training_batch_from_packed,
    torch_scene_tensors,
    torch_render_capture_training_batch,
    torch_render_capture_training_objective,
)


@dataclass(frozen=True)
class TorchOptimizationConfig:
    iterations: int = 1
    color_learning_rate: float = 0.25
    loss_weights: TrainingLossWeights = TrainingLossWeights()
    gradient_clip_norm: float | None = None
    max_samples_per_batch: int | None = None
    evolution_policy: CarrierEvolutionPolicy | None = None
    iteration_offset: int = 0
    checkpoint_interval: int | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("torch optimization iterations must be positive")
        if not 0.0 < self.color_learning_rate <= 1.0:
            raise ValueError("torch color_learning_rate must be in (0, 1]")
        if not isinstance(self.loss_weights, TrainingLossWeights):
            raise TypeError("torch loss_weights must be a TrainingLossWeights instance")
        if self.gradient_clip_norm is not None and (
            not isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("torch gradient_clip_norm must be positive when set")
        if self.max_samples_per_batch is not None and self.max_samples_per_batch <= 0:
            raise ValueError("torch max_samples_per_batch must be positive when set")
        if self.evolution_policy is not None and not isinstance(self.evolution_policy, CarrierEvolutionPolicy):
            raise TypeError("torch evolution_policy must be a CarrierEvolutionPolicy or None")
        if self.iteration_offset < 0:
            raise ValueError("torch iteration_offset must be non-negative")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("torch checkpoint_interval must be positive when set")


@dataclass(frozen=True)
class TorchOptimizationStep:
    iteration: int
    batch_index: int | None
    device: str
    sample_count: int
    target_offset: int | None
    image_loss: float
    depth_loss: float
    query_loss: float
    normal_loss: float
    mask_loss: float
    total_loss: float
    carrier_counts: dict[str, int]
    loss_weights: dict[str, float]
    optimizer: str
    gradient_norm: float
    applied_gradient_norm: float
    gradient_clip_norm: float | None
    updated_parameter_count: int
    max_samples_per_batch: int | None
    source_windows: tuple[dict[str, Any], ...] = ()
    carrier_evolution: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TorchOptimizationResult:
    scene: AuraScene
    steps: tuple[TorchOptimizationStep, ...]
    scene_checkpoints: tuple["TorchSceneCheckpoint", ...] = ()

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.name,
            "steps": [step.to_dict() for step in self.steps],
            "lossCurve": _loss_curve(self.steps),
            "checkpoints": [_checkpoint_from_step(index, step) for index, step in enumerate(self.steps)],
            "sceneCheckpoints": [checkpoint.to_dict() for checkpoint in self.scene_checkpoints],
            "finalLoss": self.steps[-1].total_loss if self.steps else None,
        }


@dataclass(frozen=True)
class TorchSceneCheckpoint:
    checkpoint_index: int
    iteration: int
    step_count: int
    scene: AuraScene

    def to_dict(self) -> dict:
        return {
            "checkpointIndex": self.checkpoint_index,
            "iteration": self.iteration,
            "stepCount": self.step_count,
            "scene": self.scene.name,
            "elementCount": len(self.scene.elements),
            "carrierCounts": _carrier_counts(self.scene.elements),
        }


@dataclass(frozen=True)
class _TorchEvolutionPrediction:
    element_id: str | None
    carrier_id: str | None
    image_loss: float
    depth_loss: float
    query_loss: float
    normal_loss: float
    target_color: tuple[float, float, float]


def torch_optimize_capture_batch(
    scene: AuraScene,
    batch: TorchCaptureTrainingBatch,
    config: TorchOptimizationConfig | None = None,
) -> TorchOptimizationResult:
    """Train native AURA carrier tensors against capture targets with torch autograd."""

    require_torch()
    if not scene.elements:
        raise ValueError("torch optimization requires at least one scene element")
    config = config or TorchOptimizationConfig()
    sample_count = _batch_sample_count(batch)
    if config.max_samples_per_batch is not None and sample_count > config.max_samples_per_batch:
        raise ValueError(
            f"torch optimization batch has {sample_count} samples, exceeding max_samples_per_batch "
            f"{config.max_samples_per_batch}"
        )
    optimized_scene, steps, scene_checkpoints = _optimize_torch_batches(
        scene,
        ((batch, None, None, ()),),
        config=config,
        device=str(batch.ray_origins.device),
    )
    return TorchOptimizationResult(scene=optimized_scene, steps=steps, scene_checkpoints=scene_checkpoints)


def torch_optimize_capture_batches(
    scene: AuraScene,
    batches: Sequence[CapturePackedRenderBatch],
    config: TorchOptimizationConfig | None = None,
    *,
    device: str | None = None,
) -> TorchOptimizationResult:
    """Stream packed capture batches through one native torch optimizer state.

    Packed batches are the tiled capture contract used by the future CUDA data
    path. This function keeps carrier tensors resident while each bounded
    source window is converted to torch and optimized in deterministic order.
    """

    require_torch()
    if not scene.elements:
        raise ValueError("torch optimization requires at least one scene element")
    if not batches:
        raise ValueError("torch optimization requires at least one packed capture batch")
    config = config or TorchOptimizationConfig()
    prepared_batches = []
    for packed in batches:
        if packed.target_count <= 0:
            continue
        if config.max_samples_per_batch is not None and packed.target_count > config.max_samples_per_batch:
            raise ValueError(
                f"packed torch optimization batch has {packed.target_count} samples, exceeding "
                f"max_samples_per_batch {config.max_samples_per_batch}"
            )
        prepared_batches.append(packed)
    if not prepared_batches:
        raise ValueError("torch optimization requires at least one non-empty packed capture batch")
    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"
    optimized_scene, steps, scene_checkpoints = _optimize_torch_batches(
        scene,
        tuple(prepared_batches),
        config=config,
        device=str(resolved_device),
    )
    return TorchOptimizationResult(scene=optimized_scene, steps=tuple(steps), scene_checkpoints=scene_checkpoints)


def _optimization_step_from_rendered(
    iteration: int,
    rendered: TorchRenderBatch,
    elements: Sequence[AuraElement],
    *,
    batch_index: int | None,
    target_offset: int | None,
    source_windows: tuple[dict[str, Any], ...],
    loss_weights: TrainingLossWeights,
    mask_loss: float,
    update: "_TorchGradientStepState",
    max_samples_per_batch: int | None,
) -> TorchOptimizationStep:
    image_loss = _mean(rendered.image_loss)
    depth_loss = _mean(rendered.depth_loss)
    query_loss = _mean(rendered.query_loss)
    normal_loss = _mean(rendered.normal_loss)
    return TorchOptimizationStep(
        iteration=iteration,
        batch_index=batch_index,
        device=rendered.device,
        sample_count=len(rendered.frame_ids),
        target_offset=target_offset,
        image_loss=image_loss,
        depth_loss=depth_loss,
        query_loss=query_loss,
        normal_loss=normal_loss,
        mask_loss=mask_loss,
        total_loss=loss_weights.total(
            image_loss=image_loss,
            depth_loss=depth_loss,
            query_loss=query_loss,
            normal_loss=normal_loss,
            mask_loss=mask_loss,
        ),
        carrier_counts=_carrier_counts(elements),
        loss_weights=loss_weights.to_dict(),
        optimizer="sgd",
        gradient_norm=update.gradient_norm,
        applied_gradient_norm=update.applied_gradient_norm,
        gradient_clip_norm=update.gradient_clip_norm,
        updated_parameter_count=update.updated_parameter_count,
        max_samples_per_batch=max_samples_per_batch,
        source_windows=source_windows,
    )


def _optimize_torch_batches(
    scene: AuraScene,
    batches: Sequence[CapturePackedRenderBatch | tuple[TorchCaptureTrainingBatch, int | None, int | None, tuple[dict[str, Any], ...]]],
    *,
    config: TorchOptimizationConfig,
    device: str,
) -> tuple[AuraScene, tuple[TorchOptimizationStep, ...], tuple[TorchSceneCheckpoint, ...]]:
    torch = require_torch()
    current_scene = scene
    steps: list[TorchOptimizationStep] = []
    scene_checkpoints: list[TorchSceneCheckpoint] = []
    rendered: TorchRenderBatch | None = None
    scene_tensors = torch_scene_tensors(current_scene, device=device)
    carrier_parameters = scene_tensors.carrier_parameters
    for iteration in range(config.iterations):
        absolute_iteration = config.iteration_offset + iteration
        iteration_rendered: list[TorchRenderBatch] = []
        for batch_item in batches:
            batch, batch_index, target_offset, source_windows = _prepared_optimization_batch(batch_item, device=device)
            sample_count = _batch_sample_count(batch)
            if config.max_samples_per_batch is not None and sample_count > config.max_samples_per_batch:
                raise ValueError(
                    f"torch optimization batch has {sample_count} samples, exceeding max_samples_per_batch "
                    f"{config.max_samples_per_batch}"
                )
            _zero_carrier_parameter_grads(carrier_parameters)
            objective = torch_render_capture_training_objective(
                current_scene,
                batch,
                carrier_parameters=carrier_parameters,
                scene_tensors=scene_tensors,
            )
            weighted_loss = _weighted_torch_loss(objective, config.loss_weights)
            weighted_loss.backward()
            update = _gradient_step_carrier_parameters(
                torch,
                carrier_parameters,
                learning_rate=config.color_learning_rate,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            _zero_carrier_parameter_grads(carrier_parameters)
            with torch.no_grad():
                checkpoint_objective = torch_render_capture_training_objective(
                    current_scene,
                    batch,
                    carrier_parameters=carrier_parameters,
                    scene_tensors=scene_tensors,
                )
                rendered = torch_render_capture_training_batch(
                    current_scene,
                    batch,
                    carrier_parameters=carrier_parameters,
                    scene_tensors=scene_tensors,
                )
            iteration_rendered.append(rendered)
            steps.append(
                _optimization_step_from_rendered(
                    absolute_iteration,
                    rendered,
                    current_scene.elements,
                    batch_index=batch_index,
                    target_offset=target_offset,
                    source_windows=source_windows,
                    loss_weights=config.loss_weights,
                    mask_loss=_tensor_scalar(checkpoint_objective.mask_loss),
                    update=update,
                    max_samples_per_batch=config.max_samples_per_batch,
                )
            )
        current_scene = _scene_from_carrier_parameters(current_scene, carrier_parameters, rendered)
        if config.evolution_policy is not None and config.evolution_policy.enabled and iteration_rendered:
            predictions = tuple(
                prediction
                for batch_rendered in iteration_rendered
                for prediction in _evolution_predictions_from_rendered(batch_rendered)
            )
            decisions = carrier_evolution_decisions(
                predictions,
                current_scene.elements,
                policy=config.evolution_policy,
                iteration=absolute_iteration,
            )
            if decisions:
                current_scene = _evolve_scene(current_scene, predictions, decisions, learning_rate=config.color_learning_rate)
                if steps:
                    steps[-1] = replace(
                        steps[-1],
                        carrier_evolution=tuple(decision.to_dict() for decision in decisions),
                        carrier_counts=_carrier_counts(current_scene.elements),
                    )
                scene_tensors = torch_scene_tensors(current_scene, device=device)
                carrier_parameters = scene_tensors.carrier_parameters
                rendered = None
        checkpoint_due = (
            config.checkpoint_interval is not None
            and (iteration + 1) % config.checkpoint_interval == 0
        )
        if checkpoint_due:
            current_scene = _scene_from_carrier_parameters(current_scene, carrier_parameters, rendered)
            scene_checkpoints.append(
                TorchSceneCheckpoint(
                    checkpoint_index=len(scene_checkpoints),
                    iteration=absolute_iteration,
                    step_count=len(steps),
                    scene=current_scene,
                )
            )
    if rendered is not None:
        current_scene = _scene_from_carrier_parameters(current_scene, carrier_parameters, rendered)
    return current_scene, tuple(steps), tuple(scene_checkpoints)


def _prepared_optimization_batch(
    batch_item: CapturePackedRenderBatch | tuple[TorchCaptureTrainingBatch, int | None, int | None, tuple[dict[str, Any], ...]],
    *,
    device: str,
) -> tuple[TorchCaptureTrainingBatch, int | None, int | None, tuple[dict[str, Any], ...]]:
    if isinstance(batch_item, CapturePackedRenderBatch):
        return (
            torch_capture_training_batch_from_packed(batch_item, device=device),
            batch_item.batch_index,
            batch_item.target_offset,
            tuple(window.to_dict() for window in batch_item.source_windows),
        )
    return batch_item


def _zero_carrier_parameter_grads(carrier_parameters: dict[str, dict[str, Any]]) -> None:
    for fields in carrier_parameters.values():
        for parameter in fields.values():
            if getattr(parameter, "grad", None) is not None:
                parameter.grad.zero_()


def _evolution_predictions_from_rendered(rendered: TorchRenderBatch) -> tuple[_TorchEvolutionPrediction, ...]:
    return tuple(
        _TorchEvolutionPrediction(
            element_id=element_id,
            carrier_id=carrier_id,
            image_loss=float(image_loss),
            depth_loss=float(depth_loss),
            query_loss=float(query_loss),
            normal_loss=float(normal_loss),
            target_color=target_color,
        )
        for element_id, carrier_id, image_loss, depth_loss, query_loss, normal_loss, target_color in zip(
            rendered.element_ids,
            rendered.carrier_ids,
            rendered.image_loss,
            rendered.depth_loss,
            rendered.query_loss,
            rendered.normal_loss,
            rendered.target_color,
        )
    )


def _evolve_scene(
    scene: AuraScene,
    predictions: Sequence[_TorchEvolutionPrediction],
    decisions: Sequence[CarrierEvolutionDecision],
    *,
    learning_rate: float,
) -> AuraScene:
    prediction_by_element = {prediction.element_id: prediction for prediction in predictions if prediction.element_id}
    decision_by_element = {decision.element_id: decision for decision in decisions}
    removed_evolved_ids = {
        decision.created_element_id
        for decision in decisions
        if decision.action in {"merge_beta_detail", "demote_neural_residual"} and decision.created_element_id is not None
    }
    elements: list[AuraElement] = []
    existing_ids = {element.id for element in scene.elements}
    for element in scene.elements:
        if element.id in removed_evolved_ids:
            continue
        prediction = prediction_by_element.get(element.id)
        decision = decision_by_element.get(element.id)
        if prediction is None:
            elements.append(element)
        else:
            elements.append(
                replace(
                    element,
                    confidence=refined_confidence(element.confidence, prediction, learning_rate=learning_rate),
                    confidence_map=updated_confidence_map(element, prediction),
                    metadata={
                        **element.metadata,
                        "optimized_by": "aura-train-torch",
                        "evolution_runtime": "training_loop",
                        **simplification_metadata(decision),
                    },
                )
            )
        if decision is not None and prediction is not None:
            evolved = evolved_element_for(element, decision, prediction)
            if evolved is not None and evolved.id not in existing_ids:
                elements.append(evolved)
                existing_ids.add(evolved.id)
    chunked_elements, chunks = carrier_lod_elements_and_chunks(tuple(elements))
    return AuraScene(name=scene.name, elements=chunked_elements, chunks=chunks, semantic_graph=scene.semantic_graph)


@dataclass(frozen=True)
class _TorchGradientStepState:
    gradient_norm: float
    applied_gradient_norm: float
    gradient_clip_norm: float | None
    updated_parameter_count: int


def _gradient_step_carrier_parameters(
    torch: Any,
    carrier_parameters: dict[str, dict[str, Any]],
    *,
    learning_rate: float,
    gradient_clip_norm: float | None,
) -> _TorchGradientStepState:
    parameters = []
    gradient_sq_sum = 0.0
    for fields in carrier_parameters.values():
        for parameter in fields.values():
            if getattr(parameter, "grad", None) is None:
                continue
            parameters.append(parameter)
            gradient_sq_sum += float(torch.sum(parameter.grad.detach() * parameter.grad.detach()).cpu().item())
    gradient_norm = sqrt(gradient_sq_sum)
    scale = 1.0
    if gradient_clip_norm is not None and gradient_norm > gradient_clip_norm and gradient_norm > 0.0:
        scale = gradient_clip_norm / gradient_norm
    with torch.no_grad():
        for fields in carrier_parameters.values():
            for name, parameter in fields.items():
                if getattr(parameter, "grad", None) is None:
                    continue
                parameter.sub_(learning_rate * scale * parameter.grad)
                if name in {"color", "opacity", "confidence", "density", "bandwidth", "residual_scale"}:
                    parameter.clamp_(0.0, 1.0)
                elif name in {"alpha", "beta"}:
                    parameter.clamp_(1e-4)
    return _TorchGradientStepState(
        gradient_norm=gradient_norm,
        applied_gradient_norm=gradient_norm * scale,
        gradient_clip_norm=gradient_clip_norm,
        updated_parameter_count=len(parameters),
    )


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
                "optimized_by": "aura-core-torch-autograd",
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


def _weighted_torch_loss(objective: Any, loss_weights: TrainingLossWeights) -> Any:
    return (
        loss_weights.image * objective.image_loss
        + loss_weights.depth * objective.depth_loss
        + loss_weights.normal * objective.normal_loss
        + loss_weights.mask * objective.mask_loss
    )


def _loss_curve(steps: Sequence[TorchOptimizationStep]) -> list[dict[str, float | int | None]]:
    return [
        {
            "iteration": step.iteration,
            "batchIndex": step.batch_index,
            "targetOffset": step.target_offset,
            "imageLoss": step.image_loss,
            "depthLoss": step.depth_loss,
            "queryLoss": step.query_loss,
            "normalLoss": step.normal_loss,
            "maskLoss": step.mask_loss,
            "totalLoss": step.total_loss,
        }
        for step in steps
    ]


def _checkpoint_from_step(index: int, step: TorchOptimizationStep) -> dict[str, Any]:
    return {
        "checkpointIndex": index,
        "iteration": step.iteration,
        "batchIndex": step.batch_index,
        "targetOffset": step.target_offset,
        "device": step.device,
        "sampleCount": step.sample_count,
        "loss": {
            "image": step.image_loss,
            "depth": step.depth_loss,
            "query": step.query_loss,
            "normal": step.normal_loss,
            "mask": step.mask_loss,
            "total": step.total_loss,
        },
        "gradientNorm": step.gradient_norm,
        "appliedGradientNorm": step.applied_gradient_norm,
        "updatedParameterCount": step.updated_parameter_count,
        "carrierCounts": dict(step.carrier_counts),
        "carrierEvolution": [dict(decision) for decision in step.carrier_evolution],
    }


def _batch_sample_count(batch: TorchCaptureTrainingBatch) -> int:
    return int(batch.frame_indices.numel())


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(float(value) for value in values) / len(values)


def _tensor_scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _tensor_vec3(value: Any) -> tuple[float, float, float]:
    items = value.detach().cpu().tolist()
    return (float(items[0]), float(items[1]), float(items[2]))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
