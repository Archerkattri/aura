from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite, sqrt
from typing import Any, Sequence

from aura.decomposition import carrier_lod_elements_and_chunks
from aura.elements import AuraElement, Bounds
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
    TorchCaptureRenderSummary,
    TorchCaptureTrainingBatch,
    TorchRenderBatch,
    require_torch,
    torch_renderer_status,
    torch_capture_training_batch_from_packed,
    torch_scene_tensors,
    torch_render_capture_training_batch,
    torch_render_capture_training_summary,
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
    confidence_loss: float
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
    target_point: tuple[float, float, float] | None = None


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
    confidence_loss: float,
    query_loss: float | None,
    update: "_TorchGradientStepState",
    max_samples_per_batch: int | None,
) -> TorchOptimizationStep:
    image_loss = _mean(rendered.image_loss)
    depth_loss = _mean(rendered.depth_loss)
    query_loss = _mean(rendered.query_loss) if query_loss is None else query_loss
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
        confidence_loss=confidence_loss,
        total_loss=loss_weights.total(
            image_loss=image_loss,
            depth_loss=depth_loss,
            query_loss=query_loss,
            normal_loss=normal_loss,
            mask_loss=mask_loss,
            confidence_loss=confidence_loss,
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


def _optimization_step_from_objective(
    iteration: int,
    objective: Any,
    elements: Sequence[AuraElement],
    *,
    batch_index: int | None,
    target_offset: int | None,
    source_windows: tuple[dict[str, Any], ...],
    sample_count: int,
    loss_weights: TrainingLossWeights,
    update: "_TorchGradientStepState",
    max_samples_per_batch: int | None,
) -> TorchOptimizationStep:
    image_loss = _tensor_scalar(objective.image_loss)
    depth_loss = _tensor_scalar(objective.depth_loss)
    query_loss = _tensor_scalar(objective.query_loss)
    normal_loss = _tensor_scalar(objective.normal_loss)
    mask_loss = _tensor_scalar(objective.mask_loss)
    confidence_loss = _tensor_scalar(objective.confidence_loss)
    return TorchOptimizationStep(
        iteration=iteration,
        batch_index=batch_index,
        device=objective.device,
        sample_count=sample_count,
        target_offset=target_offset,
        image_loss=image_loss,
        depth_loss=depth_loss,
        query_loss=query_loss,
        normal_loss=normal_loss,
        mask_loss=mask_loss,
        confidence_loss=confidence_loss,
        total_loss=loss_weights.total(
            image_loss=image_loss,
            depth_loss=depth_loss,
            query_loss=query_loss,
            normal_loss=normal_loss,
            mask_loss=mask_loss,
            confidence_loss=confidence_loss,
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
    prepared_batches = tuple(_prepared_optimization_batch(batch_item, device=device) for batch_item in batches)
    for batch, _batch_index, _target_offset, _source_windows in prepared_batches:
        sample_count = _batch_sample_count(batch)
        if config.max_samples_per_batch is not None and sample_count > config.max_samples_per_batch:
            raise ValueError(
                f"torch optimization batch has {sample_count} samples, exceeding max_samples_per_batch "
                f"{config.max_samples_per_batch}"
            )
    for iteration in range(config.iterations):
        absolute_iteration = config.iteration_offset + iteration
        iteration_summaries: list[TorchCaptureRenderSummary] = []
        evolution_enabled = config.evolution_policy is not None and config.evolution_policy.enabled
        checkpoint_due = (
            config.checkpoint_interval is not None
            and (iteration + 1) % config.checkpoint_interval == 0
        )
        for batch, batch_index, target_offset, source_windows in prepared_batches:
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
                if evolution_enabled:
                    iteration_summaries.append(
                        torch_render_capture_training_summary(
                            current_scene,
                            batch,
                            carrier_parameters=carrier_parameters,
                            scene_tensors=scene_tensors,
                        )
                    )
                if checkpoint_due:
                    rendered = torch_render_capture_training_batch(
                        current_scene,
                        batch,
                        carrier_parameters=carrier_parameters,
                        scene_tensors=scene_tensors,
                    )
                else:
                    rendered = None
            if rendered is not None:
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
                        confidence_loss=_tensor_scalar(checkpoint_objective.confidence_loss),
                        query_loss=_tensor_scalar(checkpoint_objective.query_loss),
                        update=update,
                        max_samples_per_batch=config.max_samples_per_batch,
                    )
                )
            else:
                steps.append(
                    _optimization_step_from_objective(
                        absolute_iteration,
                        checkpoint_objective,
                        current_scene.elements,
                        batch_index=batch_index,
                        target_offset=target_offset,
                        source_windows=source_windows,
                        sample_count=_batch_sample_count(batch),
                        loss_weights=config.loss_weights,
                        update=update,
                        max_samples_per_batch=config.max_samples_per_batch,
                    )
                )
        if evolution_enabled or checkpoint_due:
            current_scene = _scene_from_carrier_parameters(current_scene, carrier_parameters, rendered)
        if evolution_enabled and iteration_summaries:
            predictions = tuple(
                prediction
                for summary in iteration_summaries
                for prediction in _evolution_predictions_from_summary(summary)
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
    final_summary: TorchCaptureRenderSummary | None = None
    if rendered is None and prepared_batches:
        final_batch = prepared_batches[-1][0]
        with torch.no_grad():
            final_summary = torch_render_capture_training_summary(
                current_scene,
                final_batch,
                carrier_parameters=carrier_parameters,
                scene_tensors=scene_tensors,
            )
    if rendered is not None:
        current_scene = _scene_from_carrier_parameters(current_scene, carrier_parameters, rendered)
    elif final_summary is not None:
        current_scene = _scene_from_carrier_parameters(current_scene, carrier_parameters, None, summary=final_summary)
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
            target_point=_target_point_from_rendered(rendered, index),
        )
        for index, (element_id, carrier_id, image_loss, depth_loss, query_loss, normal_loss, target_color) in enumerate(zip(
            rendered.element_ids,
            rendered.carrier_ids,
            rendered.image_loss,
            rendered.depth_loss,
            rendered.query_loss,
            rendered.normal_loss,
            rendered.target_color,
        ))
    )


def _evolution_predictions_from_summary(summary: TorchCaptureRenderSummary) -> tuple[_TorchEvolutionPrediction, ...]:
    return tuple(
        _TorchEvolutionPrediction(
            element_id=element_id,
            carrier_id=carrier_id,
            image_loss=float(image_loss),
            depth_loss=float(depth_loss),
            query_loss=float(query_loss),
            normal_loss=float(normal_loss),
            target_color=target_color,
            target_point=target_point,
        )
        for element_id, carrier_id, image_loss, depth_loss, query_loss, normal_loss, target_color, target_point in zip(
            summary.element_ids,
            summary.carrier_ids,
            summary.image_loss,
            summary.depth_loss,
            summary.query_loss,
            summary.normal_loss,
            summary.target_color,
            summary.target_point,
        )
    )


def _target_point_from_rendered(rendered: TorchRenderBatch, index: int) -> tuple[float, float, float] | None:
    if index >= len(rendered.ray_origins) or index >= len(rendered.ray_directions) or index >= len(rendered.target_depth):
        return None
    depth = rendered.target_depth[index]
    if depth <= 0.0:
        return None
    origin = rendered.ray_origins[index]
    direction = rendered.ray_directions[index]
    return tuple(origin[axis] + direction[axis] * depth for axis in range(3))  # type: ignore[return-value]


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
    gradient_norm_tensor: Any | None
    scale_tensor: Any | None
    gradient_clip_norm: float | None
    updated_parameter_count: int

    @property
    def gradient_norm(self) -> float:
        if self.gradient_norm_tensor is None:
            return 0.0
        return _tensor_scalar(self.gradient_norm_tensor)

    @property
    def applied_gradient_norm(self) -> float:
        if self.gradient_norm_tensor is None:
            return 0.0
        if self.scale_tensor is None:
            return self.gradient_norm
        return _tensor_scalar(self.gradient_norm_tensor * self.scale_tensor)


def _gradient_step_carrier_parameters(
    torch: Any,
    carrier_parameters: dict[str, dict[str, Any]],
    *,
    learning_rate: float,
    gradient_clip_norm: float | None,
) -> _TorchGradientStepState:
    parameters = []
    gradient_sq_terms = []
    for fields in carrier_parameters.values():
        for parameter in fields.values():
            if getattr(parameter, "grad", None) is None:
                continue
            parameters.append(parameter)
            gradient = parameter.grad.detach()
            gradient_sq_terms.append(torch.sum(gradient * gradient))
    if gradient_sq_terms:
        gradient_sq_sum = torch.stack(gradient_sq_terms).sum()
        gradient_norm_tensor = torch.sqrt(gradient_sq_sum)
        if gradient_clip_norm is not None:
            clip_tensor = torch.as_tensor(float(gradient_clip_norm), dtype=gradient_norm_tensor.dtype, device=gradient_norm_tensor.device)
            scale_tensor = torch.clamp(clip_tensor / torch.clamp(gradient_norm_tensor, min=1e-12), max=1.0)
        else:
            scale_tensor = torch.ones((), dtype=gradient_norm_tensor.dtype, device=gradient_norm_tensor.device)
    else:
        gradient_norm_tensor = None
        scale_tensor = None
    with torch.no_grad():
        for fields in carrier_parameters.values():
            for name, parameter in fields.items():
                if getattr(parameter, "grad", None) is None:
                    continue
                step_scale = scale_tensor if scale_tensor is not None else 1.0
                parameter.sub_(learning_rate * step_scale * parameter.grad)
                if name in {"color", "opacity", "confidence", "density", "bandwidth", "residual_scale"}:
                    parameter.clamp_(0.0, 1.0)
                elif name in {"alpha", "beta"}:
                    parameter.clamp_(1e-4)
                elif name in {"support_radius", "gaussian_covariance_diag"}:
                    parameter.clamp_(1e-4)
        for fields in carrier_parameters.values():
            if "min_corner" in fields and "max_corner" in fields:
                min_corner = fields["min_corner"]
                max_corner = fields["max_corner"]
                lower = torch.minimum(min_corner, max_corner - 1e-4)
                upper = torch.maximum(max_corner, min_corner + 1e-4)
                min_corner.copy_(lower)
                max_corner.copy_(upper)
    return _TorchGradientStepState(
        gradient_norm_tensor=gradient_norm_tensor,
        scale_tensor=scale_tensor,
        gradient_clip_norm=gradient_clip_norm,
        updated_parameter_count=len(parameters),
    )


def _scene_from_carrier_parameters(
    scene: AuraScene,
    carrier_parameters: dict[str, dict[str, Any]],
    rendered: TorchRenderBatch | None,
    *,
    summary: TorchCaptureRenderSummary | None = None,
) -> AuraScene:
    elements = []
    loss_by_element = _loss_by_element(rendered) if rendered is not None else _loss_by_element_summary(summary)
    device = rendered.device if rendered is not None else (summary.device if summary is not None else None)
    for element in scene.elements:
        fields = carrier_parameters.get(element.id, {})
        color = element.color
        opacity = element.opacity
        confidence = element.confidence
        bounds = element.bounds
        normal = element.normal
        payload = dict(element.payload)
        if "min_corner" in fields and "max_corner" in fields:
            min_corner = _tensor_vec3(fields["min_corner"])
            max_corner = _tensor_vec3(fields["max_corner"])
            bounds = Bounds(min_corner=min_corner, max_corner=max_corner)
        if "plane_point" in fields:
            payload["plane_point"] = list(_tensor_vec3(fields["plane_point"]))
        if "normal" in fields:
            normal = _normalized_vec3(_tensor_vec3(fields["normal"]))
            payload["normal"] = list(normal)
        if "gaussian_mean" in fields:
            payload["mean"] = list(_tensor_vec3(fields["gaussian_mean"]))
        if "color" in fields:
            color = _tensor_vec3(fields["color"])
        if "opacity" in fields:
            opacity = _clamp_unit(_tensor_scalar(fields["opacity"]))
            if payload.get("type") in {"volume_cell", "neural_residual"}:
                payload["opacity"] = opacity
        if "confidence" in fields:
            confidence = _clamp_unit(_tensor_scalar(fields["confidence"]))
            if payload.get("type") in {"beta_kernel", "neural_residual", "semantic_feature"}:
                payload["confidence"] = confidence
        for name in ("density", "alpha", "beta", "phase", "bandwidth", "residual_scale"):
            if name in fields:
                payload[name] = _tensor_scalar(fields[name])
        if "frequency" in fields:
            payload["frequency"] = list(_tensor_vec3(fields["frequency"]))
        if "support_radius" in fields:
            payload["support_radius"] = list(_tensor_vec3(fields["support_radius"]))
        if "gaussian_covariance_diag" in fields:
            diag = _tensor_vec3(fields["gaussian_covariance_diag"])
            payload["covariance"] = [
                [diag[0], 0.0, 0.0],
                [0.0, diag[1], 0.0],
                [0.0, 0.0, diag[2]],
            ]
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
                "torch_device": device,
            }
        elements.append(
            replace(
                element,
                bounds=bounds,
                color=color,
                opacity=opacity,
                confidence=confidence,
                normal=normal,
                payload=payload,
                confidence_map=confidence_map,
                metadata=metadata,
            )
        )
    chunked_elements, chunks = carrier_lod_elements_and_chunks(tuple(elements))
    return AuraScene(name=scene.name, elements=chunked_elements, chunks=chunks, semantic_graph=scene.semantic_graph)


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


def _loss_by_element_summary(summary: TorchCaptureRenderSummary | None) -> dict[str, dict[str, float]]:
    if summary is None:
        return {}
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for element_id, image, depth, query, normal in zip(
        summary.element_ids,
        summary.image_loss,
        summary.depth_loss,
        summary.query_loss,
        summary.normal_loss,
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
        + loss_weights.query * objective.query_loss
        + loss_weights.normal * objective.normal_loss
        + loss_weights.mask * objective.mask_loss
        + loss_weights.confidence * objective.confidence_loss
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
            "confidenceLoss": step.confidence_loss,
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
            "confidence": step.confidence_loss,
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


def _normalized_vec3(value: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(item * item for item in value))
    if norm <= 1e-8:
        return value
    return tuple(item / norm for item in value)  # type: ignore[return-value]


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
