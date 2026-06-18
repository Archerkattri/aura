from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import exp, isfinite, log, sqrt
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
    require_torch,
    torch_renderer_status,
    torch_capture_training_batch_from_packed,
    torch_scene_tensors,
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
    # Deliverable 1: Per-attribute Adam
    optimizer_type: str = 'sgd'
    position_learning_rate: float = 1.6e-4
    scale_learning_rate: float = 5e-3
    rotation_learning_rate: float = 1e-3
    opacity_learning_rate: float = 5e-2
    feature_learning_rate: float = 2.5e-3
    # Deliverable 2: LR schedules
    position_lr_final: float = 1.6e-6
    position_lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    # Deliverable 3: AbsGS gradient accumulator
    grad_accum_window: int = 0
    # Deliverable 4: Opacity soft-reset
    opacity_reset_interval: int = 0
    opacity_reset_value: float = 0.01
    recovery_window: int = 100
    # Deliverable 6: Budget ceiling
    max_carriers: int = 0
    # Deliverable 8: Coarse-to-fine schedule
    coarse_to_fine_schedule: tuple = ()

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
        if self.optimizer_type not in ('sgd', 'adam'):
            raise ValueError("torch optimizer_type must be 'sgd' or 'adam'")
        if not (0.0 <= self.opacity_reset_value <= 1.0):
            raise ValueError("torch opacity_reset_value must be in [0, 1]")
        if self.recovery_window < 0:
            raise ValueError("torch recovery_window must be non-negative")
        if self.lr_decay_steps < 0:
            raise ValueError("torch lr_decay_steps must be non-negative")
        if self.position_lr_final <= 0.0:
            raise ValueError("torch position_lr_final must be positive")


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
    # New optional fields (all with defaults) - Deliverables 3-8
    grad_stats: tuple = ()
    opacity_reset_due: bool = False
    recovery_phase: bool = False
    importance_scores: tuple = ()
    carrier_count: int = 0
    over_budget: bool = False
    resolution_scale: float = 1.0

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


def compute_importance_scores(
    opacities: dict[str, float],
    transmittances: dict[str, float],
) -> dict[str, float]:
    """Compute per-carrier importance as max(opacity * transmittance) over views. (RadSplat arXiv:2403.13806)"""
    scores = {}
    for carrier_id, opacity in opacities.items():
        transmittance = transmittances.get(carrier_id, 1.0)
        scores[carrier_id] = opacity * transmittance
    return scores


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


# Parameter group name assignments based on parameter names
_POSITION_PARAMS = {"min_corner", "max_corner", "plane_point", "gaussian_mean"}
_SCALE_PARAMS = {"support_radius", "gaussian_covariance_diag", "bandwidth"}
_ROTATION_PARAMS = {"frequency", "normal"}
_OPACITY_PARAMS = {"opacity", "density", "alpha", "beta"}
_COLOR_PARAMS = {"color", "residual_scale"}
_FEATURE_PARAMS = {"confidence", "phase"}


def _build_adam_optimizer(
    torch: Any,
    carrier_parameters: dict[str, dict[str, Any]],
    config: TorchOptimizationConfig,
) -> Any:
    """Build Adam optimizer with per-attribute parameter groups."""
    position_params = []
    scale_params = []
    rotation_params = []
    opacity_params = []
    color_params = []
    feature_params = []

    for fields in carrier_parameters.values():
        for name, parameter in fields.items():
            if not hasattr(parameter, 'requires_grad'):
                continue
            if name in _POSITION_PARAMS:
                position_params.append(parameter)
            elif name in _SCALE_PARAMS:
                scale_params.append(parameter)
            elif name in _ROTATION_PARAMS:
                rotation_params.append(parameter)
            elif name in _OPACITY_PARAMS:
                opacity_params.append(parameter)
            elif name in _COLOR_PARAMS:
                color_params.append(parameter)
            elif name in _FEATURE_PARAMS:
                feature_params.append(parameter)
            else:
                # Default to color group for unknown params
                color_params.append(parameter)

    param_groups = []
    if position_params:
        param_groups.append({'params': position_params, 'lr': config.position_learning_rate, 'name': 'position'})
    if scale_params:
        param_groups.append({'params': scale_params, 'lr': config.scale_learning_rate, 'name': 'scale'})
    if rotation_params:
        param_groups.append({'params': rotation_params, 'lr': config.rotation_learning_rate, 'name': 'rotation'})
    if opacity_params:
        param_groups.append({'params': opacity_params, 'lr': config.opacity_learning_rate, 'name': 'opacity'})
    if color_params:
        param_groups.append({'params': color_params, 'lr': config.color_learning_rate, 'name': 'color'})
    if feature_params:
        param_groups.append({'params': feature_params, 'lr': config.feature_learning_rate, 'name': 'feature'})

    if not param_groups:
        # Fallback: empty group
        return None

    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))


def _compute_position_lr(config: TorchOptimizationConfig, step: int) -> float:
    """Compute position learning rate with exponential decay."""
    if config.lr_decay_steps <= 0:
        return config.position_learning_rate
    # Warmup
    if step < config.position_lr_warmup_steps:
        return config.position_learning_rate
    decay_step = step - config.position_lr_warmup_steps
    if decay_step >= config.lr_decay_steps:
        return config.position_lr_final
    # Exponential interpolation
    t = decay_step / config.lr_decay_steps
    lr_init = config.position_learning_rate
    lr_final = config.position_lr_final
    return lr_init * (lr_final / lr_init) ** t


def _coarse_to_fine_scale(schedule: tuple, step: int) -> float:
    """Find current resolution scale from schedule breakpoints."""
    if not schedule:
        return 1.0
    current_scale = 1.0
    for breakpoint_step, scale in schedule:
        if step >= breakpoint_step:
            current_scale = scale
    return current_scale


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
    optimizer_label: str = "sgd",
    grad_stats: tuple = (),
    opacity_reset_due: bool = False,
    recovery_phase: bool = False,
    importance_scores: tuple = (),
    max_carriers: int = 0,
    resolution_scale: float = 1.0,
) -> TorchOptimizationStep:
    image_loss = _tensor_scalar(objective.image_loss)
    depth_loss = _tensor_scalar(objective.depth_loss)
    query_loss = _tensor_scalar(objective.query_loss)
    normal_loss = _tensor_scalar(objective.normal_loss)
    mask_loss = _tensor_scalar(objective.mask_loss)
    confidence_loss = _tensor_scalar(objective.confidence_loss)
    carrier_counts = _carrier_counts(elements)
    carrier_count = sum(carrier_counts.values())
    over_budget = max_carriers > 0 and carrier_count > max_carriers
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
        carrier_counts=carrier_counts,
        loss_weights=loss_weights.to_dict(),
        optimizer=optimizer_label,
        gradient_norm=update.gradient_norm,
        applied_gradient_norm=update.applied_gradient_norm,
        gradient_clip_norm=update.gradient_clip_norm,
        updated_parameter_count=update.updated_parameter_count,
        max_samples_per_batch=max_samples_per_batch,
        source_windows=source_windows,
        grad_stats=grad_stats,
        opacity_reset_due=opacity_reset_due,
        recovery_phase=recovery_phase,
        importance_scores=importance_scores,
        carrier_count=carrier_count,
        over_budget=over_budget,
        resolution_scale=resolution_scale,
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
    materialization_summary: TorchCaptureRenderSummary | None = None
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

    # Set up optimizer for Adam path
    use_adam = config.optimizer_type == 'adam'
    adam_optimizer = None
    if use_adam:
        adam_optimizer = _build_adam_optimizer(torch, carrier_parameters, config)

    # Gradient accumulator state (Deliverable 3)
    grad_accumulator: dict[str, float] = {}
    grad_accum_step_count: int = 0

    # Opacity reset tracking (Deliverable 4)
    steps_since_reset: int = 0

    for iteration in range(config.iterations):
        absolute_iteration = config.iteration_offset + iteration
        iteration_summaries: list[TorchCaptureRenderSummary] = []
        iteration_materialization_summary: TorchCaptureRenderSummary | None = None
        evolution_enabled = config.evolution_policy is not None and config.evolution_policy.enabled
        checkpoint_due = (
            config.checkpoint_interval is not None
            and (iteration + 1) % config.checkpoint_interval == 0
        )

        # Deliverable 8: Coarse-to-fine schedule
        resolution_scale = _coarse_to_fine_scale(config.coarse_to_fine_schedule, absolute_iteration)

        # Deliverable 4: Opacity reset
        opacity_reset_due = False
        if (
            config.opacity_reset_interval > 0
            and absolute_iteration > 0
            and absolute_iteration % config.opacity_reset_interval == 0
        ):
            opacity_reset_due = True
            steps_since_reset = 0
            # Soft-reset opacity parameters
            with torch.no_grad():
                for fields in carrier_parameters.values():
                    for name, parameter in fields.items():
                        if name in _OPACITY_PARAMS:
                            parameter.fill_(config.opacity_reset_value)
        else:
            steps_since_reset += 1

        recovery_phase = (
            config.opacity_reset_interval > 0
            and steps_since_reset < config.recovery_window
            and not opacity_reset_due
        )

        # Deliverable 2: Update position LR if using Adam with decay
        if use_adam and adam_optimizer is not None and config.lr_decay_steps > 0:
            new_position_lr = _compute_position_lr(config, absolute_iteration)
            for group in adam_optimizer.param_groups:
                if group.get('name') == 'position':
                    group['lr'] = new_position_lr

        for batch, batch_index, target_offset, source_windows in prepared_batches:
            if use_adam and adam_optimizer is not None:
                adam_optimizer.zero_grad()
            else:
                _zero_carrier_parameter_grads(carrier_parameters)

            objective = torch_render_capture_training_objective(
                current_scene,
                batch,
                carrier_parameters=carrier_parameters,
                scene_tensors=scene_tensors,
            )
            weighted_loss = _weighted_torch_loss(objective, config.loss_weights)
            weighted_loss.backward()

            if use_adam and adam_optimizer is not None:
                # Clip gradients before Adam step
                all_params = [p for fields in carrier_parameters.values() for p in fields.values()]
                params_with_grad = [p for p in all_params if getattr(p, 'grad', None) is not None]
                gradient_sq_terms = []
                for p in params_with_grad:
                    gradient_sq_terms.append(torch.sum(p.grad.detach() * p.grad.detach()))

                if gradient_sq_terms:
                    gradient_sq_sum = torch.stack(gradient_sq_terms).sum()
                    gradient_norm_tensor = torch.sqrt(gradient_sq_sum)
                    if config.gradient_clip_norm is not None:
                        clip_tensor = torch.as_tensor(
                            float(config.gradient_clip_norm),
                            dtype=gradient_norm_tensor.dtype,
                            device=gradient_norm_tensor.device
                        )
                        scale_tensor = torch.clamp(
                            clip_tensor / torch.clamp(gradient_norm_tensor, min=1e-12), max=1.0
                        )
                        torch.nn.utils.clip_grad_norm_(params_with_grad, config.gradient_clip_norm)
                    else:
                        scale_tensor = None
                else:
                    gradient_norm_tensor = None
                    scale_tensor = None

                # Deliverable 3: Accumulate absolute gradients
                if config.grad_accum_window > 0:
                    grad_accum_step_count += 1
                    for element_id, fields in carrier_parameters.items():
                        for param_name, parameter in fields.items():
                            if getattr(parameter, 'grad', None) is not None:
                                key = f"{element_id}.{param_name}"
                                abs_grad = float(parameter.grad.abs().mean().item())
                                grad_accumulator[key] = grad_accumulator.get(key, 0.0) + abs_grad

                adam_optimizer.step()
                update = _TorchGradientStepState(
                    gradient_norm_tensor=gradient_norm_tensor,
                    scale_tensor=scale_tensor,
                    gradient_clip_norm=config.gradient_clip_norm,
                    updated_parameter_count=len(params_with_grad),
                )
                # Apply clamping constraints after Adam step
                with torch.no_grad():
                    for fields in carrier_parameters.values():
                        for name, parameter in fields.items():
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
            else:
                # SGD path: Deliverable 3 grad accumulation
                if config.grad_accum_window > 0:
                    grad_accum_step_count += 1
                    for element_id, fields in carrier_parameters.items():
                        for param_name, parameter in fields.items():
                            if getattr(parameter, 'grad', None) is not None:
                                key = f"{element_id}.{param_name}"
                                abs_grad = float(parameter.grad.abs().mean().item())
                                grad_accumulator[key] = grad_accumulator.get(key, 0.0) + abs_grad

                update = _gradient_step_carrier_parameters(
                    torch,
                    carrier_parameters,
                    learning_rate=config.color_learning_rate,
                    gradient_clip_norm=config.gradient_clip_norm,
                )

            # Deliverable 3: Build grad_stats snapshot and reset if window elapsed
            current_grad_stats: tuple = ()
            if config.grad_accum_window > 0 and grad_accumulator:
                window = config.grad_accum_window
                if grad_accum_step_count >= window:
                    # Summarize mean over window
                    current_grad_stats = tuple(
                        (k, v / grad_accum_step_count) for k, v in sorted(grad_accumulator.items())
                    )
                    # Reset
                    grad_accumulator.clear()
                    grad_accum_step_count = 0
                else:
                    # Partial snapshot
                    current_grad_stats = tuple(
                        (k, v / grad_accum_step_count) for k, v in sorted(grad_accumulator.items())
                    )

            if use_adam and adam_optimizer is not None:
                _zero_carrier_parameter_grads(carrier_parameters)
            else:
                _zero_carrier_parameter_grads(carrier_parameters)

            with torch.no_grad():
                checkpoint_objective = torch_render_capture_training_objective(
                    current_scene,
                    batch,
                    carrier_parameters=carrier_parameters,
                    scene_tensors=scene_tensors,
                )
                if evolution_enabled or checkpoint_due:
                    summary = torch_render_capture_training_summary(
                        current_scene,
                        batch,
                        carrier_parameters=carrier_parameters,
                        scene_tensors=scene_tensors,
                    )
                    if evolution_enabled:
                        iteration_summaries.append(summary)
                    if checkpoint_due:
                        iteration_materialization_summary = summary
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
                    optimizer_label="adam" if use_adam else "sgd",
                    grad_stats=current_grad_stats,
                    opacity_reset_due=opacity_reset_due,
                    recovery_phase=recovery_phase,
                    importance_scores=(),
                    max_carriers=config.max_carriers,
                    resolution_scale=resolution_scale,
                )
            )
        if evolution_enabled or checkpoint_due:
            current_scene = _scene_from_carrier_parameters(
                current_scene,
                carrier_parameters,
                summary=iteration_materialization_summary,
            )
            materialization_summary = iteration_materialization_summary
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
                # Rebuild Adam optimizer after evolution (carrier parameters changed)
                if use_adam:
                    adam_optimizer = _build_adam_optimizer(torch, carrier_parameters, config)
                materialization_summary = None
        if checkpoint_due:
            scene_checkpoints.append(
                TorchSceneCheckpoint(
                    checkpoint_index=len(scene_checkpoints),
                    iteration=absolute_iteration,
                    step_count=len(steps),
                    scene=current_scene,
                )
            )
    final_summary: TorchCaptureRenderSummary | None = None
    if materialization_summary is None and prepared_batches:
        final_batch = prepared_batches[-1][0]
        with torch.no_grad():
            final_summary = torch_render_capture_training_summary(
                current_scene,
                final_batch,
                carrier_parameters=carrier_parameters,
                scene_tensors=scene_tensors,
            )
    if final_summary is not None:
        current_scene = _scene_from_carrier_parameters(current_scene, carrier_parameters, summary=final_summary)
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
    *,
    summary: TorchCaptureRenderSummary | None = None,
) -> AuraScene:
    elements = []
    loss_by_element = _loss_by_element_summary(summary)
    device = summary.device if summary is not None else None
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
