from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, replace
from math import exp, isfinite, log, sqrt
from typing import Any, Optional, Sequence

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
class DensificationConfig:
    """3DGS-style adaptive densification + pruning + regularization schedule.

    All new behavior is opt-in: with enabled=False (the default) the training
    loop is completely unchanged and all existing tests remain green.
    """

    enabled: bool = False
    # Iteration range for densification
    start_iteration: int = 500
    end_iteration: int = 15000
    interval: int = 100
    # AbsGS gradient threshold: carriers above this are cloned/split
    grad_threshold: float = 0.0002
    # Carriers with scale larger than this multiple of scene median are split,
    # otherwise cloned (3DGS ADC criterion)
    split_threshold_scale: float = 1.0
    # RadSplat importance (opacity * transmittance) threshold for pruning
    prune_importance_threshold: float = 0.005
    # Opacity threshold below which carriers are pruned
    prune_opacity_threshold: float = 0.005
    # Extra opacity-reset wait before pruning starts (prevents pruning
    # carriers that are simply in a recovery phase)
    recovery_prune_delay: int = 2
    # Optional max-carrier budget override (None = use TorchOptimizationConfig.max_carriers)
    max_carriers: Optional[int] = None
    # Scale/anisotropy regularization: penalises large aspect-ratio Gaussians
    scale_reg_weight: float = 0.0
    # Opacity entropy regularization: pushes opacities toward 0 or 1
    opacity_entropy_reg_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.start_iteration < 0:
            raise ValueError("densification start_iteration must be non-negative")
        if self.end_iteration < self.start_iteration:
            raise ValueError("densification end_iteration must be >= start_iteration")
        if self.interval <= 0:
            raise ValueError("densification interval must be positive")
        if self.grad_threshold < 0:
            raise ValueError("densification grad_threshold must be non-negative")
        if not (0.0 <= self.prune_importance_threshold <= 1.0):
            raise ValueError("prune_importance_threshold must be in [0, 1]")
        if not (0.0 <= self.prune_opacity_threshold <= 1.0):
            raise ValueError("prune_opacity_threshold must be in [0, 1]")
        if self.scale_reg_weight < 0.0:
            raise ValueError("scale_reg_weight must be non-negative")
        if self.opacity_entropy_reg_weight < 0.0:
            raise ValueError("opacity_entropy_reg_weight must be non-negative")


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
    # Deliverable: Adaptive densification + pruning + regularization (3DGS ADC)
    densification: DensificationConfig = field(default_factory=DensificationConfig)

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
        if not isinstance(self.densification, DensificationConfig):
            raise TypeError("torch densification must be a DensificationConfig instance")


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
    # New optional fields (all with defaults)
    grad_stats: tuple = ()
    opacity_reset_due: bool = False
    recovery_phase: bool = False
    importance_scores: tuple = ()
    carrier_count: int = 0
    over_budget: bool = False
    # Densification reporting
    densified_count: int = 0
    pruned_count: int = 0

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


def _live_opacity(parameter: Any, fallback: float) -> float:
    """Read a scalar opacity value from a (possibly tensor) carrier parameter."""
    if parameter is None:
        return float(fallback)
    detach = getattr(parameter, "detach", None)
    if detach is not None:
        flat = parameter.detach().reshape(-1)
        return float(flat[0].item()) if flat.numel() else float(fallback)
    try:
        return float(parameter)
    except (TypeError, ValueError):
        return float(fallback)


def _carrier_importance_scores(
    carrier_parameters: dict[str, dict[str, Any]],
    elements: Sequence[AuraElement],
) -> dict[str, float]:
    """RadSplat-style per-carrier importance from the live trained opacity values.

    Batches all opacity tensor reads into a single CPU transfer instead of one
    .item() call per carrier (which caused N CUDA syncs for N carriers).
    """
    tensor_ids: list[str] = []
    tensor_list: list = []
    scalar_opacities: dict[str, float] = {}

    for element in elements:
        fields = carrier_parameters.get(element.id, {})
        param = fields.get("opacity")
        if param is not None and hasattr(param, "detach"):
            flat = param.detach().reshape(-1)
            if flat.numel():
                tensor_ids.append(element.id)
                tensor_list.append(flat[0])
                continue
        # Fallback: scalar or missing
        scalar_opacities[element.id] = _live_opacity(param, element.opacity)

    opacities: dict[str, float] = dict(scalar_opacities)
    if tensor_list:
        _torch = require_torch()
        values = _torch.stack(tensor_list).cpu().tolist()
        for eid, val in zip(tensor_ids, values):
            opacities[eid] = float(val)

    return compute_importance_scores(opacities, {})


class DensificationEngine:
    """3DGS-style adaptive densification and pruning.

    Implements:
    - Clone: duplicate carriers with high AbsGS gradient norm (under-reconstruction)
    - Split: subdivide large carriers with high gradient into two smaller children
    - Prune: remove carriers below RadSplat importance or opacity threshold

    References:
    - 3DGS ADC: Kerbl et al. arXiv:2308.04079
    - AbsGS: Ye et al. arXiv:2404.10484
    - RadSplat: Niemeyer et al. arXiv:2403.13806
    - EDC: arXiv:2411.10133
    """

    @staticmethod
    def should_run(
        absolute_iteration: int,
        densification_config: DensificationConfig,
    ) -> bool:
        """Return True when densification should run at this iteration."""
        if not densification_config.enabled:
            return False
        if absolute_iteration < densification_config.start_iteration:
            return False
        if absolute_iteration > densification_config.end_iteration:
            return False
        return (absolute_iteration - densification_config.start_iteration) % densification_config.interval == 0

    @staticmethod
    def densify_and_prune(
        scene: AuraScene,
        carrier_parameters: dict[str, dict[str, Any]],
        grad_accumulator: dict[str, float],
        absolute_iteration: int,
        densification_config: DensificationConfig,
        steps_since_reset: int,
        max_carriers_budget: int,
        *,
        torch: Any,
    ) -> tuple[AuraScene, dict[str, dict[str, Any]], int, int]:
        """Perform one densification+pruning pass.

        Returns (new_scene, new_carrier_parameters, num_densified, num_pruned).
        """
        num_densified = 0
        num_pruned = 0

        # Compute per-element grad norms from accumulator
        grad_by_element: dict[str, float] = {}
        for key, grad_val in grad_accumulator.items():
            element_id = key.split(".")[0]
            grad_by_element[element_id] = max(grad_by_element.get(element_id, 0.0), grad_val)

        # Compute per-element opacity from carrier_parameters
        opacity_by_element: dict[str, float] = {}
        for element in scene.elements:
            fields = carrier_parameters.get(element.id, {})
            opacity_by_element[element.id] = _live_opacity(fields.get("opacity"), element.opacity)

        # Compute per-element scale (use support_radius or bounds extent as proxy)
        scale_by_element: dict[str, float] = {}
        for element in scene.elements:
            fields = carrier_parameters.get(element.id, {})
            if "support_radius" in fields:
                sr = fields["support_radius"]
                detached = sr.detach().cpu().tolist() if hasattr(sr, 'detach') else list(sr)
                scale_by_element[element.id] = max(float(v) for v in detached)
            elif "gaussian_covariance_diag" in fields:
                gcd = fields["gaussian_covariance_diag"]
                detached = gcd.detach().cpu().tolist() if hasattr(gcd, 'detach') else list(gcd)
                scale_by_element[element.id] = max(float(v) ** 0.5 for v in detached)
            else:
                # Use bounds extent as a scale proxy
                b = element.bounds
                extents = [hi - lo for lo, hi in zip(b.min_corner, b.max_corner)]
                scale_by_element[element.id] = max(extents)

        # Compute median scale for split/clone decision
        scales = list(scale_by_element.values())
        if scales:
            sorted_scales = sorted(scales)
            median_scale = sorted_scales[len(sorted_scales) // 2]
        else:
            median_scale = 1.0

        effective_max = densification_config.max_carriers if densification_config.max_carriers is not None else max_carriers_budget
        current_count = len(scene.elements)

        grad_threshold = densification_config.grad_threshold

        # ---- DENSIFICATION PASS ----
        elements_to_clone: list[AuraElement] = []
        elements_to_split: list[AuraElement] = []

        for element in scene.elements:
            grad_norm = grad_by_element.get(element.id, 0.0)
            if grad_norm < grad_threshold:
                continue
            # Check budget
            if effective_max > 0 and (current_count + num_densified + len(elements_to_clone) + len(elements_to_split)) >= effective_max:
                break
            scale = scale_by_element.get(element.id, 0.0)
            if scale > densification_config.split_threshold_scale * median_scale:
                elements_to_split.append(element)
            else:
                elements_to_clone.append(element)

        # Execute clones
        new_elements: list[AuraElement] = []
        new_parameters: dict[str, dict[str, Any]] = {}
        existing_ids = {element.id for element in scene.elements}

        for element in elements_to_clone:
            clone_id = f"{element.id}_clone_{uuid.uuid4().hex[:8]}"
            if clone_id in existing_ids:
                continue
            # Clone: same bounds and parameters as parent
            clone = replace(
                element,
                id=clone_id,
                metadata={
                    **element.metadata,
                    "densification": "clone",
                    "parent": element.id,
                    "source": "aura-densify-3dgs",
                },
            )
            new_elements.append(clone)
            existing_ids.add(clone_id)
            num_densified += 1
            # Clone parameters from parent (create new tensors)
            parent_fields = carrier_parameters.get(element.id, {})
            clone_fields: dict[str, Any] = {}
            for pname, ptensor in parent_fields.items():
                if hasattr(ptensor, 'detach'):
                    new_tensor = ptensor.detach().clone().requires_grad_(ptensor.requires_grad)
                    clone_fields[pname] = new_tensor
                else:
                    clone_fields[pname] = ptensor
            new_parameters[clone_id] = clone_fields

        for element in elements_to_split:
            # Split: two children at half the parent's size, offset in opposite directions
            b = element.bounds
            center = tuple((lo + hi) / 2.0 for lo, hi in zip(b.min_corner, b.max_corner))
            half = tuple((hi - lo) / 4.0 for lo, hi in zip(b.min_corner, b.max_corner))
            # Child A: negative offset
            child_a_min = tuple(c - h for c, h in zip(center, half))
            child_a_max = tuple(c + h for c, h in zip(center, half))
            # Child B: positive offset (try axis with largest extent)
            axis = max(range(3), key=lambda i: b.max_corner[i] - b.min_corner[i])
            offset = [0.0, 0.0, 0.0]
            offset[axis] = half[axis]
            child_b_min = tuple(c - h + offset[i] for i, (c, h) in enumerate(zip(center, half)))
            child_b_max = tuple(c + h + offset[i] for i, (c, h) in enumerate(zip(center, half)))

            parent_fields = carrier_parameters.get(element.id, {})
            for child_suffix, child_min, child_max in [
                ("_split_a", child_a_min, child_a_max),
                ("_split_b", child_b_min, child_b_max),
            ]:
                child_id = f"{element.id}{child_suffix}_{uuid.uuid4().hex[:8]}"
                if child_id in existing_ids:
                    continue
                if effective_max > 0 and (current_count + num_densified + len(new_elements)) >= effective_max:
                    break
                try:
                    child_bounds = Bounds(min_corner=child_min, max_corner=child_max)
                except ValueError:
                    continue
                child = replace(
                    element,
                    id=child_id,
                    bounds=child_bounds,
                    metadata={
                        **element.metadata,
                        "densification": "split",
                        "parent": element.id,
                        "source": "aura-densify-3dgs",
                    },
                )
                new_elements.append(child)
                existing_ids.add(child_id)
                num_densified += 1
                # Split: child parameters from parent with scaled support
                child_fields: dict[str, Any] = {}
                for pname, ptensor in parent_fields.items():
                    if hasattr(ptensor, 'detach'):
                        new_tensor = ptensor.detach().clone()
                        # Scale down support_radius/covariance for split children
                        if pname in {"support_radius", "gaussian_covariance_diag", "bandwidth"}:
                            new_tensor = new_tensor * 0.7
                            new_tensor = new_tensor.clamp(min=1e-4)
                        new_tensor = new_tensor.requires_grad_(ptensor.requires_grad)
                        child_fields[pname] = new_tensor
                    else:
                        child_fields[pname] = ptensor
                new_parameters[child_id] = child_fields

        # ---- PRUNING PASS ----
        pruned_ids: set[str] = set()
        prune_importance = densification_config.prune_importance_threshold
        prune_opacity = densification_config.prune_opacity_threshold
        prune_delay = densification_config.recovery_prune_delay

        # Don't prune if we're still in the opacity reset recovery window
        in_recovery = steps_since_reset < prune_delay

        if not in_recovery:
            for element in scene.elements:
                opacity = opacity_by_element.get(element.id, element.opacity)
                # RadSplat importance proxy (opacity * transmittance, transmittance=1.0)
                importance = opacity
                if opacity < prune_opacity or importance < prune_importance:
                    pruned_ids.add(element.id)
                    num_pruned += 1

        # ---- BUILD NEW SCENE ----
        # Keep elements that weren't pruned, add newly densified elements
        retained_elements = [e for e in scene.elements if e.id not in pruned_ids]
        all_elements = retained_elements + new_elements

        if not all_elements:
            # Never prune to zero — keep the element with highest opacity
            best = max(scene.elements, key=lambda e: opacity_by_element.get(e.id, e.opacity))
            all_elements = [best]
            num_pruned = len(scene.elements) - 1

        # Update carrier_parameters: remove pruned, add new
        new_carrier_parameters = {
            eid: params
            for eid, params in carrier_parameters.items()
            if eid not in pruned_ids
        }
        new_carrier_parameters.update(new_parameters)

        chunked_elements, chunks = carrier_lod_elements_and_chunks(tuple(all_elements))
        new_scene = AuraScene(
            name=scene.name,
            elements=chunked_elements,
            chunks=chunks,
            semantic_graph=scene.semantic_graph,
        )
        return new_scene, new_carrier_parameters, num_densified, num_pruned


def _compute_regularization_loss(
    carrier_parameters: dict[str, dict[str, Any]],
    densification_config: DensificationConfig,
    *,
    torch: Any,
) -> Any:
    """Compute optional scale/anisotropy and opacity-entropy regularization terms.

    Returns a scalar tensor (or 0.0 as a plain float when both weights are zero
    and torch is not needed).
    """
    scale_weight = densification_config.scale_reg_weight
    entropy_weight = densification_config.opacity_entropy_reg_weight
    if scale_weight == 0.0 and entropy_weight == 0.0:
        return None  # Skip entirely — no extra overhead

    scale_terms = []
    opacity_terms = []

    for fields in carrier_parameters.values():
        # Scale anisotropy: penalise large ratio between max and min scale dim
        if scale_weight > 0.0:
            for pname in ("support_radius", "gaussian_covariance_diag", "bandwidth"):
                if pname in fields:
                    s = fields[pname]
                    if hasattr(s, 'reshape'):
                        sv = s.reshape(-1)
                        if sv.numel() >= 2:
                            # Penalise aspect ratio > 1: (max/min - 1)^2
                            s_max = sv.max()
                            s_min = sv.min().clamp(min=1e-8)
                            aniso = (s_max / s_min - 1.0) ** 2
                            scale_terms.append(aniso)
                    break

        # Opacity entropy: push opacities toward 0 or 1
        if entropy_weight > 0.0:
            for pname in ("opacity", "density"):
                if pname in fields:
                    o = fields[pname].reshape(-1).clamp(1e-6, 1.0 - 1e-6)
                    entropy = -(o * torch.log(o) + (1.0 - o) * torch.log(1.0 - o))
                    opacity_terms.append(entropy.mean())
                    break

    reg_loss = None
    if scale_terms and scale_weight > 0.0:
        stacked = torch.stack(scale_terms)
        contribution = scale_weight * stacked.mean()
        reg_loss = contribution if reg_loss is None else reg_loss + contribution
    if opacity_terms and entropy_weight > 0.0:
        stacked = torch.stack(opacity_terms)
        contribution = entropy_weight * stacked.mean()
        reg_loss = contribution if reg_loss is None else reg_loss + contribution

    return reg_loss


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
            if not getattr(parameter, 'requires_grad', False):
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
    densified_count: int = 0,
    pruned_count: int = 0,
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
        densified_count=densified_count,
        pruned_count=pruned_count,
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

    # Densification tracking
    densify_cfg = config.densification
    # Always accumulate gradients when densification is enabled (even if grad_accum_window=0)
    _track_grads_for_densify = densify_cfg.enabled

    for iteration in range(config.iterations):
        absolute_iteration = config.iteration_offset + iteration
        iteration_summaries: list[TorchCaptureRenderSummary] = []
        iteration_materialization_summary: TorchCaptureRenderSummary | None = None
        evolution_enabled = config.evolution_policy is not None and config.evolution_policy.enabled
        checkpoint_due = (
            config.checkpoint_interval is not None
            and (iteration + 1) % config.checkpoint_interval == 0
        )

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
            # Add optional regularization terms (scale anisotropy + opacity entropy)
            reg_loss = _compute_regularization_loss(carrier_parameters, densify_cfg, torch=torch)
            if reg_loss is not None:
                weighted_loss = weighted_loss + reg_loss
            weighted_loss.backward()

            # Replace non-finite gradients with zero so a degenerate carrier
            # (e.g. a near-singular covariance) cannot drive its parameters to
            # NaN/Inf and destabilise the whole optimisation.
            for fields in carrier_parameters.values():
                for parameter in fields.values():
                    grad = getattr(parameter, "grad", None)
                    if grad is not None:
                        torch.nan_to_num_(grad, nan=0.0, posinf=0.0, neginf=0.0)

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

                # Deliverable 3: Accumulate absolute gradients (also used by densification)
                if config.grad_accum_window > 0 or _track_grads_for_densify:
                    grad_accum_step_count += 1
                    _ga_keys: list = []
                    _ga_tensors: list = []
                    for element_id, fields in carrier_parameters.items():
                        for param_name, parameter in fields.items():
                            if getattr(parameter, 'grad', None) is not None:
                                _ga_keys.append(f"{element_id}.{param_name}")
                                _ga_tensors.append(parameter.grad.abs().mean())
                    if _ga_keys:
                        _ga_values = torch.stack(_ga_tensors).detach().cpu().tolist()
                        for _k, _v in zip(_ga_keys, _ga_values):
                            grad_accumulator[_k] = grad_accumulator.get(_k, 0.0) + _v

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
                # SGD path: Deliverable 3 grad accumulation (also used by densification)
                if config.grad_accum_window > 0 or _track_grads_for_densify:
                    grad_accum_step_count += 1
                    _ga_keys2: list = []
                    _ga_tensors2: list = []
                    for element_id, fields in carrier_parameters.items():
                        for param_name, parameter in fields.items():
                            if getattr(parameter, 'grad', None) is not None:
                                _ga_keys2.append(f"{element_id}.{param_name}")
                                _ga_tensors2.append(parameter.grad.abs().mean())
                    if _ga_keys2:
                        _ga_values2 = torch.stack(_ga_tensors2).detach().cpu().tolist()
                        for _k2, _v2 in zip(_ga_keys2, _ga_values2):
                            grad_accumulator[_k2] = grad_accumulator.get(_k2, 0.0) + _v2

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
            step_importance = tuple(
                sorted(_carrier_importance_scores(carrier_parameters, current_scene.elements).items())
            )
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
                    importance_scores=step_importance,
                    max_carriers=config.max_carriers,
                )
            )

        # ---- DENSIFICATION + PRUNING ----
        if densify_cfg.enabled and DensificationEngine.should_run(absolute_iteration, densify_cfg):
            densify_grad_accum = dict(grad_accumulator)
            # Normalize by step count for fair comparison
            if grad_accum_step_count > 0:
                densify_grad_accum = {k: v / grad_accum_step_count for k, v in densify_grad_accum.items()}
            max_bud = densify_cfg.max_carriers if densify_cfg.max_carriers is not None else config.max_carriers
            new_scene, new_carrier_parameters, num_densified, num_pruned = DensificationEngine.densify_and_prune(
                current_scene,
                carrier_parameters,
                densify_grad_accum,
                absolute_iteration,
                densify_cfg,
                steps_since_reset,
                max_bud,
                torch=torch,
            )
            if num_densified > 0 or num_pruned > 0:
                current_scene = new_scene
                carrier_parameters = new_carrier_parameters
                scene_tensors = torch_scene_tensors(current_scene, device=device)
                carrier_parameters = scene_tensors.carrier_parameters
                # Re-copy the optimized values back into scene_tensors parameters
                # (scene_tensors re-initializes from element defaults, so we need
                # to restore the in-memory trained values for retained carriers)
                _restore_trained_parameters(carrier_parameters, new_carrier_parameters)
                if use_adam:
                    adam_optimizer = _build_adam_optimizer(torch, carrier_parameters, config)
                # Patch last step with densification counts
                if steps:
                    steps[-1] = replace(
                        steps[-1],
                        densified_count=num_densified,
                        pruned_count=num_pruned,
                        carrier_counts=_carrier_counts(current_scene.elements),
                        carrier_count=len(current_scene.elements),
                    )
                materialization_summary = None

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
        if (iteration + 1) % 10 == 0 or iteration == 0:
            last_loss = steps[-1].total_loss if steps else float("nan")
            import sys as _sys
            print(
                f"iter {absolute_iteration + 1}/{config.iteration_offset + config.iterations}"
                f"  loss={last_loss:.4f}",
                file=_sys.stderr,
                flush=True,
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


def _restore_trained_parameters(
    new_carrier_parameters: dict[str, dict[str, Any]],
    trained_parameters: dict[str, dict[str, Any]],
) -> None:
    """After scene_tensors is rebuilt from element defaults, restore the in-memory
    optimized parameter values for carriers that were retained (not pruned or newly added).

    This prevents the scene_tensors re-initialization from wiping out all the training
    progress that happened before densification.
    """
    for element_id, fields in new_carrier_parameters.items():
        trained_fields = trained_parameters.get(element_id)
        if trained_fields is None:
            continue  # New carrier — keep fresh init from scene_tensors
        for pname, new_tensor in fields.items():
            trained_tensor = trained_fields.get(pname)
            if trained_tensor is None:
                continue
            if hasattr(new_tensor, 'data') and hasattr(trained_tensor, 'detach'):
                try:
                    new_tensor.data.copy_(trained_tensor.detach())
                except (RuntimeError, ValueError):
                    pass  # Shape mismatch — leave as-is


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
        # Guard against non-finite optimized parameters (a carrier whose params
        # diverged to NaN/Inf keeps its last valid values). Without this, a
        # single NaN bound poisons the chunk union and corrupts the package.
        if "min_corner" in fields and "max_corner" in fields:
            min_corner = _tensor_vec3(fields["min_corner"])
            max_corner = _tensor_vec3(fields["max_corner"])
            if _all_finite(min_corner) and _all_finite(max_corner):
                bounds = Bounds(min_corner=min_corner, max_corner=max_corner)
        if "plane_point" in fields:
            plane_point = _tensor_vec3(fields["plane_point"])
            if _all_finite(plane_point):
                payload["plane_point"] = list(plane_point)
        if "normal" in fields:
            candidate_normal = _tensor_vec3(fields["normal"])
            if _all_finite(candidate_normal):
                normal = _normalized_vec3(candidate_normal)
                payload["normal"] = list(normal)
        if "gaussian_mean" in fields:
            mean = _tensor_vec3(fields["gaussian_mean"])
            if _all_finite(mean):
                payload["mean"] = list(mean)
        if "color" in fields:
            candidate_color = _tensor_vec3(fields["color"])
            if _all_finite(candidate_color):
                color = candidate_color
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


def _all_finite(values: Sequence[float]) -> bool:
    """True when every component is a finite number (no NaN/Inf)."""
    return all(isfinite(float(item)) for item in values)


def _normalized_vec3(value: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(item * item for item in value))
    if norm <= 1e-8:
        return value
    return tuple(item / norm for item in value)  # type: ignore[return-value]


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
