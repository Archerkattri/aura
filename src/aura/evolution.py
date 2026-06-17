from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence

from aura.carrier_payloads import BetaKernelPayload, NeuralResidualPayload
from aura.elements import AuraElement, Bounds
from aura.ray import Vec3


EVOLUTION_CHILD_ACTIONS = {
    "split_beta_detail",
    "promote_neural_residual",
}
EVOLUTION_SIMPLIFICATION_ACTIONS = {
    "merge_beta_detail",
    "demote_neural_residual",
}


class EvolutionPrediction(Protocol):
    element_id: str | None
    carrier_id: str | None
    image_loss: float
    depth_loss: float
    query_loss: float
    normal_loss: float
    target_color: Vec3


@dataclass(frozen=True)
class CarrierEvolutionPolicy:
    """Deterministic thresholds for fixture-scale adaptive carrier evolution."""

    enabled: bool = True
    split_image_loss_threshold: float = 0.03
    depth_anchor_loss_threshold: float = 0.10
    merge_image_loss_threshold: float = 0.025
    merge_depth_loss_threshold: float = 0.04
    demote_after_iteration: int = 3
    demote_image_loss_threshold: float = 0.045
    demote_depth_loss_threshold: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "split_image_loss_threshold",
            "depth_anchor_loss_threshold",
            "merge_image_loss_threshold",
            "merge_depth_loss_threshold",
            "demote_image_loss_threshold",
            "demote_depth_loss_threshold",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.demote_after_iteration < 0:
            raise ValueError("demote_after_iteration must be non-negative")

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "splitImageLossThreshold": self.split_image_loss_threshold,
            "depthAnchorLossThreshold": self.depth_anchor_loss_threshold,
            "mergeImageLossThreshold": self.merge_image_loss_threshold,
            "mergeDepthLossThreshold": self.merge_depth_loss_threshold,
            "demoteAfterIteration": self.demote_after_iteration,
            "demoteImageLossThreshold": self.demote_image_loss_threshold,
            "demoteDepthLossThreshold": self.demote_depth_loss_threshold,
        }


@dataclass(frozen=True)
class CarrierEvolutionDecision:
    element_id: str
    carrier_id: str
    action: str
    reason: str
    created_element_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def carrier_evolution_decisions(
    predictions: Sequence[EvolutionPrediction],
    elements: Sequence[AuraElement],
    *,
    policy: CarrierEvolutionPolicy,
    iteration: int,
) -> tuple[CarrierEvolutionDecision, ...]:
    """Classify one deterministic evolution action per predicted element."""

    decisions = []
    seen = set()
    element_ids = {element.id for element in elements}
    element_by_id = {element.id: element for element in elements}
    for prediction in predictions:
        if prediction.element_id is None or prediction.carrier_id is None:
            continue
        if prediction.element_id not in element_by_id:
            continue
        key = (prediction.element_id, prediction.carrier_id)
        if key in seen:
            continue
        seen.add(key)
        decisions.append(
            _classify_prediction(
                prediction,
                element_by_id[prediction.element_id],
                element_ids=element_ids,
                policy=policy,
                iteration=iteration,
            )
        )
    return tuple(decisions)


def carrier_evolution_report(decisions: Sequence[CarrierEvolutionDecision]) -> dict:
    """Stable summary of evolution actions for reconstruction reports."""

    action_counts: dict[str, int] = {}
    created_ids = []
    removed_ids = []
    retained_ids = []
    for decision in decisions:
        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        if decision.action in EVOLUTION_CHILD_ACTIONS and decision.created_element_id is not None:
            created_ids.append(decision.created_element_id)
        elif decision.action in EVOLUTION_SIMPLIFICATION_ACTIONS and decision.created_element_id is not None:
            removed_ids.append(decision.created_element_id)
        elif decision.action.startswith("retain"):
            retained_ids.append(decision.element_id)
    return {
        "actionCounts": dict(sorted(action_counts.items())),
        "createdElementIds": sorted(created_ids),
        "removedElementIds": sorted(removed_ids),
        "retainedElementIds": sorted(retained_ids),
    }


def evolved_element_for(
    element: AuraElement,
    decision: CarrierEvolutionDecision,
    prediction: EvolutionPrediction,
) -> AuraElement | None:
    if decision.action == "split_beta_detail":
        bounds = _shrink_bounds(element.bounds, scale=0.45)
        return AuraElement(
            id=decision.created_element_id or f"{element.id}_beta_detail",
            carrier_id="beta",
            bounds=bounds,
            color=prediction.target_color,
            opacity=min(1.0, max(0.35, element.opacity * 0.85)),
            confidence=min(1.0, element.confidence + 0.08),
            material_id=element.material_id,
            lod=element.lod + 1,
            metadata={
                "source": "aura-core-adaptive-evolution",
                "parent": element.id,
                "evolution": decision.action,
            },
            confidence_map=evolved_confidence_map(prediction),
            edit={"source": "adaptive-carrier-evolution", "parent": element.id},
            payload=BetaKernelPayload(alpha=3.0, beta=3.0, support_radius=_half_extent(bounds)).to_dict(),
        )
    if decision.action == "promote_neural_residual":
        bounds = _shrink_bounds(element.bounds, scale=0.65)
        return AuraElement(
            id=decision.created_element_id or f"{element.id}_neural_residual",
            carrier_id="neural",
            bounds=bounds,
            color=prediction.target_color,
            opacity=min(1.0, max(0.25, element.opacity * 0.75)),
            confidence=max(0.1, element.confidence * 0.9),
            semantic_id=element.semantic_id,
            residual=True,
            lod=element.lod + 1,
            metadata={
                "source": "aura-core-adaptive-evolution",
                "parent": element.id,
                "evolution": decision.action,
            },
            confidence_map=evolved_confidence_map(prediction),
            edit={"source": "adaptive-carrier-evolution", "parent": element.id},
            payload=NeuralResidualPayload(latent_dim=16, residual_scale=min(1.0, prediction.image_loss * 4.0)).to_dict(),
        )
    return None


def created_element_id(element_id: str, action: str) -> str | None:
    if action in {"split_beta_detail", "merge_beta_detail"}:
        return f"{element_id}_beta_detail"
    if action in {"promote_neural_residual", "demote_neural_residual"}:
        return f"{element_id}_neural_residual"
    return None


def simplification_metadata(decision: CarrierEvolutionDecision | None) -> dict[str, str]:
    if decision is None or decision.action not in EVOLUTION_SIMPLIFICATION_ACTIONS:
        return {}
    return {
        "simplified_child": decision.created_element_id or "",
        "simplification": decision.action,
    }


def updated_confidence_map(element: AuraElement, prediction: EvolutionPrediction) -> dict[str, float]:
    return {
        **element.confidence_map,
        "optimization_image_loss": _clamp_unit(prediction.image_loss),
        "optimization_depth_loss": _clamp_unit(prediction.depth_loss),
        "optimization_query_loss": _clamp_unit(prediction.query_loss),
        "optimization_normal_loss": _clamp_unit(prediction.normal_loss),
        "optimization_residual": prediction_residual(prediction),
    }


def evolved_confidence_map(prediction: EvolutionPrediction) -> dict[str, float]:
    return {
        "residual": _clamp_unit(prediction.image_loss),
        "depth": _clamp_unit(prediction.depth_loss),
        "query": _clamp_unit(prediction.query_loss),
        "normal": _clamp_unit(prediction.normal_loss),
        "optimization_residual": prediction_residual(prediction),
    }


def refined_confidence(confidence: float, prediction: EvolutionPrediction, *, learning_rate: float) -> float:
    residual = prediction_residual(prediction)
    target = 1.0 - residual
    return _clamp_unit(confidence + (target - confidence) * min(1.0, learning_rate))


def prediction_residual(prediction: EvolutionPrediction) -> float:
    return _clamp_unit(
        prediction.image_loss
        + prediction.depth_loss * 0.25
        + prediction.query_loss
        + prediction.normal_loss * 0.5
    )


def _classify_prediction(
    prediction: EvolutionPrediction,
    element: AuraElement,
    *,
    element_ids: set[str],
    policy: CarrierEvolutionPolicy,
    iteration: int,
) -> CarrierEvolutionDecision:
    beta_child_id = created_element_id(prediction.element_id or "", "split_beta_detail")
    neural_child_id = created_element_id(prediction.element_id or "", "promote_neural_residual")
    metrics = _prediction_metrics(prediction)
    if (
        prediction.carrier_id == "volume"
        and beta_child_id in element_ids
        and prediction.image_loss < policy.merge_image_loss_threshold
        and prediction.depth_loss < policy.merge_depth_loss_threshold
    ):
        action = "merge_beta_detail"
        reason = "volume parent residual fell below split-detail threshold"
        thresholds = _thresholds(policy, "merge")
    elif (
        prediction.carrier_id == "semantic"
        and neural_child_id in element_ids
        and iteration >= policy.demote_after_iteration
        and prediction.image_loss < policy.demote_image_loss_threshold
        and prediction.depth_loss < policy.demote_depth_loss_threshold
    ):
        action = "demote_neural_residual"
        reason = "semantic residual no longer needs a neural child"
        thresholds = _thresholds(policy, "demote")
    elif prediction.image_loss > policy.split_image_loss_threshold and prediction.carrier_id in {"surface", "volume", "gabor", "semantic"}:
        thresholds = _thresholds(policy, "split")
        if prediction.carrier_id == "volume":
            if element.metadata.get("simplified_child") == beta_child_id:
                action = "retain_carrier"
                reason = "merged beta detail remains below re-split hysteresis"
            else:
                action = "split_beta_detail"
                reason = "volume residual benefits from compact bounded support"
        elif prediction.carrier_id == "semantic":
            if element.metadata.get("simplified_child") == neural_child_id:
                action = "retain_semantic_carrier"
                reason = "demoted neural residual remains below re-promote hysteresis"
            else:
                action = "promote_neural_residual"
                reason = "semantic object retains view-dependent photometric residual"
        else:
            action = "refine_radiance"
            reason = "photometric residual above native carrier threshold"
    elif prediction.depth_loss > policy.depth_anchor_loss_threshold and prediction.carrier_id in {"surface", "volume", "semantic"}:
        action = "anchor_carrier_depth"
        reason = "depth residual exceeds reference tolerance"
        thresholds = _thresholds(policy, "anchor")
    elif prediction.carrier_id == "gabor":
        action = "retain_frequency_carrier"
        reason = "high-frequency evidence is represented by a native carrier"
        thresholds = {}
    elif prediction.carrier_id == "semantic":
        action = "retain_semantic_carrier"
        reason = "semantic observation remains object-addressable"
        thresholds = {}
    else:
        action = "retain_carrier"
        reason = "current carrier explains fixture evidence within reference tolerance"
        thresholds = {}
    return CarrierEvolutionDecision(
        element_id=prediction.element_id or "",
        carrier_id=prediction.carrier_id or "",
        action=action,
        reason=reason,
        created_element_id=created_element_id(prediction.element_id or "", action),
        metrics=metrics,
        thresholds=thresholds,
    )


def _thresholds(policy: CarrierEvolutionPolicy, action_family: str) -> dict[str, float]:
    if action_family == "split":
        return {"splitImageLossThreshold": policy.split_image_loss_threshold}
    if action_family == "anchor":
        return {"depthAnchorLossThreshold": policy.depth_anchor_loss_threshold}
    if action_family == "merge":
        return {
            "mergeImageLossThreshold": policy.merge_image_loss_threshold,
            "mergeDepthLossThreshold": policy.merge_depth_loss_threshold,
        }
    if action_family == "demote":
        return {
            "demoteAfterIteration": float(policy.demote_after_iteration),
            "demoteImageLossThreshold": policy.demote_image_loss_threshold,
            "demoteDepthLossThreshold": policy.demote_depth_loss_threshold,
        }
    return {}


def _prediction_metrics(prediction: EvolutionPrediction) -> dict[str, float]:
    return {
        "imageLoss": prediction.image_loss,
        "depthLoss": prediction.depth_loss,
        "queryLoss": prediction.query_loss,
        "normalLoss": prediction.normal_loss,
        "residual": prediction_residual(prediction),
    }


def _shrink_bounds(bounds: Bounds, *, scale: float) -> Bounds:
    center = tuple((lo + hi) / 2.0 for lo, hi in zip(bounds.min_corner, bounds.max_corner))
    half = tuple((hi - lo) * scale / 2.0 for lo, hi in zip(bounds.min_corner, bounds.max_corner))
    return Bounds(
        min_corner=tuple(value - radius for value, radius in zip(center, half)),  # type: ignore[arg-type]
        max_corner=tuple(value + radius for value, radius in zip(center, half)),  # type: ignore[arg-type]
    )


def _half_extent(bounds: Bounds) -> Vec3:
    return tuple(max((hi - lo) / 2.0, 1e-4) for lo, hi in zip(bounds.min_corner, bounds.max_corner))  # type: ignore[return-value]


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
