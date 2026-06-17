from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import sqrt
from pathlib import Path
from typing import Sequence

from aura.assignment import RegionEvidence
from aura.carrier_payloads import BetaKernelPayload, NeuralResidualPayload
from aura.decomposition import EvidenceSample, decompose_evidence
from aura.elements import AuraElement, Bounds
from aura.ray import Ray, Vec3
from aura.scene import AuraScene


@dataclass(frozen=True)
class TrainingFrame:
    """Minimal posed training observation for the AURA-Core fixture path."""

    id: str
    camera_origin: Vec3
    look_at: Vec3
    target_color: Vec3
    target_depth: float
    semantic_label: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("training frame id is required")
        if self.target_depth <= 0.0:
            raise ValueError("target_depth must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> TrainingFrame:
        if not isinstance(payload, dict):
            raise ValueError("training frame entry must be an object")
        return cls(
            id=str(payload["id"]),
            camera_origin=_vec3_from_payload(payload["camera_origin"], "camera_origin"),
            look_at=_vec3_from_payload(payload["look_at"], "look_at"),
            target_color=_vec3_from_payload(payload["target_color"], "target_color"),
            target_depth=float(payload["target_depth"]),
            semantic_label=str(payload["semantic_label"]) if payload.get("semantic_label") is not None else None,
        )


@dataclass(frozen=True)
class ReconstructionConfig:
    iterations: int = 4
    render_width: int = 8
    render_height: int = 8
    color_learning_rate: float = 0.35
    enable_adaptive_evolution: bool = True

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render dimensions must be positive")
        if not 0.0 < self.color_learning_rate <= 1.0:
            raise ValueError("color_learning_rate must be in (0, 1]")


@dataclass(frozen=True)
class FramePrediction:
    frame_id: str
    element_id: str | None
    carrier_id: str | None
    ray_direction: Vec3
    predicted_color: Vec3
    target_color: Vec3
    predicted_depth: float | None
    target_depth: float
    image_loss: float
    depth_loss: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CarrierEvolutionDecision:
    element_id: str
    carrier_id: str
    action: str
    reason: str
    created_element_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionStep:
    iteration: int
    image_loss: float
    depth_loss: float
    carrier_counts: dict[str, int]
    total_loss: float
    predictions: tuple[FramePrediction, ...]
    carrier_evolution: tuple[CarrierEvolutionDecision, ...]

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "image_loss": self.image_loss,
            "depth_loss": self.depth_loss,
            "total_loss": self.total_loss,
            "carrier_counts": dict(self.carrier_counts),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "carrier_evolution": [decision.to_dict() for decision in self.carrier_evolution],
        }


@dataclass(frozen=True)
class ReconstructionReport:
    name: str
    frames: tuple[TrainingFrame, ...]
    stages: tuple[str, ...]
    iterations: tuple[ReconstructionStep, ...]
    final_loss: float
    native_carrier_fraction: float
    sources: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "format": "AURA_CORE_RECONSTRUCTION_REPORT",
            "name": self.name,
            "frames": [frame.to_dict() for frame in self.frames],
            "stages": list(self.stages),
            "iterations": [step.to_dict() for step in self.iterations],
            "finalLoss": self.final_loss,
            "nativeCarrierFraction": self.native_carrier_fraction,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class ReconstructionResult:
    scene: AuraScene
    report: ReconstructionReport


def synthetic_training_frames() -> tuple[TrainingFrame, ...]:
    return (
        TrainingFrame(
            id="front_left",
            camera_origin=(-0.6, -0.8, -2.0),
            look_at=(-0.35, -0.35, 0.0),
            target_color=(0.78, 0.70, 0.60),
            target_depth=2.0,
            semantic_label="wall",
        ),
        TrainingFrame(
            id="front_center",
            camera_origin=(0.1, -0.8, -2.0),
            look_at=(0.1, -0.4, 0.2),
            target_color=(0.55, 0.68, 0.88),
            target_depth=2.2,
        ),
        TrainingFrame(
            id="front_right",
            camera_origin=(0.7, -0.8, -2.0),
            look_at=(0.7, -0.4, 0.05),
            target_color=(0.92, 0.82, 0.32),
            target_depth=2.0,
        ),
        TrainingFrame(
            id="object_view",
            camera_origin=(0.1, 0.8, -2.0),
            look_at=(0.1, 0.25, 0.1),
            target_color=(0.72, 0.55, 0.92),
            target_depth=2.1,
            semantic_label="fixture_object",
        ),
    )


def load_training_frames(path: Path | str) -> tuple[TrainingFrame, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames_payload = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames_payload, list) or not frames_payload:
        raise ValueError("training frames JSON must contain a non-empty frames array")
    return tuple(TrainingFrame.from_dict(item) for item in frames_payload)


def write_synthetic_training_frames(path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "format": "AURA_TRAINING_FRAMES",
                "frames": [frame.to_dict() for frame in synthetic_training_frames()],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def reconstruct_demo_scene(
    config: ReconstructionConfig | None = None,
    *,
    frames: Sequence[TrainingFrame] | None = None,
    name: str = "reconstruct_demo",
) -> ReconstructionResult:
    """Run the deterministic AURA-Core fixture reconstruction path.

    This is a CPU reference scaffold: it creates posed observations, initializes
    native evidence cells, emits mixed AURA carriers, and records loss/adaptation
    stages. It is not a production optimizer.
    """

    config = config or ReconstructionConfig()
    training_frames = tuple(frames or synthetic_training_frames())
    scene = decompose_evidence(_initial_evidence_from_frames(training_frames), name=name)
    scene, iterations = _reference_iterations(scene, training_frames, config)
    final_loss = iterations[-1].image_loss + iterations[-1].depth_loss
    non_gaussian = sum(1 for element in scene.elements if element.carrier_id != "gaussian")
    native_fraction = 0.0 if not scene.elements else non_gaussian / len(scene.elements)
    report = ReconstructionReport(
        name=scene.name,
        frames=training_frames,
        stages=(
            "posed_synthetic_input",
            "native_evidence_initialization",
            "cpu_reference_render_loss",
            "adaptive_carrier_split_promote",
            "aura_package_export_ready",
        ),
        iterations=iterations,
        final_loss=final_loss,
        native_carrier_fraction=native_fraction,
        sources=("posed_training_frames", "depth_targets", "semantic_labels"),
    )
    return ReconstructionResult(scene=scene, report=report)


def _initial_evidence_from_frames(frames: Sequence[TrainingFrame]) -> tuple[EvidenceSample, ...]:
    if len(frames) < 4:
        raise ValueError("AURA-Core demo reconstruction requires at least four posed training frames")
    by_id = {frame.id: frame for frame in frames}
    required = {"front_left", "front_center", "front_right", "object_view"}
    missing = sorted(required.difference(by_id))
    if missing:
        raise ValueError(f"training frames missing required fixture ids: {', '.join(missing)}")
    return (
        EvidenceSample(
            id="surface_wall",
            bounds=Bounds((-0.75, -0.75, 0.0), (-0.25, -0.25, 0.1)),
            evidence=RegionEvidence(geometry_confidence=0.9, material_confidence=0.7, edit_need=0.6, ray_need=0.8),
            color=by_id["front_left"].target_color,
            opacity=0.9,
            normal=(0.0, 0.0, -1.0),
            material_id="mat_wall_plaster",
            semantic_label="wall",
            metadata={"source": "aura-core-synthetic"},
        ),
        EvidenceSample(
            id="soft_volume",
            bounds=Bounds((-0.15, -0.7, 0.0), (0.35, -0.2, 0.8)),
            evidence=RegionEvidence(fuzzy_confidence=0.85, geometry_confidence=0.25),
            color=by_id["front_center"].target_color,
            opacity=0.35,
            confidence=0.72,
            material_id="mat_soft_volume",
            metadata={"source": "aura-core-synthetic"},
        ),
        EvidenceSample(
            id="woven_frequency",
            bounds=Bounds((0.45, -0.7, 0.0), (0.95, -0.2, 0.15)),
            evidence=RegionEvidence(high_frequency=0.92, geometry_confidence=0.65),
            color=by_id["front_right"].target_color,
            opacity=0.75,
            material_id="mat_woven_detail",
            metadata={"source": "aura-core-synthetic"},
        ),
        EvidenceSample(
            id="view_residual",
            bounds=Bounds((-0.75, 0.05, 0.0), (-0.25, 0.55, 0.2)),
            evidence=RegionEvidence(view_dependent=0.88, material_confidence=0.25, image_error=0.55),
            color=(0.4, 0.75, 0.7),
            opacity=0.6,
            metadata={"source": "aura-core-synthetic"},
        ),
        EvidenceSample(
            id="semantic_object",
            bounds=Bounds((-0.1, 0.05, 0.0), (0.35, 0.5, 0.2)),
            evidence=RegionEvidence(semantic_confidence=0.95),
            color=by_id["object_view"].target_color,
            opacity=0.45,
            semantic_label="fixture_object",
            edit={"selectable": True},
            metadata={"source": "aura-core-synthetic"},
        ),
        EvidenceSample(
            id="compact_detail",
            bounds=Bounds((0.5, 0.05, 0.0), (0.8, 0.35, 0.15)),
            evidence=RegionEvidence(compact_detail=0.9, image_error=0.2),
            color=(0.95, 0.45, 0.35),
            opacity=0.85,
            metadata={"source": "aura-core-synthetic"},
        ),
        EvidenceSample(
            id="gaussian_fallback",
            bounds=Bounds((0.85, 0.3, 0.0), (1.05, 0.5, 0.2)),
            evidence=RegionEvidence(image_error=0.05, geometry_confidence=0.3, edit_need=0.1),
            color=(0.65, 0.65, 0.65),
            opacity=0.5,
            confidence=0.6,
            fallback_source="aura-core-uncertain-region",
            metadata={"source": "aura-core-synthetic"},
        ),
    )


def _reference_iterations(
    scene: AuraScene,
    frames: Sequence[TrainingFrame],
    config: ReconstructionConfig,
) -> tuple[AuraScene, tuple[ReconstructionStep, ...]]:
    steps = []
    for index in range(config.iterations):
        predictions = _predict_training_frames(scene, frames)
        image_loss = sum(prediction.image_loss for prediction in predictions) / len(predictions)
        depth_loss = sum(prediction.depth_loss for prediction in predictions) / len(predictions)
        evolution = _carrier_evolution_decisions(predictions, scene) if config.enable_adaptive_evolution else tuple()
        steps.append(
            ReconstructionStep(
                iteration=index,
                image_loss=image_loss,
                depth_loss=depth_loss,
                total_loss=image_loss + depth_loss,
                carrier_counts=_carrier_counts(scene),
                predictions=predictions,
                carrier_evolution=evolution,
            )
        )
        scene = _refine_scene_from_predictions(scene, predictions, config, evolution)
    return scene, tuple(steps)


def _predict_training_frames(scene: AuraScene, frames: Sequence[TrainingFrame]) -> tuple[FramePrediction, ...]:
    predictions = []
    element_by_id = {element.id: element for element in scene.elements}
    for frame in frames:
        direction = _normalized_direction(frame.camera_origin, frame.look_at)
        ray = Ray(origin=frame.camera_origin, direction=direction)
        result = scene.ray_query(ray)
        provenance = result.provenance.split(",", 1)[0] if result.provenance else None
        element = element_by_id.get(provenance)
        image_loss = _color_mse(result.color, frame.target_color)
        depth_loss = abs((result.depth or 0.0) - frame.target_depth) if result.depth is not None else frame.target_depth
        predictions.append(
            FramePrediction(
                frame_id=frame.id,
                element_id=element.id if element is not None else None,
                carrier_id=element.carrier_id if element is not None else None,
                ray_direction=direction,
                predicted_color=result.color,
                target_color=frame.target_color,
                predicted_depth=result.depth,
                target_depth=frame.target_depth,
                image_loss=image_loss,
                depth_loss=depth_loss,
            )
        )
    return tuple(predictions)


def _carrier_evolution_decisions(
    predictions: Sequence[FramePrediction],
    scene: AuraScene,
) -> tuple[CarrierEvolutionDecision, ...]:
    decisions = []
    seen = set()
    element_ids = {element.id for element in scene.elements}
    element_by_id = {element.id: element for element in scene.elements}
    for prediction in predictions:
        if prediction.element_id is None or prediction.carrier_id is None:
            continue
        key = (prediction.element_id, prediction.carrier_id)
        if key in seen:
            continue
        seen.add(key)
        beta_child_id = _created_element_id(prediction.element_id, "split_beta_detail")
        neural_child_id = _created_element_id(prediction.element_id, "promote_neural_residual")
        element = element_by_id[prediction.element_id]
        if (
            prediction.carrier_id == "volume"
            and beta_child_id in element_ids
            and prediction.image_loss < 0.025
            and prediction.depth_loss < 0.04
        ):
            action = "merge_beta_detail"
            reason = "volume parent residual fell below split-detail threshold"
        elif (
            prediction.carrier_id == "semantic"
            and neural_child_id in element_ids
            and prediction.image_loss < 0.045
            and prediction.depth_loss < 0.02
        ):
            action = "demote_neural_residual"
            reason = "semantic residual no longer needs a neural child"
        elif prediction.image_loss > 0.03 and prediction.carrier_id in {"surface", "volume", "gabor", "semantic"}:
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
        elif prediction.depth_loss > 0.10 and prediction.carrier_id in {"surface", "volume", "semantic"}:
            action = "anchor_carrier_depth"
            reason = "depth residual exceeds reference tolerance"
        elif prediction.carrier_id == "gabor":
            action = "retain_frequency_carrier"
            reason = "high-frequency evidence is represented by a native carrier"
        elif prediction.carrier_id == "semantic":
            action = "retain_semantic_carrier"
            reason = "semantic observation remains object-addressable"
        else:
            action = "retain_carrier"
            reason = "current carrier explains fixture evidence within reference tolerance"
        decisions.append(
            CarrierEvolutionDecision(
                element_id=prediction.element_id,
                carrier_id=prediction.carrier_id,
                action=action,
                reason=reason,
                created_element_id=_created_element_id(prediction.element_id, action),
            )
        )
    return tuple(decisions)


def _refine_scene_from_predictions(
    scene: AuraScene,
    predictions: Sequence[FramePrediction],
    config: ReconstructionConfig,
    decisions: Sequence[CarrierEvolutionDecision],
) -> AuraScene:
    targets_by_element = {prediction.element_id: prediction.target_color for prediction in predictions if prediction.element_id}
    if not targets_by_element:
        return scene
    elements = []
    existing_ids = {element.id for element in scene.elements}
    active_decisions = tuple(decisions) if config.enable_adaptive_evolution else tuple()
    decision_by_element = {decision.element_id: decision for decision in active_decisions}
    removed_evolved_ids = {
        decision.created_element_id
        for decision in active_decisions
        if decision.action in {"merge_beta_detail", "demote_neural_residual"} and decision.created_element_id is not None
    }
    for element in scene.elements:
        if element.id in removed_evolved_ids:
            continue
        target = targets_by_element.get(element.id)
        if target is None:
            elements.append(element)
            continue
        prediction = next(item for item in predictions if item.element_id == element.id)
        observation_scale = _observed_color_scale(element.color, prediction.predicted_color)
        target_unattenuated = tuple(_clamp_unit(channel / observation_scale) for channel in target)
        color = tuple(
            channel + (target_channel - channel) * config.color_learning_rate
            for channel, target_channel in zip(element.color, target_unattenuated)
        )
        bounds = element.bounds
        if prediction.predicted_depth is not None and prediction.depth_loss > 0.01:
            depth_delta = (prediction.target_depth - prediction.predicted_depth) * config.color_learning_rate
            bounds = _shift_bounds_along_ray(element.bounds, prediction.ray_direction, depth_delta)
        decision = decision_by_element.get(element.id)
        elements.append(
            replace(
                element,
                bounds=bounds,
                color=color,  # type: ignore[arg-type]
                metadata={
                    **element.metadata,
                    "optimized_by": "aura-core-reference-loop",
                    **_simplification_metadata(decision),
                },
            )
        )
        if decision is not None:
            evolved = _evolved_element_for(element, decision, prediction)
            if evolved is not None and evolved.id not in existing_ids:
                elements.append(evolved)
                existing_ids.add(evolved.id)
    element_ids = tuple(element.id for element in elements)
    chunks = tuple(replace(chunk, element_ids=element_ids) for chunk in scene.chunks)
    return AuraScene(name=scene.name, elements=tuple(elements), chunks=chunks, semantic_graph=scene.semantic_graph)


def _evolved_element_for(
    element: AuraElement,
    decision: CarrierEvolutionDecision,
    prediction: FramePrediction,
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
            confidence_map={"residual": prediction.image_loss, "depth": prediction.depth_loss},
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
            confidence_map={"residual": prediction.image_loss, "depth": prediction.depth_loss},
            edit={"source": "adaptive-carrier-evolution", "parent": element.id},
            payload=NeuralResidualPayload(latent_dim=16, residual_scale=min(1.0, prediction.image_loss * 4.0)).to_dict(),
        )
    return None


def _normalized_direction(origin: Vec3, target: Vec3) -> Vec3:
    vector = tuple(target[index] - origin[index] for index in range(3))
    norm = sqrt(sum(axis * axis for axis in vector))
    if norm <= 1e-12:
        raise ValueError("camera origin and look_at must differ")
    return tuple(axis / norm for axis in vector)  # type: ignore[return-value]


def _vec3_from_payload(payload: object, name: str) -> Vec3:
    if not isinstance(payload, list | tuple) or len(payload) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return (float(payload[0]), float(payload[1]), float(payload[2]))


def _color_mse(left: Vec3, right: Vec3) -> float:
    return sum((left_channel - right_channel) ** 2 for left_channel, right_channel in zip(left, right)) / 3.0


def _observed_color_scale(element_color: Vec3, predicted_color: Vec3) -> float:
    ratios = [
        predicted / raw
        for raw, predicted in zip(element_color, predicted_color)
        if abs(raw) > 1e-6
    ]
    if not ratios:
        return 1.0
    return max(0.05, min(1.0, sum(ratios) / len(ratios)))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _shift_bounds_along_ray(bounds: Bounds, direction: Vec3, distance: float) -> Bounds:
    offset = tuple(axis * distance for axis in direction)
    return Bounds(
        min_corner=tuple(value + delta for value, delta in zip(bounds.min_corner, offset)),  # type: ignore[arg-type]
        max_corner=tuple(value + delta for value, delta in zip(bounds.max_corner, offset)),  # type: ignore[arg-type]
    )


def _created_element_id(element_id: str, action: str) -> str | None:
    if action == "split_beta_detail":
        return f"{element_id}_beta_detail"
    if action == "promote_neural_residual":
        return f"{element_id}_neural_residual"
    if action == "merge_beta_detail":
        return f"{element_id}_beta_detail"
    if action == "demote_neural_residual":
        return f"{element_id}_neural_residual"
    return None


def _simplification_metadata(decision: CarrierEvolutionDecision | None) -> dict[str, str]:
    if decision is None or decision.action not in {"merge_beta_detail", "demote_neural_residual"}:
        return {}
    return {
        "simplified_child": decision.created_element_id or "",
        "simplification": decision.action,
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


def _carrier_counts(scene: AuraScene) -> dict[str, int]:
    counts = {carrier_id: 0 for carrier_id in scene.carrier_ids()}
    for element in scene.elements:
        counts[element.carrier_id] += 1
    return counts
