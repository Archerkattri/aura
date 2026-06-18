"""AURA-Core training frame/region contracts, reconstruction config, and reference reconstruction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from importlib import resources
from math import sqrt
from pathlib import Path
from typing import Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aura.assignment import RegionEvidence
from aura.decomposition import EvidenceSample, carrier_lod_elements_and_chunks, decompose_evidence
from aura.elements import AuraElement, Bounds
from aura.evolution import (
    CarrierEvolutionDecision,
    CarrierEvolutionPolicy,
    carrier_evolution_decisions,
    carrier_evolution_report,
    evolved_element_for,
    refined_confidence,
    simplification_metadata,
    updated_confidence_map,
)
from aura.optimize import RenderTarget, differentiate_scene_rays, gradient_descent_color_step, precondition_color_gradient
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
    image_path: str | None = None
    depth_path: str | None = None
    mask_path: str | None = None
    normal_path: str | None = None
    camera_model: str | None = None
    intrinsics: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("training frame id is required")
        if self.target_depth <= 0.0:
            raise ValueError("target_depth must be positive")
        if self.intrinsics is not None:
            required = {"fx", "fy", "cx", "cy", "width", "height"}
            missing = sorted(required.difference(self.intrinsics))
            if missing:
                raise ValueError(f"training frame intrinsics missing keys: {', '.join(missing)}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_origin": list(self.camera_origin),
            "look_at": list(self.look_at),
            "target_color": list(self.target_color),
            "target_depth": self.target_depth,
            "semantic_label": self.semantic_label,
            "image_path": self.image_path,
            "depth_path": self.depth_path,
            "mask_path": self.mask_path,
            "normal_path": self.normal_path,
            "camera_model": self.camera_model,
            "intrinsics": dict(self.intrinsics) if self.intrinsics is not None else None,
        }

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
            image_path=str(payload["image_path"]) if payload.get("image_path") is not None else None,
            depth_path=str(payload["depth_path"]) if payload.get("depth_path") is not None else None,
            mask_path=str(payload["mask_path"]) if payload.get("mask_path") is not None else None,
            normal_path=str(payload["normal_path"]) if payload.get("normal_path") is not None else None,
            camera_model=str(payload["camera_model"]) if payload.get("camera_model") is not None else None,
            intrinsics={key: float(value) for key, value in payload["intrinsics"].items()}
            if payload.get("intrinsics") is not None
            else None,
        )


@dataclass(frozen=True)
class TrainingRegion:
    """Native evidence region used to initialize AURA carriers from frame data."""

    id: str
    frame_id: str
    bounds: Bounds
    evidence: RegionEvidence
    color: Vec3 | None = None
    opacity: float = 1.0
    confidence: float = 1.0
    normal: Vec3 | None = None
    material_id: str | None = None
    semantic_label: str | None = None
    fallback_source: str = "aura-core-training-region"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("training region id is required")
        if not self.frame_id:
            raise ValueError("training region frame_id is required")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("training region opacity must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("training region confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "frame_id": self.frame_id,
            "bounds": {"min": list(self.bounds.min_corner), "max": list(self.bounds.max_corner)},
            "evidence": asdict(self.evidence),
            "color": list(self.color) if self.color is not None else None,
            "opacity": self.opacity,
            "confidence": self.confidence,
            "normal": list(self.normal) if self.normal is not None else None,
            "material_id": self.material_id,
            "semantic_label": self.semantic_label,
            "fallback_source": self.fallback_source,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TrainingRegion:
        if not isinstance(payload, dict):
            raise ValueError("training region entry must be an object")
        bounds_payload = payload["bounds"]
        if not isinstance(bounds_payload, dict):
            raise ValueError("training region bounds must be an object")
        evidence_payload = payload.get("evidence", {})
        if not isinstance(evidence_payload, dict):
            raise ValueError("training region evidence must be an object")
        return cls(
            id=str(payload["id"]),
            frame_id=str(payload["frame_id"]),
            bounds=Bounds(
                min_corner=_vec3_from_payload(bounds_payload["min"], "bounds.min"),
                max_corner=_vec3_from_payload(bounds_payload["max"], "bounds.max"),
            ),
            evidence=RegionEvidence(**{key: float(value) for key, value in evidence_payload.items()}),
            color=_vec3_from_payload(payload["color"], "color") if payload.get("color") is not None else None,
            opacity=float(payload.get("opacity", 1.0)),
            confidence=float(payload.get("confidence", 1.0)),
            normal=_vec3_from_payload(payload["normal"], "normal") if payload.get("normal") is not None else None,
            material_id=str(payload["material_id"]) if payload.get("material_id") is not None else None,
            semantic_label=str(payload["semantic_label"]) if payload.get("semantic_label") is not None else None,
            fallback_source=str(payload.get("fallback_source", "aura-core-training-region")),
        )

    def to_evidence_sample(self, frame: TrainingFrame) -> EvidenceSample:
        return EvidenceSample(
            id=self.id,
            bounds=self.bounds,
            evidence=self.evidence,
            color=self.color or frame.target_color,
            opacity=self.opacity,
            confidence=self.confidence,
            normal=self.normal,
            material_id=self.material_id,
            semantic_label=self.semantic_label,
            fallback_source=self.fallback_source,
            metadata={"source": "aura-core-training-region", "frame_id": self.frame_id},
        )


@dataclass(frozen=True)
class TrainingDataset:
    """A matched set of posed training frames and their associated evidence regions."""

    frames: tuple[TrainingFrame, ...]
    regions: tuple[TrainingRegion, ...]

    def to_dict(self) -> dict:
        return {
            "format": "AURA_TRAINING_FRAMES",
            "frames": [frame.to_dict() for frame in self.frames],
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration for the AURA-Core reference reconstruction loop."""

    iterations: int = 4
    render_width: int = 8
    render_height: int = 8
    color_learning_rate: float = 0.35
    render_backend: str = "cpu"
    torch_device: str | None = None
    require_cuda: bool = False
    enable_adaptive_evolution: bool = True
    split_image_loss_threshold: float = 0.03
    depth_anchor_loss_threshold: float = 0.10
    merge_image_loss_threshold: float = 0.025
    merge_depth_loss_threshold: float = 0.04
    demote_after_iteration: int = 3
    demote_image_loss_threshold: float = 0.045
    demote_depth_loss_threshold: float = 0.02

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render dimensions must be positive")
        if not 0.0 < self.color_learning_rate <= 1.0:
            raise ValueError("color_learning_rate must be in (0, 1]")
        if self.render_backend not in {"cpu", "torch", "auto"}:
            raise ValueError("render_backend must be one of: cpu, torch, auto")
        if self.require_cuda and self.render_backend == "cpu":
            raise ValueError("require_cuda needs render_backend='torch' or 'auto'")
        self.to_evolution_policy()

    def to_evolution_policy(self) -> CarrierEvolutionPolicy:
        return CarrierEvolutionPolicy(
            enabled=self.enable_adaptive_evolution,
            split_image_loss_threshold=self.split_image_loss_threshold,
            depth_anchor_loss_threshold=self.depth_anchor_loss_threshold,
            merge_image_loss_threshold=self.merge_image_loss_threshold,
            merge_depth_loss_threshold=self.merge_depth_loss_threshold,
            demote_after_iteration=self.demote_after_iteration,
            demote_image_loss_threshold=self.demote_image_loss_threshold,
            demote_depth_loss_threshold=self.demote_depth_loss_threshold,
        )

    def evolution_policy(self) -> dict:
        return self.to_evolution_policy().to_dict()

    def rendering_policy(self) -> dict:
        return {
            "requestedBackend": self.render_backend,
            "requestedDevice": self.torch_device,
            "requireCuda": self.require_cuda,
        }


@dataclass(frozen=True)
class FramePrediction:
    """Predicted outputs and per-frame losses for a single training frame ray."""

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
    color_jacobian: float = 0.0
    color_gradient: Vec3 = (0.0, 0.0, 0.0)
    depth_gradient: float = 0.0
    gradient_norm: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReconstructionStep:
    """Aggregated per-iteration losses, carrier counts, and evolution decisions."""

    iteration: int
    render_backend: str
    render_device: str | None
    image_loss: float
    depth_loss: float
    query_loss: float
    normal_loss: float
    carrier_counts: dict[str, int]
    total_loss: float
    predictions: tuple[FramePrediction, ...]
    carrier_evolution: tuple[CarrierEvolutionDecision, ...]

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "render_backend": self.render_backend,
            "render_device": self.render_device,
            "image_loss": self.image_loss,
            "depth_loss": self.depth_loss,
            "query_loss": self.query_loss,
            "normal_loss": self.normal_loss,
            "total_loss": self.total_loss,
            "carrier_counts": dict(self.carrier_counts),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "carrier_evolution": [decision.to_dict() for decision in self.carrier_evolution],
            "carrier_evolution_report": carrier_evolution_report(self.carrier_evolution),
        }


@dataclass(frozen=True)
class ReconstructionReport:
    """Full reconstruction history, loss trajectory, and policy records."""

    name: str
    frames: tuple[TrainingFrame, ...]
    stages: tuple[str, ...]
    iterations: tuple[ReconstructionStep, ...]
    final_loss: float
    native_carrier_fraction: float
    sources: tuple[str, ...]
    evolution_policy: dict
    rendering_policy: dict
    render_backend: str
    render_device: str | None

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
            "evolutionPolicy": dict(self.evolution_policy),
            "renderingPolicy": dict(self.rendering_policy),
            "renderBackend": self.render_backend,
            "renderDevice": self.render_device,
        }


@dataclass(frozen=True)
class ReconstructionResult:
    """Output of a reconstruction run: the final scene and its full report."""

    scene: AuraScene
    report: ReconstructionReport


def synthetic_training_frames() -> tuple[TrainingFrame, ...]:
    """Return the deterministic synthetic posed training frames for the fixture scene."""
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


def synthetic_training_regions() -> tuple[TrainingRegion, ...]:
    """Return the deterministic synthetic evidence regions for the fixture scene."""
    return (
        TrainingRegion(
            id="surface_wall",
            frame_id="front_left",
            bounds=Bounds((-0.75, -0.75, 0.0), (-0.25, -0.25, 0.1)),
            evidence=RegionEvidence(geometry_confidence=0.9, material_confidence=0.7, edit_need=0.6, ray_need=0.8),
            opacity=0.9,
            normal=(0.0, 0.0, -1.0),
            material_id="mat_wall_plaster",
            semantic_label="wall",
        ),
        TrainingRegion(
            id="soft_volume",
            frame_id="front_center",
            bounds=Bounds((-0.15, -0.7, 0.0), (0.35, -0.2, 0.8)),
            evidence=RegionEvidence(fuzzy_confidence=0.85, geometry_confidence=0.25),
            opacity=0.35,
            confidence=0.72,
            material_id="mat_soft_volume",
        ),
        TrainingRegion(
            id="woven_frequency",
            frame_id="front_right",
            bounds=Bounds((0.45, -0.7, 0.0), (0.95, -0.2, 0.15)),
            evidence=RegionEvidence(high_frequency=0.92, geometry_confidence=0.65),
            opacity=0.75,
            material_id="mat_woven_detail",
        ),
        TrainingRegion(
            id="view_residual",
            frame_id="front_left",
            bounds=Bounds((-0.75, 0.05, 0.0), (-0.25, 0.55, 0.2)),
            evidence=RegionEvidence(view_dependent=0.88, material_confidence=0.25, image_error=0.55),
            color=(0.4, 0.75, 0.7),
            opacity=0.6,
        ),
        TrainingRegion(
            id="semantic_object",
            frame_id="object_view",
            bounds=Bounds((-0.1, 0.05, 0.0), (0.35, 0.5, 0.2)),
            evidence=RegionEvidence(semantic_confidence=0.95),
            opacity=0.45,
            semantic_label="fixture_object",
        ),
        TrainingRegion(
            id="compact_detail",
            frame_id="front_right",
            bounds=Bounds((0.5, 0.05, 0.0), (0.8, 0.35, 0.15)),
            evidence=RegionEvidence(compact_detail=0.9, image_error=0.2),
            color=(0.95, 0.45, 0.35),
            opacity=0.85,
        ),
        TrainingRegion(
            id="gaussian_fallback",
            frame_id="front_right",
            bounds=Bounds((0.85, 0.3, 0.0), (1.05, 0.5, 0.2)),
            evidence=RegionEvidence(image_error=0.05, geometry_confidence=0.3, edit_need=0.1),
            color=(0.65, 0.65, 0.65),
            opacity=0.5,
            confidence=0.6,
            fallback_source="aura-core-uncertain-region",
        ),
    )


def synthetic_training_dataset() -> TrainingDataset:
    """Return the combined deterministic synthetic training dataset."""
    return TrainingDataset(frames=synthetic_training_frames(), regions=synthetic_training_regions())


def load_training_dataset(path: Path | str) -> TrainingDataset:
    """Load and validate a JSON training dataset from ``path``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training dataset JSON must be an object")
    validate_training_dataset_document(payload)
    frames_payload = payload["frames"]
    regions_payload = payload["regions"]
    dataset = TrainingDataset(
        frames=tuple(TrainingFrame.from_dict(item) for item in frames_payload),
        regions=tuple(TrainingRegion.from_dict(item) for item in regions_payload),
    )
    _validate_training_dataset_links(dataset)
    return dataset


def load_training_frames(path: Path | str) -> tuple[TrainingFrame, ...]:
    """Load and return only the frames from a JSON training dataset."""
    return load_training_dataset(path).frames


def write_synthetic_training_frames(path: Path | str) -> Path:
    """Write the deterministic synthetic training dataset to ``path`` as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "format": "AURA_TRAINING_FRAMES",
                **synthetic_training_dataset().to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def validate_training_dataset_document(payload: dict) -> None:
    """Validate the native AURA-Core training dataset JSON contract."""

    _validate_json_schema("training_dataset.schema.json", payload)


def _validate_json_schema(schema_name: str, payload: object) -> None:
    schema_path = resources.files("aura.schemas").joinpath(schema_name)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    try:
        validator.validate(payload)
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.absolute_path)
        location = f" at {path}" if path else ""
        raise ValueError(f"{schema_name} validation failed{location}: {exc.message}") from exc


def _validate_training_dataset_links(dataset: TrainingDataset) -> None:
    frame_ids = {frame.id for frame in dataset.frames}
    if len(frame_ids) != len(dataset.frames):
        raise ValueError("training dataset contains duplicate frame ids")
    region_ids = {region.id for region in dataset.regions}
    if len(region_ids) != len(dataset.regions):
        raise ValueError("training dataset contains duplicate region ids")
    missing = sorted({region.frame_id for region in dataset.regions}.difference(frame_ids))
    if missing:
        raise ValueError(f"training regions reference unknown frame ids: {', '.join(missing)}")


def reconstruct_demo_scene(
    config: ReconstructionConfig | None = None,
    *,
    frames: Sequence[TrainingFrame] | None = None,
    regions: Sequence[TrainingRegion] | None = None,
    render_targets: Sequence[RenderTarget] | None = None,
    name: str = "reconstruct_demo",
) -> ReconstructionResult:
    """Run the deterministic AURA-Core fixture reconstruction path.

    This is a CPU reference implementation: it creates posed observations, initializes
    native evidence cells, emits mixed AURA carriers, and records loss/adaptation
    stages. It is not a production optimizer.
    """

    config = config or ReconstructionConfig()
    training_frames = tuple(frames or synthetic_training_frames())
    training_regions = tuple(regions or synthetic_training_regions())
    training_targets = tuple(render_targets or ())
    _validate_render_targets(training_frames, training_targets)
    scene = decompose_evidence(_initial_evidence_from_regions(training_frames, training_regions), name=name)
    render_backend, render_device = _resolve_reconstruction_render_backend(config)
    scene, iterations = _reference_iterations(
        scene,
        training_frames,
        config,
        render_targets=training_targets or None,
        render_backend=render_backend,
        render_device=render_device,
    )
    final_loss = iterations[-1].image_loss + iterations[-1].depth_loss + iterations[-1].query_loss + iterations[-1].normal_loss
    non_gaussian = sum(1 for element in scene.elements if element.carrier_id != "gaussian")
    native_fraction = 0.0 if not scene.elements else non_gaussian / len(scene.elements)
    stages = (
        "posed_synthetic_input",
        "native_evidence_initialization",
    )
    if training_targets:
        stages += ("capture_tensor_pixel_targets",)
    if render_backend == "torch":
        stages += ("torch_native_tensor_render",)
    else:
        stages += ("cpu_differentiable_reference_render",)
    stages += ("adaptive_carrier_split_promote", "aura_package_export_ready")
    sources = (
        "posed_training_frames",
        "training_regions",
        "depth_targets",
        "semantic_labels",
    )
    if training_targets:
        sources += ("capture_tensor_pixels",)
    report = ReconstructionReport(
        name=scene.name,
        frames=training_frames,
        stages=stages,
        iterations=iterations,
        final_loss=final_loss,
        native_carrier_fraction=native_fraction,
        sources=sources,
        evolution_policy=config.evolution_policy(),
        rendering_policy=config.rendering_policy(),
        render_backend=render_backend,
        render_device=render_device,
    )
    return ReconstructionResult(scene=scene, report=report)


def _initial_evidence_from_regions(
    frames: Sequence[TrainingFrame],
    regions: Sequence[TrainingRegion],
) -> tuple[EvidenceSample, ...]:
    if not frames:
        raise ValueError("AURA-Core reconstruction requires at least one posed training frame")
    if not regions:
        raise ValueError("AURA-Core reconstruction requires at least one training region")
    by_id = {frame.id: frame for frame in frames}
    missing = sorted({region.frame_id for region in regions}.difference(by_id))
    if missing:
        raise ValueError(f"training regions reference unknown frame ids: {', '.join(missing)}")
    return tuple(region.to_evidence_sample(by_id[region.frame_id]) for region in regions)


def _validate_render_targets(frames: Sequence[TrainingFrame], render_targets: Sequence[RenderTarget]) -> None:
    frame_ids = {frame.id for frame in frames}
    missing = sorted({target.frame_id for target in render_targets}.difference(frame_ids))
    if missing:
        raise ValueError(f"render targets reference unknown frame ids: {', '.join(missing)}")


def _reference_iterations(
    scene: AuraScene,
    frames: Sequence[TrainingFrame],
    config: ReconstructionConfig,
    *,
    render_targets: Sequence[RenderTarget] | None = None,
    render_backend: str,
    render_device: str | None,
) -> tuple[AuraScene, tuple[ReconstructionStep, ...]]:
    steps = []
    for index in range(config.iterations):
        predictions = _predict_training_frames(
            scene,
            frames,
            render_targets=render_targets,
            render_backend=render_backend,
            render_device=render_device,
        )
        image_loss = sum(prediction.image_loss for prediction in predictions) / len(predictions)
        depth_loss = sum(prediction.depth_loss for prediction in predictions) / len(predictions)
        query_loss = sum(prediction.query_loss for prediction in predictions) / len(predictions)
        normal_loss = sum(prediction.normal_loss for prediction in predictions) / len(predictions)
        evolution = _carrier_evolution_decisions(predictions, scene, config=config, iteration=index) if config.enable_adaptive_evolution else tuple()
        steps.append(
            ReconstructionStep(
                iteration=index,
                render_backend=render_backend,
                render_device=render_device,
                image_loss=image_loss,
                depth_loss=depth_loss,
                query_loss=query_loss,
                normal_loss=normal_loss,
                total_loss=image_loss + depth_loss + query_loss + normal_loss,
                carrier_counts=_carrier_counts(scene),
                predictions=predictions,
                carrier_evolution=evolution,
            )
        )
        scene = _refine_scene_from_predictions(scene, predictions, config, evolution)
    return scene, tuple(steps)


def _predict_training_frames(
    scene: AuraScene,
    frames: Sequence[TrainingFrame],
    *,
    render_targets: Sequence[RenderTarget] | None = None,
    render_backend: str = "cpu",
    render_device: str | None = None,
) -> tuple[FramePrediction, ...]:
    targets = tuple(render_targets) if render_targets is not None else tuple(
        RenderTarget(
            frame_id=frame.id,
            ray=Ray(origin=frame.camera_origin, direction=_normalized_direction(frame.camera_origin, frame.look_at)),
            target_color=frame.target_color,
            target_depth=frame.target_depth,
            target_semantic_id=frame.semantic_label,
        )
        for frame in frames
    )
    if render_backend == "torch":
        return _predict_training_frames_torch(scene, frames, targets, device=render_device)
    samples = differentiate_scene_rays(scene, targets)
    predictions = []
    by_frame = {frame.id: frame for frame in frames}
    for sample in samples:
        frame = by_frame[sample.frame_id]
        predictions.append(
            FramePrediction(
                frame_id=frame.id,
                element_id=sample.element_id,
                carrier_id=sample.carrier_id,
                ray_direction=sample.ray_direction,
                predicted_color=sample.predicted_color,
                target_color=sample.target_color,
                predicted_depth=sample.predicted_depth,
                target_depth=sample.target_depth,
                target_semantic_id=sample.target_semantic_id,
                target_material_id=sample.target_material_id,
                target_normal=sample.target_normal,
                predicted_transmittance=sample.predicted_transmittance,
                predicted_opacity=sample.predicted_opacity,
                predicted_confidence=sample.predicted_confidence,
                predicted_normal=sample.predicted_normal,
                predicted_material_id=sample.predicted_material_id,
                predicted_semantic_id=sample.predicted_semantic_id,
                predicted_residual=sample.predicted_residual,
                predicted_provenance=sample.predicted_provenance,
                image_loss=sample.image_loss,
                depth_loss=sample.depth_loss,
                query_loss=sample.query_loss,
                normal_loss=sample.normal_loss,
                color_jacobian=sample.color_jacobian,
                color_gradient=sample.color_gradient,
                depth_gradient=sample.depth_gradient,
                gradient_norm=sample.gradient_norm,
            )
        )
    return tuple(predictions)


def _predict_training_frames_torch(
    scene: AuraScene,
    frames: Sequence[TrainingFrame],
    targets: Sequence[RenderTarget],
    *,
    device: str | None,
) -> tuple[FramePrediction, ...]:
    from aura.torch_renderer import torch_render_tensor_targets

    batch = torch_render_tensor_targets(
        scene,
        frame_ids=tuple(target.frame_id for target in targets),
        ray_origins=tuple(target.ray.origin for target in targets),
        ray_directions=tuple(target.ray.direction for target in targets),
        target_colors=tuple(target.target_color for target in targets),
        target_depths=tuple(target.target_depth for target in targets),
        target_normals=tuple(target.target_normal if target.target_normal is not None else (0.0, 0.0, 0.0) for target in targets),
        target_normal_present=tuple(target.target_normal is not None for target in targets),
        target_confidence=tuple(target.target_confidence if target.target_confidence is not None else 0.0 for target in targets),
        target_confidence_present=tuple(target.target_confidence is not None for target in targets),
        target_semantic_ids=tuple(target.target_semantic_id for target in targets),
        target_material_ids=tuple(target.target_material_id for target in targets),
        device=device,
    )
    predictions = []
    by_frame = {frame.id: frame for frame in frames}
    for index, frame_id in enumerate(batch.frame_ids):
        frame = by_frame[frame_id]
        predicted_depth = batch.predicted_depth[index]
        predictions.append(
            FramePrediction(
                frame_id=frame.id,
                element_id=batch.element_ids[index],
                carrier_id=batch.carrier_ids[index],
                ray_direction=targets[index].ray.direction,
                predicted_color=batch.predicted_color[index],
                target_color=batch.target_color[index],
                predicted_depth=predicted_depth,
                target_depth=batch.target_depth[index],
                target_semantic_id=batch.target_semantic_ids[index],
                target_material_id=batch.target_material_ids[index],
                target_normal=batch.target_normal[index],
                predicted_transmittance=batch.transmittance[index],
                predicted_opacity=batch.opacity[index],
                predicted_confidence=batch.confidence[index],
                predicted_normal=batch.normal[index],
                predicted_material_id=batch.material_ids[index],
                predicted_semantic_id=batch.semantic_ids[index],
                predicted_residual=batch.residual[index],
                predicted_provenance=batch.provenance[index],
                image_loss=batch.image_loss[index],
                depth_loss=batch.depth_loss[index],
                query_loss=batch.query_loss[index],
                normal_loss=batch.normal_loss[index],
                color_jacobian=max(batch.opacity[index], 1e-4),
                color_gradient=_color_gradient(batch.predicted_color[index], batch.target_color[index]),
                depth_gradient=0.0 if predicted_depth is None else predicted_depth - batch.target_depth[index],
                gradient_norm=_gradient_norm(
                    _color_gradient(batch.predicted_color[index], batch.target_color[index]),
                    0.0 if predicted_depth is None else predicted_depth - batch.target_depth[index],
                ),
            )
        )
    return tuple(predictions)


def _resolve_reconstruction_render_backend(config: ReconstructionConfig) -> tuple[str, str | None]:
    if config.render_backend == "cpu":
        return "cpu", None
    from aura.torch_renderer import torch_renderer_status

    status = torch_renderer_status()
    if config.render_backend == "auto" and not status.available:
        if config.require_cuda:
            raise RuntimeError("CUDA reconstruction was required, but torch is unavailable")
        return "cpu", None
    if not status.available:
        raise RuntimeError(status.reason or "PyTorch renderer is unavailable")
    if config.require_cuda and not status.cuda_available:
        raise RuntimeError("CUDA reconstruction was required, but torch.cuda is unavailable")
    device = config.torch_device or status.default_device or "cpu"
    if config.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"CUDA reconstruction was required, but requested device is {device!r}")
    return "torch", str(device)


def _carrier_evolution_decisions(
    predictions: Sequence[FramePrediction],
    scene: AuraScene,
    *,
    config: ReconstructionConfig,
    iteration: int,
) -> tuple[CarrierEvolutionDecision, ...]:
    return carrier_evolution_decisions(
        predictions,
        scene.elements,
        policy=config.to_evolution_policy(),
        iteration=iteration,
    )


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
        color = gradient_descent_color_step(
            element.color,
            precondition_color_gradient(prediction.color_gradient, color_jacobian=prediction.color_jacobian),
            learning_rate=config.color_learning_rate,
        )
        bounds = element.bounds
        if prediction.predicted_depth is not None and prediction.depth_loss > 0.01:
            depth_delta = -prediction.depth_gradient * prediction.depth_loss * config.color_learning_rate
            bounds = _shift_bounds_along_ray(element.bounds, prediction.ray_direction, depth_delta)
        decision = decision_by_element.get(element.id)
        confidence_map = updated_confidence_map(element, prediction)
        elements.append(
            replace(
                element,
                bounds=bounds,
                color=color,  # type: ignore[arg-type]
                confidence=refined_confidence(element.confidence, prediction, learning_rate=config.color_learning_rate),
                confidence_map=confidence_map,
                metadata={
                    **element.metadata,
                    "optimized_by": "aura-core-differentiable-reference",
                    "confidence_updated_by": "aura-core-residual-confidence",
                    **simplification_metadata(decision),
                },
            )
        )
        if decision is not None:
            evolved = evolved_element_for(element, decision, prediction)
            if evolved is not None and evolved.id not in existing_ids:
                elements.append(evolved)
                existing_ids.add(evolved.id)
    chunked_elements, chunks = carrier_lod_elements_and_chunks(tuple(elements))
    return AuraScene(name=scene.name, elements=chunked_elements, chunks=chunks, semantic_graph=scene.semantic_graph)


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


def _shift_bounds_along_ray(bounds: Bounds, direction: Vec3, distance: float) -> Bounds:
    offset = tuple(axis * distance for axis in direction)
    return Bounds(
        min_corner=tuple(value + delta for value, delta in zip(bounds.min_corner, offset)),  # type: ignore[arg-type]
        max_corner=tuple(value + delta for value, delta in zip(bounds.max_corner, offset)),  # type: ignore[arg-type]
    )


def _color_gradient(predicted: Vec3, target: Vec3) -> Vec3:
    return tuple(float(predicted[index] - target[index]) for index in range(3))  # type: ignore[return-value]


def _gradient_norm(color_gradient: Vec3, depth_gradient: float) -> float:
    return sqrt(sum(axis * axis for axis in color_gradient) + depth_gradient * depth_gradient)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _carrier_counts(scene: AuraScene) -> dict[str, int]:
    counts = {carrier_id: 0 for carrier_id in scene.carrier_ids()}
    for element in scene.elements:
        counts[element.carrier_id] += 1
    return counts
