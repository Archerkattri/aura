from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from aura.assignment import RegionEvidence
from aura.decomposition import EvidenceSample, decompose_evidence
from aura.elements import Bounds
from aura.ray import Vec3
from aura.render import render_orthographic
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


@dataclass(frozen=True)
class ReconstructionConfig:
    iterations: int = 4
    render_width: int = 8
    render_height: int = 8

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render dimensions must be positive")


@dataclass(frozen=True)
class ReconstructionStep:
    iteration: int
    image_loss: float
    depth_loss: float
    carrier_counts: dict[str, int]
    adaptation: str

    def to_dict(self) -> dict:
        return asdict(self)


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


def reconstruct_demo_scene(config: ReconstructionConfig | None = None) -> ReconstructionResult:
    """Run the deterministic AURA-Core fixture reconstruction path.

    This is a CPU reference scaffold: it creates posed observations, initializes
    native evidence cells, emits mixed AURA carriers, and records loss/adaptation
    stages. It is not a production optimizer.
    """

    config = config or ReconstructionConfig()
    frames = synthetic_training_frames()
    scene = decompose_evidence(_initial_evidence_from_frames(frames), name="reconstruct_demo")
    iterations = _reference_iterations(scene, frames, config)
    final_loss = iterations[-1].image_loss + iterations[-1].depth_loss
    non_gaussian = sum(1 for element in scene.elements if element.carrier_id != "gaussian")
    native_fraction = 0.0 if not scene.elements else non_gaussian / len(scene.elements)
    report = ReconstructionReport(
        name=scene.name,
        frames=frames,
        stages=(
            "posed_synthetic_input",
            "native_evidence_initialization",
            "cpu_reference_render_loss",
            "adaptive_carrier_evolution_report",
            "aura_package_export_ready",
        ),
        iterations=iterations,
        final_loss=final_loss,
        native_carrier_fraction=native_fraction,
        sources=("synthetic_posed_images", "synthetic_depth", "semantic_masks"),
    )
    return ReconstructionResult(scene=scene, report=report)


def _initial_evidence_from_frames(frames: Sequence[TrainingFrame]) -> tuple[EvidenceSample, ...]:
    by_id = {frame.id: frame for frame in frames}
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
) -> tuple[ReconstructionStep, ...]:
    base_image_loss = _fixture_image_loss(scene, frames, config)
    base_depth_loss = _fixture_depth_loss(scene, frames)
    counts = _carrier_counts(scene)
    steps = []
    for index in range(config.iterations):
        progress = index / max(1, config.iterations - 1)
        image_loss = base_image_loss * (1.0 - 0.55 * progress)
        depth_loss = base_depth_loss * (1.0 - 0.40 * progress)
        adaptation = "initialize" if index == 0 else "promote_or_refine_native_carriers"
        steps.append(
            ReconstructionStep(
                iteration=index,
                image_loss=image_loss,
                depth_loss=depth_loss,
                carrier_counts=counts,
                adaptation=adaptation,
            )
        )
    return tuple(steps)


def _fixture_image_loss(scene: AuraScene, frames: Sequence[TrainingFrame], config: ReconstructionConfig) -> float:
    image = render_orthographic(scene, width=config.render_width, height=config.render_height)
    predicted = _mean_color(image.pixels)
    target = _mean_color(tuple(frame.target_color for frame in frames))
    return sum((left - right) ** 2 for left, right in zip(predicted, target)) / 3.0


def _fixture_depth_loss(scene: AuraScene, frames: Sequence[TrainingFrame]) -> float:
    scene_depth = min(element.bounds.min_corner[2] for element in scene.elements) + 2.0 if scene.elements else 0.0
    target_depth = sum(frame.target_depth for frame in frames) / len(frames)
    return abs(scene_depth - target_depth)


def _mean_color(colors: Sequence[Vec3]) -> Vec3:
    if not colors:
        return (0.0, 0.0, 0.0)
    scale = 1.0 / len(colors)
    return (
        sum(color[0] for color in colors) * scale,
        sum(color[1] for color in colors) * scale,
        sum(color[2] for color in colors) * scale,
    )


def _carrier_counts(scene: AuraScene) -> dict[str, int]:
    counts = {carrier_id: 0 for carrier_id in scene.carrier_ids()}
    for element in scene.elements:
        counts[element.carrier_id] += 1
    return counts
