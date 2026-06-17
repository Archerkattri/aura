from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class CarrierKind(Enum):
    SURFACE_CELL = "surface_cell"
    VOLUME_CELL = "volume_cell"
    BETA_KERNEL = "beta_kernel"
    GABOR_FREQUENCY = "gabor_frequency"
    NEURAL_RESIDUAL = "neural_residual"
    GAUSSIAN_FALLBACK = "gaussian_fallback"
    SEMANTIC_FEATURE = "semantic_feature"


@dataclass(frozen=True)
class CarrierSpec:
    id: str
    kind: CarrierKind
    description: str
    primary_render: bool
    ray_query: bool
    collision_proxy: bool = False
    direct_relighting: bool = False
    semantic_query: bool = False
    neural_residual: bool = False
    complexity: float = 1.0


def default_registry() -> Dict[str, CarrierSpec]:
    """Return the minimal AURA carrier registry for the GPU-ready slice."""

    return {
        "surface": CarrierSpec(
            id="surface",
            kind=CarrierKind.SURFACE_CELL,
            description="Surface or thin-slab carrier for confident opaque structure.",
            primary_render=True,
            ray_query=True,
            collision_proxy=True,
            direct_relighting=True,
            complexity=1.2,
        ),
        "volume": CarrierSpec(
            id="volume",
            kind=CarrierKind.VOLUME_CELL,
            description="Volumetric density carrier for fuzzy or semi-transparent regions.",
            primary_render=True,
            ray_query=True,
            complexity=1.4,
        ),
        "beta": CarrierSpec(
            id="beta",
            kind=CarrierKind.BETA_KERNEL,
            description="Compact bounded kernel for adaptive detail.",
            primary_render=True,
            ray_query=True,
            complexity=1.1,
        ),
        "gabor": CarrierSpec(
            id="gabor",
            kind=CarrierKind.GABOR_FREQUENCY,
            description="Frequency-aware carrier for structured high-frequency texture.",
            primary_render=True,
            ray_query=True,
            complexity=1.3,
        ),
        "neural": CarrierSpec(
            id="neural",
            kind=CarrierKind.NEURAL_RESIDUAL,
            description="Local residual carrier for view-dependent or ambiguous appearance.",
            primary_render=True,
            ray_query=True,
            neural_residual=True,
            complexity=1.8,
        ),
        "gaussian": CarrierSpec(
            id="gaussian",
            kind=CarrierKind.GAUSSIAN_FALLBACK,
            description="Compatibility fallback where ordinary splats are sufficient.",
            primary_render=True,
            ray_query=True,
            complexity=0.7,
        ),
        "semantic": CarrierSpec(
            id="semantic",
            kind=CarrierKind.SEMANTIC_FEATURE,
            description="Object or language feature carrier for grouping and editing.",
            primary_render=False,
            ray_query=True,
            collision_proxy=True,
            semantic_query=True,
            complexity=0.9,
        ),
    }
