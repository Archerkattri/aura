from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional


class CarrierKind(Enum):
    """Enumeration of the AURA carrier primitives."""

    SURFACE_CELL = "surface_cell"
    VOLUME_CELL = "volume_cell"
    BETA_KERNEL = "beta_kernel"
    GABOR_FREQUENCY = "gabor_frequency"
    NEURAL_RESIDUAL = "neural_residual"
    GAUSSIAN_FALLBACK = "gaussian_fallback"
    SEMANTIC_FEATURE = "semantic_feature"


# --------------------------------------------------------------------------- #
# Carrier maturity — the honesty contract for what each carrier type can back.
#
# The registry advertises seven carrier *types* but they are NOT equally real,
# and conflating them is exactly the failure mode this repo forbids. Every carrier
# therefore carries an explicit ``maturity`` level. The publication gate
# ``carrier_registry_honesty`` (aura.publication) enforces that a "trained" claim is
# backed by committed real-scene evidence; anything advertising more than it can
# back fails the gate.
#
#   * ``trained``  — trains and renders on real scenes with committed evidence
#                    (calib_<scene>.json). Today: gaussian (gsplat) + beta (DBS-Beta).
#   * ``demo``     — has a real footprint that renders, but is only validated in a
#                    demo/2D/PRISM-extension setting, not on real full scenes. Today:
#                    gabor (2D crops) + neural (orphaned experimental footprint,
#                    composited via an explicit Gaussian fallback in hybrid.py).
#   * ``metadata`` — a typed contract / payload / graph only; no trained-carrier
#                    render family behind it. Today: surface, volume, semantic.
# --------------------------------------------------------------------------- #

MATURITY_TRAINED = "trained"
MATURITY_DEMO = "demo"
MATURITY_METADATA = "metadata"
CARRIER_MATURITIES = frozenset((MATURITY_TRAINED, MATURITY_DEMO, MATURITY_METADATA))


@dataclass(frozen=True)
class CarrierSpec:
    """Immutable descriptor for one AURA carrier type.

    Combines capability flags (ray query, collision proxy, direct relighting,
    etc.) with a relative computational complexity weight used by the
    assignment and evolution systems, plus an explicit ``maturity`` level naming
    how real the carrier is (see the maturity contract above).
    """

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
    maturity: str = MATURITY_METADATA

    def __post_init__(self) -> None:
        if self.maturity not in CARRIER_MATURITIES:
            raise ValueError(
                f"carrier {self.id!r} has unknown maturity {self.maturity!r}; "
                f"expected one of {sorted(CARRIER_MATURITIES)}"
            )


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
            maturity=MATURITY_METADATA,
        ),
        "volume": CarrierSpec(
            id="volume",
            kind=CarrierKind.VOLUME_CELL,
            description="Volumetric density carrier for fuzzy or semi-transparent regions.",
            primary_render=True,
            ray_query=True,
            complexity=1.4,
            maturity=MATURITY_METADATA,
        ),
        "beta": CarrierSpec(
            id="beta",
            kind=CarrierKind.BETA_KERNEL,
            description="Compact bounded kernel for adaptive detail.",
            primary_render=True,
            ray_query=True,
            complexity=1.1,
            maturity=MATURITY_TRAINED,
        ),
        "gabor": CarrierSpec(
            id="gabor",
            kind=CarrierKind.GABOR_FREQUENCY,
            description="Frequency-aware carrier for structured high-frequency texture.",
            primary_render=True,
            ray_query=True,
            complexity=1.3,
            maturity=MATURITY_DEMO,
        ),
        "neural": CarrierSpec(
            id="neural",
            kind=CarrierKind.NEURAL_RESIDUAL,
            description="Local residual carrier for view-dependent or ambiguous appearance.",
            primary_render=True,
            ray_query=True,
            neural_residual=True,
            complexity=1.8,
            maturity=MATURITY_DEMO,
        ),
        "gaussian": CarrierSpec(
            id="gaussian",
            kind=CarrierKind.GAUSSIAN_FALLBACK,
            description="Compatibility fallback where ordinary splats are sufficient.",
            primary_render=True,
            ray_query=True,
            complexity=0.7,
            maturity=MATURITY_TRAINED,
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
            maturity=MATURITY_METADATA,
        ),
    }


def carrier_maturity_map(
    registry: Optional[Mapping[str, CarrierSpec]] = None,
) -> Dict[str, str]:
    """Return ``{carrier_id: maturity}`` for every carrier in the registry.

    This is the honesty surface every API that lists/advertises carrier types
    exposes: it says, per type, whether it is ``trained`` (real-scene evidence),
    ``demo`` (footprint validated only in demo/2D/PRISM-extension use), or
    ``metadata`` (typed contract only, no trained render family). Defaults to
    :func:`default_registry`.
    """

    reg = registry if registry is not None else default_registry()
    return {carrier_id: spec.maturity for carrier_id, spec in reg.items()}
