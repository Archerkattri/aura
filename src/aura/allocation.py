"""Semantic-graph-governed heterogeneous carrier allocation for AURA-Core.

This module is the CORE RESEARCH NOVELTY of AURA: a language/semantic scene
graph decides WHICH typed carrier (surface/volume/beta/gabor/neural-residual/
semantic/gaussian) is assigned to each region, with differentiable inter-type
conversion scores and cross-carrier residual-correction hooks.

Key differentiators vs MP-GS (arXiv:2507.11321) and HybridNeRF (arXiv:2312.03160):
  - Semantic graph TOPOLOGY (node grouping + relations) biases carrier type choices
  - Per-region soft assignment logits (differentiable) over carrier kinds can be
    *learned* rather than only threshold-driven (opt-in via AllocationConfig)
  - Neural-residual carriers explicitly expose neighbor-relationship + residual-
    target info for cross-carrier error correction (Scaffold-GS, arXiv:2312.00109)

Architecture:
  - AllocationConfig: feature flags (graph-governed, soft-score, ablation mode)
  - SemanticCarrierBias: attribute-to-carrier score mapping from graph node attributes
  - SoftCarrierScores: per-region logits over carrier kinds (differentiable path)
  - GraphCluster: GaussianGraph-style Control-Follow clustering of region evidence
  - AllocationDecision: one allocation per region, with full provenance
  - ResidualCorrectionHook: exposes neighbor + residual-target for cross-carrier MLP
  - AllocationReport: JSON-able per-region decisions + graph + ablation metrics
  - semantic_graph_allocation(): top-level entry point
  - soft_carrier_scores(): compute differentiable logits for a region
  - residual_correction_hooks(): expose neural-residual corrector structure

Default behavior:
  When AllocationConfig.use_graph_governed=False (the default), this module
  delegates straight to the existing choose_carrier() and the output is
  identical to the legacy assignment path - no behavior change for existing code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Optional, Sequence

from aura.assignment import RegionEvidence, choose_carrier
from aura.carriers import CarrierSpec, default_registry
from aura.semantic import SemanticGraph, SemanticNode


# ---------------------------------------------------------------------------
# Carrier ordering for soft-score logits vector
# ---------------------------------------------------------------------------

#: Canonical ordered list of carrier kind IDs for the soft-score logit vector.
#: Index i in the logit vector corresponds to CARRIER_KIND_ORDER[i].
CARRIER_KIND_ORDER: tuple[str, ...] = (
    "surface",   # 0
    "volume",    # 1
    "beta",      # 2
    "gabor",     # 3
    "neural",    # 4
    "semantic",  # 5
    "gaussian",  # 6
)

_CARRIER_INDEX: dict[str, int] = {kind: i for i, kind in enumerate(CARRIER_KIND_ORDER)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AllocationConfig:
    """Feature flags for the semantic-graph-governed allocation system.

    Default values reproduce the legacy ``choose_carrier()`` hard-threshold
    behavior so existing code paths are completely unaffected.

    Attributes:
        use_graph_governed: When True, semantic graph node attributes bias
            carrier type selection beyond the heuristic thresholds.
        use_soft_scores: When True, per-region soft assignment logits are
            computed and returned in AllocationDecision. The hard-threshold
            fallback is still used for the actual carrier assignment unless
            use_learned_assignment is also True.
        use_learned_assignment: When True AND use_soft_scores is True, the
            argmax of soft_scores is used as the carrier, not the heuristic.
            In practice this requires the soft_scores to have been updated by
            an upstream optimizer; for training the soft-score logits are the
            mechanism, but AURA's optimizer updates them externally.
        graph_bias_weight: Strength in [0, 1] of graph-derived carrier bias
            relative to evidence-derived heuristic (1.0 = fully graph-driven).
        emit_residual_hooks: When True, residual-correction hook structures are
            populated for neural-residual carrier assignments.
        ablation_single_carrier: When not None, override all carrier choices
            with this carrier id (e.g. ``"gaussian"`` for ablation studies).
    """

    use_graph_governed: bool = False
    use_soft_scores: bool = False
    use_learned_assignment: bool = False
    graph_bias_weight: float = 0.5
    emit_residual_hooks: bool = False
    ablation_single_carrier: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.graph_bias_weight <= 1.0:
            raise ValueError("graph_bias_weight must be in [0, 1]")
        if self.ablation_single_carrier is not None:
            if self.ablation_single_carrier not in default_registry():
                raise ValueError(
                    f"ablation_single_carrier {self.ablation_single_carrier!r} "
                    "is not in the default carrier registry"
                )


# ---------------------------------------------------------------------------
# Semantic carrier bias: graph node attributes -> carrier score vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticCarrierBias:
    """Per-carrier score biases derived from one semantic graph node.

    All scores are in [0, 1] and represent a *soft preference* for the
    corresponding carrier kind when assigning elements that belong to this node.

    GaussianGraph (arXiv:2503.04034) Control-Follow style: graph nodes are
    "control" primitives whose attributes (surface/texture/translucency signals)
    are propagated to member element assignments.
    """

    scores: tuple[float, ...]  # length == len(CARRIER_KIND_ORDER)
    node_id: str
    node_label: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.scores) != len(CARRIER_KIND_ORDER):
            raise ValueError(
                f"SemanticCarrierBias.scores must have length {len(CARRIER_KIND_ORDER)}"
            )

    def score_for(self, carrier_id: str) -> float:
        idx = _CARRIER_INDEX.get(carrier_id)
        if idx is None:
            return 0.0
        return self.scores[idx]

    def preferred_carrier(self) -> str:
        best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return CARRIER_KIND_ORDER[best_idx]

    def to_dict(self) -> dict:
        return {
            "nodeId": self.node_id,
            "nodeLabel": self.node_label,
            "confidence": self.confidence,
            "scores": {
                CARRIER_KIND_ORDER[i]: self.scores[i]
                for i in range(len(CARRIER_KIND_ORDER))
            },
        }


def _node_attributes_to_carrier_bias(node: SemanticNode) -> SemanticCarrierBias:
    """Map SemanticNode attributes to a carrier preference score vector.

    Attribute mapping (GaussianGraph Control-Follow style):
    - "surface_flatness" / "flat" -> surface, beta
    - "texture_frequency" / "textured" -> gabor
    - "translucent" / "fuzzy" / "foliage" -> volume
    - "view_dependent" / "specular" -> neural
    - "semantic_object" / "object" -> semantic
    - "uncertain" / "complex" / "unknown" -> neural > gaussian
    - confidence modulates all scores

    Any node with confidence < 0.3 falls back to a uniform (no-bias) vector.
    """
    attrs = node.attributes
    conf = float(node.confidence)

    if conf < 0.3:
        # Low-confidence node: uniform bias (no preference)
        uniform = tuple(1.0 / len(CARRIER_KIND_ORDER) for _ in CARRIER_KIND_ORDER)
        return SemanticCarrierBias(
            scores=uniform, node_id=node.id, node_label=node.label, confidence=conf
        )

    # Start with small baseline scores (avoids zero-out of any carrier)
    scores = [0.1] * len(CARRIER_KIND_ORDER)

    def _attr(name: str, default: float = 0.0) -> float:
        v = attrs.get(name, default)
        return float(v) if isinstance(v, (int, float)) else default

    # --- Geometry / surface signals ---
    surface_flatness = _attr("surface_flatness", 0.0)
    geometry_confidence = _attr("geometry_confidence", 0.0)
    edit_need = _attr("edit_need", 0.0)

    if surface_flatness > 0.5 or bool(attrs.get("flat", False)):
        scores[_CARRIER_INDEX["surface"]] += 0.6 + surface_flatness * 0.3
        scores[_CARRIER_INDEX["beta"]] += 0.4
    elif geometry_confidence > 0.5 and edit_need > 0.3:
        scores[_CARRIER_INDEX["surface"]] += 0.5 * geometry_confidence

    # --- Texture / frequency signals ---
    texture_freq = _attr("texture_frequency", 0.0)
    high_freq = _attr("high_frequency", 0.0)

    if texture_freq > 0.5 or high_freq > 0.5 or bool(attrs.get("textured", False)):
        scores[_CARRIER_INDEX["gabor"]] += 0.6 + max(texture_freq, high_freq) * 0.3

    # --- Translucency / volume signals ---
    translucency = _attr("translucency", 0.0)
    fuzzy = _attr("fuzzy_confidence", 0.0)

    if (
        translucency > 0.4
        or fuzzy > 0.4
        or bool(attrs.get("translucent", False))
        or bool(attrs.get("foliage", False))
        or bool(attrs.get("fuzzy", False))
    ):
        scores[_CARRIER_INDEX["volume"]] += 0.6 + max(translucency, fuzzy) * 0.3

    # --- View-dependent / neural signals ---
    view_dep = _attr("view_dependent", 0.0)
    specular = _attr("specular", 0.0)
    uncertain = _attr("uncertainty", 0.0)
    complex_ = _attr("complexity", 0.0)

    if (
        view_dep > 0.4
        or specular > 0.4
        or bool(attrs.get("specular", False))
        or bool(attrs.get("view_dependent", False))
    ):
        scores[_CARRIER_INDEX["neural"]] += 0.5 + max(view_dep, specular) * 0.4

    if (
        uncertain > 0.5
        or complex_ > 0.5
        or bool(attrs.get("uncertain", False))
        or bool(attrs.get("complex", False))
    ):
        scores[_CARRIER_INDEX["neural"]] += 0.3
        scores[_CARRIER_INDEX["gaussian"]] += 0.2

    # --- Semantic object signals ---
    semantic_conf = _attr("semantic_confidence", 0.0)
    is_object = bool(attrs.get("object", False)) or bool(attrs.get("semantic_object", False))

    if semantic_conf > 0.6 or is_object:
        scores[_CARRIER_INDEX["semantic"]] += 0.5 + semantic_conf * 0.4

    # Modulate all scores by node confidence
    scores = [s * conf for s in scores]

    # Normalize to [0, 1]
    max_score = max(scores) if scores else 1.0
    if max_score > 1.0:
        scores = [s / max_score for s in scores]

    return SemanticCarrierBias(
        scores=tuple(scores),
        node_id=node.id,
        node_label=node.label,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# Graph clustering: GaussianGraph "Control-Follow" grouping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphCluster:
    """One cluster of region-evidence IDs grouped by semantic graph topology.

    Inspired by GaussianGraph (arXiv:2503.04034) Control-Follow clustering:
    a SemanticNode acts as the "control" primitive; member element_ids are
    the "follow" primitives whose carrier assignments are biased by the node.
    """

    node_id: str
    node_label: str
    element_ids: tuple[str, ...]
    carrier_bias: SemanticCarrierBias
    node_confidence: float

    def to_dict(self) -> dict:
        return {
            "nodeId": self.node_id,
            "nodeLabel": self.node_label,
            "elementIds": list(self.element_ids),
            "nodeConfidence": self.node_confidence,
            "carrierBias": self.carrier_bias.to_dict(),
        }


def build_graph_clusters(graph: SemanticGraph) -> dict[str, GraphCluster]:
    """Build a map from element_id -> GraphCluster from a SemanticGraph.

    Each node in the graph becomes one cluster; elements that appear in
    multiple nodes get the cluster of the highest-confidence node.
    """
    # element_id -> (cluster, node_confidence)
    assignment: dict[str, tuple[GraphCluster, float]] = {}

    for node in graph.nodes:
        bias = _node_attributes_to_carrier_bias(node)
        cluster = GraphCluster(
            node_id=node.id,
            node_label=node.label,
            element_ids=tuple(node.element_ids),
            carrier_bias=bias,
            node_confidence=float(node.confidence),
        )
        for eid in node.element_ids:
            existing = assignment.get(eid)
            if existing is None or node.confidence > existing[1]:
                assignment[eid] = (cluster, float(node.confidence))

    return {eid: cluster_conf[0] for eid, cluster_conf in assignment.items()}


# ---------------------------------------------------------------------------
# Soft carrier scores (differentiable path)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SoftCarrierScores:
    """Per-region soft assignment logits over all carrier kinds.

    These are the *pre-softmax* logits (unnormalised scores) for each carrier
    kind. They can be used for:
      1. Hard argmax: pick the carrier with highest score (use_learned_assignment)
      2. Differentiable soft assignment: multiply by a temperature-softmax for
         a learned allocation (upstream optimizer updates the logits)
      3. Ablation: override all scores to select a single carrier

    Real differentiability (gradient flow through logits) requires an upstream
    framework (e.g. PyTorch autograd) - this module provides the *structure*
    and computes meaningful initialization values from evidence + graph bias.
    The logits are pure Python floats here, making them harness-framework-
    agnostic; the torch optimizer can wrap them in tensors as needed.
    """

    logits: tuple[float, ...]  # len == len(CARRIER_KIND_ORDER), unnormalised
    element_id: str
    from_evidence: tuple[float, ...]   # evidence-only scores (no graph bias)
    from_graph: tuple[float, ...]      # graph-only bias scores
    blended: tuple[float, ...]         # final blended scores (= logits)
    graph_bias_weight: float

    def argmax_carrier(self) -> str:
        best_idx = max(range(len(self.logits)), key=lambda i: self.logits[i])
        return CARRIER_KIND_ORDER[best_idx]

    def softmax_probabilities(self, temperature: float = 1.0) -> tuple[float, ...]:
        """Compute softmax probabilities over logits (pure Python, no autograd)."""
        scaled = [l / max(temperature, 1e-8) for l in self.logits]
        max_l = max(scaled)
        exps = [math.exp(l - max_l) for l in scaled]
        total = sum(exps)
        return tuple(e / total for e in exps)

    def to_dict(self) -> dict:
        probs = self.softmax_probabilities()
        return {
            "elementId": self.element_id,
            "logits": {CARRIER_KIND_ORDER[i]: self.logits[i] for i in range(len(CARRIER_KIND_ORDER))},
            "probabilities": {CARRIER_KIND_ORDER[i]: probs[i] for i in range(len(CARRIER_KIND_ORDER))},
            "argmaxCarrier": self.argmax_carrier(),
            "graphBiasWeight": self.graph_bias_weight,
            "fromEvidence": {CARRIER_KIND_ORDER[i]: self.from_evidence[i] for i in range(len(CARRIER_KIND_ORDER))},
            "fromGraph": {CARRIER_KIND_ORDER[i]: self.from_graph[i] for i in range(len(CARRIER_KIND_ORDER))},
        }


def _evidence_to_logits(evidence: RegionEvidence) -> tuple[float, ...]:
    """Convert RegionEvidence to per-carrier unnormalised logit scores.

    These scores mirror the priority rules in choose_carrier() but expressed
    as continuous scores in [0, 1] per carrier rather than a hard decision.
    This allows differentiable blending with graph-derived biases.
    """
    scores = [0.05] * len(CARRIER_KIND_ORDER)  # baseline

    # surface: geometry_confidence AND edit_need
    scores[_CARRIER_INDEX["surface"]] = (
        evidence.geometry_confidence * 0.7 + evidence.edit_need * 0.3
    )

    # volume: fuzzy_confidence AND NOT geometry (soft version)
    vol_geo_penalty = max(0.0, evidence.geometry_confidence - 0.5)
    scores[_CARRIER_INDEX["volume"]] = max(
        0.0, evidence.fuzzy_confidence * 0.8 - vol_geo_penalty * 0.4
    )

    # beta: compact_detail
    scores[_CARRIER_INDEX["beta"]] = evidence.compact_detail * 0.9

    # gabor: high_frequency
    scores[_CARRIER_INDEX["gabor"]] = evidence.high_frequency * 0.9

    # neural: view_dependent AND NOT material (soft)
    mat_penalty = max(0.0, evidence.material_confidence - 0.4)
    scores[_CARRIER_INDEX["neural"]] = max(
        0.0, evidence.view_dependent * 0.8 - mat_penalty * 0.3
    )

    # semantic: semantic_confidence
    scores[_CARRIER_INDEX["semantic"]] = evidence.semantic_confidence * 0.9

    # gaussian: inverse of all other signals -> fallback when nothing fires
    max_non_gauss = max(scores[i] for i in range(len(CARRIER_KIND_ORDER) - 1))
    scores[_CARRIER_INDEX["gaussian"]] = max(0.05, 0.5 * (1.0 - max_non_gauss))

    return tuple(_clamp01(s) for s in scores)


def soft_carrier_scores(
    evidence: RegionEvidence,
    element_id: str,
    *,
    graph_cluster: Optional[GraphCluster] = None,
    config: AllocationConfig = AllocationConfig(),
) -> SoftCarrierScores:
    """Compute differentiable soft carrier assignment scores for one region.

    When graph_cluster is None (no semantic graph node covers this region),
    the graph bias is zero and scores are purely evidence-derived.

    This is the *core differentiable allocation mechanism* (Deliverable 2).
    The blended logits can be used as initialization values for an upstream
    optimizer to learn allocation via soft assignment.

    Args:
        evidence: RegionEvidence for this region.
        element_id: Unique id of the region/element.
        graph_cluster: Optional GraphCluster that covers this element.
        config: AllocationConfig controlling blend weight.

    Returns:
        SoftCarrierScores with logits, probabilities, and provenance fields.
    """
    ev_scores = _evidence_to_logits(evidence)

    if graph_cluster is not None:
        graph_scores = graph_cluster.carrier_bias.scores
    else:
        # No graph bias: flat (uniform) prior
        graph_scores = tuple(1.0 / len(CARRIER_KIND_ORDER) for _ in CARRIER_KIND_ORDER)

    w = config.graph_bias_weight if config.use_graph_governed else 0.0
    blended = tuple(
        _clamp01((1.0 - w) * ev + w * gb)
        for ev, gb in zip(ev_scores, graph_scores)
    )

    return SoftCarrierScores(
        logits=blended,
        element_id=element_id,
        from_evidence=ev_scores,
        from_graph=graph_scores,
        blended=blended,
        graph_bias_weight=w,
    )


# ---------------------------------------------------------------------------
# Residual correction hook (Deliverable 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualCorrectionHook:
    """Structure exposing neighbor-relationship and residual-target info.

    This hook is the scaffold for a neural-residual carrier to correct
    systematic errors of neighboring beta/gabor carriers (Scaffold-GS,
    arXiv:2312.00109). The actual residual MLP response lives in
    torch_kernels (not owned), so this hook exposes the conditioning data
    and neighbor relationships cleanly for consumption by that module.

    The hook is populated when:
      - A region is assigned the "neural" carrier, AND
      - AllocationConfig.emit_residual_hooks is True

    Fields:
        element_id: The neural-residual element that will perform correction.
        neighbor_element_ids: Nearby beta/gabor/surface carrier element IDs
            whose systematic errors this neural carrier should correct.
        residual_target_keys: Evidence keys that indicate the residual
            signal magnitude (to be read by the MLP as conditioning).
        carrier_types_to_correct: The carrier type IDs of the neighbors.
        conditioning_data: Additional float scalars for MLP conditioning
            (e.g. image_error, view_dependent) that the kernel module can
            consume without needing to re-read the full evidence.
    """

    element_id: str
    neighbor_element_ids: tuple[str, ...]
    residual_target_keys: tuple[str, ...]
    carrier_types_to_correct: tuple[str, ...]
    conditioning_data: Mapping[str, float]

    def to_dict(self) -> dict:
        return {
            "elementId": self.element_id,
            "neighborElementIds": list(self.neighbor_element_ids),
            "residualTargetKeys": list(self.residual_target_keys),
            "carrierTypesToCorrect": list(self.carrier_types_to_correct),
            "conditioningData": dict(self.conditioning_data),
        }


def residual_correction_hooks(
    decisions: Sequence["AllocationDecision"],
    *,
    config: AllocationConfig = AllocationConfig(),
    neighbor_radius_factor: float = 2.0,
) -> tuple[ResidualCorrectionHook, ...]:
    """Build residual correction hooks for all neural-residual allocations.

    For each neural-residual decision, find nearby beta/gabor/surface carriers
    within the same semantic cluster (if available) and expose them as
    neighbors that this MLP-residual element should correct.

    Args:
        decisions: All allocation decisions (from semantic_graph_allocation).
        config: AllocationConfig - hooks are only populated when
            emit_residual_hooks=True.
        neighbor_radius_factor: Unused (structural parameter for future
            spatial neighbor lookup when 3D bounds are available).

    Returns:
        Tuple of ResidualCorrectionHook, one per neural-residual element.
        Empty tuple when emit_residual_hooks is False.
    """
    if not config.emit_residual_hooks:
        return ()

    # Group decisions by semantic cluster node
    cluster_members: dict[Optional[str], list[AllocationDecision]] = {}
    for decision in decisions:
        node_id = decision.graph_cluster_node_id
        cluster_members.setdefault(node_id, []).append(decision)

    hooks: list[ResidualCorrectionHook] = []
    correctable_carriers = {"beta", "gabor", "surface"}

    for decision in decisions:
        if decision.carrier_id != "neural":
            continue

        # Find neighbors: same cluster, different carrier (beta/gabor/surface)
        node_id = decision.graph_cluster_node_id
        cluster = cluster_members.get(node_id, [])
        neighbors = [
            d for d in cluster
            if d.element_id != decision.element_id
            and d.carrier_id in correctable_carriers
        ]

        neighbor_ids = tuple(n.element_id for n in neighbors)
        carrier_types = tuple(dict.fromkeys(n.carrier_id for n in neighbors))

        # Conditioning data for the MLP (from evidence)
        ev = decision.evidence
        conditioning: dict[str, float] = {
            "image_error": ev.image_error,
            "view_dependent": ev.view_dependent,
            "material_confidence": ev.material_confidence,
            "geometry_confidence": ev.geometry_confidence,
        }

        hooks.append(
            ResidualCorrectionHook(
                element_id=decision.element_id,
                neighbor_element_ids=neighbor_ids,
                residual_target_keys=(
                    "optimization_image_loss",
                    "image_residual",
                    "view_residual",
                ),
                carrier_types_to_correct=carrier_types,
                conditioning_data=conditioning,
            )
        )

    return tuple(hooks)


# ---------------------------------------------------------------------------
# Allocation decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AllocationDecision:
    """One carrier allocation decision for one region.

    Captures the full provenance: which carrier was chosen, why, whether
    the graph governed the decision, the soft scores, and the graph cluster.
    """

    element_id: str
    carrier_id: str
    carrier_spec: CarrierSpec
    evidence: RegionEvidence
    reason: str
    governed_by_graph: bool
    graph_cluster_node_id: Optional[str]
    graph_cluster_label: Optional[str]
    soft_scores: Optional[SoftCarrierScores]
    is_ablation_override: bool = False

    def to_dict(self) -> dict:
        return {
            "elementId": self.element_id,
            "carrierId": self.carrier_id,
            "carrierKind": self.carrier_spec.kind.value,
            "reason": self.reason,
            "governedByGraph": self.governed_by_graph,
            "graphClusterNodeId": self.graph_cluster_node_id,
            "graphClusterLabel": self.graph_cluster_label,
            "isAblationOverride": self.is_ablation_override,
            "softScores": self.soft_scores.to_dict() if self.soft_scores is not None else None,
        }


# ---------------------------------------------------------------------------
# Allocation report (Deliverable 4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AllocationReport:
    """JSON-able allocation report for one decomposition pass.

    Contains:
    - Per-region carrier-type decisions with full provenance
    - The semantic graph that governed them (serialized)
    - Carrier-type distribution counts
    - Ablation metadata (which mode was active, single-carrier override)
    - Residual correction hooks (if enabled)

    This report enables downstream comparison of typed-mix vs single-carrier
    ablation conditions.
    """

    decisions: tuple[AllocationDecision, ...]
    graph_dict: dict
    carrier_counts: dict[str, int]
    graph_governed_count: int
    heuristic_count: int
    ablation_mode: Optional[str]
    residual_hooks: tuple[ResidualCorrectionHook, ...]
    config_dict: dict

    def to_dict(self) -> dict:
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "semanticGraph": self.graph_dict,
            "carrierCounts": self.carrier_counts,
            "graphGovernedCount": self.graph_governed_count,
            "heuristicCount": self.heuristic_count,
            "ablationMode": self.ablation_mode,
            "residualHooks": [h.to_dict() for h in self.residual_hooks],
            "config": self.config_dict,
        }

    @property
    def total_regions(self) -> int:
        return len(self.decisions)

    @property
    def graph_coverage_fraction(self) -> float:
        if self.total_regions == 0:
            return 0.0
        return self.graph_governed_count / self.total_regions


# ---------------------------------------------------------------------------
# Top-level allocation entry point (Deliverable 1)
# ---------------------------------------------------------------------------

def semantic_graph_allocation(
    evidences: Sequence[tuple[str, RegionEvidence]],
    graph: Optional[SemanticGraph] = None,
    *,
    config: AllocationConfig = AllocationConfig(),
    registry: Optional[Mapping[str, CarrierSpec]] = None,
) -> AllocationReport:
    """Semantic-graph-governed carrier allocation for a set of regions.

    This is the headline AURA novelty: a language/semantic scene graph
    decides which carrier type each region receives, with differentiable
    inter-type conversion scores.

    Default behavior (config.use_graph_governed=False):
        Falls straight through to choose_carrier() - identical to legacy
        assignment path. Zero behavior change for existing callers.

    Graph-governed behavior (config.use_graph_governed=True):
        1. Build GaussianGraph-style Control-Follow clusters from graph nodes.
        2. For each region: compute soft carrier scores blending evidence
           heuristics with graph-node attribute biases.
        3. If use_learned_assignment: carrier = argmax(soft_scores).
           Otherwise: use choose_carrier() heuristic but record that graph
           governed the choice for provenance.
        4. If ablation_single_carrier is set: override all choices.
        5. Emit ResidualCorrectionHooks for neural-residual allocations.
        6. Build and return AllocationReport.

    Args:
        evidences: Sequence of (element_id, RegionEvidence) pairs.
        graph: Optional SemanticGraph. When None, allocation is purely
            evidence-heuristic (same as legacy).
        config: AllocationConfig controlling all opt-in features.
        registry: Optional carrier registry; defaults to default_registry().

    Returns:
        AllocationReport with full provenance and JSON-able records.
    """
    reg = registry or default_registry()

    # Build clusters from graph (empty dict when no graph or not graph-governed)
    clusters: dict[str, GraphCluster] = {}
    if graph is not None and config.use_graph_governed:
        clusters = build_graph_clusters(graph)

    decisions: list[AllocationDecision] = []

    for element_id, evidence in evidences:
        cluster = clusters.get(element_id)

        # --- Ablation single-carrier override ---
        if config.ablation_single_carrier is not None:
            carrier_id = config.ablation_single_carrier
            spec = reg[carrier_id]
            soft = None
            if config.use_soft_scores:
                soft = soft_carrier_scores(evidence, element_id, graph_cluster=cluster, config=config)
            decisions.append(AllocationDecision(
                element_id=element_id,
                carrier_id=carrier_id,
                carrier_spec=spec,
                evidence=evidence,
                reason=f"ablation_single_carrier={carrier_id}",
                governed_by_graph=False,
                graph_cluster_node_id=None,
                graph_cluster_label=None,
                soft_scores=soft,
                is_ablation_override=True,
            ))
            continue

        # --- Compute soft scores when requested ---
        soft = None
        if config.use_soft_scores:
            soft = soft_carrier_scores(evidence, element_id, graph_cluster=cluster, config=config)

        # --- Carrier selection ---
        governed_by_graph = False
        if config.use_graph_governed and config.use_learned_assignment and soft is not None:
            # Use soft-score argmax (learned assignment path)
            carrier_id = soft.argmax_carrier()
            spec = reg.get(carrier_id) or reg["gaussian"]
            reason = (
                f"soft_score_argmax:cluster={cluster.node_label if cluster else 'none'}"
                f",score={soft.logits[_CARRIER_INDEX[carrier_id]]:.3f}"
            )
            governed_by_graph = cluster is not None
        elif config.use_graph_governed and cluster is not None:
            # Graph-biased but still heuristic: use choose_carrier() but
            # potentially override when graph signal is very strong
            heuristic_spec = choose_carrier(evidence, reg)
            bias = cluster.carrier_bias
            graph_preferred = bias.preferred_carrier()
            graph_score = bias.score_for(graph_preferred)
            heuristic_id = heuristic_spec.id

            # Override only when graph signal is decisive AND high confidence
            if (
                graph_preferred != heuristic_id
                and graph_score > 0.6
                and cluster.node_confidence > 0.5
            ):
                # Strong graph override
                carrier_id = graph_preferred
                spec = reg.get(carrier_id, heuristic_spec)
                reason = (
                    f"graph_override:{cluster.node_label}"
                    f"->carrier={carrier_id}"
                    f",graph_score={graph_score:.3f}"
                )
                governed_by_graph = True
            else:
                # Weak graph signal: heuristic wins, graph recorded for provenance
                carrier_id = heuristic_id
                spec = heuristic_spec
                reason = (
                    f"heuristic_with_graph_context:{cluster.node_label}"
                    f"->carrier={carrier_id}"
                )
                governed_by_graph = True  # graph was consulted even if it didn't override
        else:
            # Pure heuristic (default path - identical to legacy)
            spec = choose_carrier(evidence, reg)
            carrier_id = spec.id
            reason = "heuristic:choose_carrier"
            governed_by_graph = False

        decisions.append(AllocationDecision(
            element_id=element_id,
            carrier_id=carrier_id,
            carrier_spec=spec,
            evidence=evidence,
            reason=reason,
            governed_by_graph=governed_by_graph,
            graph_cluster_node_id=cluster.node_id if cluster else None,
            graph_cluster_label=cluster.node_label if cluster else None,
            soft_scores=soft,
            is_ablation_override=False,
        ))

    # --- Residual correction hooks ---
    hooks = residual_correction_hooks(decisions, config=config)

    # --- Build report ---
    carrier_counts: dict[str, int] = {}
    for d in decisions:
        carrier_counts[d.carrier_id] = carrier_counts.get(d.carrier_id, 0) + 1

    graph_governed_count = sum(1 for d in decisions if d.governed_by_graph)
    heuristic_count = len(decisions) - graph_governed_count

    config_dict = {
        "useGraphGoverned": config.use_graph_governed,
        "useSoftScores": config.use_soft_scores,
        "useLearnedAssignment": config.use_learned_assignment,
        "graphBiasWeight": config.graph_bias_weight,
        "emitResidualHooks": config.emit_residual_hooks,
        "ablationSingleCarrier": config.ablation_single_carrier,
    }

    return AllocationReport(
        decisions=tuple(decisions),
        graph_dict=graph.to_dict() if graph is not None else {"nodes": [], "edges": []},
        carrier_counts=dict(sorted(carrier_counts.items())),
        graph_governed_count=graph_governed_count,
        heuristic_count=heuristic_count,
        ablation_mode=config.ablation_single_carrier,
        residual_hooks=hooks,
        config_dict=config_dict,
    )


# ---------------------------------------------------------------------------
# Differentiable evolution scores (Deliverable 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SoftEvolutionScores:
    """Per-element soft carrier transition scores for differentiable evolution.

    These logits represent how strongly each possible transition (promote,
    demote, split, merge, retain) is recommended for a given element based on
    its current evidence and soft carrier assignment scores.

    This is the differentiable complement to the hard-threshold policy in
    evolution.py. The hard policy remains the default; soft scores are only
    computed and used when AllocationConfig.use_soft_scores=True.

    Inspired by HybridNeRF (arXiv:2312.03160) soft surfaceness field and
    MP-GS (arXiv:2507.11321) inter-type conversion.
    """

    element_id: str
    current_carrier: str
    # Soft scores for each possible target carrier (transition logits)
    transition_logits: dict[str, float]
    # Recommended action based on argmax
    recommended_action: str
    # Confidence in the recommendation
    confidence: float

    def to_dict(self) -> dict:
        return {
            "elementId": self.element_id,
            "currentCarrier": self.current_carrier,
            "transitionLogits": self.transition_logits,
            "recommendedAction": self.recommended_action,
            "confidence": self.confidence,
        }


def soft_evolution_scores(
    element_id: str,
    current_carrier: str,
    evidence: RegionEvidence,
    *,
    image_loss: float = 0.0,
    depth_loss: float = 0.0,
    graph_cluster: Optional[GraphCluster] = None,
    config: AllocationConfig = AllocationConfig(),
) -> SoftEvolutionScores:
    """Compute soft carrier transition scores for one element.

    Used as a *complement* to the hard-threshold evolution policy in
    evolution.py. When config.use_soft_scores is False, this function is
    not called (the hard policy runs as before).

    The transition logits represent:
      - "retain_<current>": keep the current carrier
      - "promote_to_<target>": upgrade to a higher-complexity carrier
      - "demote_to_<target>": downgrade to a lower-complexity carrier
      - "split": add a detail child carrier

    Args:
        element_id: Element ID.
        current_carrier: Current carrier ID.
        evidence: RegionEvidence for the element.
        image_loss: Current image reconstruction loss.
        depth_loss: Current depth reconstruction loss.
        graph_cluster: Optional graph cluster (influences transition bias).
        config: AllocationConfig.

    Returns:
        SoftEvolutionScores with transition logits and recommended action.
    """
    # Get soft scores for this element's evidence
    scores = soft_carrier_scores(
        evidence, element_id, graph_cluster=graph_cluster, config=config
    )

    # Current carrier's score
    current_score = scores.logits[_CARRIER_INDEX.get(current_carrier, -1)] \
        if current_carrier in _CARRIER_INDEX else 0.0

    # Compute residual pressure (high loss -> higher promote/split pressure)
    residual = _clamp01(image_loss + depth_loss * 0.25)

    # Build transition logits
    carrier_complexities = {
        "surface": 1.2, "volume": 1.4, "beta": 1.1,
        "gabor": 1.3, "neural": 1.8, "gaussian": 0.7, "semantic": 0.9,
    }
    current_complexity = carrier_complexities.get(current_carrier, 1.0)

    transition_logits: dict[str, float] = {}
    best_action = f"retain_{current_carrier}"
    best_score = current_score * (1.0 - residual)

    for target_carrier in CARRIER_KIND_ORDER:
        if target_carrier == current_carrier:
            # Retain: score is inversely proportional to residual
            retain_score = _clamp01(current_score * (1.0 - residual * 0.8))
            transition_logits[f"retain_{current_carrier}"] = retain_score
            if retain_score > best_score:
                best_score = retain_score
                best_action = f"retain_{current_carrier}"
        else:
            target_complexity = carrier_complexities.get(target_carrier, 1.0)
            target_score = scores.logits[_CARRIER_INDEX[target_carrier]]

            if target_complexity > current_complexity:
                # Promote: favoured when residual is high AND target score is high
                promote_score = _clamp01(target_score * residual * 1.2)
                action = f"promote_to_{target_carrier}"
                transition_logits[action] = promote_score
                if promote_score > best_score:
                    best_score = promote_score
                    best_action = action
            else:
                # Demote: favoured when residual is low AND target score is higher
                demote_score = _clamp01(target_score * (1.0 - residual) * 0.8)
                action = f"demote_to_{target_carrier}"
                transition_logits[action] = demote_score
                if demote_score > best_score:
                    best_score = demote_score
                    best_action = action

    # Split: add a detail child (beta or neural) when residual is high
    if residual > 0.3:
        split_score = _clamp01(residual * 0.9)
        transition_logits["split_detail"] = split_score
        if split_score > best_score:
            best_score = split_score
            best_action = "split_detail"

    return SoftEvolutionScores(
        element_id=element_id,
        current_carrier=current_carrier,
        transition_logits=transition_logits,
        recommended_action=best_action,
        confidence=_clamp01(best_score),
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))
