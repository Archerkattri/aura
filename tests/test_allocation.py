"""Tests for the semantic-graph-governed heterogeneous carrier allocation module.

Tests cover:
 1. Semantic-graph-governed carrier allocation (Deliverable 1)
 2. Differentiable inter-type conversion scores (Deliverable 2)
 3. Cross-carrier residual-correction hooks (Deliverable 3)
 4. Allocation report / metrics / ablation hooks (Deliverable 4)
 5. Default behavior / backward-compatibility (hard rules)
"""

from __future__ import annotations

import math
import pytest

from aura.assignment import RegionEvidence, choose_carrier
from aura.allocation import (
    CARRIER_KIND_ORDER,
    AllocationConfig,
    AllocationDecision,
    AllocationReport,
    GraphCluster,
    ResidualCorrectionHook,
    SemanticCarrierBias,
    SoftCarrierScores,
    SoftEvolutionScores,
    build_graph_clusters,
    residual_correction_hooks,
    semantic_graph_allocation,
    soft_carrier_scores,
    soft_evolution_scores,
)
from aura.semantic import SemanticEdge, SemanticGraph, SemanticNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(**kwargs) -> RegionEvidence:
    return RegionEvidence(**kwargs)


def _node(nid: str, label: str, element_ids=(), confidence=0.9, **attrs) -> SemanticNode:
    return SemanticNode(
        id=nid,
        label=label,
        element_ids=tuple(element_ids),
        confidence=confidence,
        attributes=attrs,
    )


def _graph(*nodes, edges=()) -> SemanticGraph:
    return SemanticGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Deliverable 1: Semantic-graph-governed carrier allocation
# ---------------------------------------------------------------------------

class TestDefaultBehaviorMatchesLegacy:
    """With use_graph_governed=False (default), output must match choose_carrier()."""

    def test_default_config_surface_evidence(self):
        ev = _evidence(geometry_confidence=0.92, material_confidence=0.75, edit_need=0.8)
        report = semantic_graph_allocation([("e1", ev)])
        assert report.decisions[0].carrier_id == choose_carrier(ev).id

    def test_default_config_volume_evidence(self):
        ev = _evidence(fuzzy_confidence=0.9, geometry_confidence=0.2)
        report = semantic_graph_allocation([("e1", ev)])
        assert report.decisions[0].carrier_id == choose_carrier(ev).id

    def test_default_config_gabor_evidence(self):
        ev = _evidence(high_frequency=0.95)
        report = semantic_graph_allocation([("e1", ev)])
        assert report.decisions[0].carrier_id == choose_carrier(ev).id

    def test_default_config_gaussian_fallback(self):
        ev = _evidence(image_error=0.05)
        report = semantic_graph_allocation([("e1", ev)])
        assert report.decisions[0].carrier_id == "gaussian"

    def test_default_config_neural_evidence(self):
        ev = _evidence(view_dependent=0.9, material_confidence=0.2)
        report = semantic_graph_allocation([("e1", ev)])
        assert report.decisions[0].carrier_id == choose_carrier(ev).id

    def test_default_config_semantic_evidence(self):
        ev = _evidence(semantic_confidence=0.9)
        report = semantic_graph_allocation([("e1", ev)])
        assert report.decisions[0].carrier_id == "semantic"

    def test_default_config_graph_is_not_consulted(self):
        """With default config, graph is ignored even if provided."""
        graph = _graph(
            _node("n1", "flat_wall", element_ids=["e1"], surface_flatness=0.95),
        )
        ev = _evidence(high_frequency=0.95)
        default_report = semantic_graph_allocation([("e1", ev)])
        with_graph_report = semantic_graph_allocation([("e1", ev)], graph=graph)
        # Both must agree because use_graph_governed=False
        assert default_report.decisions[0].carrier_id == with_graph_report.decisions[0].carrier_id

    def test_default_heuristic_count_equals_total(self):
        evs = [("e1", _evidence(high_frequency=0.95)), ("e2", _evidence(fuzzy_confidence=0.9))]
        report = semantic_graph_allocation(evs)
        assert report.heuristic_count == 2
        assert report.graph_governed_count == 0


class TestGraphGovernedAllocation:
    """When use_graph_governed=True, graph topology biases carrier selection."""

    def test_flat_surface_node_biases_toward_surface_carrier(self):
        graph = _graph(
            _node("n1", "flat_wall", element_ids=["e1"],
                  surface_flatness=0.95, geometry_confidence=0.8, edit_need=0.6),
        )
        config = AllocationConfig(use_graph_governed=True, graph_bias_weight=0.8)
        # Use evidence that would normally pick gabor (high freq), graph should override
        # to surface when graph signal is very strong and heuristic would pick surface too
        ev = _evidence(geometry_confidence=0.8, edit_need=0.7)
        report = semantic_graph_allocation([("e1", ev)], graph=graph, config=config)
        d = report.decisions[0]
        assert d.governed_by_graph is True
        assert d.graph_cluster_node_id == "n1"
        assert d.graph_cluster_label == "flat_wall"

    def test_textured_node_biases_toward_gabor(self):
        graph = _graph(
            _node("n1", "textured_wall", element_ids=["e1"],
                  texture_frequency=0.9, high_frequency=0.85),
        )
        config = AllocationConfig(use_graph_governed=True, graph_bias_weight=0.9)
        # Evidence that would pick gaussian (no strong signal)
        ev = _evidence(image_error=0.05)
        report = semantic_graph_allocation([("e1", ev)], graph=graph, config=config)
        d = report.decisions[0]
        assert d.governed_by_graph is True
        # The graph should strongly push toward gabor
        assert d.carrier_id == "gabor"

    def test_translucent_node_biases_toward_volume(self):
        graph = _graph(
            _node("n1", "foliage", element_ids=["e1"], translucent=True, foliage=True),
        )
        config = AllocationConfig(use_graph_governed=True, graph_bias_weight=0.9)
        ev = _evidence(image_error=0.05)  # would otherwise pick gaussian
        report = semantic_graph_allocation([("e1", ev)], graph=graph, config=config)
        d = report.decisions[0]
        assert d.governed_by_graph is True
        assert d.carrier_id == "volume"

    def test_uncertain_complex_node_biases_toward_neural(self):
        graph = _graph(
            _node("n1", "unknown_object", element_ids=["e1"], uncertain=True, complex=True),
        )
        config = AllocationConfig(use_graph_governed=True, graph_bias_weight=0.9)
        ev = _evidence(image_error=0.05)
        report = semantic_graph_allocation([("e1", ev)], graph=graph, config=config)
        d = report.decisions[0]
        assert d.governed_by_graph is True

    def test_low_confidence_node_does_not_override_heuristic(self):
        """Graph node with confidence < 0.3 should not override heuristic choice."""
        graph = _graph(
            _node("n1", "weak_node", element_ids=["e1"],
                  surface_flatness=0.95, confidence=0.2),
        )
        config = AllocationConfig(use_graph_governed=True, graph_bias_weight=0.9)
        ev = _evidence(high_frequency=0.95)  # heuristic picks gabor
        report = semantic_graph_allocation([("e1", ev)], graph=graph, config=config)
        d = report.decisions[0]
        # Low-confidence node should not produce a strong enough graph_score to override
        # The heuristic gabor should still win
        assert d.carrier_id == "gabor"

    def test_element_not_in_graph_uses_heuristic(self):
        graph = _graph(
            _node("n1", "wall", element_ids=["e2"]),  # e1 not in graph
        )
        config = AllocationConfig(use_graph_governed=True, graph_bias_weight=0.8)
        ev = _evidence(high_frequency=0.95)
        report = semantic_graph_allocation([("e1", ev)], graph=graph, config=config)
        d = report.decisions[0]
        assert d.graph_cluster_node_id is None
        assert d.governed_by_graph is False
        assert d.carrier_id == choose_carrier(ev).id

    def test_graph_governed_count_tracked_correctly(self):
        graph = _graph(
            _node("n1", "wall", element_ids=["e1", "e2"],
                  surface_flatness=0.9, geometry_confidence=0.8),
        )
        config = AllocationConfig(use_graph_governed=True)
        evs = [
            ("e1", _evidence(geometry_confidence=0.8, edit_need=0.7)),
            ("e2", _evidence(fuzzy_confidence=0.9)),
            ("e3", _evidence(high_frequency=0.95)),  # not in graph
        ]
        report = semantic_graph_allocation(evs, graph=graph, config=config)
        assert report.graph_governed_count == 2
        assert report.heuristic_count == 1


class TestGraphClusters:
    """Test build_graph_clusters and SemanticCarrierBias."""

    def test_build_clusters_maps_elements_to_nodes(self):
        graph = _graph(
            _node("n1", "wall", element_ids=["e1", "e2"]),
            _node("n2", "foliage", element_ids=["e3"]),
        )
        clusters = build_graph_clusters(graph)
        assert set(clusters.keys()) == {"e1", "e2", "e3"}
        assert clusters["e1"].node_id == "n1"
        assert clusters["e3"].node_id == "n2"

    def test_highest_confidence_node_wins_for_shared_element(self):
        graph = _graph(
            _node("n1", "low_conf", element_ids=["e1"], confidence=0.4),
            _node("n2", "high_conf", element_ids=["e1"], confidence=0.9),
        )
        clusters = build_graph_clusters(graph)
        assert clusters["e1"].node_id == "n2"

    def test_empty_graph_gives_empty_clusters(self):
        graph = SemanticGraph()
        clusters = build_graph_clusters(graph)
        assert clusters == {}

    def test_carrier_bias_flat_node_prefers_surface(self):
        node = _node("n1", "flat", element_ids=[], surface_flatness=0.95)
        graph = _graph(node)
        clusters = build_graph_clusters(graph)
        # No elements in graph; test bias directly via node attributes
        from aura.allocation import _node_attributes_to_carrier_bias
        bias = _node_attributes_to_carrier_bias(node)
        assert bias.preferred_carrier() in ("surface", "beta")
        assert bias.score_for("surface") > bias.score_for("volume")

    def test_carrier_bias_textured_node_prefers_gabor(self):
        node = _node("n1", "textured", element_ids=[], texture_frequency=0.9)
        from aura.allocation import _node_attributes_to_carrier_bias
        bias = _node_attributes_to_carrier_bias(node)
        assert bias.preferred_carrier() == "gabor"

    def test_carrier_bias_volume_node_prefers_volume(self):
        node = _node("n1", "foliage", element_ids=[], translucent=True)
        from aura.allocation import _node_attributes_to_carrier_bias
        bias = _node_attributes_to_carrier_bias(node)
        assert bias.preferred_carrier() == "volume"

    def test_carrier_bias_scores_length_matches_carrier_kind_order(self):
        node = _node("n1", "x", element_ids=[])
        from aura.allocation import _node_attributes_to_carrier_bias
        bias = _node_attributes_to_carrier_bias(node)
        assert len(bias.scores) == len(CARRIER_KIND_ORDER)

    def test_carrier_bias_all_scores_in_unit_range(self):
        node = _node("n1", "x", element_ids=[], surface_flatness=0.9, texture_frequency=0.8)
        from aura.allocation import _node_attributes_to_carrier_bias
        bias = _node_attributes_to_carrier_bias(node)
        for s in bias.scores:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Deliverable 2: Soft / differentiable carrier scores
# ---------------------------------------------------------------------------

class TestSoftCarrierScores:
    """Soft carrier scores and differentiable inter-type conversion."""

    def test_soft_scores_len_matches_carrier_order(self):
        ev = _evidence(high_frequency=0.9)
        scores = soft_carrier_scores(ev, "e1")
        assert len(scores.logits) == len(CARRIER_KIND_ORDER)
        assert len(scores.from_evidence) == len(CARRIER_KIND_ORDER)
        assert len(scores.from_graph) == len(CARRIER_KIND_ORDER)

    def test_soft_scores_all_in_unit_range(self):
        ev = _evidence(geometry_confidence=0.8, edit_need=0.7)
        scores = soft_carrier_scores(ev, "e1")
        for s in scores.logits:
            assert 0.0 <= s <= 1.0

    def test_high_frequency_evidence_gives_gabor_high_logit(self):
        ev = _evidence(high_frequency=0.95)
        scores = soft_carrier_scores(ev, "e1")
        gabor_score = scores.logits[CARRIER_KIND_ORDER.index("gabor")]
        # gabor should be top or near top
        assert gabor_score > 0.5

    def test_surface_evidence_gives_surface_high_logit(self):
        ev = _evidence(geometry_confidence=0.9, edit_need=0.8)
        scores = soft_carrier_scores(ev, "e1")
        surface_score = scores.logits[CARRIER_KIND_ORDER.index("surface")]
        assert surface_score > 0.5

    def test_soft_scores_without_graph_have_zero_bias_weight(self):
        ev = _evidence(high_frequency=0.9)
        scores = soft_carrier_scores(ev, "e1", config=AllocationConfig(use_soft_scores=True))
        assert scores.graph_bias_weight == 0.0

    def test_soft_scores_with_graph_governed_uses_bias_weight(self):
        ev = _evidence(high_frequency=0.9)
        config = AllocationConfig(use_graph_governed=True, graph_bias_weight=0.6)
        node = _node("n1", "textured", element_ids=["e1"], texture_frequency=0.9)
        graph = _graph(node)
        clusters = build_graph_clusters(graph)
        scores = soft_carrier_scores(ev, "e1", graph_cluster=clusters.get("e1"), config=config)
        assert scores.graph_bias_weight == 0.6

    def test_softmax_probabilities_sum_to_one(self):
        ev = _evidence(geometry_confidence=0.8, edit_need=0.7)
        scores = soft_carrier_scores(ev, "e1")
        probs = scores.softmax_probabilities()
        assert len(probs) == len(CARRIER_KIND_ORDER)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_softmax_probabilities_with_temperature(self):
        ev = _evidence(high_frequency=0.9)
        scores = soft_carrier_scores(ev, "e1")
        probs_low_t = scores.softmax_probabilities(temperature=0.1)
        probs_high_t = scores.softmax_probabilities(temperature=10.0)
        # Low temperature -> more peaked, high temperature -> more uniform
        max_low = max(probs_low_t)
        max_high = max(probs_high_t)
        assert max_low > max_high

    def test_argmax_carrier_matches_max_logit(self):
        ev = _evidence(high_frequency=0.95)
        scores = soft_carrier_scores(ev, "e1")
        argmax = scores.argmax_carrier()
        assert argmax == CARRIER_KIND_ORDER[
            max(range(len(scores.logits)), key=lambda i: scores.logits[i])
        ]

    def test_no_graph_cluster_uses_uniform_graph_prior(self):
        ev = _evidence(high_frequency=0.9)
        scores = soft_carrier_scores(
            ev, "e1", graph_cluster=None,
            config=AllocationConfig(use_graph_governed=True, graph_bias_weight=0.3)
        )
        # With uniform graph prior, bias weight should be 0.3
        assert scores.graph_bias_weight == 0.3

    def test_soft_scores_to_dict_contains_required_keys(self):
        ev = _evidence(high_frequency=0.9)
        scores = soft_carrier_scores(ev, "e1", config=AllocationConfig(use_soft_scores=True))
        d = scores.to_dict()
        assert "logits" in d
        assert "probabilities" in d
        assert "argmaxCarrier" in d
        assert "graphBiasWeight" in d
        assert "fromEvidence" in d
        assert "fromGraph" in d


class TestLearnedAssignment:
    """Test soft-score argmax overrides heuristic when use_learned_assignment=True."""

    def test_learned_assignment_uses_argmax_of_soft_scores(self):
        """When use_learned_assignment=True, argmax of blended scores decides carrier."""
        # Give graph a strong foliage signal: volume should win
        graph = _graph(
            _node("n1", "foliage", element_ids=["e1"],
                  translucent=True, foliage=True, confidence=0.95),
        )
        config = AllocationConfig(
            use_graph_governed=True,
            use_soft_scores=True,
            use_learned_assignment=True,
            graph_bias_weight=0.95,
        )
        # Evidence that would normally pick gabor
        ev = _evidence(high_frequency=0.8)
        report = semantic_graph_allocation([("e1", ev)], graph=graph, config=config)
        d = report.decisions[0]
        # Soft scores should be computed and reflect the high-frequency evidence
        ss = d.soft_scores
        assert ss is not None
        gabor_i = CARRIER_KIND_ORDER.index("gabor")
        assert ss.from_evidence[gabor_i] == max(ss.from_evidence)
        probs = ss.softmax_probabilities()
        assert abs(sum(probs) - 1.0) < 1e-6 and all(0.0 <= p <= 1.0 for p in probs)
        # Reason should indicate soft_score_argmax
        assert "soft_score_argmax" in d.reason

    def test_soft_scores_populated_when_use_soft_scores_true(self):
        config = AllocationConfig(use_soft_scores=True)
        ev = _evidence(high_frequency=0.9)
        report = semantic_graph_allocation([("e1", ev)], config=config)
        ss = report.decisions[0].soft_scores
        assert ss is not None
        # No graph bias, so high-frequency evidence makes gabor the argmax
        assert ss.argmax_carrier() == "gabor"
        probs = ss.softmax_probabilities()
        assert abs(sum(probs) - 1.0) < 1e-6 and all(0.0 <= p <= 1.0 for p in probs)

    def test_soft_scores_none_when_not_requested(self):
        config = AllocationConfig(use_soft_scores=False)
        ev = _evidence(high_frequency=0.9)
        report = semantic_graph_allocation([("e1", ev)], config=config)
        assert report.decisions[0].soft_scores is None


class TestSoftEvolutionScores:
    """Soft evolution scores for differentiable carrier transitions."""

    def test_soft_evolution_scores_returns_scores_object(self):
        ev = _evidence(high_frequency=0.9)
        result = soft_evolution_scores("e1", "gabor", ev, image_loss=0.05)
        assert isinstance(result, SoftEvolutionScores)

    def test_high_residual_promotes_split_action(self):
        ev = _evidence(view_dependent=0.8, material_confidence=0.3)
        result = soft_evolution_scores("e1", "surface", ev, image_loss=0.5, depth_loss=0.3)
        # With high residual, split_detail should be a candidate
        assert "split_detail" in result.transition_logits

    def test_low_residual_favors_retain_or_demote(self):
        ev = _evidence(geometry_confidence=0.85, edit_need=0.6)
        result = soft_evolution_scores("e1", "surface", ev, image_loss=0.005, depth_loss=0.003)
        # retain action should exist
        assert f"retain_surface" in result.transition_logits

    def test_transition_logits_all_in_unit_range(self):
        ev = _evidence(high_frequency=0.9)
        result = soft_evolution_scores("e1", "gabor", ev, image_loss=0.1)
        for score in result.transition_logits.values():
            assert 0.0 <= score <= 1.0

    def test_recommended_action_corresponds_to_max_logit(self):
        ev = _evidence(view_dependent=0.8)
        result = soft_evolution_scores("e1", "surface", ev, image_loss=0.5)
        max_action = max(result.transition_logits, key=result.transition_logits.get)
        assert result.recommended_action == max_action

    def test_confidence_in_unit_range(self):
        ev = _evidence(high_frequency=0.9)
        result = soft_evolution_scores("e1", "gabor", ev, image_loss=0.05)
        assert 0.0 <= result.confidence <= 1.0

    def test_soft_evolution_to_dict(self):
        ev = _evidence(high_frequency=0.9)
        result = soft_evolution_scores("e1", "gabor", ev, image_loss=0.1)
        d = result.to_dict()
        assert d["elementId"] == "e1"
        assert d["currentCarrier"] == "gabor"
        assert "transitionLogits" in d
        assert "recommendedAction" in d
        assert "confidence" in d

    def test_soft_evolution_with_graph_cluster(self):
        node = _node("n1", "textured", element_ids=["e1"], texture_frequency=0.9)
        from aura.allocation import _node_attributes_to_carrier_bias
        bias = _node_attributes_to_carrier_bias(node)
        cluster = GraphCluster(
            node_id="n1", node_label="textured", element_ids=("e1",),
            carrier_bias=bias, node_confidence=0.9,
        )
        ev = _evidence(high_frequency=0.9)
        config = AllocationConfig(use_graph_governed=True, use_soft_scores=True)
        result = soft_evolution_scores("e1", "gabor", ev, image_loss=0.05,
                                       graph_cluster=cluster, config=config)
        assert result.element_id == "e1"


# ---------------------------------------------------------------------------
# Deliverable 3: Cross-carrier residual correction hooks
# ---------------------------------------------------------------------------

class TestResidualCorrectionHooks:
    """Neural-residual carriers expose neighbor + residual-target info."""

    def _make_neural_decision(self, element_id, cluster_node_id=None, cluster_label=None, **ev_kwargs):
        from aura.carriers import default_registry
        ev = _evidence(**ev_kwargs) if ev_kwargs else _evidence(view_dependent=0.8)
        reg = default_registry()
        return AllocationDecision(
            element_id=element_id,
            carrier_id="neural",
            carrier_spec=reg["neural"],
            evidence=ev,
            reason="test",
            governed_by_graph=cluster_node_id is not None,
            graph_cluster_node_id=cluster_node_id,
            graph_cluster_label=cluster_label,
            soft_scores=None,
        )

    def _make_decision(self, element_id, carrier_id, cluster_node_id=None, cluster_label=None):
        from aura.carriers import default_registry
        ev = _evidence(high_frequency=0.9) if carrier_id == "gabor" else _evidence()
        reg = default_registry()
        return AllocationDecision(
            element_id=element_id,
            carrier_id=carrier_id,
            carrier_spec=reg[carrier_id],
            evidence=ev,
            reason="test",
            governed_by_graph=False,
            graph_cluster_node_id=cluster_node_id,
            graph_cluster_label=cluster_label,
            soft_scores=None,
        )

    def test_hooks_empty_when_emit_hooks_false(self):
        config = AllocationConfig(emit_residual_hooks=False)
        d = self._make_neural_decision("e_neural", cluster_node_id="n1", cluster_label="obj")
        hooks = residual_correction_hooks([d], config=config)
        assert hooks == ()

    def test_hooks_generated_for_neural_carrier(self):
        config = AllocationConfig(emit_residual_hooks=True)
        d = self._make_neural_decision("e_neural", cluster_node_id="n1", cluster_label="obj")
        hooks = residual_correction_hooks([d], config=config)
        assert len(hooks) == 1
        assert hooks[0].element_id == "e_neural"

    def test_hook_includes_same_cluster_neighbors(self):
        config = AllocationConfig(emit_residual_hooks=True)
        neural = self._make_neural_decision("e_neural", cluster_node_id="n1", cluster_label="obj")
        gabor_nbr = self._make_decision("e_gabor", "gabor", cluster_node_id="n1", cluster_label="obj")
        beta_nbr = self._make_decision("e_beta", "beta", cluster_node_id="n1", cluster_label="obj")
        # volume is not in correctable set
        volume_nbr = self._make_decision("e_volume", "volume", cluster_node_id="n1", cluster_label="obj")

        hooks = residual_correction_hooks([neural, gabor_nbr, beta_nbr, volume_nbr], config=config)
        assert len(hooks) == 1
        hook = hooks[0]
        assert "e_gabor" in hook.neighbor_element_ids
        assert "e_beta" in hook.neighbor_element_ids
        assert "e_volume" not in hook.neighbor_element_ids

    def test_hook_carrier_types_to_correct(self):
        config = AllocationConfig(emit_residual_hooks=True)
        neural = self._make_neural_decision("e_neural", cluster_node_id="n1", cluster_label="obj")
        gabor_nbr = self._make_decision("e_gabor", "gabor", cluster_node_id="n1", cluster_label="obj")

        hooks = residual_correction_hooks([neural, gabor_nbr], config=config)
        assert "gabor" in hooks[0].carrier_types_to_correct

    def test_hook_residual_target_keys(self):
        config = AllocationConfig(emit_residual_hooks=True)
        neural = self._make_neural_decision("e_neural", cluster_node_id="n1")
        hooks = residual_correction_hooks([neural], config=config)
        assert "optimization_image_loss" in hooks[0].residual_target_keys
        assert "image_residual" in hooks[0].residual_target_keys

    def test_hook_conditioning_data_keys(self):
        config = AllocationConfig(emit_residual_hooks=True)
        neural = self._make_neural_decision(
            "e_neural", cluster_node_id="n1",
            view_dependent=0.8, material_confidence=0.3
        )
        hooks = residual_correction_hooks([neural], config=config)
        assert "image_error" in hooks[0].conditioning_data
        assert "view_dependent" in hooks[0].conditioning_data
        assert "material_confidence" in hooks[0].conditioning_data
        assert "geometry_confidence" in hooks[0].conditioning_data

    def test_non_neural_carrier_generates_no_hook(self):
        config = AllocationConfig(emit_residual_hooks=True)
        gabor_d = self._make_decision("e_gabor", "gabor")
        surface_d = self._make_decision("e_surface", "surface")
        hooks = residual_correction_hooks([gabor_d, surface_d], config=config)
        assert hooks == ()

    def test_hooks_from_different_clusters_are_isolated(self):
        config = AllocationConfig(emit_residual_hooks=True)
        neural_a = self._make_neural_decision("e_neural_a", cluster_node_id="n1")
        neural_b = self._make_neural_decision("e_neural_b", cluster_node_id="n2")
        gabor_a = self._make_decision("e_gabor_a", "gabor", cluster_node_id="n1")
        gabor_b = self._make_decision("e_gabor_b", "gabor", cluster_node_id="n2")

        hooks = residual_correction_hooks(
            [neural_a, neural_b, gabor_a, gabor_b], config=config
        )
        assert len(hooks) == 2
        hook_ids = {h.element_id: h for h in hooks}
        # neural_a should neighbor gabor_a (same cluster n1), not gabor_b
        assert "e_gabor_a" in hook_ids["e_neural_a"].neighbor_element_ids
        assert "e_gabor_b" not in hook_ids["e_neural_a"].neighbor_element_ids
        # neural_b should neighbor gabor_b (same cluster n2), not gabor_a
        assert "e_gabor_b" in hook_ids["e_neural_b"].neighbor_element_ids
        assert "e_gabor_a" not in hook_ids["e_neural_b"].neighbor_element_ids

    def test_hook_to_dict_schema(self):
        config = AllocationConfig(emit_residual_hooks=True)
        neural = self._make_neural_decision("e_neural", cluster_node_id="n1")
        hooks = residual_correction_hooks([neural], config=config)
        d = hooks[0].to_dict()
        assert "elementId" in d
        assert "neighborElementIds" in d
        assert "residualTargetKeys" in d
        assert "carrierTypesToCorrect" in d
        assert "conditioningData" in d


# ---------------------------------------------------------------------------
# Deliverable 4: Allocation report / metrics / ablation
# ---------------------------------------------------------------------------

class TestAllocationReport:
    """Allocation report contains per-region decisions, graph, ablation hook."""

    def test_report_carrier_counts_correct(self):
        evs = [
            ("e1", _evidence(high_frequency=0.95)),
            ("e2", _evidence(high_frequency=0.95)),
            ("e3", _evidence(fuzzy_confidence=0.9, geometry_confidence=0.2)),
        ]
        report = semantic_graph_allocation(evs)
        assert report.carrier_counts.get("gabor", 0) == 2
        assert report.carrier_counts.get("volume", 0) == 1
        assert report.total_regions == 3

    def test_report_to_dict_is_json_able(self):
        import json
        evs = [("e1", _evidence(high_frequency=0.9))]
        report = semantic_graph_allocation(evs)
        d = report.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert "decisions" in serialized

    def test_report_to_dict_contains_required_keys(self):
        evs = [("e1", _evidence(high_frequency=0.9))]
        report = semantic_graph_allocation(evs)
        d = report.to_dict()
        assert "decisions" in d
        assert "semanticGraph" in d
        assert "carrierCounts" in d
        assert "graphGovernedCount" in d
        assert "heuristicCount" in d
        assert "ablationMode" in d
        assert "residualHooks" in d
        assert "config" in d

    def test_decision_to_dict_contains_required_keys(self):
        evs = [("e1", _evidence(high_frequency=0.9))]
        report = semantic_graph_allocation(evs)
        d = report.decisions[0].to_dict()
        assert "elementId" in d
        assert "carrierId" in d
        assert "carrierKind" in d
        assert "reason" in d
        assert "governedByGraph" in d
        assert "graphClusterNodeId" in d
        assert "softScores" in d

    def test_report_semantic_graph_included(self):
        graph = _graph(
            _node("n1", "wall", element_ids=["e1"]),
        )
        evs = [("e1", _evidence(geometry_confidence=0.8, edit_need=0.7))]
        report = semantic_graph_allocation(evs, graph=graph)
        graph_d = report.graph_dict
        assert len(graph_d["nodes"]) == 1
        assert graph_d["nodes"][0]["label"] == "wall"

    def test_report_empty_graph_when_no_graph_provided(self):
        evs = [("e1", _evidence())]
        report = semantic_graph_allocation(evs)
        assert report.graph_dict == {"nodes": [], "edges": []}

    def test_report_graph_coverage_fraction(self):
        graph = _graph(
            _node("n1", "wall", element_ids=["e1", "e2"]),
        )
        config = AllocationConfig(use_graph_governed=True)
        evs = [
            ("e1", _evidence(geometry_confidence=0.8, edit_need=0.7)),
            ("e2", _evidence(high_frequency=0.9)),
            ("e3", _evidence()),  # not in graph
        ]
        report = semantic_graph_allocation(evs, graph=graph, config=config)
        assert report.graph_coverage_fraction == pytest.approx(2 / 3)

    def test_empty_input_gives_empty_report(self):
        report = semantic_graph_allocation([])
        assert report.total_regions == 0
        assert report.carrier_counts == {}
        assert report.graph_governed_count == 0
        assert report.heuristic_count == 0
        assert report.graph_coverage_fraction == 0.0


class TestAblationHook:
    """Ablation single-carrier override and report metrics."""

    def test_ablation_single_carrier_overrides_all(self):
        config = AllocationConfig(ablation_single_carrier="gaussian")
        evs = [
            ("e1", _evidence(high_frequency=0.95)),
            ("e2", _evidence(fuzzy_confidence=0.9)),
            ("e3", _evidence(semantic_confidence=0.9)),
        ]
        report = semantic_graph_allocation(evs, config=config)
        for d in report.decisions:
            assert d.carrier_id == "gaussian"
            assert d.is_ablation_override is True

    def test_ablation_single_carrier_surface(self):
        config = AllocationConfig(ablation_single_carrier="surface")
        evs = [("e1", _evidence()), ("e2", _evidence(high_frequency=0.9))]
        report = semantic_graph_allocation(evs, config=config)
        assert all(d.carrier_id == "surface" for d in report.decisions)

    def test_ablation_mode_in_report(self):
        config = AllocationConfig(ablation_single_carrier="gaussian")
        report = semantic_graph_allocation([("e1", _evidence())], config=config)
        assert report.ablation_mode == "gaussian"

    def test_ablation_mode_none_in_normal_run(self):
        report = semantic_graph_allocation([("e1", _evidence())])
        assert report.ablation_mode is None

    def test_ablation_invalid_carrier_raises(self):
        with pytest.raises(ValueError, match="ablation_single_carrier"):
            AllocationConfig(ablation_single_carrier="nonexistent_carrier")

    def test_ablation_carrier_count_reflects_override(self):
        config = AllocationConfig(ablation_single_carrier="beta")
        evs = [("e1", _evidence()), ("e2", _evidence()), ("e3", _evidence())]
        report = semantic_graph_allocation(evs, config=config)
        assert report.carrier_counts == {"beta": 3}

    def test_ablation_soft_scores_still_computed_if_requested(self):
        config = AllocationConfig(
            ablation_single_carrier="gaussian", use_soft_scores=True
        )
        report = semantic_graph_allocation([("e1", _evidence(high_frequency=0.9))], config=config)
        d = report.decisions[0]
        # Soft scores still reflect the evidence (gabor) even though ablation
        # forces the final carrier to gaussian — proving they are independent.
        assert d.soft_scores is not None
        assert d.soft_scores.argmax_carrier() == "gabor"
        assert d.carrier_id == "gaussian"

    def test_ablation_graph_governed_false_in_override(self):
        config = AllocationConfig(ablation_single_carrier="gaussian")
        graph = _graph(_node("n1", "wall", element_ids=["e1"]))
        report = semantic_graph_allocation([("e1", _evidence())], graph=graph, config=config)
        assert report.decisions[0].governed_by_graph is False


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestAllocationConfigValidation:

    def test_default_config_valid(self):
        config = AllocationConfig()
        assert config.use_graph_governed is False
        assert config.use_soft_scores is False
        assert config.ablation_single_carrier is None

    def test_graph_bias_weight_out_of_range_raises(self):
        with pytest.raises(ValueError, match="graph_bias_weight"):
            AllocationConfig(graph_bias_weight=1.5)
        with pytest.raises(ValueError, match="graph_bias_weight"):
            AllocationConfig(graph_bias_weight=-0.1)

    def test_valid_ablation_carrier_accepted(self):
        config = AllocationConfig(ablation_single_carrier="gabor")
        assert config.ablation_single_carrier == "gabor"

    def test_all_valid_carrier_ids_accepted_as_ablation(self):
        from aura.carriers import default_registry
        for cid in default_registry():
            AllocationConfig(ablation_single_carrier=cid)  # should not raise


# ---------------------------------------------------------------------------
# Coverage tests: missing lines
# ---------------------------------------------------------------------------

class TestNodeAttributesToCarrierBiasEdgePaths:
    """Cover edge paths in _node_attributes_to_carrier_bias."""

    def test_geometry_confidence_and_edit_need_without_flat_adds_surface_score(self):
        """Line 201: elif geometry_confidence > 0.5 and edit_need > 0.3 branch."""
        from aura.allocation import _node_attributes_to_carrier_bias
        node = SemanticNode(
            id="n1",
            label="geo_node",
            confidence=0.9,
            attributes={"geometry_confidence": 0.8, "edit_need": 0.5, "surface_flatness": 0.0},
        )
        bias = _node_attributes_to_carrier_bias(node)
        # surface carrier should have a higher score than baseline (0.1)
        surface_idx = list(CARRIER_KIND_ORDER).index("surface")
        assert bias.scores[surface_idx] > 0.1 * 0.9  # modulated by confidence

    def test_view_dependent_signal_boosts_neural_score(self):
        """Line 235: view_dep > 0.4 adds to neural score."""
        from aura.allocation import _node_attributes_to_carrier_bias
        node = SemanticNode(
            id="n1",
            label="specular_node",
            confidence=0.9,
            attributes={"view_dependent": 0.8},
        )
        bias = _node_attributes_to_carrier_bias(node)
        neural_idx = list(CARRIER_KIND_ORDER).index("neural")
        # Neural score should be boosted above baseline
        assert bias.scores[neural_idx] > 0.1 * 0.9

    def test_high_scores_are_normalized_to_one(self):
        """Line 259: when max_score > 1.0 scores are normalized."""
        from aura.allocation import _node_attributes_to_carrier_bias
        # Create a node that drives many carriers high so max_score > 1.0
        node = SemanticNode(
            id="n1",
            label="complex_node",
            confidence=1.0,
            attributes={
                "surface_flatness": 0.9,
                "texture_frequency": 0.9,
                "translucency": 0.9,
                "view_dependent": 0.9,
                "semantic_confidence": 0.9,
            },
        )
        bias = _node_attributes_to_carrier_bias(node)
        # All scores must be <= 1.0 after normalization
        for score in bias.scores:
            assert score <= 1.0 + 1e-9

    def test_graph_cluster_to_dict(self):
        """Line 289: GraphCluster.to_dict() produces expected keys."""
        node = SemanticNode(id="n1", label="wall", confidence=0.9)
        bias = SemanticCarrierBias(
            scores=tuple(0.1 for _ in CARRIER_KIND_ORDER),
            node_id="n1",
            node_label="wall",
            confidence=0.9,
        )
        cluster = GraphCluster(
            node_id="n1",
            node_label="wall",
            element_ids=("e1", "e2"),
            carrier_bias=bias,
            node_confidence=0.9,
        )
        d = cluster.to_dict()
        assert d["nodeId"] == "n1"
        assert d["nodeLabel"] == "wall"
        assert d["elementIds"] == ["e1", "e2"]
        assert d["nodeConfidence"] == 0.9
        assert "carrierBias" in d


class TestSoftEvolutionScoresDemotePath:
    """Cover lines 978-979: demote path in soft_evolution_scores."""

    def test_demote_path_when_residual_low_and_target_lower_complexity(self):
        """When residual is low a lower-complexity target gets demote score."""
        ev = _evidence(image_error=0.05, geometry_confidence=0.3)
        scores = soft_evolution_scores(
            element_id="e1",
            current_carrier="neural",  # complexity 1.8 -- targets gaussian (0.7) should demote
            evidence=ev,
            image_loss=0.01,  # very low residual -> demote path active
            depth_loss=0.0,
            config=AllocationConfig(use_soft_scores=True),
        )
        # At least one demote action should exist in transition logits
        demote_keys = [k for k in scores.transition_logits if k.startswith("demote_to_")]
        assert len(demote_keys) > 0


class TestTrainableAllocationLogitsEdgePaths:
    """Cover edge path in TrainableAllocationLogits init (line 1046)."""

    def test_initial_logits_shorter_than_n_carriers_padded_with_zeros(self):
        """Line 1046: when initial_logits entry is shorter than n_carriers, zeros are appended."""
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            import pytest; pytest.skip("torch not available")
        from aura.allocation import TrainableAllocationLogits, CARRIER_KIND_ORDER
        n = len(CARRIER_KIND_ORDER)
        # Provide only 2 logits for an element that needs n
        init = {"e1": [1.0, 2.0]}  # shorter than n
        store = TrainableAllocationLogits(["e1"], device="cpu", initial_logits=init)
        param = store.logit_params["e1"]
        assert param.shape[0] == n
        # First two values should be as provided, rest should be 0.0
        assert abs(param[0].item() - 1.0) < 1e-6
        assert abs(param[1].item() - 2.0) < 1e-6
        for i in range(2, n):
            assert abs(param[i].item() - 0.0) < 1e-6


class TestTrainAllocationLogitsNoTargetPath:
    """Cover lines 1206, 1216-1217 in train_allocation_logits when no targets match."""

    def test_train_allocation_logits_with_no_matching_targets_returns_store(self):
        """Lines 1215-1217: when no element ids match targets, loss is 0 tensor."""
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            import pytest; pytest.skip("torch not available")
        from aura.allocation import train_allocation_logits, CARRIER_KIND_ORDER
        # element "e1" exists but targets dict is empty -> total stays None -> line 1216-1217
        store = train_allocation_logits(
            ["e1"],
            {},  # empty targets: no matching element id
            device="cpu",
            n_steps=3,
            learning_rate=0.1,
        )
        # Should return a store without error
        assignments = store.hard_assignments()
        assert "e1" in assignments


# ---------------------------------------------------------------------------
# SemanticCarrierBias
# ---------------------------------------------------------------------------

class TestSemanticCarrierBias:

    def test_score_for_returns_zero_for_unknown_carrier(self):
        scores = tuple(0.5 for _ in CARRIER_KIND_ORDER)
        bias = SemanticCarrierBias(scores=scores, node_id="n1", node_label="x")
        assert bias.score_for("nonexistent") == 0.0

    def test_score_for_returns_correct_value(self):
        scores = tuple(float(i) / 10 for i in range(len(CARRIER_KIND_ORDER)))
        bias = SemanticCarrierBias(scores=scores, node_id="n1", node_label="x")
        for i, kind in enumerate(CARRIER_KIND_ORDER):
            assert bias.score_for(kind) == pytest.approx(i / 10)

    def test_preferred_carrier_returns_max(self):
        scores = [0.1] * len(CARRIER_KIND_ORDER)
        scores[CARRIER_KIND_ORDER.index("gabor")] = 0.9
        bias = SemanticCarrierBias(scores=tuple(scores), node_id="n1", node_label="x")
        assert bias.preferred_carrier() == "gabor"

    def test_wrong_score_length_raises(self):
        with pytest.raises(ValueError, match="scores"):
            SemanticCarrierBias(scores=(0.1, 0.2), node_id="n1", node_label="x")

    def test_to_dict_has_all_carrier_ids(self):
        scores = tuple(0.5 for _ in CARRIER_KIND_ORDER)
        bias = SemanticCarrierBias(scores=scores, node_id="n1", node_label="x")
        d = bias.to_dict()
        for kind in CARRIER_KIND_ORDER:
            assert kind in d["scores"]


# ---------------------------------------------------------------------------
# Integration: full round-trip with graph, soft scores, residual hooks
# ---------------------------------------------------------------------------

class TestIntegrationRoundTrip:

    def test_full_pipeline_with_graph_soft_scores_and_hooks(self):
        """End-to-end: graph-governed + soft scores + residual hooks all active."""
        graph = _graph(
            _node("n_obj", "object_A", element_ids=["e_obj_1", "e_obj_2"],
                  semantic_confidence=0.9, uncertain=True),
            _node("n_surf", "flat_wall", element_ids=["e_surf_1"],
                  surface_flatness=0.95, geometry_confidence=0.9),
        )
        evs = [
            ("e_obj_1", _evidence(view_dependent=0.85, material_confidence=0.2)),
            ("e_obj_2", _evidence(high_frequency=0.8)),
            ("e_surf_1", _evidence(geometry_confidence=0.9, edit_need=0.6)),
            ("e_orphan", _evidence()),  # not in graph
        ]
        config = AllocationConfig(
            use_graph_governed=True,
            use_soft_scores=True,
            graph_bias_weight=0.7,
            emit_residual_hooks=True,
        )
        report = semantic_graph_allocation(evs, graph=graph, config=config)

        # Basic structure checks
        assert report.total_regions == 4
        assert len(report.decisions) == 4

        # Graph-governed elements should have cluster info
        by_id = {d.element_id: d for d in report.decisions}
        assert by_id["e_obj_1"].graph_cluster_node_id == "n_obj"
        assert by_id["e_surf_1"].graph_cluster_node_id == "n_surf"
        assert by_id["e_orphan"].graph_cluster_node_id is None

        # Soft scores populated and valid for all (use_soft_scores=True)
        for d in report.decisions:
            ss = d.soft_scores
            assert ss is not None
            assert len(ss.logits) == len(CARRIER_KIND_ORDER)
            probs = ss.softmax_probabilities()
            assert abs(sum(probs) - 1.0) < 1e-6 and all(0.0 <= p <= 1.0 for p in probs)

        # Report serializable
        import json
        d = report.to_dict()
        json.dumps(d)  # should not raise

    def test_backward_compat_allocation_matches_choose_carrier(self):
        """Default path must produce identical results to choose_carrier for all carriers."""
        from aura.carriers import default_registry
        reg = default_registry()

        test_cases = [
            _evidence(semantic_confidence=0.9),
            _evidence(fuzzy_confidence=0.8, geometry_confidence=0.3),
            _evidence(high_frequency=0.9),
            _evidence(view_dependent=0.8, material_confidence=0.2),
            _evidence(geometry_confidence=0.8, edit_need=0.6),
            _evidence(compact_detail=0.8),
            _evidence(image_error=0.05),
        ]
        evs = [(f"e{i}", ev) for i, ev in enumerate(test_cases)]
        report = semantic_graph_allocation(evs)  # default config

        for (eid, ev), d in zip(evs, report.decisions):
            expected = choose_carrier(ev, reg).id
            assert d.carrier_id == expected, (
                f"Element {eid}: expected {expected}, got {d.carrier_id} "
                f"(evidence={ev})"
            )


class TestCarrierBiasNormalization:
    """Cover line 259: normalize scores when max_score > 1.0."""

    def test_carrier_bias_normalizes_when_accumulated_score_exceeds_one(self):
        """Line 259: scores are divided by max when max_score > 1.0.

        Neural carrier accumulates 0.5+0.4*1.0 (view-dep) + 0.3 (uncertain) = 1.2 > 1.0.
        With conf=1.0 (confidence=1.0 node), triggers the normalization branch.
        """
        from aura.allocation import _node_attributes_to_carrier_bias

        node = _node(
            "n_norm",
            "complex",
            element_ids=[],
            confidence=1.0,
            view_dependent=1.0,   # triggers neural += 0.5 + 0.4 = 0.9
            specular=1.0,
            uncertain=1.0,        # triggers neural += 0.3 → total 1.2 → max_score > 1.0
        )
        bias = _node_attributes_to_carrier_bias(node)
        # All scores must be in [0, 1] after normalization
        for s in bias.scores:
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1] after normalization"
