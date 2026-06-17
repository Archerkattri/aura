from types import SimpleNamespace

import pytest

from aura.elements import AuraElement, Bounds
from aura.evolution import (
    CarrierEvolutionPolicy,
    carrier_evolution_decisions,
    carrier_evolution_report,
    evolved_element_for,
    simplification_metadata,
)


def _element(element_id: str, carrier_id: str, **kwargs) -> AuraElement:
    return AuraElement(
        id=element_id,
        carrier_id=carrier_id,
        bounds=kwargs.pop("bounds", Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
        color=kwargs.pop("color", (0.4, 0.5, 0.6)),
        opacity=kwargs.pop("opacity", 0.7),
        confidence=kwargs.pop("confidence", 0.8),
        semantic_id=kwargs.pop("semantic_id", None),
        material_id=kwargs.pop("material_id", None),
        metadata=kwargs.pop("metadata", {}),
        confidence_map=kwargs.pop("confidence_map", {}),
        payload=kwargs.pop("payload", {}),
        **kwargs,
    )


def _prediction(element_id: str, carrier_id: str, **kwargs):
    return SimpleNamespace(
        element_id=element_id,
        carrier_id=carrier_id,
        image_loss=kwargs.pop("image_loss", 0.2),
        depth_loss=kwargs.pop("depth_loss", 0.0),
        query_loss=kwargs.pop("query_loss", 0.0),
        normal_loss=kwargs.pop("normal_loss", 0.0),
        target_color=kwargs.pop("target_color", (0.9, 0.2, 0.1)),
    )


def test_policy_splits_high_residual_volume_into_beta_child():
    parent = _element("soft_volume", "volume", material_id="mat_soft")
    prediction = _prediction("soft_volume", "volume", image_loss=0.2)

    decisions = carrier_evolution_decisions((prediction,), (parent,), policy=CarrierEvolutionPolicy(), iteration=0)
    decision = decisions[0]
    child = evolved_element_for(parent, decision, prediction)

    assert decision.action == "split_beta_detail"
    assert decision.created_element_id == "soft_volume_beta_detail"
    assert decision.metrics == {
        "imageLoss": 0.2,
        "depthLoss": 0.0,
        "queryLoss": 0.0,
        "normalLoss": 0.0,
        "residual": 0.2,
    }
    assert decision.thresholds == {"splitImageLossThreshold": 0.03}
    assert child is not None
    assert child.id == "soft_volume_beta_detail"
    assert child.carrier_id == "beta"
    assert child.payload["type"] == "beta_kernel"
    assert child.metadata == {
        "source": "aura-core-adaptive-evolution",
        "parent": "soft_volume",
        "evolution": "split_beta_detail",
    }
    assert child.confidence_map["optimization_residual"] == 0.2


def test_policy_promotes_semantic_residual_into_neural_child():
    parent = _element("semantic_object", "semantic", semantic_id="fixture_object", opacity=0.45)
    prediction = _prediction("semantic_object", "semantic", image_loss=0.18)

    decision = carrier_evolution_decisions((prediction,), (parent,), policy=CarrierEvolutionPolicy(), iteration=0)[0]
    child = evolved_element_for(parent, decision, prediction)

    assert decision.action == "promote_neural_residual"
    assert decision.created_element_id == "semantic_object_neural_residual"
    assert decision.thresholds == {"splitImageLossThreshold": 0.03}
    assert child is not None
    assert child.carrier_id == "neural"
    assert child.semantic_id == "fixture_object"
    assert child.residual is True
    assert child.payload["type"] == "neural_residual"
    assert child.payload["residual_scale"] == pytest.approx(0.72)


def test_policy_merges_converged_beta_detail_and_reports_removed_child():
    parent = _element("soft_volume", "volume")
    child = _element("soft_volume_beta_detail", "beta")
    prediction = _prediction("soft_volume", "volume", image_loss=0.01, depth_loss=0.01)

    decision = carrier_evolution_decisions((prediction,), (parent, child), policy=CarrierEvolutionPolicy(), iteration=4)[0]
    report = carrier_evolution_report((decision,))

    assert decision.action == "merge_beta_detail"
    assert decision.created_element_id == "soft_volume_beta_detail"
    assert decision.thresholds == {
        "mergeImageLossThreshold": 0.025,
        "mergeDepthLossThreshold": 0.04,
    }
    assert simplification_metadata(decision) == {
        "simplified_child": "soft_volume_beta_detail",
        "simplification": "merge_beta_detail",
    }
    assert report == {
        "actionCounts": {"merge_beta_detail": 1},
        "createdElementIds": [],
        "removedElementIds": ["soft_volume_beta_detail"],
        "retainedElementIds": [],
    }


def test_policy_demotes_converged_neural_residual_after_iteration_gate():
    parent = _element("semantic_object", "semantic")
    child = _element("semantic_object_neural_residual", "neural")
    prediction = _prediction("semantic_object", "semantic", image_loss=0.01, depth_loss=0.01)
    policy = CarrierEvolutionPolicy(demote_after_iteration=3)

    before_gate = carrier_evolution_decisions((prediction,), (parent, child), policy=policy, iteration=2)[0]
    after_gate = carrier_evolution_decisions((prediction,), (parent, child), policy=policy, iteration=3)[0]

    assert before_gate.action == "retain_semantic_carrier"
    assert after_gate.action == "demote_neural_residual"
    assert after_gate.created_element_id == "semantic_object_neural_residual"
    assert after_gate.thresholds == {
        "demoteAfterIteration": 3.0,
        "demoteImageLossThreshold": 0.045,
        "demoteDepthLossThreshold": 0.02,
    }


def test_policy_hysteresis_prevents_immediate_recreate_after_simplification():
    merged_volume = _element(
        "soft_volume",
        "volume",
        metadata={"simplified_child": "soft_volume_beta_detail", "simplification": "merge_beta_detail"},
    )
    demoted_semantic = _element(
        "semantic_object",
        "semantic",
        metadata={
            "simplified_child": "semantic_object_neural_residual",
            "simplification": "demote_neural_residual",
        },
    )

    decisions = carrier_evolution_decisions(
        (
            _prediction("soft_volume", "volume", image_loss=0.2),
            _prediction("semantic_object", "semantic", image_loss=0.2),
        ),
        (merged_volume, demoted_semantic),
        policy=CarrierEvolutionPolicy(),
        iteration=5,
    )

    assert [decision.action for decision in decisions] == ["retain_carrier", "retain_semantic_carrier"]
    assert carrier_evolution_report(decisions)["retainedElementIds"] == ["semantic_object", "soft_volume"]


def test_policy_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="split_image_loss_threshold"):
        CarrierEvolutionPolicy(split_image_loss_threshold=-0.1)
    with pytest.raises(ValueError, match="demote_after_iteration"):
        CarrierEvolutionPolicy(demote_after_iteration=-1)
