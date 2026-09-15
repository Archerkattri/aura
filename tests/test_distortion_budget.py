"""Tests for finite-family, render-distortion-aware stream certificates."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aura import (
    StreamCandidate,
    calibrate_distortion_budget,
    choose_stream_level,
    evaluate_stream_certificate,
    make_stream_metadata,
    metadata_overhead_ratio,
    validate_stream_metadata,
)


def _candidates():
    return [
        StreamCandidate("q_low", 100, 1.0, retained_fraction=0.10, parameters={"bits": 4}),
        StreamCandidate("q_mid", 300, 1.5, retained_fraction=0.50, parameters={"bits": 8}),
        StreamCandidate("full", 1000, 2.0, retained_fraction=1.0, is_full_asset=True),
    ]


def _plan(**kwargs):
    n = 200
    return calibrate_distortion_budget(
        _candidates(),
        {
            "q_low": np.full(n, 0.020),
            "q_mid": np.full(n, 0.010),
            "full": np.zeros(n),
        },
        alpha=0.1,
        risk_budget=0.10,
        asset_id="scene-a",
        renderer_id="aura-cpu",
        renderer_version="test",
        codec_id="spz",
        codec_version="4",
        calibration_view_ids=[f"cal-{i}" for i in range(n)],
        **kwargs,
    )


def test_calibration_is_joint_family_corrected_and_keeps_real_image_separate():
    n = 200
    plan = _plan(
        evaluation_view_ids=[f"eval-{i}" for i in range(n)],
        evaluation_image_losses={
            "q_low": np.full(n, 0.08),
            "q_mid": np.full(n, 0.03),
            "full": np.zeros(n),
        },
        source_digest="source-1",
    )

    assert plan["format"] == "AURA_STREAM_DISTORTION_CERTIFICATE"
    assert plan["n_nontrivial_levels"] == 2
    assert plan["alpha_per_level"] == pytest.approx(0.05)
    assert plan["family_wise_confidence"] == pytest.approx(0.9)
    assert plan["calibration_evidence"]["source_digest"] == "source-1"
    assert plan["levels"][-1]["trivial"] is True
    assert plan["levels"][-1]["epsilon_certified"] == 0.0
    assert plan["levels"][0]["real_image_evaluation"]["mean"] == pytest.approx(0.08)
    assert plan["loss"]["certificate_target"] == "frozen_full_asset_render_distortion"
    json.loads(json.dumps(plan))


def test_selection_uses_only_predeclared_levels_and_falls_back_or_abstains():
    plan = _plan()
    selected = choose_stream_level(plan, distortion_budget=0.10)
    # q_low's UCB is just above .10; q_mid's is below it.
    assert selected["action"] == "select"
    assert selected["selected_level_id"] == "q_mid"
    assert selected["used_full_asset_fallback"] is False

    fallback = choose_stream_level(plan, distortion_budget=0.0)
    assert fallback["action"] == "select_full_asset"
    assert fallback["selected_level_id"] == "full"
    assert fallback["used_full_asset_fallback"] is True

    abstain = choose_stream_level(plan, distortion_budget=0.0, max_bytes=500, allow_full_asset=True)
    assert abstain["action"] == "abstain"
    assert abstain["selected_level_id"] is None


def test_heldout_evaluation_reports_full_asset_and_real_image_metrics_separately():
    plan = _plan()
    n = 80
    report = evaluate_stream_certificate(
        plan,
        {
            "q_low": np.full(n, 0.04),
            "q_mid": np.full(n, 0.02),
            "full": np.zeros(n),
        },
        heldout_view_ids=[f"heldout-{i}" for i in range(n)],
        heldout_real_image_losses={
            "q_low": np.full(n, 0.12),
            "q_mid": np.full(n, 0.04),
            "full": np.zeros(n),
        },
    )
    assert report["format"] == "AURA_STREAM_DISTORTION_EVALUATION"
    assert report["all_levels_hold"] is True
    low = next(level for level in report["levels"] if level["level_id"] == "q_low")
    assert low["heldout_full_asset_distortion"]["mean"] == pytest.approx(0.04)
    assert low["heldout_real_image_loss"]["mean"] == pytest.approx(0.12)

    with pytest.raises(ValueError, match="overlap"):
        evaluate_stream_certificate(
            plan,
            {"q_low": [0.01], "q_mid": [0.01], "full": [0.0]},
            heldout_view_ids=["cal-0"],
        )


def test_metadata_binds_identity_level_set_and_expiry():
    plan = _plan()
    metadata = make_stream_metadata(
        plan,
        asset_digest="asset-sha",
        renderer_id="aura-cpu",
        renderer_version="test",
        codec_id="spz",
        codec_version="4",
        permitted_levels=["q_mid", "full"],
        expires_at=200.0,
    )
    result = validate_stream_metadata(
        metadata,
        asset_id="scene-a",
        asset_digest="asset-sha",
        renderer_id="aura-cpu",
        renderer_version="test",
        codec_id="spz",
        codec_version="4",
        level_id="q_mid",
        now=100.0,
    )
    assert result["valid"] is True
    assert result["permitted_levels"] == ["full", "q_mid"]
    # A normal splat asset is much larger than the certificate JSON; the helper
    # makes the overhead target explicit without pretending tiny fixtures are a
    # production-size asset.
    assert metadata_overhead_ratio(metadata, 300_000) < 0.01

    with pytest.raises(ValueError, match="binding mismatch"):
        validate_stream_metadata(metadata, asset_digest="other-sha")
    with pytest.raises(ValueError, match="not permitted"):
        validate_stream_metadata(metadata, level_id="q_low")
    with pytest.raises(ValueError, match="expired"):
        validate_stream_metadata(metadata, now=200.0)

    tampered = json.loads(json.dumps(metadata))
    tampered["risk"]["risk_budget"] = 0.99
    with pytest.raises(ValueError, match="certificate_digest"):
        validate_stream_metadata(tampered)


def test_rejects_invalid_or_leaky_measurements_and_ladder():
    with pytest.raises(ValueError, match="non-finite"):
        calibrate_distortion_budget(
            _candidates(),
            {"q_low": [np.nan], "q_mid": [0.1], "full": [0.0]},
        )
    with pytest.raises(ValueError, match="normalized"):
        calibrate_distortion_budget(
            _candidates(),
            {"q_low": [1.1], "q_mid": [0.1], "full": [0.0]},
        )
    with pytest.raises(ValueError, match="exactly zero"):
        calibrate_distortion_budget(
            _candidates(),
            {"q_low": [0.1], "q_mid": [0.1], "full": [1e-12]},
        )
    with pytest.raises(ValueError, match="disjoint"):
        _plan(evaluation_view_ids=["cal-0"] + [f"eval-{i}" for i in range(199)],
              evaluation_image_losses={
                  "q_low": np.full(200, 0.1),
                  "q_mid": np.full(200, 0.1),
                  "full": np.zeros(200),
              })
    with pytest.raises(ValueError, match="increase strictly"):
        calibrate_distortion_budget(
            [
                StreamCandidate("a", 100, 1.0, retained_fraction=0.5),
                StreamCandidate("b", 200, 1.0, retained_fraction=0.5),
            ],
            {"a": [0.1], "b": [0.1]},
        )


def test_mapping_candidates_and_render_time_objective():
    candidates = [
        {"id": "slow-small", "bytes": 100, "render_time": 5.0, "keep_fraction": 0.1},
        {"id": "fast-large", "bytes": 200, "render_time": 1.0, "keep_fraction": 0.5},
        {"id": "full", "bytes": 300, "render_time": 2.0, "keep_fraction": 1.0, "full_asset": True},
    ]
    plan = calibrate_distortion_budget(
        candidates,
        {"slow-small": [0.01, 0.01], "fast-large": [0.005, 0.005], "full": [0.0, 0.0]},
        risk_budget=1.0,
    )
    choice = choose_stream_level(plan, objective="render_time_ms")
    assert choice["selected_level_id"] == "fast-large"


def test_invalid_cost_arguments_are_rejected():
    plan = _plan()
    with pytest.raises(ValueError, match="max_bytes"):
        choose_stream_level(plan, max_bytes=-1)
    with pytest.raises(ValueError, match="objective"):
        choose_stream_level(plan, objective="quality")
    with pytest.raises(ValueError, match="positive integer"):
        metadata_overhead_ratio({}, 0)
