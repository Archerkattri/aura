import json

from aura.sota import SotaProviderResult, SotaUpgradeCandidate, latest_sota_ab_artifact, sota_ab_report


def test_sota_provider_result_serializes_blocked_provider():
    result = SotaProviderResult(
        provider_id="dinov3",
        task="semantics",
        status="blocked",
        version=None,
        device=None,
        artifact=None,
        metrics={},
        notes=("missing gated weights",),
    )

    assert result.to_dict() == {
        "providerId": "dinov3",
        "task": "semantics",
        "status": "blocked",
        "version": None,
        "device": None,
        "artifact": None,
        "metrics": {},
        "notes": ["missing gated weights"],
    }


def test_sota_ab_report_promotes_quality_winner():
    baseline = SotaUpgradeCandidate(
        provider_id="current_semantic",
        task="semantics",
        role="baseline",
        primary_metric="queryConsistency",
        higher_is_better=True,
        metrics={"queryConsistency": 0.62, "gpu": 1.0},
        hard_constraints={"gpu": True},
        official_evidence=False,
    )
    candidate = SotaUpgradeCandidate(
        provider_id="dinov3",
        task="semantics",
        role="candidate",
        primary_metric="queryConsistency",
        higher_is_better=True,
        metrics={"queryConsistency": 0.71, "gpu": 1.0},
        hard_constraints={"gpu": True},
        official_evidence=False,
    )

    report = sota_ab_report([baseline, candidate])

    assert report["sotaReady"] is True
    assert report["summary"]["promotedProviderIds"] == ["dinov3"]
    assert report["comparisons"][0]["winnerProviderId"] == "dinov3"


def test_sota_ab_report_keeps_losing_candidate_unpromoted():
    baseline = SotaUpgradeCandidate(
        provider_id="current_semantic",
        task="semantics",
        role="baseline",
        primary_metric="queryConsistency",
        higher_is_better=True,
        metrics={"queryConsistency": 0.62, "gpu": 1.0},
        hard_constraints={"gpu": True},
        official_evidence=False,
    )
    candidate = SotaUpgradeCandidate(
        provider_id="dinov3",
        task="semantics",
        role="candidate",
        primary_metric="queryConsistency",
        higher_is_better=True,
        metrics={"queryConsistency": 0.50, "gpu": 1.0},
        hard_constraints={"gpu": True},
        official_evidence=False,
    )

    report = sota_ab_report([baseline, candidate])

    assert report["summary"]["promotedProviderIds"] == []
    assert report["comparisons"][0]["winnerProviderId"] == "current_semantic"
    assert report["comparisons"][0]["promoted"] is False


def test_sota_ab_report_promotes_official_replacement_evidence():
    baseline = SotaUpgradeCandidate(
        provider_id="local_2dgs_style_smoke",
        task="surface_baseline",
        role="baseline",
        primary_metric="officialEvidenceScore",
        higher_is_better=True,
        metrics={"officialEvidenceScore": 0.0, "gpu": 1.0},
        hard_constraints={"gpu": True},
        official_evidence=False,
    )
    candidate = SotaUpgradeCandidate(
        provider_id="official_2dgs",
        task="surface_baseline",
        role="candidate",
        primary_metric="officialEvidenceScore",
        higher_is_better=True,
        metrics={"officialEvidenceScore": 1.0, "gpu": 1.0},
        hard_constraints={"gpu": True},
        official_evidence=True,
    )

    report = sota_ab_report([baseline, candidate])

    assert report["comparisons"][0]["winnerProviderId"] == "official_2dgs"
    assert report["comparisons"][0]["reason"] == "candidate provides official replacement evidence"


def test_latest_sota_ab_artifact_reads_newest_report(tmp_path, monkeypatch):
    old = tmp_path / "sota_ab_validation_2026-06-24.json"
    new = tmp_path / "sota_ab_validation_2026-06-25.json"
    old.write_text(json.dumps({"format": "AURA_SOTA_AB_REPORT", "sotaReady": False}))
    new.write_text(json.dumps({"format": "AURA_SOTA_AB_REPORT", "sotaReady": True}))
    monkeypatch.setattr("aura.sota.RESULTS", tmp_path)

    assert latest_sota_ab_artifact()["sotaReady"] is True


def test_fixture_sota_report_keeps_losing_dinov3_candidate_unpromoted():
    from experiments.sota_ab_validation import fixture_candidates

    report = sota_ab_report(fixture_candidates(dinov3_query_consistency=0.5))

    semantic = next(item for item in report["comparisons"] if item["task"] == "semantics")
    assert semantic["winnerProviderId"] == "current_semantic"
    assert semantic["promoted"] is False


def test_fixture_sota_report_does_not_promote_unmeasured_upgrades():
    from experiments.sota_ab_validation import fixture_candidates

    report = sota_ab_report(fixture_candidates())

    assert report["abReady"] is True
    assert report["sotaReady"] is False
    assert report["summary"]["promotedProviderIds"] == []
    assert {item["winnerProviderId"] for item in report["comparisons"]} == {
        "current_semantic",
        "current_colmap_capture",
        "local_ray_query",
        "local_2dgs_style_smoke",
    }


def test_measured_sota_report_promotes_dinov3_after_diversity_parity_fix():
    from experiments.sota_ab_validation import measured_candidates_2026_06_25

    report = sota_ab_report(measured_candidates_2026_06_25())

    semantic = next(item for item in report["comparisons"] if item["task"] == "semantics")
    dinov3 = next(item for item in semantic["candidates"] if item["providerId"] == "dinov3_small_timm")
    assert semantic["winnerProviderId"] == "dinov3_small_timm"
    assert semantic["promoted"] is True
    assert dinov3["hardConstraints"]["semanticDiversityParity"] is True
    assert dinov3["metrics"]["queryDiversity"] == 1.0


def test_measured_geometry_priors_are_larger_than_smoke_and_keep_colmap_default():
    from experiments.sota_ab_validation import measured_candidates_2026_06_25

    report = sota_ab_report(measured_candidates_2026_06_25())

    geometry = next(item for item in report["comparisons"] if item["task"] == "geometry_priors")
    assert geometry["winnerProviderId"] == "current_colmap_capture"
    assert geometry["promoted"] is False
    for provider_id in ("vggt_1b", "depth_anything_3_small"):
        candidate = next(item for item in geometry["candidates"] if item["providerId"] == provider_id)
        assert candidate["metrics"]["images"] >= 12.0
        assert candidate["metrics"]["validPriorCoverage"] == 1.0
        assert candidate["hardConstraints"]["repairEvidence"] is True
