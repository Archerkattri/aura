"""SOTA provider A/B reporting for optional AURA method upgrades."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"


@dataclass(frozen=True)
class SotaProviderResult:
    """Runtime status for one optional SOTA provider."""

    provider_id: str
    task: str
    status: str
    version: str | None
    device: str | None
    artifact: str | None
    metrics: Mapping[str, float] = field(default_factory=dict)
    notes: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "providerId": self.provider_id,
            "task": self.task,
            "status": self.status,
            "version": self.version,
            "device": self.device,
            "artifact": self.artifact,
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SotaUpgradeCandidate:
    """One baseline or candidate row in a task-specific A/B comparison."""

    provider_id: str
    task: str
    role: str
    primary_metric: str
    higher_is_better: bool
    metrics: Mapping[str, float]
    hard_constraints: Mapping[str, bool] = field(default_factory=dict)
    official_evidence: bool = False
    notes: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "providerId": self.provider_id,
            "task": self.task,
            "role": self.role,
            "primaryMetric": self.primary_metric,
            "higherIsBetter": self.higher_is_better,
            "metrics": dict(self.metrics),
            "hardConstraints": dict(self.hard_constraints),
            "officialEvidence": self.official_evidence,
            "notes": list(self.notes),
        }


def sota_ab_report(candidates: Sequence[SotaUpgradeCandidate]) -> dict[str, Any]:
    """Compare optional SOTA upgrades against current AURA baselines."""

    comparisons: list[dict[str, Any]] = []
    promoted: list[str] = []
    blocked: list[str] = []
    by_task: dict[str, list[SotaUpgradeCandidate]] = {}
    for candidate in candidates:
        by_task.setdefault(candidate.task, []).append(candidate)

    for task, task_candidates in sorted(by_task.items()):
        baseline = next((candidate for candidate in task_candidates if candidate.role == "baseline"), None)
        challengers = [candidate for candidate in task_candidates if candidate.role == "candidate"]
        if baseline is None or not challengers:
            blocked.append(task)
            comparisons.append({
                "task": task,
                "winnerProviderId": None,
                "promoted": False,
                "reason": "missing baseline or candidate",
                "candidates": [candidate.to_dict() for candidate in task_candidates],
            })
            continue

        winner = _winner(baseline, challengers)
        promoted_flag = winner.provider_id != baseline.provider_id
        if promoted_flag:
            promoted.append(winner.provider_id)
        comparisons.append({
            "task": task,
            "winnerProviderId": winner.provider_id,
            "promoted": promoted_flag,
            "reason": _promotion_reason(baseline, winner),
            "candidates": [candidate.to_dict() for candidate in (baseline, *challengers)],
        })

    ab_ready = not blocked and bool(comparisons)
    candidate_coverage_ready = bool(comparisons) and all(
        any(candidate.role == "candidate" and all(candidate.hard_constraints.values()) for candidate in task_candidates)
        for task_candidates in by_task.values()
    )
    return {
        "format": "AURA_SOTA_AB_REPORT",
        "abReady": ab_ready,
        "sotaReady": ab_ready and candidate_coverage_ready,
        "summary": {
            "taskCount": len(by_task),
            "comparisonCount": len(comparisons),
            "promotedProviderIds": promoted,
            "blockedTaskIds": blocked,
            "candidateCoverageReady": candidate_coverage_ready,
        },
        "comparisons": comparisons,
        "claimBoundary": (
            "SOTA readiness is limited to providers with passing A/B artifacts; "
            "blocked optional providers cannot support leaderboard claims."
        ),
    }


def latest_sota_ab_artifact(results_dir: Path | None = None) -> dict[str, Any]:
    """Load the newest SOTA A/B validation artifact, or a failing placeholder."""

    root = results_dir or RESULTS
    matches = sorted(root.glob("sota_ab_validation*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        return {
            "format": "AURA_SOTA_AB_REPORT",
            "abReady": False,
            "sotaReady": False,
            "summary": {
                "taskCount": 0,
                "comparisonCount": 0,
                "promotedProviderIds": [],
                "blockedTaskIds": ["missing_artifact"],
                "candidateCoverageReady": False,
            },
            "comparisons": [],
            "claimBoundary": "SOTA A/B validation artifact is missing.",
        }
    return json.loads(matches[0].read_text())


def _winner(
    baseline: SotaUpgradeCandidate,
    challengers: Sequence[SotaUpgradeCandidate],
) -> SotaUpgradeCandidate:
    winner = baseline
    for candidate in challengers:
        if not all(candidate.hard_constraints.values()):
            continue
        if candidate.official_evidence and not baseline.official_evidence:
            winner = candidate
            continue
        candidate_value = candidate.metrics.get(candidate.primary_metric)
        winner_value = winner.metrics.get(winner.primary_metric)
        if candidate_value is None or winner_value is None:
            continue
        if candidate.higher_is_better and candidate_value > winner_value:
            winner = candidate
        elif not candidate.higher_is_better and candidate_value < winner_value:
            winner = candidate
    return winner


def _promotion_reason(baseline: SotaUpgradeCandidate, winner: SotaUpgradeCandidate) -> str:
    if winner.provider_id == baseline.provider_id:
        return "baseline retained"
    if winner.official_evidence and not baseline.official_evidence:
        return "candidate provides official replacement evidence"
    return f"candidate wins {winner.primary_metric}"
