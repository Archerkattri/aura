"""Leaderboard-grade A/B report contracts for AURA.

These contracts are intentionally stricter than local publication summaries:
fixture rows, missing artifacts, or partial scene coverage must not produce a
leaderboard-SOTA claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    dataset: str
    split: str
    image_scale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "sceneId": self.scene_id,
            "dataset": self.dataset,
            "split": self.split,
            "imageScale": self.image_scale,
        }


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    role: str
    backend: str
    command: str
    environment: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "methodId": self.method_id,
            "role": self.role,
            "backend": self.backend,
            "command": self.command,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class LeaderboardMetric:
    name: str
    value: float
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "higherIsBetter": bool(self.higher_is_better),
        }


@dataclass(frozen=True)
class LeaderboardRun:
    scene_id: str
    method_id: str
    metrics: Sequence[LeaderboardMetric]
    artifacts: Sequence[str]
    measured: bool
    notes: Sequence[str] = field(default_factory=tuple)

    def metric(self, name: str) -> LeaderboardMetric | None:
        return next((metric for metric in self.metrics if metric.name == name), None)

    @property
    def promotable(self) -> bool:
        return bool(self.measured) and bool(self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "methodId": self.method_id,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "artifacts": list(self.artifacts),
            "measured": bool(self.measured),
            "promotable": self.promotable,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LeaderboardReport:
    benchmark_id: str
    task: str
    scenes: Sequence[SceneSpec]
    methods: Sequence[MethodSpec]
    runs: Sequence[LeaderboardRun]
    primary_metric: str

    def to_dict(self) -> dict[str, Any]:
        comparisons = self._comparisons()
        missing_scenes = self._missing_scenes()
        promoted = sorted(
            {
                row["winnerMethodId"]
                for row in comparisons
                if row["promoted"] and row["winnerMethodId"] is not None
            }
        )
        leaderboard_ready = not missing_scenes and bool(promoted) and all(row["ready"] for row in comparisons)
        return {
            "format": "AURA_LEADERBOARD_ABLATION_REPORT",
            "benchmarkId": self.benchmark_id,
            "task": self.task,
            "primaryMetric": self.primary_metric,
            "leaderboardReady": leaderboard_ready,
            "missingScenes": missing_scenes,
            "promotedMethodIds": promoted,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "methods": [method.to_dict() for method in self.methods],
            "runs": [run.to_dict() for run in self.runs],
            "comparisons": comparisons,
            "claimBoundary": {
                "canClaim": [
                    "fair same-split ablation rows for scenes with measured baseline and candidate artifacts",
                    "per-task candidate promotion when a measured candidate beats the measured baseline",
                ],
                "cannotClaim": [
                    "leaderboard SOTA when any required scene is missing",
                    "leaderboard SOTA from fixture rows or rows without artifacts",
                    "cross-paper superiority when dataset split, metric, or image scale differs",
                ],
            },
        }

    def _missing_scenes(self) -> list[str]:
        missing: list[str] = []
        for scene in self.scenes:
            scene_runs = [run for run in self.runs if run.scene_id == scene.scene_id and run.promotable]
            roles = {self._role_for(run.method_id) for run in scene_runs}
            if "baseline" not in roles or "candidate" not in roles:
                missing.append(scene.scene_id)
        return sorted(missing)

    def _comparisons(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scene in self.scenes:
            baseline = self._best_run(scene.scene_id, role="baseline")
            candidates = self._runs_for(scene.scene_id, role="candidate")
            winner = baseline
            reason = "missing measured baseline or candidate"
            if baseline is not None and candidates:
                for candidate in candidates:
                    winner = _metric_winner(winner, candidate, self.primary_metric) or winner
                if winner is baseline:
                    reason = "baseline retained"
                else:
                    reason = f"candidate wins {self.primary_metric}"
            rows.append(
                {
                    "sceneId": scene.scene_id,
                    "baselineMethodId": None if baseline is None else baseline.method_id,
                    "winnerMethodId": None if winner is None else winner.method_id,
                    "candidateMethodIds": [run.method_id for run in candidates],
                    "promoted": bool(baseline is not None and winner is not None and winner.method_id != baseline.method_id),
                    "ready": bool(baseline is not None and candidates),
                    "reason": reason,
                }
            )
        return rows

    def _runs_for(self, scene_id: str, *, role: str) -> list[LeaderboardRun]:
        return [
            run for run in self.runs
            if run.scene_id == scene_id and run.promotable and self._role_for(run.method_id) == role
        ]

    def _best_run(self, scene_id: str, *, role: str) -> LeaderboardRun | None:
        runs = self._runs_for(scene_id, role=role)
        if not runs:
            return None
        winner = runs[0]
        for run in runs[1:]:
            winner = _metric_winner(winner, run, self.primary_metric) or winner
        return winner

    def _role_for(self, method_id: str) -> str | None:
        method = next((method for method in self.methods if method.method_id == method_id), None)
        return None if method is None else method.role


def _metric_winner(
    left: LeaderboardRun | None,
    right: LeaderboardRun,
    metric_name: str,
) -> LeaderboardRun | None:
    if left is None:
        return right
    left_metric = left.metric(metric_name)
    right_metric = right.metric(metric_name)
    if left_metric is None or right_metric is None:
        return left
    if right_metric.higher_is_better:
        return right if right_metric.value > left_metric.value else left
    return right if right_metric.value < left_metric.value else left
