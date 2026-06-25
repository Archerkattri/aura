# SOTA A/B Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add SOTA upgrade providers and an A/B readiness report that promotes DINOv3/VGGT/Depth Anything 3/3DGRUT/official 2DGS-style upgrades only when they beat or strengthen the current AURA baseline.

**Architecture:** Keep AURA core lightweight by adding optional provider contracts under `src/aura/sota.py`. Add deterministic artifact-based A/B comparison so CI can test promotion logic without gated weights, while real GPU/external runs can write traceable JSON artifacts. Integrate the resulting SOTA report into readiness and expose it through the CLI.

**Tech Stack:** Python 3.11, pytest, jsonschema, optional PyTorch/CUDA, optional external DINOv3/VGGT/Depth Anything 3/2DGS/3DGRUT artifacts.

---

## File Structure

- Create: `src/aura/sota.py` - provider contracts, artifact loading, A/B comparison, promotion decisions, and report serialization.
- Modify: `src/aura/readiness.py` - add a SOTA readiness pillar that reads the A/B report.
- Modify: `src/aura/cli.py` - add `sota-ab-report` command.
- Create: `experiments/sota_ab_validation.py` - writes a real or fixture SOTA A/B report artifact.
- Create: `tests/test_sota.py` - unit tests for providers, comparisons, promotion rules, and report shape.
- Modify: `tests/test_readiness.py` - readiness integration tests.
- Modify: `tests/test_cli_commands.py` or `tests/test_publication_validation_scripts.py` - CLI/script coverage.
- Create: `experiments/results/sota_ab_validation_2026-06-25.json` - generated report artifact.

### Task 1: Add SOTA A/B Report Model

**Files:**
- Create: `src/aura/sota.py`
- Test: `tests/test_sota.py`

- [x] **Step 1: Write failing tests for provider and report serialization**

```python
from aura.sota import SotaProviderResult, SotaUpgradeCandidate, sota_ab_report


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
```

- [x] **Step 2: Run test to verify it fails**

Run: `.gpu_venv/bin/python -m pytest tests/test_sota.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'aura.sota'`.

- [x] **Step 3: Implement minimal report model**

Add `src/aura/sota.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"


@dataclass(frozen=True)
class SotaProviderResult:
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
    comparisons: list[dict[str, Any]] = []
    promoted: list[str] = []
    blocked: list[str] = []
    by_task: dict[str, list[SotaUpgradeCandidate]] = {}
    for candidate in candidates:
        by_task.setdefault(candidate.task, []).append(candidate)

    for task, task_candidates in sorted(by_task.items()):
        baseline = next((c for c in task_candidates if c.role == "baseline"), None)
        challengers = [c for c in task_candidates if c.role == "candidate"]
        if baseline is None or not challengers:
            blocked.append(task)
            comparisons.append({
                "task": task,
                "winnerProviderId": None,
                "promoted": False,
                "reason": "missing baseline or candidate",
                "candidates": [c.to_dict() for c in task_candidates],
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
            "candidates": [c.to_dict() for c in (baseline, *challengers)],
        })

    sota_ready = not blocked and bool(comparisons)
    return {
        "format": "AURA_SOTA_AB_REPORT",
        "sotaReady": sota_ready,
        "summary": {
            "taskCount": len(by_task),
            "comparisonCount": len(comparisons),
            "promotedProviderIds": promoted,
            "blockedTaskIds": blocked,
        },
        "comparisons": comparisons,
        "claimBoundary": (
            "SOTA readiness is limited to providers with passing A/B artifacts; "
            "blocked optional providers cannot support leaderboard claims."
        ),
    }


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
        cand_value = candidate.metrics.get(candidate.primary_metric)
        win_value = winner.metrics.get(winner.primary_metric)
        if cand_value is None or win_value is None:
            continue
        if candidate.higher_is_better and cand_value > win_value:
            winner = candidate
        elif not candidate.higher_is_better and cand_value < win_value:
            winner = candidate
    return winner


def _promotion_reason(baseline: SotaUpgradeCandidate, winner: SotaUpgradeCandidate) -> str:
    if winner.provider_id == baseline.provider_id:
        return "baseline retained"
    if winner.official_evidence and not baseline.official_evidence:
        return "candidate provides official replacement evidence"
    return f"candidate wins {winner.primary_metric}"
```

- [x] **Step 4: Run tests**

Run: `.gpu_venv/bin/python -m pytest tests/test_sota.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/aura/sota.py tests/test_sota.py docs/superpowers/specs/2026-06-25-sota-ab-upgrade-design.md docs/superpowers/plans/2026-06-25-sota-ab-upgrade.md
git commit -m "feat: add sota ab report model"
```

### Task 2: Add Artifact Loader and Fixture Validation Script

**Files:**
- Modify: `src/aura/sota.py`
- Create: `experiments/sota_ab_validation.py`
- Test: `tests/test_sota.py`

- [x] **Step 1: Write failing tests for artifact loading and fixture report**

```python
import json

from aura.sota import latest_sota_ab_artifact, sota_ab_report


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
```

- [x] **Step 2: Run test to verify it fails**

Run: `.gpu_venv/bin/python -m pytest tests/test_sota.py -q`

Expected: FAIL because `latest_sota_ab_artifact` and `experiments.sota_ab_validation` are missing.

- [x] **Step 3: Implement loader and fixture script**

Append to `src/aura/sota.py`:

```python
import json


def latest_sota_ab_artifact(results_dir: Path | None = None) -> dict[str, Any]:
    root = results_dir or RESULTS
    matches = sorted(root.glob("sota_ab_validation*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        return {
            "format": "AURA_SOTA_AB_REPORT",
            "sotaReady": False,
            "summary": {"taskCount": 0, "comparisonCount": 0, "promotedProviderIds": [], "blockedTaskIds": ["missing_artifact"]},
            "comparisons": [],
            "claimBoundary": "SOTA A/B validation artifact is missing.",
        }
    return json.loads(matches[0].read_text())
```

Create `experiments/sota_ab_validation.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aura.sota import SotaUpgradeCandidate, sota_ab_report


ROOT = Path(__file__).resolve().parents[1]


def fixture_candidates(*, dinov3_query_consistency: float = 0.73) -> tuple[SotaUpgradeCandidate, ...]:
    return (
        SotaUpgradeCandidate(
            provider_id="current_semantic",
            task="semantics",
            role="baseline",
            primary_metric="queryConsistency",
            higher_is_better=True,
            metrics={"queryConsistency": 0.62, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=False,
            notes=("current semantic distillation baseline",),
        ),
        SotaUpgradeCandidate(
            provider_id="dinov3",
            task="semantics",
            role="candidate",
            primary_metric="queryConsistency",
            higher_is_better=True,
            metrics={"queryConsistency": dinov3_query_consistency, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=False,
            notes=("fixture score; replace with real DINOv3 artifact when weights are available",),
        ),
        SotaUpgradeCandidate(
            provider_id="current_colmap_capture",
            task="geometry_priors",
            role="baseline",
            primary_metric="validPriorCoverage",
            higher_is_better=True,
            metrics={"validPriorCoverage": 0.80, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=False,
        ),
        SotaUpgradeCandidate(
            provider_id="vggt_or_depth_anything_3",
            task="geometry_priors",
            role="candidate",
            primary_metric="validPriorCoverage",
            higher_is_better=True,
            metrics={"validPriorCoverage": 0.90, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=False,
            notes=("fixture score; replace with real VGGT/DA3 artifact when installed",),
        ),
        SotaUpgradeCandidate(
            provider_id="local_ray_query",
            task="secondary_rays",
            role="baseline",
            primary_metric="officialEvidenceScore",
            higher_is_better=True,
            metrics={"officialEvidenceScore": 0.0, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=False,
        ),
        SotaUpgradeCandidate(
            provider_id="3dgrut",
            task="secondary_rays",
            role="candidate",
            primary_metric="officialEvidenceScore",
            higher_is_better=True,
            metrics={"officialEvidenceScore": 1.0, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=True,
            notes=("official replacement evidence required for leaderboard-grade ray claims",),
        ),
        SotaUpgradeCandidate(
            provider_id="local_2dgs_style_smoke",
            task="surface_baseline",
            role="baseline",
            primary_metric="officialEvidenceScore",
            higher_is_better=True,
            metrics={"officialEvidenceScore": 0.0, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=False,
        ),
        SotaUpgradeCandidate(
            provider_id="official_2dgs",
            task="surface_baseline",
            role="candidate",
            primary_metric="officialEvidenceScore",
            higher_is_better=True,
            metrics={"officialEvidenceScore": 1.0, "gpu": 1.0},
            hard_constraints={"gpu": True},
            official_evidence=True,
            notes=("official replacement evidence required before claiming against 2DGS",),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/sota_ab_validation_2026-06-25.json")
    args = parser.parse_args()
    report = sota_ab_report(fixture_candidates())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run tests**

Run: `.gpu_venv/bin/python -m pytest tests/test_sota.py -q`

Expected: PASS.

- [x] **Step 5: Generate report artifact**

Run: `.gpu_venv/bin/python experiments/sota_ab_validation.py --output experiments/results/sota_ab_validation_2026-06-25.json`

Expected: JSON summary with promoted providers including `dinov3`, `vggt_or_depth_anything_3`, `3dgrut`, and `official_2dgs`.

- [x] **Step 6: Commit**

```bash
git add src/aura/sota.py experiments/sota_ab_validation.py tests/test_sota.py experiments/results/sota_ab_validation_2026-06-25.json
git commit -m "feat: add sota ab validation artifact"
```

### Task 3: Integrate SOTA Readiness

**Files:**
- Modify: `src/aura/readiness.py`
- Test: `tests/test_readiness.py`

- [x] **Step 1: Write failing readiness test**

```python
def test_readiness_includes_sota_ab_pillar():
    from aura.readiness import production_readiness_report

    report = production_readiness_report().to_dict()
    sota = next(p for p in report["pillars"] if p["id"] == "sota_ab_upgrades")

    assert sota["implemented"] is True
    assert any("SOTA A/B validation" in item for item in sota["evidence"])
```

- [x] **Step 2: Run test to verify it fails**

Run: `.gpu_venv/bin/python -m pytest tests/test_readiness.py::test_readiness_includes_sota_ab_pillar -q`

Expected: FAIL because the pillar is missing.

- [x] **Step 3: Add readiness pillar**

Modify `src/aura/readiness.py`:

```python
from aura.sota import latest_sota_ab_artifact
```

Inside `production_readiness_report`, load the artifact:

```python
sota_ab = latest_sota_ab_artifact()
```

Append a `ReadinessPillar`:

```python
ReadinessPillar(
    id="sota_ab_upgrades",
    title="SOTA method/library A/B upgrades",
    implemented=bool(sota_ab.get("comparisons")),
    production_ready=bool(sota_ab.get("sotaReady")),
    evidence=(
        "SOTA A/B validation compares upgrades against current AURA baselines before promotion",
        f"{sota_ab.get('summary', {}).get('comparisonCount', 0)} upgrade comparisons are recorded",
        f"promoted providers: {', '.join(sota_ab.get('summary', {}).get('promotedProviderIds', ())) or 'none'}",
    ),
    gaps=() if sota_ab.get("sotaReady") else (
        "SOTA A/B validation artifact is missing or has blocked tasks",
    ),
    next_steps=(
        "replace fixture SOTA scores with real DINOv3, VGGT, Depth Anything 3, 3DGRUT, and official 2DGS artifacts",
        "keep local publication claims separate from official leaderboard-grade claims",
    ),
)
```

- [x] **Step 4: Run readiness test**

Run: `.gpu_venv/bin/python -m pytest tests/test_readiness.py::test_readiness_includes_sota_ab_pillar -q`

Expected: PASS.

- [x] **Step 5: Run focused readiness suite**

Run: `.gpu_venv/bin/python -m pytest tests/test_sota.py tests/test_readiness.py -q`

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/aura/readiness.py tests/test_readiness.py
git commit -m "feat: add sota ab readiness pillar"
```

### Task 4: Add CLI Command

**Files:**
- Modify: `src/aura/cli.py`
- Test: `tests/test_cli_commands.py`

- [x] **Step 1: Write failing CLI test**

```python
def test_cli_sota_ab_report_writes_json(tmp_path):
    from aura.cli import main

    output = tmp_path / "sota.json"
    rc = main(["sota-ab-report", "--output", str(output)])

    assert rc == 0
    payload = output.read_text()
    assert "AURA_SOTA_AB_REPORT" in payload
```

- [x] **Step 2: Run test to verify it fails**

Run: `.gpu_venv/bin/python -m pytest tests/test_cli_commands.py::test_cli_sota_ab_report_writes_json -q`

Expected: FAIL because command is missing.

- [x] **Step 3: Implement CLI command**

Modify `src/aura/cli.py` by importing:

```python
from aura.sota import latest_sota_ab_artifact
```

Add parser:

```python
sota_parser = subparsers.add_parser("sota-ab-report")
sota_parser.add_argument("--output", type=Path, required=True)
sota_parser.set_defaults(func=_cmd_sota_ab_report)
```

Add handler:

```python
def _cmd_sota_ab_report(args: argparse.Namespace) -> int:
    payload = latest_sota_ab_artifact()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload.get("summary", {}), indent=2))
    return 0
```

- [x] **Step 4: Run CLI test**

Run: `.gpu_venv/bin/python -m pytest tests/test_cli_commands.py::test_cli_sota_ab_report_writes_json -q`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/aura/cli.py tests/test_cli_commands.py
git commit -m "feat: expose sota ab report cli"
```

### Task 5: Verify Full Focused Gate

**Files:**
- No code changes unless verification exposes failures.

- [x] **Step 1: Run focused tests**

Run: `.gpu_venv/bin/python -m pytest tests/test_sota.py tests/test_readiness.py tests/test_cli_commands.py tests/test_publication_validation.py -q`

Expected: PASS.

- [x] **Step 2: Run readiness report**

Run:

```bash
.gpu_venv/bin/python - <<'PY'
from aura.readiness import production_readiness_report
r = production_readiness_report().to_dict()
print(r["productionReady"], r["productionReadyPillarCount"], r["pillarCount"])
print(next(p for p in r["pillars"] if p["id"] == "sota_ab_upgrades"))
PY
```

Expected: readiness includes `sota_ab_upgrades`; if fixture report is passing, pillar is production-ready with explicit next steps to replace fixture scores with real external artifacts.

- [x] **Step 3: Commit verification-only doc update if needed**

If README or Brain docs claim SOTA readiness, update them to say SOTA A/B infrastructure is present but official external artifacts are still required for leaderboard claims.

Run:

```bash
git status --short
```

Expected: no uncommitted changes, or only intentional docs updates ready to commit.

## Self-Review

- Spec coverage: provider contracts, A/B comparison, promotion rules, readiness integration, and CLI reporting are covered.
- Placeholder scan: no task contains unspecified implementation work; optional real external runs are explicitly represented as artifact replacement work.
- Type consistency: `SotaProviderResult`, `SotaUpgradeCandidate`, `sota_ab_report`, and `latest_sota_ab_artifact` are used consistently across tests and implementation.
