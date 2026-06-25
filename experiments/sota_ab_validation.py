from __future__ import annotations

import argparse
import json
from pathlib import Path

from aura.sota import SotaUpgradeCandidate, sota_ab_report

ROOT = Path(__file__).resolve().parents[1]


def fixture_candidates(*, dinov3_query_consistency: float = 0.73) -> tuple[SotaUpgradeCandidate, ...]:
    """Fixture A/B rows used until real external provider artifacts are imported."""

    return (
        SotaUpgradeCandidate(
            provider_id="current_semantic",
            task="semantics",
            role="baseline",
            primary_metric="queryConsistency",
            higher_is_better=True,
            metrics={"queryConsistency": 0.62, "gpu": 1.0},
            hard_constraints={"gpu": True, "realArtifact": True},
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
            hard_constraints={"gpu": True, "realArtifact": False},
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
            hard_constraints={"gpu": True, "realArtifact": True},
            official_evidence=False,
        ),
        SotaUpgradeCandidate(
            provider_id="vggt_or_depth_anything_3",
            task="geometry_priors",
            role="candidate",
            primary_metric="validPriorCoverage",
            higher_is_better=True,
            metrics={"validPriorCoverage": 0.90, "gpu": 1.0},
            hard_constraints={"gpu": True, "realArtifact": False},
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
            hard_constraints={"gpu": True, "realArtifact": True},
            official_evidence=False,
        ),
        SotaUpgradeCandidate(
            provider_id="3dgrut",
            task="secondary_rays",
            role="candidate",
            primary_metric="officialEvidenceScore",
            higher_is_better=True,
            metrics={"officialEvidenceScore": 1.0, "gpu": 1.0},
            hard_constraints={"gpu": True, "realArtifact": False},
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
            hard_constraints={"gpu": True, "realArtifact": True},
            official_evidence=False,
        ),
        SotaUpgradeCandidate(
            provider_id="official_2dgs",
            task="surface_baseline",
            role="candidate",
            primary_metric="officialEvidenceScore",
            higher_is_better=True,
            metrics={"officialEvidenceScore": 1.0, "gpu": 1.0},
            hard_constraints={"gpu": True, "realArtifact": False},
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
