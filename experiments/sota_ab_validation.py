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


def measured_candidates_2026_06_25() -> tuple[SotaUpgradeCandidate, ...]:
    """Measured A/B rows from the 2026-06-25 GPU SOTA provider pass."""

    return (
        SotaUpgradeCandidate(
            provider_id="current_semantic_dinov2",
            task="semantics",
            role="baseline",
            primary_metric="aggregateQueryMargin",
            higher_is_better=True,
            metrics={
                "carrierCoverage": 0.965416,
                "aggregateQueryMargin": 1.125,
                "queryDiversity": 0.75,
                "wheelQueryMargin": 3.0,
            },
            hard_constraints={"gpu": True, "realArtifact": True, "semanticDiversity": True},
            official_evidence=False,
            notes=(
                "DINOv2 distilled 965416/1000000 Truck carriers with view_stride=32",
                "CLIP group-query best groups for truck/wheel/ground/building were 5/5/2/6",
            ),
        ),
        SotaUpgradeCandidate(
            provider_id="dinov3_small_timm",
            task="semantics",
            role="candidate",
            primary_metric="aggregateQueryMargin",
            higher_is_better=True,
            metrics={
                "carrierCoverage": 0.965416,
                "aggregateQueryMargin": 1.175,
                "queryDiversity": 0.50,
                "wheelQueryMargin": 3.1,
            },
            hard_constraints={"gpu": True, "realArtifact": True, "semanticDiversity": False},
            official_evidence=False,
            notes=(
                "pretrained timm vit_small_patch16_dinov3 loaded and ran on cuda",
                "not promoted: quick run collapsed truck/wheel/ground to group 6 despite a slightly higher aggregate margin",
            ),
        ),
        SotaUpgradeCandidate(
            provider_id="current_colmap_capture",
            task="geometry_priors",
            role="baseline",
            primary_metric="validPriorCoverage",
            higher_is_better=True,
            metrics={"validPriorCoverage": 1.0, "images": 251.0},
            hard_constraints={"gpu": True, "realArtifact": True},
            official_evidence=False,
            notes=("COLMAP capture path remains the default for already-posed Truck data",),
        ),
        SotaUpgradeCandidate(
            provider_id="vggt_1b",
            task="geometry_priors",
            role="candidate",
            primary_metric="validPriorCoverage",
            higher_is_better=True,
            metrics={"validPriorCoverage": 1.0, "images": 4.0, "seconds": 0.4366},
            hard_constraints={"gpu": True, "realArtifact": True},
            official_evidence=False,
            notes=(
                "VGGT-1B loaded from Hugging Face and produced finite pose/depth/world-point tensors on 4 Truck images",
                "kept as optional feed-forward prior; not promoted over COLMAP for already-posed local data",
            ),
        ),
        SotaUpgradeCandidate(
            provider_id="depth_anything_3_small",
            task="geometry_priors",
            role="candidate",
            primary_metric="validPriorCoverage",
            higher_is_better=True,
            metrics={"validPriorCoverage": 1.0, "images": 4.0, "seconds": 0.5923},
            hard_constraints={"gpu": True, "realArtifact": True},
            official_evidence=False,
            notes=(
                "Depth Anything 3 Small produced finite depth/confidence/extrinsic/intrinsic tensors on 4 Truck images",
                "kept as optional depth/pose prior; not promoted over COLMAP for already-posed local data",
            ),
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
            provider_id="3dgrut_3dgut_official",
            task="secondary_rays",
            role="candidate",
            primary_metric="officialEvidenceScore",
            higher_is_better=True,
            metrics={"officialEvidenceScore": 1.0, "trainingSteps": 1.0, "iterationSpeed": 5.50},
            hard_constraints={"gpu": True, "realArtifact": True},
            official_evidence=True,
            notes=(
                "official 3DGRUT repo cloned; 3DGUT CUDA/Slang extension compiled for sm_120",
                "one-iteration Truck COLMAP smoke completed and wrote ckpt_last.pt",
            ),
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
            metrics={"officialEvidenceScore": 0.0, "buildReady": 1.0, "trainingSmokeCompleted": 0.0},
            hard_constraints={"gpu": True, "realArtifact": False},
            official_evidence=True,
            notes=(
                "official 2DGS repo cloned and CUDA submodules built after adding cfloat include to simple-knn",
                "not promoted: one-iteration Truck train smoke did not finish after roughly three minutes and was stopped",
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/sota_ab_validation_2026-06-25.json")
    parser.add_argument("--fixture", action="store_true", help="write fixture candidates instead of measured rows")
    args = parser.parse_args()

    candidates = fixture_candidates() if args.fixture else measured_candidates_2026_06_25()
    report = sota_ab_report(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
