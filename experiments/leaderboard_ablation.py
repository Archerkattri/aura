#!/usr/bin/env python3
"""Generate AURA's strict leaderboard-ablation report.

This script intentionally starts from existing measured artifacts. It does not
launch long jobs; later ablation scripts can append measured candidate rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aura.leaderboard import (
    LeaderboardMetric,
    LeaderboardReport,
    LeaderboardRun,
    MethodSpec,
    SceneSpec,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
SCENES = ("truck", "room", "bicycle", "bonsai", "counter", "garden", "kitchen", "stump")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _scene_specs() -> tuple[SceneSpec, ...]:
    return tuple(
        SceneSpec(
            scene_id=scene,
            dataset="Tanks and Temples" if scene == "truck" else "Mip-NeRF 360",
            split="llffhold8",
            image_scale="native" if scene == "truck" else "images_2/downsample_factor=2",
        )
        for scene in SCENES
    )


def _metric(name: str, value: float, *, higher: bool) -> LeaderboardMetric:
    return LeaderboardMetric(name=name, value=float(value), higher_is_better=higher)


def _multiscene_runs() -> list[LeaderboardRun]:
    payload = _read_json(RESULTS / "multiscene.json") or {}
    runs: list[LeaderboardRun] = []
    for row in payload.get("scenes", ()):
        scene = str(row["scene"])
        beta_psnr = row.get("beta_psnr")
        gauss_psnr = row.get("gaussian_psnr", row.get("gauss_psnr"))
        if beta_psnr is not None:
            runs.append(
                LeaderboardRun(
                    scene_id=scene,
                    method_id="aura_beta",
                    metrics=(_metric("psnr", float(beta_psnr), higher=True),),
                    artifacts=("experiments/results/multiscene.json",),
                    measured=True,
                    notes=("AURA DBS-Beta local multiscene row",),
                )
            )
        if gauss_psnr is not None:
            runs.append(
                LeaderboardRun(
                    scene_id=scene,
                    method_id="fixed_gaussian_control",
                    metrics=(_metric("psnr", float(gauss_psnr), higher=True),),
                    artifacts=("experiments/results/multiscene.json",),
                    measured=True,
                    notes=("AURA fixed-Gaussian local control row",),
                )
            )
    return runs


def _official_runs() -> list[LeaderboardRun]:
    payload = _read_json(RESULTS / "official_multiscene_baselines_2026-06-25.json") or {}
    runs: list[LeaderboardRun] = []
    for method in payload.get("methods", ()):
        method_id = str(method.get("method", ""))
        if method_id not in {"official_2dgs", "official_3dgut"}:
            continue
        for row in method.get("rows", ()):
            metrics = [
                _metric("psnr", float(row["psnr"]), higher=True),
                _metric("ssim", float(row["ssim"]), higher=True),
                _metric("lpips", float(row["lpips"]), higher=False),
            ]
            if "meanInferenceMs" in row:
                metrics.append(_metric("render_ms", float(row["meanInferenceMs"]), higher=False))
            runs.append(
                LeaderboardRun(
                    scene_id=str(row["scene"]),
                    method_id=method_id,
                    metrics=tuple(metrics),
                    artifacts=("experiments/results/official_multiscene_baselines_2026-06-25.json",),
                    measured=True,
                    notes=(str(row.get("output", "")),),
                )
            )
    return runs


def _gsplat_main_mcmc_runs() -> list[LeaderboardRun]:
    runs: list[LeaderboardRun] = []
    for artifact in sorted(RESULTS.glob("gsplat_main_mcmc_truck_*_2026-06-25.json")):
        payload = _read_json(artifact) or {}
        if not payload:
            continue
        if payload.get("format") != "AURA_GSPLAT_MAIN_MCMC_TRUCK_ABLATION":
            continue
        metrics = payload.get("metrics", {})
        notes = [str(payload.get("leaderboardImpact", ""))]
        output = payload.get("output")
        if output:
            notes.append(str(output))
        runs.append(
            LeaderboardRun(
                scene_id=str(payload["scene"]),
                method_id="gsplat_main_mcmc",
                metrics=(
                    _metric("psnr", float(metrics["psnr"]), higher=True),
                    _metric("ssim", float(metrics["ssim"]), higher=True),
                    _metric("lpips", float(metrics["lpips"]), higher=False),
                    _metric("render_ms", float(metrics["secondsPerImage"]) * 1000.0, higher=False),
                    _metric("num_gaussians", float(metrics["numGaussians"]), higher=False),
                ),
                artifacts=(str(artifact.relative_to(ROOT)),),
                measured=True,
                notes=tuple(note for note in notes if note),
            )
        )
    return runs


def _higs_inference_runs() -> list[LeaderboardRun]:
    artifact = RESULTS / "gsplat_main_higs_truck_30000_2026-06-25.json"
    payload = _read_json(artifact) or {}
    if payload.get("format") != "AURA_GSPLAT_MAIN_HIGS_TRUCK_BENCHMARK":
        return []
    inference = payload["statefulInferenceRenderer"]
    reference = payload["referenceRasterization"]
    quality = payload["qualityVsReference"]
    return [
        LeaderboardRun(
            scene_id=str(payload["scene"]),
            method_id="higs_inference",
            metrics=(
                _metric("render_ms", float(inference["medianMs"]), higher=False),
                _metric("reference_render_ms", float(reference["medianMs"]), higher=False),
                _metric("speedup_vs_reference", float(payload["speedupVsReference"]), higher=True),
                _metric("reference_psnr", float(quality["psnrMean"]), higher=True),
                _metric("reference_lpips", float(quality["lpipsAlexMean"]), higher=False),
            ),
            artifacts=("experiments/results/gsplat_main_higs_truck_30000_2026-06-25.json",),
            measured=True,
            notes=(
                str(payload.get("leaderboardImpact", "")),
                str(payload.get("promotionScope", "")),
            ),
        )
    ]


def leaderboard_ablation_report(out: Path) -> dict[str, Any]:
    methods = (
        MethodSpec(method_id="aura_beta", role="baseline", backend="dbs-beta", command="experiments/run_multiscene.sh"),
        MethodSpec(method_id="fixed_gaussian_control", role="candidate", backend="dbs-gaussian-control", command="experiments/run_multiscene.sh"),
        MethodSpec(method_id="official_2dgs", role="candidate", backend="official-2dgs", command="external 30k run"),
        MethodSpec(method_id="official_3dgut", role="candidate", backend="official-3dgut", command="external 30k run"),
        MethodSpec(method_id="gsplat_main_mcmc", role="candidate", backend="gsplat-main", command="pending measured ablation"),
        MethodSpec(method_id="radsplat_pruned", role="candidate", backend="radsplat-style", command="pending measured ablation"),
        MethodSpec(method_id="higs_inference", role="candidate", backend="gsplat-main-higs", command="pending measured ablation"),
    )
    report = LeaderboardReport(
        benchmark_id="aura_leaderboard_v1",
        task="novel_view_synthesis",
        scenes=_scene_specs(),
        methods=methods,
        runs=tuple(_multiscene_runs() + _official_runs() + _gsplat_main_mcmc_runs() + _higs_inference_runs()),
        primary_metric="psnr",
    )
    payload = report.to_dict()
    # Current rows are same-split evidence, not a leaderboard-superiority claim:
    # candidates do not beat AURA Beta on the primary metric across all scenes.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS / "leaderboard_ablation_2026-06-25.json")
    args = parser.parse_args()
    payload = leaderboard_ablation_report(args.out)
    print(json.dumps({
        "leaderboardReady": payload["leaderboardReady"],
        "missingScenes": payload["missingScenes"],
        "promotedMethodIds": payload["promotedMethodIds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
