import json
import subprocess
import sys

from aura import (
    apply_ablation,
    default_benchmark_suite,
    load_package,
    native_demo_ray_query_expectations,
    package_scene,
    run_ablation_benchmarks,
    run_core_reconstruction_benchmark,
    run_ray_query_correctness_benchmark,
    run_reference_benchmark,
)
from aura.cli import native_demo_scene


def test_default_benchmark_suite_covers_required_mvp_axes():
    suite = default_benchmark_suite()
    case_ids = {case.id for case in suite.cases}
    ablation_ids = {ablation.id for ablation in suite.ablations}

    assert {
        "visual_quality_vs_3dgs",
        "ray_query_correctness",
        "geometry_collision_proxy",
        "shadow_reflection_queries",
        "package_size",
        "render_query_speed",
        "mixed_carrier_behavior",
        "confidence_calibration",
        "aura_core_reconstruction",
    }.issubset(case_ids)
    visual_case = next(case for case in suite.cases if case.id == "visual_quality_vs_3dgs")
    speed_case = next(case for case in suite.cases if case.id == "render_query_speed")
    assert "ssim" in visual_case.metrics
    assert "lpips_proxy" in visual_case.metrics
    assert "ssim_placeholder" not in visual_case.metrics
    assert "frames_per_second" in speed_case.metrics
    assert {"gaussian_only", "no_neural_residual", "no_frequency_carrier", "no_semantic_graph"}.issubset(ablation_ids)


def test_benchmark_plan_cli_prints_reproducible_json():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "benchmark-plan"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert "cases" in payload
    assert "ablations" in payload
    assert payload["cases"][0]["id"] == "visual_quality_vs_3dgs"


def test_reference_benchmark_reports_native_package_metrics(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)
    package = load_package(tmp_path)

    payload = run_reference_benchmark(package, package_dir=tmp_path, render_width=8, render_height=8)

    assert payload["format"] == "AURA_REFERENCE_BENCHMARK"
    assert payload["asset"] == "native_demo"
    assert payload["elementCount"] == 7
    assert payload["semanticObjectCount"] == 2
    assert payload["nonGaussianFraction"] > 0.5
    assert payload["confidenceQuality"]["meanElementConfidence"] > 0.0
    assert payload["confidenceQuality"]["confidenceWithinBoundsRate"] == 1.0
    assert payload["confidenceQuality"]["confidenceMapCoverageRate"] > 0.0
    assert payload["packageBytes"] > 0
    assert payload["rayQuery"]["probeCount"] > 0
    assert payload["rayQuery"]["querySeconds"] >= 0.0
    assert payload["rayQuery"]["raysPerSecond"] >= 0.0
    assert payload["rayQuery"]["queryP50Ms"] >= 0.0
    assert payload["rayQuery"]["queryP95Ms"] >= payload["rayQuery"]["queryP50Ms"]
    assert payload["rayQueryCorrectness"]["format"] == "AURA_RAY_QUERY_CORRECTNESS_BENCHMARK"
    assert payload["rayQueryCorrectness"]["firstHitAccuracy"] > 0.0
    assert payload["interactionQuality"]["hitPointReadyRate"] == 1.0
    assert payload["interactionQuality"]["shadowTransmittanceReadyRate"] == 1.0
    assert payload["interactionQuality"]["shadowTransmittanceWithinBoundsRate"] == 1.0
    assert payload["interactionQuality"]["collisionDistanceReadyRate"] == 1.0
    assert payload["interactionQuality"]["reflectionVectorReadyRate"] > 0.0
    assert payload["carrierEntropy"] > 0.0
    assert payload["previewRender"]["pixelCount"] == 64
    assert payload["previewRender"]["renderSeconds"] >= 0.0
    assert payload["previewRender"]["framesPerSecond"] >= 0.0
    assert payload["previewRender"]["pixelsPerSecond"] >= 0.0
    assert payload["previewRender"]["referenceVisualQuality"]["psnrInfinite"] is True
    assert payload["previewRender"]["referenceVisualQuality"]["ssim"] == 1.0
    assert payload["previewRender"]["referenceVisualQuality"]["lpipsProxy"] == 0.0


def test_apply_ablation_disables_requested_carriers(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)
    package = load_package(tmp_path)
    ablation = next(item for item in default_benchmark_suite().ablations if item.id == "no_frequency_carrier")

    ablated = apply_ablation(package, ablation)

    assert "gabor" not in ablated.scene.carrier_ids()
    assert len(ablated.scene.elements) == len(package.scene.elements) - 1


def test_ablation_benchmark_reports_deltas(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)
    package = load_package(tmp_path)

    payload = run_ablation_benchmarks(package, package_dir=tmp_path, render_width=8, render_height=8)
    by_id = {item["id"]: item for item in payload["ablations"]}

    assert payload["format"] == "AURA_ABLATION_BENCHMARK"
    assert by_id["gaussian_only"]["metrics"]["carrierCounts"] == {"gaussian": 1}
    assert by_id["gaussian_only"]["delta"]["elementCount"] == -6
    assert by_id["no_semantic_graph"]["delta"]["semanticObjectCount"] == -1


def test_core_reconstruction_benchmark_compares_adaptive_and_static_runs():
    payload = run_core_reconstruction_benchmark(iterations=6)

    assert payload["format"] == "AURA_CORE_RECONSTRUCTION_BENCHMARK"
    assert payload["scene"] == "reconstruct_demo"
    assert payload["adaptive"]["finalLoss"] > 0.0
    assert payload["adaptive"]["finalQueryLoss"] == 0.0
    assert payload["static"]["finalQueryLoss"] == 0.0
    assert payload["adaptive"]["confidenceQuality"]["optimizationResidualMapRate"] > 0.0
    assert payload["adaptive"]["confidenceQuality"]["lowResidualHighConfidenceRate"] > 0.0
    assert payload["static"]["confidenceQuality"]["confidenceWithinBoundsRate"] == 1.0
    assert payload["adaptive"]["lossReduction"] > 0.0
    assert payload["static"]["lossReduction"] > 0.0
    assert payload["adaptive"]["evolutionActionCounts"]["split_beta_detail"] > 0
    assert payload["adaptive"]["evolutionActionCounts"]["promote_neural_residual"] > 0
    assert payload["adaptive"]["evolutionActionCounts"]["merge_beta_detail"] > 0
    assert payload["adaptive"]["evolutionActionCounts"]["demote_neural_residual"] > 0
    assert payload["static"]["evolvedElementCount"] == 0
    assert payload["static"]["evolutionActionCounts"] == {}
    assert payload["delta"]["adaptiveEvolutionActions"] == payload["adaptive"]["evolutionActionCounts"]
    assert payload["delta"]["queryLoss"] == 0.0


def test_ray_query_correctness_benchmark_scores_native_demo_contract():
    payload = run_ray_query_correctness_benchmark(native_demo_scene(), native_demo_ray_query_expectations())
    by_label = {probe["label"]: probe for probe in payload["probes"]}

    assert payload["format"] == "AURA_RAY_QUERY_CORRECTNESS_BENCHMARK"
    assert payload["scene"] == "native_demo"
    assert payload["passed"] is True
    assert payload["passRate"] == 1.0
    assert payload["firstHitAccuracy"] == 1.0
    assert payload["carrierAccuracy"] == 1.0
    assert payload["depthWithinToleranceRate"] == 1.0
    assert payload["transmittanceWithinBoundsRate"] == 1.0
    assert by_label["surface_first_hit"]["actual"]["carrierId"] == "surface"
    assert by_label["surface_first_hit"]["actual"]["materialId"] == "mat_wall_plaster"
    assert by_label["surface_first_hit"]["checks"]["normal"]["passed"] is True
    assert by_label["semantic_object"]["actual"]["semanticId"] == "fixture_object"
    assert by_label["neural_residual"]["actual"]["residual"] is True
    assert by_label["empty_space_control"]["actual"]["firstHit"] is False


def test_core_benchmark_cli_prints_reconstruction_metrics():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "benchmark-core", "--iterations", "6"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_CORE_RECONSTRUCTION_BENCHMARK"
    assert payload["adaptive"]["nativeCarrierFraction"] > 0.8
    assert payload["static"]["nativeCarrierFraction"] > 0.8


def test_ray_query_benchmark_cli_prints_correctness_json(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-ray-query",
            str(tmp_path),
            "--native-demo-expectations",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_RAY_QUERY_CORRECTNESS_BENCHMARK"
    assert payload["passed"] is True
    assert payload["carrierAccuracy"] == 1.0


def test_reference_benchmark_cli_prints_result_json(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "benchmark-reference", str(tmp_path), "--width", "8", "--height", "8"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["asset"] == "native_demo"
    assert payload["confidenceQuality"]["meanElementConfidence"] > 0.0
    assert payload["rayQuery"]["shadowReadyCount"] > 0
    assert payload["interactionQuality"]["shadowTransmittanceReadyRate"] == 1.0
    assert payload["rayQueryCorrectness"]["format"] == "AURA_RAY_QUERY_CORRECTNESS_BENCHMARK"
    assert payload["rayQuery"]["raysPerSecond"] >= 0.0
    assert payload["previewRender"]["width"] == 8
    assert payload["previewRender"]["renderSeconds"] >= 0.0


def test_reference_benchmark_cli_can_include_ablation_results(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-reference",
            str(tmp_path),
            "--width",
            "8",
            "--height",
            "8",
            "--include-ablations",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_ABLATION_BENCHMARK"
    assert payload["baseline"]["asset"] == "native_demo"
    assert {item["id"] for item in payload["ablations"]} >= {"gaussian_only", "no_frequency_carrier"}
