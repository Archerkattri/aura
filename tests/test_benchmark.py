import json
import subprocess
import sys

from aura import apply_ablation, default_benchmark_suite, load_package, package_scene, run_ablation_benchmarks, run_reference_benchmark
from aura.cli import native_demo_scene


def test_default_benchmark_suite_covers_required_mvp_axes():
    suite = default_benchmark_suite()
    case_ids = {case.id for case in suite.cases}
    ablation_ids = {ablation.id for ablation in suite.ablations}

    assert {
        "visual_quality_vs_3dgs",
        "ray_query_correctness",
        "geometry_collision_proxy",
        "package_size",
        "render_query_speed",
        "mixed_carrier_behavior",
    }.issubset(case_ids)
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
    assert payload["packageBytes"] > 0
    assert payload["rayQuery"]["probeCount"] > 0
    assert payload["carrierEntropy"] > 0.0
    assert payload["previewRender"]["pixelCount"] == 64


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
    assert payload["rayQuery"]["shadowReadyCount"] > 0
    assert payload["previewRender"]["width"] == 8


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
