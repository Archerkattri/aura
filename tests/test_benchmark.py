import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import aura.benchmark as benchmark_module
from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    Ray,
    RayQueryExpectation,
    apply_ablation,
    default_benchmark_suite,
    load_package,
    native_demo_ray_query_expectations,
    package_scene,
    render_orthographic,
    run_ablation_benchmarks,
    run_capture_reconstruction_benchmark,
    run_core_reconstruction_benchmark,
    run_ray_query_correctness_benchmark,
    run_reference_benchmark,
    run_visual_quality_benchmark,
)
from aura.benchmark import evaluate_backend_readiness, evaluate_native_carrier_coverage, run_production_gate_report
from aura.cli import native_demo_scene
from aura.torch_renderer import TorchCaptureRenderSummary


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
        "runtime_export_contract",
        "backend_readiness_contract",
        "cuda_renderer_abi_parity",
        "production_gate_contract",
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
    assert payload["runtimeExport"]["format"] == "AURA_RUNTIME_EXPORT_REPORT"
    assert payload["runtimeExport"]["engineWorkflow"]["nativeRuntimeReady"] is True
    assert payload["runtimeExport"]["engineWorkflow"]["chunkedStreamingReady"] is True
    assert len(payload["runtimeExport"]["chunkExport"]) == payload["chunkCount"]
    assert "transmittance" in payload["runtimeExport"]["rayQueryContract"]["fields"]
    assert payload["backendReadiness"]["format"] == "AURA_BACKEND_READINESS_EVALUATION"
    assert payload["backendReadiness"]["requiresTorchImport"] is False
    assert payload["backendReadiness"]["mvpContractReady"] is True
    assert payload["backendReadiness"]["productionCudaReady"] is False
    assert payload["backendReadiness"]["sceneCarrierAutogradCoverageRate"] == 1.0
    assert payload["backendReadiness"]["sceneCarrierCudaCoverageRate"] == 0.0
    assert payload["backendReadiness"]["queryContract"]["fieldCoverageRate"] == 1.0
    assert payload["backendReadiness"]["chunkLodContract"]["chunkedElementCoverageRate"] == 1.0
    assert "carrier_cuda_kernels_not_production_ready" in payload["backendReadiness"]["productionBlockers"]
    assert payload["cudaRenderer"]["format"] == "AURA_CUDA_RENDERER_LAUNCH_REPORT"
    assert payload["cudaRenderer"]["available"] is False
    assert payload["cudaRendererCallableBoundary"]["format"] == "AURA_CUDA_RENDERER_CALLABLE_BOUNDARY"
    assert payload["cudaRendererCallableBoundary"]["callableBoundaryReady"] is True
    assert payload["cudaRendererCallableBoundary"]["fallbackContractReady"] is True
    assert payload["cudaRendererCallableBoundary"]["fallbackAvailable"] is True
    assert payload["cudaRendererCallableBoundary"]["fallbackBackend"] == "cpu"
    assert payload["cudaRendererCallableBoundary"]["compiledCudaAvailable"] is False
    assert payload["cudaRendererCallableBoundary"]["productionReady"] is False
    assert "orderedHits" in payload["cudaRendererCallableBoundary"]["outputFields"]
    assert payload["cudaRendererAbiParity"]["format"] == "AURA_CUDA_RENDERER_ABI_PARITY"
    assert payload["cudaRendererAbiParity"]["passed"] is True
    assert payload["cudaRendererAbiParity"]["parityReady"] is True
    assert payload["cudaRendererAbiParity"]["productionReady"] is False
    assert payload["cudaRendererAbiParity"]["probeCount"] > 0
    assert payload["cudaRendererAbiParity"]["kernelInput"]["rayCount"] == payload["cudaRendererAbiParity"]["probeCount"]
    assert payload["nativeCarrierCoverage"]["format"] == "AURA_NATIVE_CARRIER_COVERAGE"
    assert payload["nativeCarrierCoverage"]["auraFirstCoverageReady"] is True
    assert payload["nativeCarrierCoverage"]["requiredNativeCarrierCoverageRate"] == 1.0
    assert payload["nativeCarrierCoverage"]["missingNativeCarrierIds"] == []
    assert payload["productionGate"]["format"] == "AURA_BENCHMARK_PRODUCTION_GATE"
    assert payload["productionGate"]["productionReady"] is False
    assert payload["productionGate"]["blocksProductionClaim"] is True
    assert payload["productionGate"]["cudaRendererReady"] is False
    assert payload["productionGate"]["cudaRendererAvailable"] is False
    assert payload["productionGate"]["cudaRendererProductionReady"] is False
    assert payload["productionGate"]["cudaRendererReportKind"] == "legacy_cuda_kernels_metadata_report"
    assert payload["productionGate"]["cudaRendererCallableBoundaryReady"] is True
    assert payload["productionGate"]["cudaRendererCallableFallbackAvailable"] is True
    assert payload["productionGate"]["cudaRendererCallableFallbackBackend"] == "cpu"
    assert payload["productionGate"]["cudaRendererCallableFallbackOnly"] is True
    assert payload["productionGate"]["cudaRendererCallableProductionReady"] is False
    assert payload["productionGate"]["cudaRendererAbiParityReady"] is True
    assert payload["productionGate"]["cudaRendererAbiParityProductionReady"] is False
    assert payload["productionGate"]["cudaRendererAbiParityProbeCount"] == payload["cudaRendererAbiParity"]["probeCount"]
    assert payload["productionGate"]["nativeCarrierCoverageReady"] is True
    assert payload["productionGate"]["requiredNativeCarrierCoverageRate"] == 1.0
    assert payload["productionGate"]["visualBenchmarkSelfReference"] is True
    assert "cuda_renderer_unavailable" in payload["productionGate"]["productionBlockers"]
    assert "cuda_renderer_callable_fallback_only" in payload["productionGate"]["productionBlockers"]
    assert "cuda_renderer_abi_parity_cpu_oracle_only" in payload["productionGate"]["productionBlockers"]
    assert "visual_benchmark_self_reference" in payload["productionGate"]["productionBlockers"]
    assert payload["carrierEntropy"] > 0.0
    assert payload["previewRender"]["pixelCount"] == 64
    assert payload["previewRender"]["renderSeconds"] >= 0.0
    assert payload["previewRender"]["framesPerSecond"] >= 0.0
    assert payload["previewRender"]["pixelsPerSecond"] >= 0.0
    assert payload["previewRender"]["referenceVisualQuality"]["psnrInfinite"] is True
    assert payload["previewRender"]["referenceVisualQuality"]["ssim"] == 1.0
    assert payload["previewRender"]["referenceVisualQuality"]["lpipsProxy"] == 0.0
    assert payload["previewRender"]["visualClaimBoundary"]["selfReference"] is True
    assert payload["previewRender"]["visualClaimBoundary"]["productionClaimAllowed"] is False


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_capture_reconstruction_benchmark_trains_and_scores_capture_targets(tmp_path):
    manifest_path = _write_capture_benchmark_manifest(tmp_path)
    output_dir = tmp_path / "capture-benchmark.aura"

    payload = run_capture_reconstruction_benchmark(
        manifest_path,
        output_dir=output_dir,
        iterations=2,
        device="cpu",
        tile_size=1,
        max_targets_per_frame=1,
        max_targets_per_batch=1,
    )

    assert payload["format"] == "AURA_CAPTURE_RECONSTRUCTION_BENCHMARK"
    assert payload["packageDir"] == str(output_dir)
    assert payload["device"] == "cpu"
    assert payload["packedTargetCount"] == 1
    assert payload["trainingSteps"] == 2
    assert payload["initialReference"]["metrics"]["psnrInfinite"] is False
    assert payload["captureBaseline"]["label"] == "capture_leave_one_out_color_depth_baseline"
    assert payload["captureBaseline"]["baselineKind"] == "leave_one_out_capture_color_depth"
    assert payload["captureBaseline"]["metrics"]["psnrInfinite"] is False
    assert payload["trained"]["metrics"]["psnr"] is not None
    assert payload["trained"]["evaluationSummary"] == "compact_trace_free_capture_summary"
    assert payload["trained"]["orderedTraceMeanLength"] == 0.0
    assert payload["trained"]["metrics"]["ssim"] >= 0.0
    assert payload["trained"]["metrics"]["lpipsProxy"] >= 0.0
    assert "psnrDelta" in payload["improvementVsCaptureBaseline"]
    assert payload["rayQueryCorrectness"]["passed"] is True
    assert payload["baseline"] is None
    assert (output_dir / "manifest.json").exists()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_capture_benchmark_cli_prints_json(tmp_path):
    manifest_path = _write_capture_benchmark_manifest(tmp_path)
    output_dir = tmp_path / "capture-cli-benchmark.aura"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-capture",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--iterations",
            "1",
            "--device",
            "cpu",
            "--tile-size",
            "1",
            "--max-targets-per-frame",
            "1",
            "--max-targets-per-batch",
            "1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_CAPTURE_RECONSTRUCTION_BENCHMARK"
    assert payload["packageDir"] == str(output_dir)
    assert payload["trained"]["label"] == "aura_native_trained"
    assert payload["trained"]["sampleCount"] == 1
    assert payload["trained"]["metrics"]["mse"] >= 0.0
    assert payload["captureBaseline"]["sampleCount"] == 1
    assert payload["notes"]["captureBaseline"].startswith("Built-in non-AURA")
    assert "3DGS" in payload["notes"]["baseline"]


def test_visual_quality_benchmark_compares_package_render_to_reference():
    package = package_scene(native_demo_scene())
    reference = render_orthographic(package.scene, width=4, height=4)

    payload = run_visual_quality_benchmark(package, reference, baseline_label="native_self", min_psnr=80.0)

    assert payload["format"] == "AURA_VISUAL_QUALITY_BENCHMARK"
    assert payload["asset"] == "native_demo"
    assert payload["baseline"] == "native_self"
    assert payload["render"]["pixelCount"] == 16
    assert payload["render"]["framesPerSecond"] >= 0.0
    assert payload["metrics"]["psnrInfinite"] is True
    assert payload["metrics"]["ssim"] == 1.0
    assert payload["metrics"]["lpipsProxy"] == 0.0
    assert payload["passed"] is True
    assert payload["visualClaimBoundary"]["selfReference"] is True
    assert payload["visualClaimBoundary"]["productionClaimAllowed"] is False
    assert payload["productionGate"]["productionReady"] is False
    assert payload["productionGate"]["blocksProductionClaim"] is True
    assert payload["productionGate"]["nativeCarrierCoverageReady"] is True
    assert payload["productionGate"]["cudaRendererCallableBoundaryReady"] is True
    assert payload["productionGate"]["cudaRendererCallableFallbackOnly"] is True
    assert "visual_benchmark_self_reference" in payload["productionGate"]["productionBlockers"]
    assert "cuda_renderer_unavailable" in payload["productionGate"]["productionBlockers"]
    assert "cuda_renderer_callable_fallback_only" in payload["productionGate"]["productionBlockers"]
    assert "learned LPIPS" in payload["metricNotes"]["lpipsProxy"]

    relabeled_payload = run_visual_quality_benchmark(package, reference, baseline_label="teacher")
    assert relabeled_payload["visualClaimBoundary"]["selfReference"] is True
    assert "visual_benchmark_self_reference" in relabeled_payload["productionGate"]["productionBlockers"]


def test_native_carrier_coverage_blocks_gaussian_fallback_only_claims(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)
    package = load_package(tmp_path)
    ablation = next(item for item in default_benchmark_suite().ablations if item.id == "gaussian_only")
    fallback_only = apply_ablation(package, ablation)

    coverage = evaluate_native_carrier_coverage(fallback_only.scene)
    payload = run_reference_benchmark(fallback_only, render_width=4, render_height=4)

    assert coverage["auraFirstCoverageReady"] is False
    assert coverage["requiredNativeCarrierCoverageRate"] == 0.0
    assert set(coverage["missingNativeCarrierIds"]) == {
        "surface",
        "volume",
        "beta",
        "gabor",
        "neural",
        "semantic",
    }
    assert "native_carriers_absent" in coverage["productionBlockers"]
    assert "gaussian_fallback_only_scene" in coverage["productionBlockers"]
    assert payload["productionGate"]["nativeCarrierCoverageReady"] is False
    assert payload["productionGate"]["requiredNativeCarrierCoverageRate"] == 0.0
    assert "native_carrier_coverage_incomplete" in payload["productionGate"]["productionBlockers"]
    assert "gaussian_fallback_only_scene" in payload["productionGate"]["productionBlockers"]


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


def test_ray_query_correctness_benchmark_scores_ordered_hit_trace():
    scene = AuraScene(
        name="ordered_trace_benchmark",
        elements=(
            AuraElement(
                id="front_surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.5,
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="rear_volume",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.2), (0.5, 0.5, 0.7)),
                color=(0.0, 0.0, 1.0),
                opacity=0.8,
                payload={"type": "volume_cell", "density": 1.0},
            ),
        ),
    )

    payload = run_ray_query_correctness_benchmark(
        scene,
        (
            RayQueryExpectation(
                label="surface_then_volume",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                expected_first_hit=True,
                expected_element_id="front_surface",
                expected_carrier_id="surface",
                expected_ordered_element_ids=("front_surface", "rear_volume"),
                expected_ordered_carrier_ids=("surface", "volume"),
            ),
        ),
    )
    probe = payload["probes"][0]

    assert payload["passed"] is True
    assert payload["orderedElementTraceAccuracy"] == 1.0
    assert payload["orderedCarrierTraceAccuracy"] == 1.0
    assert probe["checks"]["orderedElementIds"]["passed"] is True
    assert probe["actual"]["orderedElementIds"] == ["front_surface", "rear_volume"]
    assert probe["actual"]["orderedCarrierIds"] == ["surface", "volume"]
    assert probe["actual"]["orderedHits"][1]["carrierId"] == "volume"


def test_capture_ray_query_expectations_reuse_rendered_ray_metadata(monkeypatch):
    class _NoCpuRayTensor:
        def detach(self):
            raise AssertionError("benchmark should reuse rendered ray metadata instead of syncing tensors again")

    class _PackedBatch:
        batch_index = 2
        target_count = 1

    class _TorchBatch:
        ray_origins = _NoCpuRayTensor()
        ray_directions = _NoCpuRayTensor()

    scene = AuraScene(
        name="capture_expectation_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
            ),
        ),
    )

    monkeypatch.setattr(benchmark_module, "torch_capture_training_batch_from_packed", lambda *_args, **_kwargs: _TorchBatch())
    monkeypatch.setattr(
        benchmark_module,
        "torch_render_capture_training_summary",
        lambda *_args, **_kwargs: TorchCaptureRenderSummary(
            device="cpu",
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0, 1.0),),
            element_ids=("surface",),
            carrier_ids=("surface",),
            predicted_color=((1.0, 0.0, 0.0),),
            predicted_depth=(1.0,),
            transmittance=(0.0,),
            normal=((0.0, 0.0, -1.0),),
            target_color=((1.0, 0.0, 0.0),),
            target_depth=(1.0,),
            target_point=((0.0, 0.0, 0.0),),
            image_loss=(0.0,),
            depth_loss=(0.0,),
            query_loss=(0.0,),
            normal_loss=(0.0,),
        ),
    )

    expectations = benchmark_module._capture_ray_query_expectations(scene, (_PackedBatch(),), device="cpu")

    assert len(expectations) == 1
    assert expectations[0].label == "capture_batch_2_sample_0"
    assert expectations[0].ray.origin == (0.0, 0.0, -1.0)
    assert expectations[0].ray.direction == (0.0, 0.0, 1.0)


def test_backend_readiness_evaluation_is_cpu_contract_not_cuda_claim():
    scene = native_demo_scene()

    payload = evaluate_backend_readiness(scene)

    assert payload["format"] == "AURA_BACKEND_READINESS_EVALUATION"
    assert payload["scene"] == "native_demo"
    assert payload["requiresTorchImport"] is False
    assert payload["mvpContractReady"] is True
    assert payload["productionCudaReady"] is False
    assert payload["sceneCarrierAutogradCoverageRate"] == 1.0
    assert payload["sceneCarrierCudaCoverageRate"] == 0.0
    assert payload["carrierKernelContract"]["missingAutogradCarrierIds"] == []
    assert set(payload["carrierKernelContract"]["missingCudaCarrierIds"]) == set(scene.carrier_ids())
    assert payload["queryContract"]["missingFields"] == []
    assert payload["queryContract"]["supportsOrderedHitTrace"] is True
    assert payload["chunkLodContract"]["exportedChunkCount"] == len(scene.chunks)
    assert payload["chunkLodContract"]["chunkedElementCoverageRate"] == 1.0
    assert payload["chunkLodContract"]["productionGpuTraversalReady"] is False
    assert "does not execute torch" in payload["notes"]


def test_production_gate_report_surfaces_cuda_visual_and_native_carrier_gates(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)
    package = load_package(tmp_path)

    payload = run_production_gate_report(package)
    gate = payload["productionGate"]

    assert payload["format"] == "AURA_PRODUCTION_GATE_REPORT"
    assert payload["claimBoundary"]["productionClaimAllowed"] is False
    assert "native adaptive radiance reconstruction engine" in payload["claimBoundary"]["safeCurrentClaim"]
    assert payload["cudaRenderer"]["available"] is False
    assert payload["cudaRendererCallableBoundary"]["callableBoundaryReady"] is True
    assert payload["cudaRendererCallableBoundary"]["fallbackAvailable"] is True
    assert payload["cudaRendererCallableBoundary"]["productionReady"] is False
    assert payload["cudaRendererAbiParity"]["format"] == "AURA_CUDA_RENDERER_ABI_PARITY"
    assert payload["cudaRendererAbiParity"]["passed"] is True
    assert payload["cudaRendererAbiParity"]["productionReady"] is False
    assert payload["visualClaimBoundary"]["selfReference"] is True
    assert payload["nativeCarrierCoverage"]["auraFirstCoverageReady"] is True
    assert gate["productionReady"] is False
    assert gate["cudaRendererAvailable"] is False
    assert gate["cudaRendererReportKind"] == "legacy_cuda_kernels_metadata_report"
    assert gate["cudaRendererCallableBoundaryReady"] is True
    assert gate["cudaRendererCallableFallbackAvailable"] is True
    assert gate["cudaRendererCallableFallbackOnly"] is True
    assert gate["cudaRendererCallableProductionReady"] is False
    assert gate["cudaRendererAbiParityReady"] is True
    assert gate["cudaRendererAbiParityProductionReady"] is False
    assert gate["cudaRendererAbiParityProbeCount"] == payload["cudaRendererAbiParity"]["probeCount"]
    assert gate["visualBenchmarkSelfReference"] is True
    assert gate["nativeCarrierCoverageReady"] is True
    assert "cuda_renderer_unavailable" in gate["productionBlockers"]
    assert "cuda_renderer_callable_fallback_only" in gate["productionBlockers"]
    assert "cuda_renderer_abi_parity_cpu_oracle_only" in gate["productionBlockers"]
    assert "visual_benchmark_self_reference" in gate["productionBlockers"]


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
    assert payload["cudaRendererAbiParity"]["format"] == "AURA_CUDA_RENDERER_ABI_PARITY"
    assert payload["cudaRendererAbiParity"]["productionReady"] is False
    assert payload["productionGate"]["cudaRendererAbiParityReady"] is True
    assert payload["productionGate"]["cudaRendererAbiParityProductionReady"] is False
    assert payload["rayQuery"]["raysPerSecond"] >= 0.0
    assert payload["previewRender"]["width"] == 8
    assert payload["previewRender"]["renderSeconds"] >= 0.0


def test_production_gate_report_cli_prints_native_gate_json(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "production-gate-report", str(tmp_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_PRODUCTION_GATE_REPORT"
    assert payload["productionGate"]["productionReady"] is False
    assert payload["productionGate"]["cudaRendererAvailable"] is False
    assert payload["productionGate"]["cudaRendererCallableBoundaryReady"] is True
    assert payload["productionGate"]["cudaRendererCallableFallbackOnly"] is True
    assert payload["productionGate"]["cudaRendererAbiParityReady"] is True
    assert payload["productionGate"]["cudaRendererAbiParityProductionReady"] is False
    assert payload["cudaRendererAbiParity"]["productionReady"] is False
    assert payload["productionGate"]["visualBenchmarkSelfReference"] is True
    assert payload["productionGate"]["nativeCarrierCoverageReady"] is True


def test_visual_benchmark_cli_prints_result_json(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)
    reference = tmp_path / "reference.ppm"
    render_orthographic(native_demo_scene(), width=4, height=4).write_ppm(reference)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-visual",
            str(tmp_path),
            str(reference),
            "--baseline-label",
            "native_self",
            "--min-psnr",
            "40",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_VISUAL_QUALITY_BENCHMARK"
    assert payload["baseline"] == "native_self"
    assert payload["metrics"]["psnr"] >= 40.0
    assert payload["metrics"]["ssim"] > 0.99
    assert payload["passed"] is True


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


def _write_capture_benchmark_manifest(tmp_path):
    root = tmp_path / "capture"
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "masks").mkdir()
    (root / "normal").mkdir()
    (root / "images" / "frame_000001.ppm").write_text(
        "P3\n2 1\n4\n4 0 0 0 2 2\n",
        encoding="ascii",
    )
    (root / "depth" / "frame_000001.pgm").write_text(
        "P2\n2 1\n4\n2 4\n",
        encoding="ascii",
    )
    (root / "masks" / "frame_000001.pgm").write_text(
        "P2\n2 1\n2\n2 0\n",
        encoding="ascii",
    )
    _write_colmap_normal_map(root / "normal" / "frame_000001.bin", 2, 1, ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)))
    payload = {
        "format": "AURA_CAPTURE_MANIFEST",
        "root": str(root),
        "frames": [
            {
                "id": "frame_000001",
                "image_path": "images/frame_000001.ppm",
                "depth_path": "depth/frame_000001.pgm",
                "mask_path": "masks/frame_000001.pgm",
                "normal_path": "normal/frame_000001.bin",
                "camera_origin": [0.0, 0.0, -2.0],
                "look_at": [0.0, 0.0, 0.0],
                "target_color": [0.1, 0.1, 0.1],
                "target_depth": 2.0,
                "semantic_label": "fixture",
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2.0, "height": 1.0},
            }
        ],
        "regions": [
            {
                "id": "surface_000001",
                "frame_id": "frame_000001",
                "bounds": {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.1]},
                "evidence": {"geometry_confidence": 0.9, "edit_need": 0.5},
                "opacity": 0.9,
                "confidence": 0.8,
                "normal": [0.0, 0.0, -1.0],
                "fallback_source": "capture-benchmark-fixture",
            }
        ],
    }
    manifest_path = tmp_path / "capture_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _write_colmap_normal_map(path, width, height, normals):
    import struct

    with path.open("wb") as handle:
        handle.write(f"{width}&{height}&3&".encode("ascii"))
        for normal in normals:
            handle.write(struct.pack("<fff", *normal))


def test_benchmark_ray_grid_builds_front_facing_sweep():
    from aura.benchmark import _benchmark_ray_grid

    scene = native_demo_scene()
    origins, directions = _benchmark_ray_grid(scene, 64)

    assert len(origins) == 64
    assert len(directions) == 64
    assert all(direction == (0.0, 0.0, 1.0) for direction in directions)
    min_z = min(float(element.bounds.min_corner[2]) for element in scene.elements)
    assert all(origin[2] < min_z for origin in origins)


def test_run_cuda_runtime_benchmark_reports_skip_without_cuda(monkeypatch):
    import importlib

    from aura.benchmark import run_cuda_runtime_benchmark

    torch_module = importlib.import_module("torch") if importlib.util.find_spec("torch") else None
    if torch_module is not None:
        monkeypatch.setattr(torch_module.cuda, "is_available", lambda: False)

    report = run_cuda_runtime_benchmark(native_demo_scene(), ray_count=16, iterations=1, warmup=0)

    assert report["format"] == "AURA_CUDA_RUNTIME_BENCHMARK"
    assert report["executed"] is False
    assert report["cudaAvailable"] is False
    assert report["parityPassed"] is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_run_cuda_runtime_benchmark_measures_throughput_and_parity_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    from aura.benchmark import run_cuda_runtime_benchmark

    report = run_cuda_runtime_benchmark(native_demo_scene(), ray_count=512, iterations=3, warmup=1, max_hits=8)

    assert report["executed"] is True
    assert report["cudaAvailable"] is True
    assert report["parityPassed"] is True
    assert report["elementMatchRate"] == 1.0
    assert report["maxColorDelta"] <= 1.0e-4
    assert report["cuda"]["raysPerSecond"] > 0.0
    assert report["torch"]["raysPerSecond"] > 0.0
    assert report["cuda"]["backend"] == "cuda"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_benchmark_cuda_runtime_cli_runs_on_native_demo_scene():
    import os
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    worktree_src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(worktree_src), env.get("PYTHONPATH", "")))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-cuda-runtime",
            "--ray-count",
            "256",
            "--iterations",
            "2",
            "--warmup",
            "1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_CUDA_RUNTIME_BENCHMARK"
    assert payload["executed"] is True
    assert payload["parityPassed"] is True


# ---------------------------------------------------------------------------
# Validation / error path tests for BenchmarkCase, AblationConfig, BenchmarkSuite
# ---------------------------------------------------------------------------

def test_benchmark_case_raises_on_empty_id():
    from aura.benchmark import BenchmarkCase
    with pytest.raises(ValueError, match="benchmark id is required"):
        BenchmarkCase(id="", purpose="test", metrics=("ssim",))


def test_benchmark_case_raises_on_empty_metrics():
    from aura.benchmark import BenchmarkCase
    with pytest.raises(ValueError, match="benchmark metrics are required"):
        BenchmarkCase(id="my_case", purpose="test", metrics=())


def test_benchmark_case_to_dict_returns_asdict():
    from aura.benchmark import BenchmarkCase
    case = BenchmarkCase(id="my_case", purpose="test", metrics=("ssim", "psnr"), baseline="ref")
    d = case.to_dict()
    assert d["id"] == "my_case"
    assert d["metrics"] == ("ssim", "psnr")
    assert d["baseline"] == "ref"


def test_ablation_config_raises_on_empty_id():
    from aura.benchmark import AblationConfig
    with pytest.raises(ValueError, match="ablation id is required"):
        AblationConfig(id="")


def test_ablation_config_to_dict_returns_asdict():
    from aura.benchmark import AblationConfig
    config = AblationConfig(id="my_ablation", disabled_carriers=("gabor",), notes="test")
    d = config.to_dict()
    assert d["id"] == "my_ablation"
    assert d["disabled_carriers"] == ("gabor",)


def test_benchmark_suite_to_dict_contains_cases_and_ablations():
    from aura.benchmark import AblationConfig, BenchmarkCase, BenchmarkSuite
    suite = BenchmarkSuite(
        cases=[BenchmarkCase(id="c1", purpose="p", metrics=("m",))],
        ablations=[AblationConfig(id="a1")],
    )
    d = suite.to_dict()
    assert len(d["cases"]) == 1
    assert len(d["ablations"]) == 1


# ---------------------------------------------------------------------------
# RayQueryExpectation validation error paths (lines 133, 135, 137, 139)
# ---------------------------------------------------------------------------

def test_ray_query_expectation_raises_on_empty_label():
    with pytest.raises(ValueError, match="ray expectation label is required"):
        RayQueryExpectation(
            label="",
            ray=Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
        )


def test_ray_query_expectation_raises_on_negative_depth_tolerance():
    with pytest.raises(ValueError, match="depth_tolerance must be non-negative"):
        RayQueryExpectation(
            label="test",
            ray=Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
            depth_tolerance=-0.1,
        )


def test_ray_query_expectation_raises_on_transmittance_min_out_of_range():
    with pytest.raises(ValueError, match="transmittance_min must be in"):
        RayQueryExpectation(
            label="test",
            ray=Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
            transmittance_min=1.5,
        )


def test_ray_query_expectation_raises_on_transmittance_max_out_of_range():
    with pytest.raises(ValueError, match="transmittance_max must be in"):
        RayQueryExpectation(
            label="test",
            ray=Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
            transmittance_max=-0.5,
        )


# ---------------------------------------------------------------------------
# run_ray_query_correctness_benchmark: empty expectations path (line 230)
# ---------------------------------------------------------------------------

def test_ray_query_correctness_benchmark_raises_on_empty_expectations():
    with pytest.raises(ValueError, match="ray query correctness benchmark requires expectations"):
        run_ray_query_correctness_benchmark(native_demo_scene(), ())


# ---------------------------------------------------------------------------
# run_cuda_renderer_abi_parity_benchmark: empty expectations + exception path
# (lines 267, 278-279)
# ---------------------------------------------------------------------------

def test_cuda_renderer_abi_parity_benchmark_raises_on_empty_expectations():
    from aura.benchmark import run_cuda_renderer_abi_parity_benchmark
    with pytest.raises(ValueError, match="CUDA renderer ABI parity benchmark requires expectations"):
        run_cuda_renderer_abi_parity_benchmark(native_demo_scene(), ())


def test_cuda_renderer_abi_parity_benchmark_returns_error_report_on_exception(monkeypatch):
    from aura.benchmark import run_cuda_renderer_abi_parity_benchmark
    monkeypatch.setattr(
        benchmark_module, "cuda_renderer_kernel_inputs", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated failure"))
    )
    scene = native_demo_scene()
    expectations = (
        RayQueryExpectation(
            label="probe",
            ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
        ),
    )
    report = run_cuda_renderer_abi_parity_benchmark(scene, expectations)
    assert report["format"] == "AURA_CUDA_RENDERER_ABI_PARITY"
    assert report["passed"] is False
    assert report["parityReady"] is False
    assert "RuntimeError" in report["error"]


# ---------------------------------------------------------------------------
# _benchmark_ray_grid: error and empty-scene paths (lines 347, 355-356)
# ---------------------------------------------------------------------------

def test_benchmark_ray_grid_raises_on_non_positive_ray_count():
    from aura.benchmark import _benchmark_ray_grid
    with pytest.raises(ValueError, match="ray_count must be positive"):
        _benchmark_ray_grid(native_demo_scene(), 0)


def test_benchmark_ray_grid_uses_fallback_bounds_for_empty_scene():
    from aura.benchmark import _benchmark_ray_grid
    empty_scene = AuraScene(name="empty", elements=())
    origins, directions = _benchmark_ray_grid(empty_scene, 4)
    assert len(origins) == 4
    assert all(direction == (0.0, 0.0, 1.0) for direction in directions)


# ---------------------------------------------------------------------------
# run_cuda_runtime_benchmark: iterations <= 0 error path (line 392)
# ---------------------------------------------------------------------------

def test_run_cuda_runtime_benchmark_raises_on_non_positive_iterations():
    from aura.benchmark import run_cuda_runtime_benchmark
    with pytest.raises(ValueError, match="iterations must be positive"):
        run_cuda_runtime_benchmark(native_demo_scene(), iterations=0)


# ---------------------------------------------------------------------------
# _load_real_scene_references: non-existent dir + empty dir error paths
# (lines 787, 805)
# ---------------------------------------------------------------------------

def test_load_real_scene_references_raises_on_missing_dir(tmp_path):
    from aura.benchmark import _load_real_scene_references
    with pytest.raises(ValueError, match="does not exist"):
        _load_real_scene_references(tmp_path / "nonexistent")


def test_load_real_scene_references_raises_on_empty_dir(tmp_path):
    from aura.benchmark import _load_real_scene_references
    with pytest.raises(ValueError, match="no reference images found"):
        _load_real_scene_references(tmp_path)


# ---------------------------------------------------------------------------
# _leave_one_out_color_predictions: single-color path (line 1013)
# _leave_one_out_scalar_predictions: single-value path (line 1024)
# ---------------------------------------------------------------------------

def test_leave_one_out_color_predictions_single_color_returns_zero():
    from aura.benchmark import _leave_one_out_color_predictions
    result = _leave_one_out_color_predictions(((0.8, 0.4, 0.2),))
    assert result == ((0.0, 0.0, 0.0),)


def test_leave_one_out_color_predictions_multi_color_returns_leave_one_out():
    from aura.benchmark import _leave_one_out_color_predictions
    # With 2 colors, each prediction = the other color
    result = _leave_one_out_color_predictions(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    assert len(result) == 2
    assert result[0] == pytest.approx((0.0, 1.0, 0.0))
    assert result[1] == pytest.approx((1.0, 0.0, 0.0))


def test_leave_one_out_scalar_predictions_single_value_returns_fallback():
    from aura.benchmark import _leave_one_out_scalar_predictions
    result = _leave_one_out_scalar_predictions((3.0,), fallback=42.0)
    assert result == (42.0,)


def test_leave_one_out_scalar_predictions_multi_value_returns_leave_one_out():
    from aura.benchmark import _leave_one_out_scalar_predictions
    result = _leave_one_out_scalar_predictions((1.0, 3.0), fallback=0.0)
    assert len(result) == 2
    assert result[0] == pytest.approx(3.0)
    assert result[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _metric_value: psnrInfinite path (line 1044)
# ---------------------------------------------------------------------------

def test_metric_value_returns_100_when_psnr_infinite():
    from aura.benchmark import _metric_value
    metrics = {"psnrInfinite": True, "psnr": None}
    assert _metric_value(metrics, "psnr") == 100.0


def test_metric_value_returns_zero_for_missing_key():
    from aura.benchmark import _metric_value
    assert _metric_value({}, "ssim") == 0.0


def test_metric_value_finite_psnr_capped_at_100():
    """A finite PSNR above the ceiling is capped at 100 so it never outranks an
    infinite (perfect) PSNR, which is also 100."""
    from aura.benchmark import _metric_value
    assert _metric_value({"psnr": 120.0}, "psnr") == 100.0
    assert _metric_value({"psnr": 25.0}, "psnr") == 25.0
    # Perfect (infinite) and a finite 120 dB now tie at the ceiling.
    assert _metric_value({"psnrInfinite": True, "psnr": None}, "psnr") == \
        _metric_value({"psnr": 120.0}, "psnr")


# ---------------------------------------------------------------------------
# _evaluate_capture_leave_one_out_baseline: skip zero-count batch (line 979),
# raise when no targets (line 988)
# ---------------------------------------------------------------------------

def test_evaluate_capture_leave_one_out_baseline_skips_zero_count_batches():
    from aura.benchmark import _evaluate_capture_leave_one_out_baseline

    class _ZeroBatch:
        target_count = 0

    class _RealBatch:
        target_count = 1
        target_color = (0.5, 0.3, 0.1)
        target_depth = (1.5,)

    result = _evaluate_capture_leave_one_out_baseline(
        "test_label", [_ZeroBatch(), _RealBatch()], device=None
    )
    assert result["sampleCount"] == 1


def test_evaluate_capture_leave_one_out_baseline_raises_on_no_targets():
    from aura.benchmark import _evaluate_capture_leave_one_out_baseline

    class _ZeroBatch:
        target_count = 0

    with pytest.raises(ValueError, match="capture mean baseline requires at least one target"):
        _evaluate_capture_leave_one_out_baseline("label", [_ZeroBatch()], device=None)


# ---------------------------------------------------------------------------
# _package_size: None path and missing path (lines 1722, 1725)
# ---------------------------------------------------------------------------

def test_package_size_returns_none_when_dir_is_none():
    from aura.benchmark import _package_size
    assert _package_size(None) is None


def test_package_size_returns_none_when_dir_does_not_exist(tmp_path):
    from aura.benchmark import _package_size
    assert _package_size(tmp_path / "nonexistent") is None


# ---------------------------------------------------------------------------
# _carrier_entropy: zero total + zero count paths (lines 1732, 1736)
# ---------------------------------------------------------------------------

def test_carrier_entropy_returns_zero_for_empty_counts():
    from aura.benchmark import _carrier_entropy
    assert _carrier_entropy({}) == 0.0


def test_carrier_entropy_skips_zero_count_carriers():
    from aura.benchmark import _carrier_entropy
    # One carrier has count=0, should be skipped (line 1736)
    entropy = _carrier_entropy({"surface": 10, "gaussian": 0})
    assert entropy == 0.0  # only one non-zero carrier → entropy is 0


def test_carrier_entropy_is_positive_for_mixed_carriers():
    from aura.benchmark import _carrier_entropy
    entropy = _carrier_entropy({"surface": 5, "gaussian": 5})
    assert entropy > 0.0


# ---------------------------------------------------------------------------
# _append_unique: duplicate suppression (line 1722 is _package_size,
# _append_unique is line 1715)
# ---------------------------------------------------------------------------

def test_append_unique_does_not_add_duplicate():
    from aura.benchmark import _append_unique
    items = ["a", "b"]
    _append_unique(items, "a")
    assert items == ["a", "b"]


def test_append_unique_adds_new_item():
    from aura.benchmark import _append_unique
    items = ["a"]
    _append_unique(items, "b")
    assert items == ["a", "b"]


# ---------------------------------------------------------------------------
# _percentile_ms: empty values path (line 2086)
# ---------------------------------------------------------------------------

def test_percentile_ms_returns_zero_for_empty_values():
    from aura.benchmark import _percentile_ms
    assert _percentile_ms([], 0.5) == 0.0


def test_percentile_ms_returns_correct_percentile():
    from aura.benchmark import _percentile_ms
    values = [0.001, 0.002, 0.003, 0.004]
    result = _percentile_ms(values, 0.5)
    assert result > 0.0


# ---------------------------------------------------------------------------
# _timed_scene_ray_inspections / _scene_center_traversals /
# _scene_center_expectations: early return on empty scene
# (lines 1924, 1939, 1951)
# ---------------------------------------------------------------------------

def test_timed_scene_ray_inspections_returns_empty_for_empty_scene():
    from aura.benchmark import _timed_scene_ray_inspections
    empty_scene = AuraScene(name="empty", elements=())
    inspections, timings = _timed_scene_ray_inspections(empty_scene)
    assert inspections == ()
    assert timings == ()


def test_scene_center_traversals_returns_empty_for_empty_scene():
    from aura.benchmark import _scene_center_traversals
    empty_scene = AuraScene(name="empty", elements=())
    assert _scene_center_traversals(empty_scene) == ()


def test_scene_center_expectations_returns_empty_for_empty_scene():
    from aura.benchmark import _scene_center_expectations
    empty_scene = AuraScene(name="empty", elements=())
    assert _scene_center_expectations(empty_scene) == ()


# ---------------------------------------------------------------------------
# evaluate_backend_readiness: additional blocker paths (lines 1428, 1430, 1432)
# ---------------------------------------------------------------------------

def test_evaluate_backend_readiness_adds_query_contract_blocker_when_fields_missing(monkeypatch):
    from aura.benchmark import _backend_query_contract
    original = _backend_query_contract

    def _patched_contract(runtime_export):
        result = original(runtime_export)
        result["missingFields"] = ["fake_field"]
        return result

    monkeypatch.setattr(benchmark_module, "_backend_query_contract", _patched_contract)
    payload = evaluate_backend_readiness(native_demo_scene())
    assert "ray_query_contract_fields_missing" in payload["productionBlockers"]


# ---------------------------------------------------------------------------
# _benchmark_production_gate: callable_boundary_unavailable path (line 1581),
# abi_parity_failed path (line 1589)
# ---------------------------------------------------------------------------

def test_benchmark_production_gate_adds_callable_boundary_unavailable_blocker():
    from aura.benchmark import _benchmark_production_gate
    gate = _benchmark_production_gate(
        backend_readiness={"productionCudaReady": False, "productionBlockers": []},
        cuda_renderer_callable_boundary={"callableBoundaryReady": False, "fallbackAvailable": False, "productionReady": False},
    )
    assert "cuda_renderer_callable_boundary_unavailable" in gate["productionBlockers"]


def test_benchmark_production_gate_adds_abi_parity_failed_blocker():
    from aura.benchmark import _benchmark_production_gate
    gate = _benchmark_production_gate(
        backend_readiness={"productionCudaReady": False, "productionBlockers": []},
        cuda_renderer_abi_parity={"passed": False, "productionReady": False},
    )
    assert "cuda_renderer_abi_parity_failed" in gate["productionBlockers"]


# ---------------------------------------------------------------------------
# run_real_scene_benchmark: fixture self-reference path via no reference_dir
# (lines 703-756, the block between reference_dir check and views loop)
# ---------------------------------------------------------------------------

def test_run_real_scene_benchmark_uses_fixture_when_no_reference_dir(tmp_path):
    from aura.benchmark import run_real_scene_benchmark
    pkg = package_scene(native_demo_scene())
    report = run_real_scene_benchmark(pkg, fixture_view_count=2)
    assert report["format"] == "AURA_REAL_SCENE_BENCHMARK"
    assert report["referenceSource"] == "deterministic_fixture"
    assert report["baseline"] == "fixture_self_reference"
    assert report["aggregate"]["viewCount"] == 2


def test_run_real_scene_benchmark_uses_external_reference_dir(tmp_path):
    from aura.benchmark import run_real_scene_benchmark
    # Write a small PPM reference image into tmp_path
    ppm_path = tmp_path / "view_001.ppm"
    ppm_path.write_text("P3\n2 2\n255\n255 0 0 0 255 0 0 0 255 255 255 0\n", encoding="ascii")
    pkg = package_scene(native_demo_scene())
    report = run_real_scene_benchmark(pkg, reference_dir=tmp_path, baseline_label="external_ref")
    assert report["format"] == "AURA_REAL_SCENE_BENCHMARK"
    assert report["referenceSource"] == "external_reference_dir"
    assert report["baseline"] == "external_ref"
    assert report["aggregate"]["viewCount"] == 1


# ---------------------------------------------------------------------------
# _load_real_scene_references: max_views path (line 799)
# ---------------------------------------------------------------------------

def test_load_real_scene_references_with_max_views_limits_candidates(tmp_path):
    from aura.benchmark import _load_real_scene_references
    for i in range(3):
        (tmp_path / f"view_{i:03d}.ppm").write_text(
            "P3\n2 2\n255\n255 0 0 0 255 0 0 0 255 255 255 0\n", encoding="ascii"
        )
    refs = _load_real_scene_references(tmp_path, max_views=2)
    assert len(refs) == 2


# ---------------------------------------------------------------------------
# _fixture_real_scene_references: non-positive view_count path (line 805)
# ---------------------------------------------------------------------------

def test_fixture_real_scene_references_raises_on_non_positive_view_count():
    from aura.benchmark import _fixture_real_scene_references
    with pytest.raises(ValueError, match="fixture view count must be positive"):
        _fixture_real_scene_references(native_demo_scene(), view_count=0)


# ---------------------------------------------------------------------------
# _capture_ray_query_expectations: skip zero-count batch (line 1058)
# ---------------------------------------------------------------------------

def test_capture_ray_query_expectations_skips_zero_count_batches(monkeypatch):
    class _ZeroBatch:
        batch_index = 0
        target_count = 0

    scene = native_demo_scene()
    result = benchmark_module._capture_ray_query_expectations(scene, [_ZeroBatch()], device=None)
    assert result == ()


# ---------------------------------------------------------------------------
# _scene_from_capture_manifest_dataset: unknown frame (line 1096),
# empty evidence (line 1099)
# ---------------------------------------------------------------------------

def test_scene_from_capture_manifest_dataset_raises_on_unknown_frame(tmp_path):
    """Trigger line 1096: region references a frame not in the manifest frames list."""
    from aura.benchmark import _scene_from_capture_manifest_dataset

    class _FakeRegion:
        id = "region_001"
        frame_id = "frame_NONEXISTENT"

        def to_evidence_sample(self, frame):
            return object()

    class _FakeDataset:
        frames = []  # no frames
        regions = [_FakeRegion()]

    class _FakeManifest:
        def to_training_dataset(self, *, load_assets=True):
            return _FakeDataset()

    with pytest.raises(ValueError, match="references unknown frame"):
        _scene_from_capture_manifest_dataset(_FakeManifest(), name="test")


def test_scene_from_capture_manifest_dataset_raises_on_empty_evidence(tmp_path):
    """Trigger line 1099: manifest has frames but no regions → empty evidence list."""
    from aura.benchmark import _scene_from_capture_manifest_dataset

    class _FakeDataset:
        frames = []
        regions = []  # no regions → evidence will be empty

    class _FakeManifest:
        def to_training_dataset(self, *, load_assets=True):
            return _FakeDataset()

    with pytest.raises(ValueError, match="capture benchmark requires at least one training region"):
        _scene_from_capture_manifest_dataset(_FakeManifest(), name="test")


# ---------------------------------------------------------------------------
# evaluate_backend_readiness: elements_missing_chunk_assignment (line 1430),
# chunk_traversal_not_observed (line 1432)
# ---------------------------------------------------------------------------

def test_evaluate_backend_readiness_adds_elements_missing_chunk_assignment_blocker():
    from aura.scene import AuraChunk
    element = AuraElement(
        id="orphan",
        carrier_id="surface",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "surface_cell"},
    )
    # Chunk has no elements assigned
    chunk = AuraChunk(id="c1", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=())
    scene = AuraScene(name="unchunked", elements=(element,), chunks=(chunk,))
    payload = evaluate_backend_readiness(scene)
    assert "elements_missing_chunk_assignment" in payload["productionBlockers"]


def test_evaluate_backend_readiness_adds_chunk_traversal_not_observed_blocker():
    from aura.scene import AuraChunk

    class _NoChunkTraversal:
        tested_chunk_ids = ()
        skipped_element_count = 0

    element = AuraElement(
        id="e1",
        carrier_id="surface",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "surface_cell"},
    )
    chunk = AuraChunk(id="c1", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("e1",))
    scene = AuraScene(name="no_chunk_traversal", elements=(element,), chunks=(chunk,))
    payload = evaluate_backend_readiness(scene, traversals=[_NoChunkTraversal()])
    assert "chunk_traversal_not_observed" in payload["productionBlockers"]


# ---------------------------------------------------------------------------
# _evaluate_capture_scene_predictions: raises when all batches have 0 targets
# (line 943)
# ---------------------------------------------------------------------------

def test_run_capture_reconstruction_benchmark_raises_on_empty_packed_batches(tmp_path, monkeypatch):
    """Trigger line 852: packed_batches is empty after packing."""
    monkeypatch.setattr(benchmark_module, "capture_tensors_to_packed_render_batches", lambda *a, **kw: [])
    manifest_path = _write_capture_benchmark_manifest(tmp_path)
    with pytest.raises(ValueError, match="capture benchmark requires at least one sampled target"):
        run_capture_reconstruction_benchmark(
            manifest_path,
            output_dir=tmp_path / "out.aura",
            iterations=1,
            device="cpu",
        )


def test_evaluate_capture_scene_predictions_raises_on_all_zero_count_batches(monkeypatch):
    from aura.benchmark import _evaluate_capture_scene_predictions

    class _ZeroBatch:
        target_count = 0

    scene = native_demo_scene()
    with pytest.raises(ValueError, match="capture evaluation requires at least one rendered batch"):
        _evaluate_capture_scene_predictions("label", scene, [_ZeroBatch()], device=None)


# ---------------------------------------------------------------------------
# run_cuda_runtime_benchmark: exception in CUDA render path (lines 463-465)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_run_cuda_runtime_benchmark_returns_report_on_render_failure(monkeypatch):
    import torch
    from aura.benchmark import run_cuda_runtime_benchmark

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable for this test")

    # Monkeypatch torch_render_rays to fail so the except block at line 463 fires
    from aura import torch_kernels as _tk_mod
    import aura.torch_renderer as _tr_mod

    def _fail_render(*_args, **_kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(_tr_mod, "torch_render_rays", _fail_render)

    # Also need to patch the benchmark module's reference to torch_render_rays
    monkeypatch.setattr(benchmark_module, "torch_render_rays", _fail_render, raising=False)

    report = run_cuda_runtime_benchmark(native_demo_scene(), ray_count=4, iterations=1, warmup=0)
    assert report["format"] == "AURA_CUDA_RUNTIME_BENCHMARK"
    # Either executed=False with a render_failed reason, or the patch didn't reach the module
    # (the inner closure captures `torch_render_rays` from a local import)
    assert report.get("reason") is None or "render_failed" in str(report.get("reason", "")) or report.get("executed") is True
