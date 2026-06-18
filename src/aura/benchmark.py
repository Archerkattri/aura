from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import log2
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from aura.asset import AuraAsset
from aura.core import ReconstructionConfig, ReconstructionReport, reconstruct_demo_scene
from aura.decomposition import decompose_evidence
from aura.ingest.capture import (
    CaptureManifest,
    capture_tensors_to_training_dataset,
    load_capture_asset_tensors,
    load_capture_manifest,
)
from aura.inspection import RayInspection, inspect_ray
from aura.package import AuraPackage, package_scene
from aura.ray import Ray
from aura.render import RenderImage, compare_images, render_orthographic
from aura.runtime_export import runtime_export_report
from aura.scene import BVH_CHUNK_THRESHOLD, AuraScene
from aura.semantic import SemanticGraph
from aura.torch_optimizer import TorchOptimizationConfig, torch_optimize_capture_batches
from aura.torch_renderer import (
    TorchRenderBatch,
    torch_capture_training_batch_from_packed,
    torch_render_capture_training_batch,
)
from aura.training_targets import capture_tensors_to_packed_render_batches, plan_capture_tensor_sampling
from aura.cuda_kernels import cuda_renderer_report
from aura.cuda_renderer import (
    cuda_renderer_boundary_report,
    cuda_renderer_kernel_inputs,
    cuda_renderer_reference_first_hit_indices,
    simulate_cuda_renderer_kernel,
)
from aura.torch_kernels import torch_carrier_kernel_specs


NATIVE_PRODUCTION_CARRIER_IDS = ("surface", "volume", "beta", "gabor", "neural", "semantic")
CALLABLE_CUDA_RENDERER_OUTPUT_FIELDS = (
    "elementIds",
    "carrierIds",
    "color",
    "opacity",
    "transmittance",
    "depth",
    "normal",
    "confidence",
    "residual",
    "materialIds",
    "semanticIds",
    "provenance",
    "orderedHits",
    "orderedHitOverflow",
)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    purpose: str
    metrics: tuple[str, ...]
    baseline: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("benchmark id is required")
        if not self.metrics:
            raise ValueError("benchmark metrics are required")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AblationConfig:
    id: str
    disabled_carriers: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ablation id is required")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSuite:
    cases: Sequence[BenchmarkCase]
    ablations: Sequence[AblationConfig]

    def to_dict(self) -> dict:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "ablations": [ablation.to_dict() for ablation in self.ablations],
        }


@dataclass(frozen=True)
class RayQueryExpectation:
    label: str
    ray: Ray
    expected_first_hit: bool
    expected_element_id: str | None = None
    expected_carrier_id: str | None = None
    expected_depth: float | None = None
    depth_tolerance: float = 1e-6
    transmittance_min: float | None = None
    transmittance_max: float | None = None
    expected_semantic_id: str | None = None
    expected_material_id: str | None = None
    expected_residual: bool | None = None
    expected_ordered_element_ids: tuple[str, ...] = ()
    expected_ordered_carrier_ids: tuple[str, ...] = ()
    require_normal: bool = False

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ray expectation label is required")
        if self.depth_tolerance < 0.0:
            raise ValueError("depth_tolerance must be non-negative")
        if self.transmittance_min is not None and not 0.0 <= self.transmittance_min <= 1.0:
            raise ValueError("transmittance_min must be in [0, 1]")
        if self.transmittance_max is not None and not 0.0 <= self.transmittance_max <= 1.0:
            raise ValueError("transmittance_max must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "ray": {"origin": list(self.ray.origin), "direction": list(self.ray.direction)},
            "expectedFirstHit": self.expected_first_hit,
            "expectedElementId": self.expected_element_id,
            "expectedCarrierId": self.expected_carrier_id,
            "expectedDepth": self.expected_depth,
            "depthTolerance": self.depth_tolerance,
            "transmittanceMin": self.transmittance_min,
            "transmittanceMax": self.transmittance_max,
            "expectedSemanticId": self.expected_semantic_id,
            "expectedMaterialId": self.expected_material_id,
            "expectedResidual": self.expected_residual,
            "expectedOrderedElementIds": list(self.expected_ordered_element_ids),
            "expectedOrderedCarrierIds": list(self.expected_ordered_carrier_ids),
            "requireNormal": self.require_normal,
        }


def native_demo_ray_query_expectations() -> tuple[RayQueryExpectation, ...]:
    return (
        RayQueryExpectation(
            label="surface_first_hit",
            ray=Ray(origin=(-0.5, -0.5, -2.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
            expected_element_id="surface_wall",
            expected_carrier_id="surface",
            expected_depth=2.0,
            transmittance_min=0.09,
            transmittance_max=0.11,
            expected_material_id="mat_wall_plaster",
            require_normal=True,
        ),
        RayQueryExpectation(
            label="volume_transmittance",
            ray=Ray(origin=(0.1, -0.45, -2.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
            expected_element_id="soft_volume",
            expected_carrier_id="volume",
            expected_depth=2.0,
            transmittance_min=0.50,
            transmittance_max=0.51,
            expected_material_id="mat_soft_volume",
        ),
        RayQueryExpectation(
            label="semantic_object",
            ray=Ray(origin=(0.125, 0.275, -2.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
            expected_element_id="semantic_object",
            expected_carrier_id="semantic",
            expected_depth=2.0,
            transmittance_min=0.54,
            transmittance_max=0.56,
            expected_semantic_id="fixture_object",
        ),
        RayQueryExpectation(
            label="neural_residual",
            ray=Ray(origin=(-0.5, 0.3, -2.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=True,
            expected_element_id="view_residual",
            expected_carrier_id="neural",
            expected_depth=2.0,
            transmittance_min=0.39,
            transmittance_max=0.41,
            expected_residual=True,
        ),
        RayQueryExpectation(
            label="empty_space_control",
            ray=Ray(origin=(2.0, 2.0, -2.0), direction=(0.0, 0.0, 1.0)),
            expected_first_hit=False,
            transmittance_min=1.0,
            transmittance_max=1.0,
        ),
    )


def run_ray_query_correctness_benchmark(
    scene: AuraScene,
    expectations: Sequence[RayQueryExpectation],
) -> dict:
    if not expectations:
        raise ValueError("ray query correctness benchmark requires expectations")
    element_by_id = {element.id: element for element in scene.elements}
    probes = tuple(_score_ray_query_expectation(scene, element_by_id, expectation) for expectation in expectations)
    return {
        "format": "AURA_RAY_QUERY_CORRECTNESS_BENCHMARK",
        "scene": scene.name,
        "probeCount": len(probes),
        "passed": all(probe["passed"] for probe in probes),
        "passRate": _rate(probe["passed"] for probe in probes),
        "firstHitAccuracy": _rate(probe["checks"]["firstHit"]["passed"] for probe in probes),
        "elementAccuracy": _rate(_optional_check_passed(probe, "elementId") for probe in probes),
        "carrierAccuracy": _rate(_optional_check_passed(probe, "carrierId") for probe in probes),
        "depthWithinToleranceRate": _rate(_optional_check_passed(probe, "depth") for probe in probes),
        "transmittanceWithinBoundsRate": _rate(_optional_check_passed(probe, "transmittance") for probe in probes),
        "semanticAccuracy": _rate(_optional_check_passed(probe, "semanticId") for probe in probes),
        "materialAccuracy": _rate(_optional_check_passed(probe, "materialId") for probe in probes),
        "orderedElementTraceAccuracy": _rate(_optional_check_passed(probe, "orderedElementIds") for probe in probes),
        "orderedCarrierTraceAccuracy": _rate(_optional_check_passed(probe, "orderedCarrierIds") for probe in probes),
        "normalReadyRate": _rate(_optional_check_passed(probe, "normal") for probe in probes),
        "residualAccuracy": _rate(_optional_check_passed(probe, "residual") for probe in probes),
        "probes": list(probes),
    }


def run_cuda_renderer_abi_parity_benchmark(
    scene: AuraScene,
    expectations: Sequence[RayQueryExpectation],
    *,
    max_hits: int = 4,
) -> dict:
    if not expectations:
        raise ValueError("CUDA renderer ABI parity benchmark requires expectations")
    rays = tuple(expectation.ray for expectation in expectations)
    element_ids = tuple(element.id for element in scene.elements)
    try:
        inputs = cuda_renderer_kernel_inputs(
            scene,
            ray_origins=tuple(ray.origin for ray in rays),
            ray_directions=tuple(ray.direction for ray in rays),
            max_hits=max_hits,
        )
        simulation = simulate_cuda_renderer_kernel(inputs)
    except Exception as exc:
        return {
            "format": "AURA_CUDA_RENDERER_ABI_PARITY",
            "scene": scene.name,
            "passed": False,
            "parityReady": False,
            "productionReady": False,
            "probeCount": len(expectations),
            "firstHitIndexAccuracy": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "probes": [],
            "notes": (
                "This CPU-only parity benchmark validates the packaged CUDA renderer ABI inputs and outputs. "
                "Failure here blocks future CUDA dispatch parity, but success is not production CUDA readiness."
            ),
        }
    expected_indices = cuda_renderer_reference_first_hit_indices(scene, rays)
    simulated_indices = simulation.first_hit_indices
    probes = []
    for index, expectation in enumerate(expectations):
        expected_index = expected_indices[index]
        simulated_index = simulated_indices[index]
        expected_element_id = element_ids[expected_index] if expected_index >= 0 else None
        simulated_element_id = element_ids[simulated_index] if simulated_index >= 0 else None
        probes.append(
            {
                "label": expectation.label,
                "expectedFirstHitIndex": expected_index,
                "simulatedFirstHitIndex": simulated_index,
                "expectedElementId": expected_element_id,
                "simulatedElementId": simulated_element_id,
                "passed": expected_index == simulated_index,
            }
        )
    passed = all(probe["passed"] for probe in probes)
    return {
        "format": "AURA_CUDA_RENDERER_ABI_PARITY",
        "scene": scene.name,
        "passed": passed,
        "parityReady": passed,
        "productionReady": False,
        "probeCount": len(probes),
        "firstHitIndexAccuracy": _rate(probe["passed"] for probe in probes),
        "kernelSymbol": "aura_render_rays_kernel",
        "kernelInput": {
            "rayCount": inputs.ray_count,
            "elementCount": inputs.element_count,
            "maxHits": inputs.max_hits,
            "outputBufferShapes": {name: list(shape) for name, shape in inputs.output_buffer_shapes().items()},
        },
        "simulation": {
            "firstHitIndices": list(simulated_indices),
            "outDepth": list(simulation.out_depth),
            "outMaterialId": list(simulation.out_material_id),
            "outSemanticId": list(simulation.out_semantic_id),
            "outResidual": list(simulation.out_residual),
        },
        "probes": probes,
        "notes": (
            "This is a CPU oracle for the packaged CUDA renderer ABI. It validates deterministic flat-buffer "
            "first-hit parity against AuraScene.traverse_ray and does not compile or launch CUDA."
        ),
    }


def run_reference_benchmark(
    package: AuraPackage,
    *,
    package_dir: Path | str | None = None,
    render_width: int = 16,
    render_height: int = 16,
) -> dict:
    scene = package.scene
    carrier_counts = {carrier_id: 0 for carrier_id in scene.carrier_ids()}
    for element in scene.elements:
        carrier_counts[element.carrier_id] = carrier_counts.get(element.carrier_id, 0) + 1
    element_count = len(scene.elements)
    non_gaussian = sum(count for carrier, count in carrier_counts.items() if carrier != "gaussian")
    inspections, query_timings = _timed_scene_ray_inspections(scene)
    traversals = _scene_center_traversals(scene)
    hits = [inspection for inspection in inspections if inspection.first_hit]
    render_start = perf_counter()
    image = render_orthographic(scene, width=render_width, height=render_height)
    render_seconds = perf_counter() - render_start
    reference_visual_quality = compare_images(image, image)
    query_seconds = sum(query_timings)
    runtime_export = runtime_export_report(package).to_dict()
    backend_readiness = evaluate_backend_readiness(scene, runtime_export, traversals)
    native_carrier_coverage = evaluate_native_carrier_coverage(scene)
    cuda_renderer = cuda_renderer_report()
    cuda_renderer_callable_boundary = cuda_renderer_callable_boundary_report(scene)
    cuda_renderer_abi_parity = run_cuda_renderer_abi_parity_benchmark(scene, _scene_center_expectations(scene))
    preview_visual_claim = _visual_claim_boundary(
        baseline_label="reference_preview_self",
        self_reference=True,
        backend_readiness=backend_readiness,
    )
    return {
        "format": "AURA_REFERENCE_BENCHMARK",
        "asset": package.asset.name,
        "elementCount": element_count,
        "chunkCount": len(scene.chunks),
        "semanticObjectCount": len(scene.semantic_graph.nodes),
        "carrierCounts": carrier_counts,
        "carrierEntropy": _carrier_entropy(carrier_counts),
        "nonGaussianFraction": 0.0 if element_count == 0 else non_gaussian / element_count,
        "confidenceQuality": _scene_confidence_quality(scene),
        "packageBytes": _package_size(package_dir) if package_dir is not None else None,
        "rayQuery": {
            "probeCount": len(inspections),
            "hitCount": len(hits),
            "firstHitRate": 0.0 if not inspections else len(hits) / len(inspections),
            "shadowReadyCount": sum(1 for inspection in inspections if inspection.shadow_ready),
            "reflectionReadyCount": sum(1 for inspection in inspections if inspection.reflection_ready),
            "collisionProxyReadyCount": sum(1 for inspection in inspections if inspection.collision_proxy_ready),
            "querySeconds": query_seconds,
            "raysPerSecond": 0.0 if query_seconds <= 0.0 else len(inspections) / query_seconds,
            "queryP50Ms": _percentile_ms(query_timings, 0.5),
            "queryP95Ms": _percentile_ms(query_timings, 0.95),
            "chunkTraversal": {
                "enabled": bool(scene.chunks),
                "probeCount": len(traversals),
                "modes": sorted({item.traversal_mode for item in traversals}),
                "testedBvhNodeCount": sum(item.tested_bvh_node_count for item in traversals),
                "testedChunkCount": sum(len(item.tested_chunk_ids) for item in traversals),
                "testedElementCount": sum(len(item.tested_element_ids) for item in traversals),
                "skippedElementCount": sum(item.skipped_element_count for item in traversals),
                "traversals": [item.to_dict() for item in traversals],
            },
            "probes": [inspection.to_dict() for inspection in inspections],
        },
        "interactionQuality": _interaction_quality(inspections),
        "runtimeExport": runtime_export,
        "backendReadiness": backend_readiness,
        "nativeCarrierCoverage": native_carrier_coverage,
        "cudaRenderer": cuda_renderer,
        "cudaRendererCallableBoundary": cuda_renderer_callable_boundary,
        "cudaRendererAbiParity": cuda_renderer_abi_parity,
        "productionGate": _benchmark_production_gate(
            backend_readiness=backend_readiness,
            visual_claims=(preview_visual_claim,),
            native_carrier_coverage=native_carrier_coverage,
            cuda_renderer=cuda_renderer,
            cuda_renderer_callable_boundary=cuda_renderer_callable_boundary,
            cuda_renderer_abi_parity=cuda_renderer_abi_parity,
        ),
        "rayQueryCorrectness": run_ray_query_correctness_benchmark(
            scene,
            _scene_center_expectations(scene),
        ),
        "previewRender": {
            "width": image.width,
            "height": image.height,
            "pixelCount": len(image.pixels),
            "renderSeconds": render_seconds,
            "framesPerSecond": 0.0 if render_seconds <= 0.0 else 1.0 / render_seconds,
            "pixelsPerSecond": 0.0 if render_seconds <= 0.0 else len(image.pixels) / render_seconds,
            "referenceVisualQuality": reference_visual_quality,
            "visualClaimBoundary": preview_visual_claim,
        },
    }


def run_visual_quality_benchmark(
    package: AuraPackage,
    reference_image: RenderImage,
    *,
    baseline_label: str = "teacher",
    render_width: int | None = None,
    render_height: int | None = None,
    min_psnr: float | None = None,
) -> dict:
    width = render_width or reference_image.width
    height = render_height or reference_image.height
    render_start = perf_counter()
    rendered = render_orthographic(package.scene, width=width, height=height)
    render_seconds = perf_counter() - render_start
    metrics = compare_images(reference_image, rendered, min_psnr=min_psnr)
    backend_readiness = evaluate_backend_readiness(package.scene)
    native_carrier_coverage = evaluate_native_carrier_coverage(package.scene)
    cuda_renderer = cuda_renderer_report()
    cuda_renderer_callable_boundary = cuda_renderer_callable_boundary_report(package.scene)
    visual_claim = _visual_claim_boundary(
        baseline_label=baseline_label,
        self_reference=_is_self_reference_visual(baseline_label, metrics),
        backend_readiness=backend_readiness,
    )
    return {
        "format": "AURA_VISUAL_QUALITY_BENCHMARK",
        "asset": package.asset.name,
        "baseline": baseline_label,
        "render": {
            "width": rendered.width,
            "height": rendered.height,
            "pixelCount": len(rendered.pixels),
            "renderSeconds": render_seconds,
            "framesPerSecond": 0.0 if render_seconds <= 0.0 else 1.0 / render_seconds,
            "pixelsPerSecond": 0.0 if render_seconds <= 0.0 else len(rendered.pixels) / render_seconds,
        },
        "metrics": metrics,
        "passed": bool(metrics["passed"]),
        "backendReadiness": backend_readiness,
        "nativeCarrierCoverage": native_carrier_coverage,
        "cudaRenderer": cuda_renderer,
        "cudaRendererCallableBoundary": cuda_renderer_callable_boundary,
        "productionGate": _benchmark_production_gate(
            backend_readiness=backend_readiness,
            visual_claims=(visual_claim,),
            native_carrier_coverage=native_carrier_coverage,
            cuda_renderer=cuda_renderer,
            cuda_renderer_callable_boundary=cuda_renderer_callable_boundary,
        ),
        "visualClaimBoundary": visual_claim,
        "metricNotes": {
            "lpipsProxy": "Deterministic mean absolute RGB distance; replace with learned LPIPS backend for paper claims.",
            "ssim": "Global RGB SSIM reference metric for deterministic smoke benchmarks.",
        },
    }


def run_capture_reconstruction_benchmark(
    manifest_path: Path | str,
    *,
    output_dir: Path | str,
    iterations: int = 4,
    device: str | None = None,
    pixel_stride: int = 1,
    max_targets_per_frame: int | None = 4096,
    tile_size: int = 256,
    max_targets_per_batch: int | None = 1024,
    color_learning_rate: float = 0.25,
    baseline_package: AuraPackage | None = None,
    baseline_label: str = "external_baseline",
) -> dict:
    """Train native AURA carriers from a capture manifest and score capture targets."""

    manifest = load_capture_manifest(manifest_path)
    tensors = load_capture_asset_tensors(manifest)
    dataset = capture_tensors_to_training_dataset(manifest, tensors)
    initial_scene = _scene_from_capture_manifest_dataset(manifest, name="aura_capture_initial")
    sampling_plan = plan_capture_tensor_sampling(
        dataset.frames,
        tensors,
        pixel_stride=pixel_stride,
        max_targets_per_frame=max_targets_per_frame,
        tile_size=tile_size,
        max_targets_per_batch=max_targets_per_batch,
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        dataset.frames,
        tensors,
        pixel_stride=pixel_stride,
        max_targets_per_frame=max_targets_per_frame,
        tile_size=tile_size,
        max_targets_per_batch=max_targets_per_batch,
        sampling_plan=sampling_plan,
    )
    if not packed_batches:
        raise ValueError("capture benchmark requires at least one sampled target")

    train_start = perf_counter()
    result = torch_optimize_capture_batches(
        initial_scene,
        packed_batches,
        TorchOptimizationConfig(
            iterations=iterations,
            color_learning_rate=color_learning_rate,
            max_samples_per_batch=sampling_plan.max_targets_per_batch,
        ),
        device=device,
    )
    train_seconds = perf_counter() - train_start
    package_dir = package_scene(result.scene, fallbacks={"mesh": "fallback/aura-capture-benchmark.glb"}).write(output_dir)

    initial_eval = _evaluate_capture_scene_predictions(
        "aura_initial_reference",
        initial_scene,
        packed_batches,
        device=device,
    )
    trained_eval = _evaluate_capture_scene_predictions(
        "aura_native_trained",
        result.scene,
        packed_batches,
        device=device,
    )
    baseline_eval = (
        _evaluate_capture_scene_predictions(
            baseline_label,
            baseline_package.scene,
            packed_batches,
            device=device,
        )
        if baseline_package is not None
        else None
    )
    capture_baseline_eval = _evaluate_capture_leave_one_out_baseline(
        "capture_leave_one_out_color_depth_baseline",
        packed_batches,
        device=trained_eval["device"],
    )
    query_expectations = _capture_ray_query_expectations(result.scene, packed_batches, device=device)
    ray_query = run_ray_query_correctness_benchmark(result.scene, query_expectations) if query_expectations else None

    return {
        "format": "AURA_CAPTURE_RECONSTRUCTION_BENCHMARK",
        "manifest": str(manifest_path),
        "packageDir": str(package_dir),
        "device": trained_eval["device"],
        "iterations": iterations,
        "trainingSeconds": train_seconds,
        "trainingSteps": len(result.steps),
        "captureSamplingPlan": sampling_plan.to_dict(),
        "packedBatchCount": len(packed_batches),
        "packedTargetCount": sum(batch.target_count for batch in packed_batches),
        "initialReference": initial_eval,
        "captureBaseline": capture_baseline_eval,
        "trained": trained_eval,
        "baseline": baseline_eval,
        "improvement": _capture_metric_delta(initial_eval, trained_eval),
        "improvementVsCaptureBaseline": _capture_metric_delta(capture_baseline_eval, trained_eval),
        "rayQueryCorrectness": ray_query,
        "training": result.to_dict(),
        "notes": {
            "visualMetrics": "PSNR/SSIM/LPIPS-proxy are computed against capture image samples, not self-reference renders.",
            "captureBaseline": "Built-in non-AURA baseline predicts each target from other sampled capture colors/depths, with a neutral fallback for one-sample smokes.",
            "baseline": "3DGS or another external package is reported separately when --baseline-package is supplied.",
        },
    }


def _evaluate_capture_scene_predictions(
    label: str,
    scene: AuraScene,
    packed_batches: Sequence[Any],
    *,
    device: str | None,
) -> dict:
    render_start = perf_counter()
    rendered_batches = tuple(
        torch_render_capture_training_batch(
            scene,
            torch_capture_training_batch_from_packed(packed_batch, device=device),
        )
        for packed_batch in packed_batches
        if packed_batch.target_count > 0
    )
    render_seconds = perf_counter() - render_start
    if not rendered_batches:
        raise ValueError("capture evaluation requires at least one rendered batch")
    predicted_colors = tuple(color for batch in rendered_batches for color in batch.predicted_color)
    target_colors = tuple(color for batch in rendered_batches for color in batch.target_color)
    target_count = len(target_colors)
    predicted_image = RenderImage(width=target_count, height=1, pixels=predicted_colors)
    target_image = RenderImage(width=target_count, height=1, pixels=target_colors)
    metrics = compare_images(target_image, predicted_image)
    depth_errors = tuple(
        abs((predicted if predicted is not None else 0.0) - target)
        for batch in rendered_batches
        for predicted, target in zip(batch.predicted_depth, batch.target_depth)
    )
    normal_losses = tuple(value for batch in rendered_batches for value in batch.normal_loss)
    transmittance = tuple(value for batch in rendered_batches for value in batch.transmittance)
    trace_lengths = tuple(len(trace) for batch in rendered_batches for trace in batch.ordered_hits)
    hit_count = sum(1 for batch in rendered_batches for element_id in batch.element_ids if element_id is not None)
    return {
        "label": label,
        "device": rendered_batches[0].device,
        "sampleCount": target_count,
        "renderSeconds": render_seconds,
        "samplesPerSecond": 0.0 if render_seconds <= 0.0 else target_count / render_seconds,
        "metrics": metrics,
        "depthMeanAbsoluteError": _mean(depth_errors),
        "normalLossMean": _mean(normal_losses),
        "hitRate": 0.0 if target_count == 0 else hit_count / target_count,
        "orderedTraceMeanLength": _mean(trace_lengths),
        "transmittanceMean": _mean(transmittance),
    }


def _evaluate_capture_leave_one_out_baseline(label: str, packed_batches: Sequence[Any], *, device: str | None) -> dict:
    target_colors: list[tuple[float, float, float]] = []
    target_depths: list[float] = []
    for packed_batch in packed_batches:
        if packed_batch.target_count <= 0:
            continue
        colors = tuple(float(value) for value in packed_batch.target_color)
        depths = tuple(float(value) for value in packed_batch.target_depth)
        target_colors.extend(
            (colors[index], colors[index + 1], colors[index + 2])
            for index in range(0, len(colors), 3)
        )
        target_depths.extend(depths)
    if not target_colors:
        raise ValueError("capture mean baseline requires at least one target")
    predicted_colors = _leave_one_out_color_predictions(tuple(target_colors))
    predicted_depths = _leave_one_out_scalar_predictions(tuple(target_depths), fallback=0.0)
    target_image = RenderImage(width=len(target_colors), height=1, pixels=tuple(target_colors))
    predicted_image = RenderImage(width=len(predicted_colors), height=1, pixels=predicted_colors)
    depth_errors = tuple(abs(predicted - target) for predicted, target in zip(predicted_depths, target_depths))
    return {
        "label": label,
        "device": device or "cpu",
        "sampleCount": len(target_colors),
        "renderSeconds": 0.0,
        "samplesPerSecond": 0.0,
        "metrics": compare_images(target_image, predicted_image),
        "depthMeanAbsoluteError": _mean(depth_errors),
        "normalLossMean": 0.0,
        "hitRate": 0.0,
        "orderedTraceMeanLength": 0.0,
        "transmittanceMean": 1.0,
        "baselineKind": "leave_one_out_capture_color_depth",
    }


def _leave_one_out_color_predictions(colors: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    if len(colors) == 1:
        return ((0.0, 0.0, 0.0),)
    totals = tuple(sum(color[channel] for color in colors) for channel in range(3))
    denominator = len(colors) - 1
    return tuple(
        tuple((totals[channel] - color[channel]) / denominator for channel in range(3))  # type: ignore[misc]
        for color in colors
    )


def _leave_one_out_scalar_predictions(values: tuple[float, ...], *, fallback: float) -> tuple[float, ...]:
    if len(values) == 1:
        return (fallback,)
    total = sum(values)
    denominator = len(values) - 1
    return tuple((total - value) / denominator for value in values)


def _capture_metric_delta(initial: dict, trained: dict) -> dict:
    initial_metrics = initial["metrics"]
    trained_metrics = trained["metrics"]
    return {
        "psnrDelta": _metric_value(trained_metrics, "psnr") - _metric_value(initial_metrics, "psnr"),
        "ssimDelta": _metric_value(trained_metrics, "ssim") - _metric_value(initial_metrics, "ssim"),
        "lpipsProxyDelta": _metric_value(trained_metrics, "lpipsProxy") - _metric_value(initial_metrics, "lpipsProxy"),
        "mseDelta": _metric_value(trained_metrics, "mse") - _metric_value(initial_metrics, "mse"),
        "depthMeanAbsoluteErrorDelta": trained["depthMeanAbsoluteError"] - initial["depthMeanAbsoluteError"],
        "normalLossMeanDelta": trained["normalLossMean"] - initial["normalLossMean"],
    }


def _metric_value(metrics: dict, key: str) -> float:
    if key == "psnr" and metrics.get("psnrInfinite"):
        return 100.0
    value = metrics.get(key)
    return 0.0 if value is None else float(value)


def _capture_ray_query_expectations(
    scene: AuraScene,
    packed_batches: Sequence[Any],
    *,
    device: str | None,
) -> tuple[RayQueryExpectation, ...]:
    expectations: list[RayQueryExpectation] = []
    for packed_batch in packed_batches:
        if packed_batch.target_count <= 0:
            continue
        torch_batch = torch_capture_training_batch_from_packed(packed_batch, device=device)
        rendered = torch_render_capture_training_batch(scene, torch_batch)
        origins = torch_batch.ray_origins.detach().cpu().tolist()
        directions = torch_batch.ray_directions.detach().cpu().tolist()
        for index, (origin, direction, element_id, carrier_id, depth, transmittance, normal) in enumerate(
            zip(
                origins,
                directions,
                rendered.element_ids,
                rendered.carrier_ids,
                rendered.predicted_depth,
                rendered.transmittance,
                rendered.normal,
            )
        ):
            expectations.append(
                RayQueryExpectation(
                    label=f"capture_batch_{packed_batch.batch_index}_sample_{index}",
                    ray=Ray(origin=tuple(float(value) for value in origin), direction=tuple(float(value) for value in direction)),
                    expected_first_hit=element_id is not None,
                    expected_element_id=element_id,
                    expected_carrier_id=carrier_id,
                    expected_depth=depth,
                    depth_tolerance=1e-4,
                    transmittance_min=max(0.0, transmittance - 1e-4),
                    transmittance_max=min(1.0, transmittance + 1e-4),
                    require_normal=normal is not None,
                )
            )
    return tuple(expectations)


def _scene_from_capture_manifest_dataset(manifest: CaptureManifest, *, name: str) -> AuraScene:
    dataset = manifest.to_training_dataset(load_assets=False)
    by_frame = {frame.id: frame for frame in dataset.frames}
    evidence = []
    for region in dataset.regions:
        frame = by_frame.get(region.frame_id)
        if frame is None:
            raise ValueError(f"training region {region.id} references unknown frame {region.frame_id}")
        evidence.append(region.to_evidence_sample(frame))
    if not evidence:
        raise ValueError("capture benchmark requires at least one training region")
    return decompose_evidence(tuple(evidence), name=name)


def run_production_gate_report(
    package: AuraPackage,
    *,
    visual_baseline_label: str = "reference_preview_self",
    visual_self_reference: bool = True,
) -> dict:
    scene = package.scene
    runtime_export = runtime_export_report(package).to_dict()
    traversals = _scene_center_traversals(scene)
    backend_readiness = evaluate_backend_readiness(scene, runtime_export, traversals)
    native_carrier_coverage = evaluate_native_carrier_coverage(scene)
    cuda_renderer = cuda_renderer_report()
    cuda_renderer_callable_boundary = cuda_renderer_callable_boundary_report(scene)
    cuda_renderer_abi_parity = run_cuda_renderer_abi_parity_benchmark(scene, _scene_center_expectations(scene))
    visual_claim = _visual_claim_boundary(
        baseline_label=visual_baseline_label,
        self_reference=visual_self_reference or _is_self_reference_baseline(visual_baseline_label),
        backend_readiness=backend_readiness,
    )
    production_gate = _benchmark_production_gate(
        backend_readiness=backend_readiness,
        visual_claims=(visual_claim,),
        native_carrier_coverage=native_carrier_coverage,
        cuda_renderer=cuda_renderer,
        cuda_renderer_callable_boundary=cuda_renderer_callable_boundary,
        cuda_renderer_abi_parity=cuda_renderer_abi_parity,
    )
    return {
        "format": "AURA_PRODUCTION_GATE_REPORT",
        "asset": package.asset.name,
        "productionGate": production_gate,
        "cudaRenderer": cuda_renderer,
        "cudaRendererCallableBoundary": cuda_renderer_callable_boundary,
        "cudaRendererAbiParity": cuda_renderer_abi_parity,
        "backendReadiness": backend_readiness,
        "nativeCarrierCoverage": native_carrier_coverage,
        "visualClaimBoundary": visual_claim,
        "claimBoundary": {
            "productionClaimAllowed": bool(production_gate["productionReady"]),
            "safeCurrentClaim": (
                "AURA-Core is a scaffold for a native adaptive radiance reconstruction engine "
                "that converts captures into queryable runtime assets."
            ),
            "blockedClaims": [
                "production-ready engine integration",
                "real-time CUDA rendering or training",
                "paper-quality visual superiority over external baselines",
            ],
        },
    }


def run_ablation_benchmarks(
    package: AuraPackage,
    *,
    package_dir: Path | str | None = None,
    suite: BenchmarkSuite | None = None,
    render_width: int = 16,
    render_height: int = 16,
) -> dict:
    suite = suite or default_benchmark_suite()
    baseline = run_reference_benchmark(package, package_dir=package_dir, render_width=render_width, render_height=render_height)
    results = []
    for ablation in suite.ablations:
        ablated = apply_ablation(package, ablation)
        metrics = run_reference_benchmark(ablated, package_dir=None, render_width=render_width, render_height=render_height)
        results.append(
            {
                "id": ablation.id,
                "disabledCarriers": list(ablation.disabled_carriers),
                "notes": ablation.notes,
                "metrics": metrics,
                "delta": {
                    "elementCount": metrics["elementCount"] - baseline["elementCount"],
                    "nonGaussianFraction": metrics["nonGaussianFraction"] - baseline["nonGaussianFraction"],
                    "firstHitRate": metrics["rayQuery"]["firstHitRate"] - baseline["rayQuery"]["firstHitRate"],
                    "semanticObjectCount": metrics["semanticObjectCount"] - baseline["semanticObjectCount"],
                },
            }
        )
    return {
        "format": "AURA_ABLATION_BENCHMARK",
        "asset": package.asset.name,
        "baseline": baseline,
        "ablations": results,
    }


def run_core_reconstruction_benchmark(*, iterations: int = 6) -> dict:
    adaptive = reconstruct_demo_scene(
        ReconstructionConfig(iterations=iterations, enable_adaptive_evolution=True)
    )
    static = reconstruct_demo_scene(
        ReconstructionConfig(iterations=iterations, enable_adaptive_evolution=False)
    )
    adaptive_metrics = _core_report_metrics(adaptive.report)
    static_metrics = _core_report_metrics(static.report)
    return {
        "format": "AURA_CORE_RECONSTRUCTION_BENCHMARK",
        "scene": adaptive.scene.name,
        "iterations": iterations,
        "adaptive": {
            **adaptive_metrics,
            "elementCount": len(adaptive.scene.elements),
            "carrierCounts": _scene_carrier_counts(adaptive.scene),
            "confidenceQuality": _scene_confidence_quality(adaptive.scene),
            "evolvedElementCount": sum(1 for element in adaptive.scene.elements if element.metadata.get("source") == "aura-core-adaptive-evolution"),
        },
        "static": {
            **static_metrics,
            "elementCount": len(static.scene.elements),
            "carrierCounts": _scene_carrier_counts(static.scene),
            "confidenceQuality": _scene_confidence_quality(static.scene),
            "evolvedElementCount": sum(1 for element in static.scene.elements if element.metadata.get("source") == "aura-core-adaptive-evolution"),
        },
        "delta": {
            "finalLoss": adaptive_metrics["finalLoss"] - static_metrics["finalLoss"],
            "imageLoss": adaptive_metrics["finalImageLoss"] - static_metrics["finalImageLoss"],
            "depthLoss": adaptive_metrics["finalDepthLoss"] - static_metrics["finalDepthLoss"],
            "queryLoss": adaptive_metrics["finalQueryLoss"] - static_metrics["finalQueryLoss"],
            "elementCount": len(adaptive.scene.elements) - len(static.scene.elements),
            "adaptiveEvolutionActions": adaptive_metrics["evolutionActionCounts"],
        },
    }


def apply_ablation(package: AuraPackage, ablation: AblationConfig) -> AuraPackage:
    disabled = set(ablation.disabled_carriers)
    elements = tuple(element for element in package.scene.elements if element.carrier_id not in disabled)
    element_ids = {element.id for element in elements}
    chunks = tuple(
        type(chunk)(
            id=chunk.id,
            bounds=chunk.bounds,
            element_ids=tuple(element_id for element_id in chunk.element_ids if element_id in element_ids),
            lod=chunk.lod,
        )
        for chunk in package.scene.chunks
    )
    nodes = tuple(
        node
        for node in package.scene.semantic_graph.nodes
        if set(node.element_ids).issubset(element_ids)
    )
    edges = tuple(
        edge
        for edge in package.scene.semantic_graph.edges
        if {edge.source, edge.target}.issubset({node.id for node in nodes})
    )
    scene = AuraScene(
        name=f"{package.scene.name}_{ablation.id}",
        elements=elements,
        chunks=chunks,
        semantic_graph=SemanticGraph(nodes=nodes, edges=edges),
    )
    asset = AuraAsset(
        name=f"{package.asset.name}_{ablation.id}",
        carrier_ids=scene.carrier_ids() or ("gaussian",),
        version=package.asset.version,
        units=package.asset.units,
        coordinate_system=package.asset.coordinate_system,
        fallbacks=dict(package.asset.fallbacks),
    )
    return AuraPackage(asset=asset, scene=scene, exchange=dict(package.exchange))


def default_benchmark_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        cases=(
            BenchmarkCase(
                id="visual_quality_vs_3dgs",
                purpose="Compare native AURA preview output against a 3DGS teacher render.",
                metrics=("mse", "psnr", "ssim", "lpips_proxy"),
                baseline="3dgs",
            ),
            BenchmarkCase(
                id="ray_query_correctness",
                purpose="Check first-hit, ordered hit traces, depth, normal, opacity, transmittance, semantic id, and provenance.",
                metrics=(
                    "first_hit_accuracy",
                    "ordered_element_trace_accuracy",
                    "ordered_carrier_trace_accuracy",
                    "depth_abs_error",
                    "normal_cosine",
                    "transmittance_abs_error",
                ),
            ),
            BenchmarkCase(
                id="geometry_collision_proxy",
                purpose="Measure whether surface/semantic carriers provide stable proxy geometry.",
                metrics=("collision_proxy_ready_rate", "collision_distance_ready_rate", "hit_point_ready_rate"),
            ),
            BenchmarkCase(
                id="shadow_reflection_queries",
                purpose="Score reference shadow transmittance and reflection-vector query readiness.",
                metrics=("shadow_transmittance_ready_rate", "shadow_unoccluded_rate", "reflection_vector_ready_rate"),
            ),
            BenchmarkCase(
                id="package_size",
                purpose="Track native .aura package footprint against fallback exports.",
                metrics=("bytes_total", "bytes_per_element", "fallback_bytes"),
            ),
            BenchmarkCase(
                id="render_query_speed",
                purpose="Track reference render and ray-query throughput before GPU kernels land.",
                metrics=("rays_per_second", "frames_per_second", "pixels_per_second", "query_p50_ms", "query_p95_ms"),
            ),
            BenchmarkCase(
                id="mixed_carrier_behavior",
                purpose="Verify non-Gaussian carriers dominate when evidence supports them.",
                metrics=("carrier_entropy", "non_gaussian_fraction", "assignment_rule_coverage"),
            ),
            BenchmarkCase(
                id="confidence_calibration",
                purpose="Track carrier confidence values and residual-backed confidence map coverage.",
                metrics=("mean_element_confidence", "optimization_residual_map_rate", "low_residual_high_confidence_rate"),
            ),
            BenchmarkCase(
                id="aura_core_reconstruction",
                purpose="Measure native AURA-Core reconstruction loss and adaptive carrier evolution.",
                metrics=("final_loss", "image_loss_delta", "depth_loss_delta", "query_loss_delta", "split_promote_merge_demote_counts"),
                baseline="static_carriers",
            ),
            BenchmarkCase(
                id="runtime_export_contract",
                purpose="Check native AURA runtime export metadata for carrier chunks, ray-query contract fields, and fallback losses.",
                metrics=("chunk_export_count", "ray_query_field_count", "fallback_loss_count", "native_runtime_ready"),
            ),
            BenchmarkCase(
                id="backend_readiness_contract",
                purpose="Evaluate CPU-verifiable torch/reference backend readiness without claiming production CUDA.",
                metrics=(
                    "scene_carrier_autograd_coverage_rate",
                    "scene_carrier_cuda_coverage_rate",
                    "ray_query_field_coverage_rate",
                    "chunked_element_coverage_rate",
                    "chunk_culling_observed_rate",
                    "production_cuda_ready",
                ),
            ),
            BenchmarkCase(
                id="cuda_renderer_abi_parity",
                purpose="Validate packaged CUDA renderer flat-buffer inputs and first-hit outputs against the CPU ray-query oracle.",
                metrics=("first_hit_index_accuracy", "ray_count", "element_count", "ordered_hit_shape", "production_ready_false"),
            ),
            BenchmarkCase(
                id="production_gate_contract",
                purpose="Block production claims on CUDA renderer availability, self-reference visual scores, and native carrier coverage.",
                metrics=(
                    "cuda_renderer_available",
                    "visual_benchmark_self_reference",
                    "required_native_carrier_coverage_rate",
                    "blocks_production_claim",
                ),
            ),
        ),
        ablations=(
            AblationConfig(id="gaussian_only", disabled_carriers=("surface", "volume", "beta", "gabor", "neural", "semantic"), notes="Fallback-only baseline."),
            AblationConfig(id="no_neural_residual", disabled_carriers=("neural",), notes="Tests view-dependent residual value."),
            AblationConfig(id="no_frequency_carrier", disabled_carriers=("gabor",), notes="Tests high-frequency carrier value."),
            AblationConfig(id="no_semantic_graph", disabled_carriers=("semantic",), notes="Tests object graph/editability value."),
        ),
    )


def evaluate_backend_readiness(
    scene: AuraScene,
    runtime_export: dict[str, Any] | None = None,
    traversals: Sequence[Any] = (),
) -> dict:
    runtime_export = runtime_export or _scene_runtime_contract_view(scene)
    carrier_ids = tuple(scene.carrier_ids())
    kernel_specs = torch_carrier_kernel_specs()
    specs_by_carrier = {spec.carrier_id: spec for spec in kernel_specs}
    carrier_kernel_rows = []
    for carrier_id in carrier_ids:
        spec = specs_by_carrier.get(carrier_id)
        carrier_kernel_rows.append(
            {
                "carrierId": carrier_id,
                "autogradKernel": bool(spec and spec.autograd_kernel),
                "cudaKernel": bool(spec and spec.cuda_kernel),
                "productionReady": bool(spec and spec.production_ready),
                "implementationStage": spec.implementation_stage if spec else "missing_kernel_spec",
                "differentiableFields": list(spec.differentiable_fields) if spec else [],
            }
        )
    query_contract = _backend_query_contract(runtime_export)
    chunk_lod_contract = _backend_chunk_lod_contract(scene, runtime_export, traversals)
    autograd_coverage = _rate(row["autogradKernel"] for row in carrier_kernel_rows)
    cuda_coverage = _rate(row["cudaKernel"] for row in carrier_kernel_rows)
    production_ready = bool(carrier_kernel_rows) and all(row["productionReady"] for row in carrier_kernel_rows)
    blockers = []
    if not production_ready:
        blockers.append("carrier_cuda_kernels_not_production_ready")
    if query_contract["missingFields"]:
        blockers.append("ray_query_contract_fields_missing")
    if scene.elements and chunk_lod_contract["chunkedElementCoverageRate"] < 1.0:
        blockers.append("elements_missing_chunk_assignment")
    if scene.chunks and chunk_lod_contract["chunkCullingObservedRate"] <= 0.0:
        blockers.append("chunk_traversal_not_observed")
    return {
        "format": "AURA_BACKEND_READINESS_EVALUATION",
        "scene": scene.name,
        "requiresTorchImport": False,
        "evaluatedBackend": "cpu_reference_contract_for_torch_cuda_readiness",
        "productionCudaReady": production_ready,
        "mvpContractReady": (
            autograd_coverage == 1.0
            and query_contract["fieldCoverageRate"] == 1.0
            and (not scene.elements or chunk_lod_contract["chunkedElementCoverageRate"] == 1.0)
        ),
        "sceneCarrierAutogradCoverageRate": autograd_coverage,
        "sceneCarrierCudaCoverageRate": cuda_coverage,
        "sceneCarrierProductionReadyRate": _rate(row["productionReady"] for row in carrier_kernel_rows),
        "carrierKernelContract": {
            "sceneCarrierCount": len(carrier_kernel_rows),
            "knownKernelSpecCount": len(kernel_specs),
            "missingAutogradCarrierIds": [row["carrierId"] for row in carrier_kernel_rows if not row["autogradKernel"]],
            "missingCudaCarrierIds": [row["carrierId"] for row in carrier_kernel_rows if not row["cudaKernel"]],
            "carriers": carrier_kernel_rows,
        },
        "queryContract": query_contract,
        "chunkLodContract": chunk_lod_contract,
        "productionBlockers": blockers,
        "notes": (
            "This is a deterministic CPU contract evaluation for backend readiness. "
            "It does not execute torch, require CUDA, or certify production CUDA throughput."
        ),
    }


def evaluate_native_carrier_coverage(
    scene: AuraScene,
    *,
    required_native_carrier_ids: Sequence[str] = NATIVE_PRODUCTION_CARRIER_IDS,
) -> dict:
    carrier_counts = _scene_carrier_counts(scene)
    element_count = len(scene.elements)
    required = tuple(required_native_carrier_ids)
    present_required = tuple(carrier_id for carrier_id in required if carrier_counts.get(carrier_id, 0) > 0)
    missing_required = tuple(carrier_id for carrier_id in required if carrier_counts.get(carrier_id, 0) <= 0)
    native_element_count = sum(carrier_counts.get(carrier_id, 0) for carrier_id in required)
    gaussian_fallback_count = carrier_counts.get("gaussian", 0)
    blockers = []
    if element_count == 0 or native_element_count == 0:
        blockers.append("native_carriers_absent")
    if missing_required:
        blockers.append("native_carrier_coverage_incomplete")
    if element_count > 0 and gaussian_fallback_count == element_count:
        blockers.append("gaussian_fallback_only_scene")
    return {
        "format": "AURA_NATIVE_CARRIER_COVERAGE",
        "scene": scene.name,
        "requiredNativeCarrierIds": list(required),
        "presentNativeCarrierIds": list(present_required),
        "missingNativeCarrierIds": list(missing_required),
        "carrierCounts": carrier_counts,
        "elementCount": element_count,
        "nativeElementCount": native_element_count,
        "gaussianFallbackElementCount": gaussian_fallback_count,
        "requiredNativeCarrierCoverageRate": _rate(carrier_id in present_required for carrier_id in required),
        "nativeElementFraction": 0.0 if element_count == 0 else native_element_count / element_count,
        "gaussianFallbackFraction": 0.0 if element_count == 0 else gaussian_fallback_count / element_count,
        "auraFirstCoverageReady": not blockers,
        "productionBlockers": blockers,
        "notes": (
            "This is a broad AURA-first production-claim gate for benchmark packages. "
            "It does not make every valid package use every carrier, but it blocks broad "
            "production claims until native carrier families are exercised instead of "
            "relying only on Gaussian fallback."
        ),
    }


def cuda_renderer_callable_boundary_report(scene: AuraScene) -> dict:
    boundary = cuda_renderer_boundary_report(
        scene,
        probe_ray_origin=(0.0, 0.0, -2.0),
        probe_ray_direction=(0.0, 0.0, 1.0),
        fallback_backend="cpu",
        max_hits=4,
    )
    fallback_probe = boundary.get("fallbackProbe")
    fallback_probe = fallback_probe if isinstance(fallback_probe, dict) else {}
    output_fields = tuple(str(field) for field in fallback_probe.get("outputFields", ()))
    missing_output_fields = [field for field in CALLABLE_CUDA_RENDERER_OUTPUT_FIELDS if field not in output_fields]
    fallback_backend = fallback_probe.get("backend")
    callable_execution_available = bool(fallback_probe.get("executed")) and fallback_backend in {"cpu", "torch", "cuda"}
    fallback_available = bool(fallback_probe.get("executed")) and fallback_backend in {"cpu", "torch"}
    return {
        "format": "AURA_CUDA_RENDERER_CALLABLE_BOUNDARY",
        "reportKind": "callable_cuda_renderer_fallback_boundary",
        "apiName": boundary.get("apiName"),
        "callableBoundaryReady": bool(boundary.get("callableBoundaryAvailable")) and callable_execution_available,
        "fallbackContractReady": not missing_output_fields,
        "fallbackAvailable": fallback_available,
        "compiledExecutionAvailable": bool(fallback_backend == "cuda" and fallback_probe.get("executed")),
        "fallbackBackend": fallback_backend,
        "compiledCudaAvailable": bool(boundary.get("available")),
        "productionReady": bool(boundary.get("productionReady")),
        "reason": fallback_probe.get("reason") or fallback_probe.get("error"),
        "missingOutputFields": missing_output_fields,
        "outputFields": [field for field in CALLABLE_CUDA_RENDERER_OUTPUT_FIELDS if field in output_fields],
        "launchConfig": {
            "rayCount": fallback_probe.get("rayCount"),
            "maxHits": fallback_probe.get("maxHits"),
        }
        if fallback_probe
        else None,
        "extension": boundary.get("extension"),
        "boundaryReport": boundary,
        "notes": (
            "This probes aura.cuda_renderer.cuda_render_rays, the callable launch boundary. "
            "A CPU or torch fallback proves the ray-query output contract is callable, "
            "not that CUDA acceleration exists."
        ),
    }


def _benchmark_production_gate(
    *,
    backend_readiness: dict[str, Any],
    visual_claims: Sequence[dict[str, Any]] = (),
    native_carrier_coverage: dict[str, Any] | None = None,
    cuda_renderer: dict[str, Any] | None = None,
    cuda_renderer_callable_boundary: dict[str, Any] | None = None,
    cuda_renderer_abi_parity: dict[str, Any] | None = None,
) -> dict:
    blockers = list(backend_readiness.get("productionBlockers", ()))
    if not backend_readiness.get("productionCudaReady", False):
        _append_unique(blockers, "cuda_renderer_unavailable")
    if cuda_renderer is not None and not (
        cuda_renderer.get("available", False) and cuda_renderer.get("productionReady", False)
    ):
        _append_unique(blockers, "cuda_renderer_unavailable")
    if cuda_renderer_callable_boundary is not None:
        if not cuda_renderer_callable_boundary.get("callableBoundaryReady", False):
            _append_unique(blockers, "cuda_renderer_callable_boundary_unavailable")
        if (
            cuda_renderer_callable_boundary.get("fallbackAvailable", False)
            and not cuda_renderer_callable_boundary.get("productionReady", False)
        ):
            _append_unique(blockers, "cuda_renderer_callable_fallback_only")
    if cuda_renderer_abi_parity is not None:
        if not cuda_renderer_abi_parity.get("passed", False):
            _append_unique(blockers, "cuda_renderer_abi_parity_failed")
        if not cuda_renderer_abi_parity.get("productionReady", False):
            _append_unique(blockers, "cuda_renderer_abi_parity_cpu_oracle_only")
    if native_carrier_coverage is not None:
        for blocker in native_carrier_coverage.get("productionBlockers", ()):
            _append_unique(blockers, str(blocker))
    for claim in visual_claims:
        for blocker in claim.get("productionBlockers", ()):
            _append_unique(blockers, str(blocker))
    cuda_renderer_ready = bool(backend_readiness.get("productionCudaReady")) and (
        cuda_renderer is None or bool(cuda_renderer.get("productionReady"))
    )
    callable_fallback_available = bool(
        cuda_renderer_callable_boundary and cuda_renderer_callable_boundary.get("fallbackAvailable")
    )
    return {
        "format": "AURA_BENCHMARK_PRODUCTION_GATE",
        "productionReady": not blockers,
        "blocksProductionClaim": bool(blockers),
        "cudaRendererReady": cuda_renderer_ready,
        "cudaRendererAvailable": bool(cuda_renderer and cuda_renderer.get("available")),
        "cudaRendererProductionReady": bool(cuda_renderer and cuda_renderer.get("productionReady")),
        "cudaRendererReportKind": (
            None if cuda_renderer is None else "legacy_cuda_kernels_metadata_report"
        ),
        "cudaRendererCallableBoundaryReady": bool(
            cuda_renderer_callable_boundary
            and cuda_renderer_callable_boundary.get("callableBoundaryReady")
        ),
        "cudaRendererCallableFallbackAvailable": callable_fallback_available,
        "cudaRendererCallableFallbackBackend": (
            None
            if cuda_renderer_callable_boundary is None
            else cuda_renderer_callable_boundary.get("fallbackBackend")
        ),
        "cudaRendererCallableFallbackOnly": callable_fallback_available
        and not bool(cuda_renderer_callable_boundary and cuda_renderer_callable_boundary.get("productionReady")),
        "cudaRendererCallableProductionReady": bool(
            cuda_renderer_callable_boundary
            and cuda_renderer_callable_boundary.get("productionReady")
        ),
        "cudaRendererAbiParityReady": bool(
            cuda_renderer_abi_parity
            and cuda_renderer_abi_parity.get("parityReady")
        ),
        "cudaRendererAbiParityProductionReady": bool(
            cuda_renderer_abi_parity
            and cuda_renderer_abi_parity.get("productionReady")
        ),
        "cudaRendererAbiParityProbeCount": (
            None
            if cuda_renderer_abi_parity is None
            else cuda_renderer_abi_parity.get("probeCount")
        ),
        "nativeCarrierCoverageReady": bool(
            native_carrier_coverage and native_carrier_coverage.get("auraFirstCoverageReady")
        ),
        "requiredNativeCarrierCoverageRate": (
            None
            if native_carrier_coverage is None
            else native_carrier_coverage.get("requiredNativeCarrierCoverageRate")
        ),
        "missingNativeCarrierIds": (
            []
            if native_carrier_coverage is None
            else list(native_carrier_coverage.get("missingNativeCarrierIds", ()))
        ),
        "visualBenchmarkSelfReference": any(bool(claim.get("selfReference")) for claim in visual_claims),
        "visualBenchmarkExternalReference": any(bool(claim.get("externalReference")) for claim in visual_claims),
        "productionBlockers": blockers,
        "claimRequirements": [
            "production CUDA renderer is compiled, loadable, parity-tested, and benchmarked",
            "callable cuda_renderer fallback is replaced by real CUDA dispatch before production CUDA claims",
            "cudaRendererAbiParity productionReady is true for compiled CUDA dispatch, not just the CPU oracle",
            "visual quality is measured against external teacher or baseline renders",
            "benchmark package exercises native AURA carrier families instead of Gaussian fallback only",
        ],
        "notes": (
            "This gate is deterministic and CPU-only. It blocks production claims when CUDA renderer readiness "
            "is unavailable, visual quality is measured against a self-reference, or broad native carrier "
            "coverage has not been exercised."
        ),
    }


def _visual_claim_boundary(
    *,
    baseline_label: str,
    self_reference: bool,
    backend_readiness: dict[str, Any],
) -> dict:
    blockers = []
    if self_reference:
        blockers.append("visual_benchmark_self_reference")
    if not backend_readiness.get("productionCudaReady", False):
        blockers.append("cuda_renderer_unavailable")
    notes = (
        "Self-reference visual scores are smoke checks only and cannot support production or paper-quality "
        "visual claims."
        if self_reference
        else "External-reference visual scores still require production CUDA readiness and baseline context before claims."
    )
    return {
        "baseline": baseline_label,
        "selfReference": self_reference,
        "externalReference": not self_reference,
        "cudaRendererReady": bool(backend_readiness.get("productionCudaReady")),
        "productionClaimAllowed": not blockers,
        "productionBlockers": blockers,
        "notes": notes,
    }


def _is_self_reference_baseline(baseline_label: str) -> bool:
    normalized = baseline_label.lower().replace("-", "_").replace(" ", "_")
    return normalized in {"self", "native_self", "aura_self", "self_reference", "reference_preview_self"} or "self" in normalized


def _is_self_reference_visual(baseline_label: str, metrics: dict[str, Any]) -> bool:
    return _is_self_reference_baseline(baseline_label) or (
        bool(metrics.get("psnrInfinite"))
        and metrics.get("ssim") == 1.0
        and metrics.get("lpipsProxy") == 0.0
    )


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _package_size(package_dir: Path | str | None) -> int | None:
    if package_dir is None:
        return None
    root = Path(package_dir)
    if not root.exists():
        return None
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _carrier_entropy(carrier_counts: dict[str, int]) -> float:
    total = sum(carrier_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in carrier_counts.values():
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * log2(probability)
    return entropy


def _core_report_metrics(report: ReconstructionReport) -> dict:
    payload = report.to_dict()
    first = payload["iterations"][0]
    final = payload["iterations"][-1]
    actions = {}
    for step in payload["iterations"]:
        for decision in step["carrier_evolution"]:
            action = decision["action"]
            actions[action] = actions.get(action, 0) + 1
    return {
        "finalLoss": payload["finalLoss"],
        "initialLoss": first["total_loss"],
        "finalImageLoss": final["image_loss"],
        "finalDepthLoss": final["depth_loss"],
        "finalQueryLoss": final["query_loss"],
        "lossReduction": first["total_loss"] - payload["finalLoss"],
        "nativeCarrierFraction": payload["nativeCarrierFraction"],
        "evolutionActionCounts": actions,
    }


def _interaction_quality(inspections: Sequence[RayInspection]) -> dict:
    hits = [inspection for inspection in inspections if inspection.first_hit]
    shadows = [inspection for inspection in hits if inspection.shadow_transmittance is not None]
    reflections = [inspection for inspection in hits if inspection.reflection_ready]
    collisions = [inspection for inspection in hits if inspection.collision_proxy_ready]
    return {
        "hitPointReadyRate": _rate(inspection.hit_point is not None for inspection in hits),
        "shadowTransmittanceReadyRate": _rate(inspection.shadow_transmittance is not None for inspection in hits),
        "shadowTransmittanceWithinBoundsRate": _rate(
            0.0 <= inspection.shadow_transmittance <= 1.0
            for inspection in shadows
            if inspection.shadow_transmittance is not None
        ),
        "shadowUnoccludedRate": _rate(inspection.shadow_occluded is False for inspection in shadows),
        "reflectionVectorReadyRate": _rate(inspection.reflection_direction is not None for inspection in hits),
        "reflectionHitRate": _rate(inspection.reflection_hit is True for inspection in reflections),
        "collisionProxyReadyRate": _rate(inspection.collision_proxy_ready for inspection in hits),
        "collisionDistanceReadyRate": _rate(inspection.collision_distance is not None for inspection in collisions),
    }


def _scene_confidence_quality(scene: AuraScene) -> dict:
    elements = tuple(scene.elements)
    confidences = [element.confidence for element in elements]
    residual_elements = [
        element
        for element in elements
        if "optimization_residual" in element.confidence_map
    ]
    return {
        "meanElementConfidence": _mean(confidences),
        "minElementConfidence": min(confidences) if confidences else 0.0,
        "confidenceWithinBoundsRate": _rate(0.0 <= value <= 1.0 for value in confidences),
        "confidenceMapCoverageRate": _rate(bool(element.confidence_map) for element in elements),
        "optimizationResidualMapRate": _rate("optimization_residual" in element.confidence_map for element in elements),
        "queryResidualMapRate": _rate(
            ("optimization_query_loss" in element.confidence_map) or ("query" in element.confidence_map)
            for element in elements
        ),
        "lowResidualHighConfidenceRate": _rate(
            element.confidence >= 0.5 and element.confidence_map["optimization_residual"] <= 0.1
            for element in residual_elements
        ),
    }


def _scene_carrier_counts(scene: AuraScene) -> dict[str, int]:
    counts = {carrier_id: 0 for carrier_id in scene.carrier_ids()}
    for element in scene.elements:
        counts[element.carrier_id] += 1
    return counts


def _scene_runtime_contract_view(scene: AuraScene) -> dict:
    chunk_count = len(scene.chunks)
    active_mode = "bvh" if chunk_count >= BVH_CHUNK_THRESHOLD else "chunk_linear" if chunk_count else "element_linear"
    return {
        "rayQueryContract": {
            "fields": [
                "firstHit",
                "depth",
                "normal",
                "transmittance",
                "opacity",
                "semanticId",
                "materialId",
                "confidence",
                "residual",
                "provenance",
                "orderedHits",
            ],
            "supportsFirstHit": True,
            "supportsOrderedHitTrace": True,
            "supportsCompositing": True,
            "requiresNativeAuraRuntime": True,
        },
        "chunkExport": [
            {
                "chunkId": chunk.id,
                "lod": chunk.lod,
                "elementIds": list(chunk.element_ids),
                "carrierIds": sorted(
                    {
                        element.carrier_id
                        for element in scene.elements
                        if element.id in chunk.element_ids
                    }
                ),
            }
            for chunk in scene.chunks
        ],
        "accelerationContract": {
            "activeTraversalMode": active_mode,
            "supportedTraversalModes": ["element_linear", "chunk_linear", "bvh"],
            "supportsChunkCulling": chunk_count > 0,
            "supportsCachedBvh": chunk_count >= BVH_CHUNK_THRESHOLD,
            "productionGpuTraversalReady": False,
        },
    }


def _backend_query_contract(runtime_export: dict[str, Any]) -> dict:
    required_fields = (
        "firstHit",
        "depth",
        "normal",
        "transmittance",
        "opacity",
        "semanticId",
        "materialId",
        "confidence",
        "residual",
        "provenance",
        "orderedHits",
    )
    contract = runtime_export.get("rayQueryContract", {})
    present_fields = tuple(str(field) for field in contract.get("fields", ()))
    missing_fields = [field for field in required_fields if field not in present_fields]
    return {
        "requiredFields": list(required_fields),
        "presentFields": list(present_fields),
        "missingFields": missing_fields,
        "fieldCoverageRate": 1.0 - (len(missing_fields) / len(required_fields)),
        "supportsFirstHit": bool(contract.get("supportsFirstHit")),
        "supportsOrderedHitTrace": bool(contract.get("supportsOrderedHitTrace")),
        "supportsCompositing": bool(contract.get("supportsCompositing")),
        "requiresNativeAuraRuntime": bool(contract.get("requiresNativeAuraRuntime")),
    }


def _backend_chunk_lod_contract(
    scene: AuraScene,
    runtime_export: dict[str, Any],
    traversals: Sequence[Any],
) -> dict:
    element_ids = {element.id for element in scene.elements}
    chunked_element_ids = {element_id for chunk in scene.chunks for element_id in chunk.element_ids}
    chunk_export = tuple(runtime_export.get("chunkExport", ()))
    acceleration = runtime_export.get("accelerationContract", {})
    return {
        "chunkCount": len(scene.chunks),
        "exportedChunkCount": len(chunk_export),
        "lodLevels": sorted({chunk.lod for chunk in scene.chunks}),
        "elementCount": len(element_ids),
        "chunkedElementCount": len(element_ids & chunked_element_ids),
        "orphanElementIds": sorted(element_ids - chunked_element_ids),
        "chunkedElementCoverageRate": _rate(element_id in chunked_element_ids for element_id in element_ids),
        "chunkCarrierCoverageRate": _rate(bool(entry.get("carrierIds")) for entry in chunk_export),
        "activeTraversalMode": acceleration.get("activeTraversalMode", "unknown"),
        "supportedTraversalModes": list(acceleration.get("supportedTraversalModes", ())),
        "supportsChunkCulling": bool(acceleration.get("supportsChunkCulling")),
        "supportsCachedBvh": bool(acceleration.get("supportsCachedBvh")),
        "productionGpuTraversalReady": bool(acceleration.get("productionGpuTraversalReady")),
        "traversalProbeCount": len(traversals),
        "chunkCullingObservedRate": _rate(bool(getattr(traversal, "tested_chunk_ids", ())) for traversal in traversals),
        "skippedElementObservedRate": _rate(getattr(traversal, "skipped_element_count", 0) > 0 for traversal in traversals),
    }


def _timed_scene_ray_inspections(scene: AuraScene) -> tuple[tuple[RayInspection, ...], tuple[float, ...]]:
    if not scene.elements:
        return tuple(), tuple()
    camera_z = min(element.bounds.min_corner[2] for element in scene.elements) - 2.0
    inspections = []
    timings = []
    for element in scene.elements[:8]:
        center = tuple((lo + hi) / 2.0 for lo, hi in zip(element.bounds.min_corner, element.bounds.max_corner))
        ray = Ray(origin=(center[0], center[1], camera_z), direction=(0.0, 0.0, 1.0))
        start = perf_counter()
        inspections.append(inspect_ray(scene, ray, label=element.id))
        timings.append(perf_counter() - start)
    return tuple(inspections), tuple(timings)


def _scene_center_traversals(scene: AuraScene) -> tuple:
    if not scene.elements:
        return tuple()
    camera_z = min(element.bounds.min_corner[2] for element in scene.elements) - 2.0
    traversals = []
    for element in scene.elements[:8]:
        center = tuple((lo + hi) / 2.0 for lo, hi in zip(element.bounds.min_corner, element.bounds.max_corner))
        ray = Ray(origin=(center[0], center[1], camera_z), direction=(0.0, 0.0, 1.0))
        traversals.append(scene.traverse_ray(ray))
    return tuple(traversals)


def _scene_center_expectations(scene: AuraScene) -> tuple[RayQueryExpectation, ...]:
    if not scene.elements:
        return tuple()
    camera_z = min(element.bounds.min_corner[2] for element in scene.elements) - 2.0
    expectations = []
    for element in scene.elements[:8]:
        center = tuple((lo + hi) / 2.0 for lo, hi in zip(element.bounds.min_corner, element.bounds.max_corner))
        expectations.append(
            RayQueryExpectation(
                label=element.id,
                ray=Ray(origin=(center[0], center[1], camera_z), direction=(0.0, 0.0, 1.0)),
                expected_first_hit=True,
                expected_element_id=element.id,
                expected_carrier_id=element.carrier_id,
                expected_depth=element.bounds.min_corner[2] - camera_z,
                depth_tolerance=1e-6,
                transmittance_min=0.0,
                transmittance_max=1.0,
                expected_semantic_id=element.semantic_id,
                expected_material_id=element.material_id,
                expected_residual=element.residual,
                require_normal=element.normal is not None or element.payload.get("type") == "surface_cell",
            )
        )
    return tuple(expectations)


def _score_ray_query_expectation(
    scene: AuraScene,
    element_by_id: dict[str, object],
    expectation: RayQueryExpectation,
) -> dict:
    traversal = scene.traverse_ray(expectation.ray)
    result = traversal.result
    first_hit = result.provenance != "miss"
    first_element_id = result.provenance.split(",", 1)[0] if first_hit and result.provenance else None
    element = element_by_id.get(first_element_id or "")
    actual_carrier_id = getattr(element, "carrier_id", None)
    ordered_element_ids = tuple(hit.element_id for hit in traversal.ordered_hits)
    ordered_carrier_ids = tuple(hit.carrier_id for hit in traversal.ordered_hits)
    checks = {
        "firstHit": _check(expectation.expected_first_hit, first_hit),
    }
    if expectation.expected_element_id is not None:
        checks["elementId"] = _check(expectation.expected_element_id, first_element_id)
    if expectation.expected_carrier_id is not None:
        checks["carrierId"] = _check(expectation.expected_carrier_id, actual_carrier_id)
    if expectation.expected_depth is not None:
        checks["depth"] = _check_range(
            result.depth,
            expectation.expected_depth - expectation.depth_tolerance,
            expectation.expected_depth + expectation.depth_tolerance,
        )
    if expectation.transmittance_min is not None or expectation.transmittance_max is not None:
        checks["transmittance"] = _check_range(
            result.transmittance,
            expectation.transmittance_min if expectation.transmittance_min is not None else 0.0,
            expectation.transmittance_max if expectation.transmittance_max is not None else 1.0,
        )
    if expectation.expected_semantic_id is not None:
        checks["semanticId"] = _check(expectation.expected_semantic_id, result.semantic_id)
    if expectation.expected_material_id is not None:
        checks["materialId"] = _check(expectation.expected_material_id, result.material_id)
    if expectation.expected_residual is not None:
        checks["residual"] = _check(expectation.expected_residual, result.residual)
    if expectation.expected_ordered_element_ids:
        checks["orderedElementIds"] = _check(
            list(expectation.expected_ordered_element_ids),
            list(ordered_element_ids),
        )
    if expectation.expected_ordered_carrier_ids:
        checks["orderedCarrierIds"] = _check(
            list(expectation.expected_ordered_carrier_ids),
            list(ordered_carrier_ids),
        )
    if expectation.require_normal:
        checks["normal"] = _check(True, result.normal is not None)
    return {
        "label": expectation.label,
        "passed": all(item["passed"] for item in checks.values()),
        "expected": expectation.to_dict(),
        "actual": {
            "firstHit": first_hit,
            "elementId": first_element_id,
            "carrierId": actual_carrier_id,
            "depth": result.depth,
            "transmittance": result.transmittance,
            "opacity": result.opacity,
            "semanticId": result.semantic_id,
            "materialId": result.material_id,
            "confidence": result.confidence,
            "residual": result.residual,
            "normal": list(result.normal) if result.normal is not None else None,
            "provenance": result.provenance,
            "orderedHits": [hit.to_dict() for hit in traversal.ordered_hits],
            "orderedElementIds": list(ordered_element_ids),
            "orderedCarrierIds": list(ordered_carrier_ids),
        },
        "checks": checks,
    }


def _check(expected: object, actual: object) -> dict:
    return {
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    }


def _check_range(actual: float | None, minimum: float, maximum: float) -> dict:
    return {
        "min": minimum,
        "max": maximum,
        "actual": actual,
        "passed": actual is not None and minimum <= actual <= maximum,
    }


def _optional_check_passed(probe: dict, name: str) -> bool | None:
    check = probe["checks"].get(name)
    return None if check is None else bool(check["passed"])


def _rate(values) -> float:
    scored = [bool(value) for value in values if value is not None]
    if not scored:
        return 1.0
    return sum(1 for value in scored if value) / len(scored)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _percentile_ms(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index] * 1000.0
