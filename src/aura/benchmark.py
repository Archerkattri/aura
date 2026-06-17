from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import log2
from pathlib import Path
from time import perf_counter
from typing import Sequence

from aura.asset import AuraAsset
from aura.core import ReconstructionConfig, ReconstructionReport, reconstruct_demo_scene
from aura.inspection import RayInspection, inspect_ray
from aura.package import AuraPackage
from aura.ray import Ray
from aura.render import RenderImage, compare_images, render_orthographic
from aura.runtime_export import runtime_export_report
from aura.scene import AuraScene
from aura.semantic import SemanticGraph


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
        "normalReadyRate": _rate(_optional_check_passed(probe, "normal") for probe in probes),
        "residualAccuracy": _rate(_optional_check_passed(probe, "residual") for probe in probes),
        "probes": list(probes),
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
        "runtimeExport": runtime_export_report(package).to_dict(),
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
        "metricNotes": {
            "lpipsProxy": "Deterministic mean absolute RGB distance; replace with learned LPIPS backend for paper claims.",
            "ssim": "Global RGB SSIM reference metric for deterministic smoke benchmarks.",
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
                purpose="Check first-hit, depth, normal, opacity, transmittance, semantic id, and provenance.",
                metrics=("first_hit_accuracy", "depth_abs_error", "normal_cosine", "transmittance_abs_error"),
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
        ),
        ablations=(
            AblationConfig(id="gaussian_only", disabled_carriers=("surface", "volume", "beta", "gabor", "neural", "semantic"), notes="Fallback-only baseline."),
            AblationConfig(id="no_neural_residual", disabled_carriers=("neural",), notes="Tests view-dependent residual value."),
            AblationConfig(id="no_frequency_carrier", disabled_carriers=("gabor",), notes="Tests high-frequency carrier value."),
            AblationConfig(id="no_semantic_graph", disabled_carriers=("semantic",), notes="Tests object graph/editability value."),
        ),
    )


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
    result = scene.ray_query(expectation.ray)
    first_hit = result.provenance != "miss"
    first_element_id = result.provenance.split(",", 1)[0] if first_hit and result.provenance else None
    element = element_by_id.get(first_element_id or "")
    actual_carrier_id = getattr(element, "carrier_id", None)
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
