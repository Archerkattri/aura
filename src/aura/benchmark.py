from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from aura.inspection import inspect_scene_rays
from aura.package import AuraPackage
from aura.render import render_orthographic


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
    inspections = inspect_scene_rays(scene)
    hits = [inspection for inspection in inspections if inspection.first_hit]
    image = render_orthographic(scene, width=render_width, height=render_height)
    return {
        "format": "AURA_REFERENCE_BENCHMARK",
        "asset": package.asset.name,
        "elementCount": element_count,
        "chunkCount": len(scene.chunks),
        "semanticObjectCount": len(scene.semantic_graph.nodes),
        "carrierCounts": carrier_counts,
        "nonGaussianFraction": 0.0 if element_count == 0 else non_gaussian / element_count,
        "packageBytes": _package_size(package_dir) if package_dir is not None else None,
        "rayQuery": {
            "probeCount": len(inspections),
            "hitCount": len(hits),
            "firstHitRate": 0.0 if not inspections else len(hits) / len(inspections),
            "shadowReadyCount": sum(1 for inspection in inspections if inspection.shadow_ready),
            "reflectionReadyCount": sum(1 for inspection in inspections if inspection.reflection_ready),
            "collisionProxyReadyCount": sum(1 for inspection in inspections if inspection.collision_proxy_ready),
            "probes": [inspection.to_dict() for inspection in inspections],
        },
        "previewRender": {
            "width": image.width,
            "height": image.height,
            "pixelCount": len(image.pixels),
        },
    }


def default_benchmark_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        cases=(
            BenchmarkCase(
                id="visual_quality_vs_3dgs",
                purpose="Compare native AURA preview output against a 3DGS teacher render.",
                metrics=("mse", "psnr", "ssim_placeholder"),
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
                metrics=("proxy_iou", "collision_false_positive_rate", "collision_false_negative_rate"),
            ),
            BenchmarkCase(
                id="package_size",
                purpose="Track native .aura package footprint against fallback exports.",
                metrics=("bytes_total", "bytes_per_element", "fallback_bytes"),
            ),
            BenchmarkCase(
                id="render_query_speed",
                purpose="Track reference render and ray-query throughput before GPU kernels land.",
                metrics=("rays_per_second", "render_seconds", "query_p50_ms", "query_p95_ms"),
            ),
            BenchmarkCase(
                id="mixed_carrier_behavior",
                purpose="Verify non-Gaussian carriers dominate when evidence supports them.",
                metrics=("carrier_entropy", "non_gaussian_fraction", "assignment_rule_coverage"),
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
