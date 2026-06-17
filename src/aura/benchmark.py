from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


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
