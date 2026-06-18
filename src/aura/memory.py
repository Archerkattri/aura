"""Long-run memory stability probes for AURA render/query loops.

Production rendering and interactive querying run for thousands of iterations.
This module drives a configurable render/query workload while sampling memory so
unbounded growth (leaked tensors, accumulating Python objects) is caught early.

Two signals are tracked:

* ``tracemalloc`` peak/current Python allocation across iterations.
* ``torch.cuda.memory_allocated`` when a CUDA torch device is in use.

A linear-growth heuristic compares the average allocation of the final samples
against the early samples; a leak shows up as sustained growth proportional to
iteration count. The probe is intentionally cheap enough for CI yet structured
so that a genuine per-iteration leak would trip the threshold.
"""

from __future__ import annotations

import gc
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from aura.ray import Ray
from aura.render import render_orthographic
from aura.scene import AuraScene

__all__ = [
    "MemoryStabilityReport",
    "run_memory_stability_probe",
]


@dataclass(frozen=True)
class MemoryStabilityReport:
    iterations: int
    samples: tuple[int, ...]
    baseline_bytes: int
    peak_bytes: int
    final_bytes: int
    early_mean_bytes: float
    late_mean_bytes: float
    growth_bytes_per_iteration: float
    threshold_bytes_per_iteration: float
    stable: bool
    cuda_allocated_start: int | None
    cuda_allocated_end: int | None
    backend: str

    def to_dict(self) -> dict:
        return {
            "format": "AURA_MEMORY_STABILITY",
            "iterations": self.iterations,
            "backend": self.backend,
            "samples": list(self.samples),
            "baselineBytes": self.baseline_bytes,
            "peakBytes": self.peak_bytes,
            "finalBytes": self.final_bytes,
            "earlyMeanBytes": self.early_mean_bytes,
            "lateMeanBytes": self.late_mean_bytes,
            "growthBytesPerIteration": self.growth_bytes_per_iteration,
            "thresholdBytesPerIteration": self.threshold_bytes_per_iteration,
            "stable": self.stable,
            "cudaAllocatedStart": self.cuda_allocated_start,
            "cudaAllocatedEnd": self.cuda_allocated_end,
        }


def _default_workload(
    scene: AuraScene,
    *,
    width: int,
    height: int,
) -> Callable[[int], None]:
    bounds = None
    rays = (
        Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        Ray(origin=(0.1, -0.1, -2.0), direction=(0.0, 0.0, 1.0)),
    )

    def _step(_iteration: int) -> None:
        image = render_orthographic(scene, width=width, height=height, bounds=bounds)
        # touch a pixel so the render is not optimized away
        _ = image.pixel(0, 0)
        for ray in rays:
            _ = scene.ray_query(ray)

    return _step


def run_memory_stability_probe(
    scene: AuraScene,
    *,
    iterations: int = 64,
    width: int = 16,
    height: int = 16,
    workload: Callable[[int], None] | None = None,
    sample_interval: int = 8,
    growth_threshold_bytes_per_iteration: float = 4096.0,
    device: str | None = None,
    backend: str = "cpu",
) -> MemoryStabilityReport:
    """Run a render/query loop and assess whether memory stays bounded.

    ``growth_threshold_bytes_per_iteration`` is the maximum tolerated sustained
    Python allocation growth per iteration; a steady leak of even a few hundred
    bytes per loop accumulates well past this over the iteration count.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    step = workload or _default_workload(scene, width=width, height=height)

    cuda = _maybe_cuda(device)
    cuda_start: int | None = None
    cuda_end: int | None = None
    if cuda is not None:
        cuda.synchronize()
        cuda.empty_cache()
        cuda_start = int(cuda.memory_allocated())

    gc.collect()
    tracing_already = tracemalloc.is_tracing()
    if not tracing_already:
        tracemalloc.start()
    try:
        # Warm up so one-time allocations are not counted as growth.
        step(0)
        gc.collect()
        baseline_current, _ = tracemalloc.get_traced_memory()
        samples: list[int] = []
        for index in range(1, iterations + 1):
            step(index)
            if index % sample_interval == 0 or index == iterations:
                current, _ = tracemalloc.get_traced_memory()
                samples.append(int(current))
        peak_current, peak = tracemalloc.get_traced_memory()
    finally:
        if not tracing_already:
            tracemalloc.stop()

    if cuda is not None:
        cuda.synchronize()
        cuda_end = int(cuda.memory_allocated())

    if not samples:
        samples = [int(peak_current)]
    half = max(1, len(samples) // 2)
    early = samples[:half]
    late = samples[-half:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    span = max(iterations - sample_interval, 1)
    growth_per_iteration = (late_mean - early_mean) / span

    stable = growth_per_iteration <= growth_threshold_bytes_per_iteration
    if cuda_start is not None and cuda_end is not None:
        # A leaked CUDA tensor would grow allocated memory with iterations.
        cuda_growth_per_iteration = (cuda_end - cuda_start) / iterations
        stable = stable and cuda_growth_per_iteration <= growth_threshold_bytes_per_iteration

    return MemoryStabilityReport(
        iterations=iterations,
        samples=tuple(samples),
        baseline_bytes=int(baseline_current),
        peak_bytes=int(peak),
        final_bytes=int(samples[-1]),
        early_mean_bytes=early_mean,
        late_mean_bytes=late_mean,
        growth_bytes_per_iteration=growth_per_iteration,
        threshold_bytes_per_iteration=growth_threshold_bytes_per_iteration,
        stable=stable,
        cuda_allocated_start=cuda_start,
        cuda_allocated_end=cuda_end,
        backend=backend,
    )


def _maybe_cuda(device: str | None) -> Any | None:
    if device is None or not str(device).startswith("cuda"):
        return None
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda
