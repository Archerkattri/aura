#!/usr/bin/env python3
"""Validate the production CUDA renderer path.

This gate is intentionally stricter than the CUDA boundary smoke tests: it only
passes when the compiled Python binding dispatches on CUDA, no fallback backend
is used, the result matches the torch GPU reference within tolerance, and the
measured compiled throughput clears a declared floor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any


FORMAT = "AURA_CUDA_PRODUCTION_BACKEND_REPORT"
FAILED_PARITY_SENTINEL = 1.0e30


def _json_number(value: float, *, fallback: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(fallback)


def summarize_cuda_gate(
    *,
    compiled_cuda_dispatch: bool,
    fallback_used: bool,
    device: str | None,
    max_abs_error: float,
    parity_threshold: float,
    rays_per_second: float,
    min_rays_per_second: float,
    extra_failures: tuple[str, ...] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical CUDA production gate summary."""

    failures = list(extra_failures)
    if not compiled_cuda_dispatch:
        failures.append("compiled CUDA dispatch was not used")
    if fallback_used:
        failures.append("fallback backend was used")
    if str(device) != "cuda" and not str(device).startswith("cuda:"):
        failures.append("device was not cuda")
    max_abs_error = _json_number(max_abs_error, fallback=FAILED_PARITY_SENTINEL)
    parity_threshold = _json_number(parity_threshold, fallback=-1.0)
    rays_per_second = _json_number(rays_per_second, fallback=0.0)
    min_rays_per_second = _json_number(min_rays_per_second, fallback=FAILED_PARITY_SENTINEL)

    if max_abs_error > parity_threshold:
        failures.append("CUDA parity threshold was exceeded")
    if rays_per_second < min_rays_per_second:
        failures.append("CUDA throughput floor was not reached")

    report: dict[str, Any] = {
        "format": FORMAT,
        "passed": not failures,
        "compiledCudaDispatch": bool(compiled_cuda_dispatch),
        "fallbackUsed": bool(fallback_used),
        "device": device,
        "parity": {
            "maxAbsError": max_abs_error,
            "threshold": parity_threshold,
            "passed": max_abs_error <= parity_threshold,
        },
        "throughput": {
            "raysPerSecond": rays_per_second,
            "minRaysPerSecond": min_rays_per_second,
            "passed": rays_per_second >= min_rays_per_second,
        },
        "failures": failures,
    }
    if extra:
        report.update(extra)
    return report


def _readiness_scene():
    from aura.core import synthetic_training_frames, synthetic_training_regions
    from aura.decomposition import decompose_evidence

    frames = {frame.id: frame for frame in synthetic_training_frames()}
    evidence = tuple(region.to_evidence_sample(frames[region.frame_id]) for region in synthetic_training_regions())
    return decompose_evidence(evidence, name="cuda_production_validation")


def _ray_grid(scene: Any, ray_count: int) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...]]:
    from aura.benchmark import _benchmark_ray_grid

    return _benchmark_ray_grid(scene, ray_count)


def _batch_max_abs_error(cuda_batch: Any, torch_batch: Any) -> float:
    max_error = 0.0
    for cuda_color, torch_color in zip(cuda_batch.color, torch_batch.predicted_color):
        for cuda_value, torch_value in zip(cuda_color, torch_color):
            max_error = max(max_error, abs(float(cuda_value) - float(torch_value)))
    for cuda_value, torch_value in zip(cuda_batch.opacity, torch_batch.opacity):
        max_error = max(max_error, abs(float(cuda_value) - float(torch_value)))
    for cuda_value, torch_value in zip(cuda_batch.transmittance, torch_batch.transmittance):
        max_error = max(max_error, abs(float(cuda_value) - float(torch_value)))
    for cuda_value, torch_value in zip(cuda_batch.depth, torch_batch.predicted_depth):
        if cuda_value is None or torch_value is None:
            max_error = max(max_error, 0.0 if cuda_value is None and torch_value is None else float("inf"))
        else:
            max_error = max(max_error, abs(float(cuda_value) - float(torch_value)))
    return max_error


def run_validation(
    *,
    device: str,
    ray_count: int,
    iterations: int,
    warmup: int,
    max_hits: int,
    parity_threshold: float,
    min_rays_per_second: float,
) -> dict[str, Any]:
    if ray_count <= 0:
        raise ValueError("ray_count must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")

    scene = _readiness_scene()
    origins, directions = _ray_grid(scene, ray_count)
    extra: dict[str, Any] = {
        "scene": scene.name,
        "rayCount": ray_count,
        "iterations": iterations,
        "warmup": warmup,
        "maxHits": max_hits,
        "claimBoundary": (
            "Passes only for compiled CUDA renderer dispatch without CPU/torch fallback; "
            "fixture-scale parity and throughput are not a full real-dataset FPS claim."
        ),
    }

    try:
        import torch
    except Exception as exc:
        return summarize_cuda_gate(
            compiled_cuda_dispatch=False,
            fallback_used=False,
            device=device,
            max_abs_error=float("inf"),
            parity_threshold=parity_threshold,
            rays_per_second=0.0,
            min_rays_per_second=min_rays_per_second,
            extra_failures=(f"torch import failed: {type(exc).__name__}: {exc}",),
            extra=extra,
        )

    if not str(device).startswith("cuda"):
        return summarize_cuda_gate(
            compiled_cuda_dispatch=False,
            fallback_used=False,
            device=device,
            max_abs_error=float("inf"),
            parity_threshold=parity_threshold,
            rays_per_second=0.0,
            min_rays_per_second=min_rays_per_second,
            extra_failures=("CUDA device was not requested",),
            extra=extra,
        )
    if not bool(torch.cuda.is_available()):
        return summarize_cuda_gate(
            compiled_cuda_dispatch=False,
            fallback_used=False,
            device=device,
            max_abs_error=float("inf"),
            parity_threshold=parity_threshold,
            rays_per_second=0.0,
            min_rays_per_second=min_rays_per_second,
            extra_failures=("torch.cuda.is_available() was false",),
            extra=extra,
        )

    from aura.cuda_renderer import cuda_render_rays
    from aura.torch_renderer import torch_render_rays

    origin_tensor = torch.tensor(origins, dtype=torch.float32, device=device).contiguous()
    direction_tensor = torch.tensor(directions, dtype=torch.float32, device=device).contiguous()
    extra["cudaDeviceName"] = torch.cuda.get_device_name(0)

    def cuda_call() -> Any:
        return cuda_render_rays(
            scene,
            ray_origins=origin_tensor,
            ray_directions=direction_tensor,
            device=device,
            require_cuda=True,
            fallback_backend="none",
            max_hits=max_hits,
        )

    try:
        cuda_batch = cuda_call()
    except Exception as exc:
        return summarize_cuda_gate(
            compiled_cuda_dispatch=False,
            fallback_used=False,
            device=device,
            max_abs_error=float("inf"),
            parity_threshold=parity_threshold,
            rays_per_second=0.0,
            min_rays_per_second=min_rays_per_second,
            extra_failures=(f"compiled CUDA dispatch failed: {type(exc).__name__}: {exc}",),
            extra=extra,
        )

    compiled_cuda_dispatch = cuda_batch.backend == "cuda" and cuda_batch.production_ready
    fallback_used = cuda_batch.backend != "cuda"
    try:
        torch_batch = torch_render_rays(scene, origin_tensor, direction_tensor, device=device, collect_traces=False)
        torch.cuda.synchronize()
        max_abs_error = _batch_max_abs_error(cuda_batch, torch_batch)
    except Exception as exc:
        return summarize_cuda_gate(
            compiled_cuda_dispatch=compiled_cuda_dispatch,
            fallback_used=fallback_used,
            device=cuda_batch.device,
            max_abs_error=float("inf"),
            parity_threshold=parity_threshold,
            rays_per_second=0.0,
            min_rays_per_second=min_rays_per_second,
            extra_failures=(f"torch GPU reference parity failed: {type(exc).__name__}: {exc}",),
            extra=extra,
        )

    for _ in range(warmup):
        cuda_call()
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(iterations):
        cuda_call()
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    rays_per_second = (ray_count * iterations) / seconds if seconds > 0.0 else float("inf")

    return summarize_cuda_gate(
        compiled_cuda_dispatch=compiled_cuda_dispatch,
        fallback_used=fallback_used,
        device=cuda_batch.device,
        max_abs_error=max_abs_error,
        parity_threshold=parity_threshold,
        rays_per_second=rays_per_second,
        min_rays_per_second=min_rays_per_second,
        extra=extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "--output", dest="out", type=Path, default=Path("experiments/results/cuda_production_backend_2026-06-24.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ray-count", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-hits", type=int, default=8)
    parser.add_argument("--parity-threshold", type=float, default=1e-3)
    parser.add_argument("--min-rays-per-second", type=float, default=1.0)
    args = parser.parse_args()

    payload = run_validation(
        device=args.device,
        ray_count=args.ray_count,
        iterations=args.iterations,
        warmup=args.warmup,
        max_hits=args.max_hits,
        parity_threshold=args.parity_threshold,
        min_rays_per_second=args.min_rays_per_second,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
