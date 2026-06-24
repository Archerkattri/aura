#!/usr/bin/env python3
"""Validate the AURA torch backend on bounded real-capture CUDA tensors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any


FORMAT = "AURA_TORCH_BACKEND_VALIDATION"
ROOT = Path(__file__).resolve().parent.parent


def summarize_torch_backend_gate(
    *,
    device: str | None,
    manifest_frame_count: int,
    manifest_region_count: int,
    loaded_frame_count: int,
    scene_element_count: int,
    packed_batch_count: int,
    packed_target_count: int,
    max_batch_target_count: int,
    finite_losses: bool,
    render_seconds: float,
    max_allowed_batch_targets: int,
    min_manifest_regions: int,
    min_packed_targets: int,
) -> dict[str, Any]:
    failures = []
    if str(device) != "cuda" and not str(device).startswith("cuda:"):
        failures.append("torch backend did not run on cuda")
    if manifest_region_count < min_manifest_regions:
        failures.append("real capture manifest region count is below the required floor")
    if loaded_frame_count <= 0:
        failures.append("no real capture frames were loaded")
    if scene_element_count <= 0:
        failures.append("no real capture scene elements were rendered")
    if packed_batch_count <= 0:
        failures.append("no packed render batches were produced")
    if packed_target_count < min_packed_targets:
        failures.append("packed target count is below the required floor")
    if max_batch_target_count > max_allowed_batch_targets:
        failures.append("packed batch exceeded the configured target limit")
    if not finite_losses:
        failures.append("torch render losses were not finite")
    if not math.isfinite(float(render_seconds)) or float(render_seconds) <= 0.0:
        failures.append("torch render timing was invalid")

    return {
        "format": FORMAT,
        "passed": not failures,
        "device": device,
        "manifestFrameCount": int(manifest_frame_count),
        "manifestRegionCount": int(manifest_region_count),
        "loadedFrameCount": int(loaded_frame_count),
        "sceneElementCount": int(scene_element_count),
        "packedBatchCount": int(packed_batch_count),
        "packedTargetCount": int(packed_target_count),
        "maxBatchTargetCount": int(max_batch_target_count),
        "maxAllowedBatchTargets": int(max_allowed_batch_targets),
        "minManifestRegions": int(min_manifest_regions),
        "minPackedTargets": int(min_packed_targets),
        "finiteLosses": bool(finite_losses),
        "renderSeconds": float(render_seconds),
        "failures": failures,
    }


def _bounded_manifest(manifest: Any, *, max_frames: int, max_regions: int) -> Any:
    from aura.ingest.capture import CaptureManifest

    selected_frames = tuple(manifest.frames[:max_frames])
    frame_ids = {frame.id for frame in selected_frames}
    selected_regions = tuple(region for region in manifest.regions if region.frame_id in frame_ids)[:max_regions]
    return CaptureManifest(root=manifest.root, frames=selected_frames, regions=selected_regions)


def _finite_summary_losses(summary: Any) -> bool:
    values = (
        *summary.image_loss,
        *summary.depth_loss,
        *summary.normal_loss,
        *summary.query_loss,
    )
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def run_validation(
    *,
    manifest_path: Path,
    device: str,
    max_frames: int,
    max_regions: int,
    pixel_stride: int,
    max_targets_per_frame: int,
    tile_size: int,
    max_targets_per_batch: int,
    min_manifest_regions: int,
    min_packed_targets: int,
) -> dict[str, Any]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if max_regions <= 0:
        raise ValueError("max_regions must be positive")

    import torch
    from aura.benchmark import _scene_from_capture_manifest_dataset
    from aura.ingest.capture import load_capture_asset_tensors, load_capture_manifest
    from aura.torch_renderer import torch_capture_training_batch_from_packed, torch_render_capture_training_summary
    from aura.training_targets import capture_tensors_to_packed_render_batches, plan_capture_tensor_sampling

    if not str(device).startswith("cuda"):
        return summarize_torch_backend_gate(
            device=device,
            manifest_frame_count=0,
            manifest_region_count=0,
            loaded_frame_count=0,
            scene_element_count=0,
            packed_batch_count=0,
            packed_target_count=0,
            max_batch_target_count=0,
            finite_losses=False,
            render_seconds=0.0,
            max_allowed_batch_targets=max_targets_per_batch,
            min_manifest_regions=min_manifest_regions,
            min_packed_targets=min_packed_targets,
        )
    if not bool(torch.cuda.is_available()):
        payload = summarize_torch_backend_gate(
            device=device,
            manifest_frame_count=0,
            manifest_region_count=0,
            loaded_frame_count=0,
            scene_element_count=0,
            packed_batch_count=0,
            packed_target_count=0,
            max_batch_target_count=0,
            finite_losses=False,
            render_seconds=0.0,
            max_allowed_batch_targets=max_targets_per_batch,
            min_manifest_regions=min_manifest_regions,
            min_packed_targets=min_packed_targets,
        )
        payload["failures"].append("torch.cuda.is_available() was false")
        return payload

    manifest = load_capture_manifest(manifest_path, validate=False)
    bounded = _bounded_manifest(manifest, max_frames=max_frames, max_regions=max_regions)
    tensors = load_capture_asset_tensors(bounded, max_loaded_bytes=512 * 1024 * 1024, max_frame_bytes=128 * 1024 * 1024)
    sampling_plan = plan_capture_tensor_sampling(
        bounded.frames,
        tensors,
        pixel_stride=pixel_stride,
        max_targets_per_frame=max_targets_per_frame,
        tile_size=tile_size,
        max_targets_per_batch=max_targets_per_batch,
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        bounded.frames,
        tensors,
        pixel_stride=pixel_stride,
        max_targets_per_frame=max_targets_per_frame,
        tile_size=tile_size,
        max_targets_per_batch=max_targets_per_batch,
        sampling_plan=sampling_plan,
    )
    scene = _scene_from_capture_manifest_dataset(bounded, name="torch_backend_validation")
    start = perf_counter()
    summaries = tuple(
        torch_render_capture_training_summary(
            scene,
            torch_capture_training_batch_from_packed(batch, device=device),
        )
        for batch in packed_batches
        if batch.target_count > 0
    )
    torch.cuda.synchronize()
    render_seconds = perf_counter() - start
    finite_losses = all(_finite_summary_losses(summary) for summary in summaries)
    packed_target_count = sum(batch.target_count for batch in packed_batches)
    max_batch_target_count = max((batch.target_count for batch in packed_batches), default=0)

    payload = summarize_torch_backend_gate(
        device=summaries[0].device if summaries else device,
        manifest_frame_count=len(manifest.frames),
        manifest_region_count=len(manifest.regions),
        loaded_frame_count=len(tensors),
        scene_element_count=len(scene.elements),
        packed_batch_count=len(packed_batches),
        packed_target_count=packed_target_count,
        max_batch_target_count=max_batch_target_count,
        finite_losses=finite_losses,
        render_seconds=render_seconds,
        max_allowed_batch_targets=max_targets_per_batch,
        min_manifest_regions=min_manifest_regions,
        min_packed_targets=min_packed_targets,
    )
    payload.update(
        {
            "manifest": str(manifest_path),
            "maxFrames": max_frames,
            "maxRegions": max_regions,
            "samplingPlan": sampling_plan.to_dict(),
            "cudaDeviceName": torch.cuda.get_device_name(0),
            "claimBoundary": (
                "Validates real-capture asset loading, bounded packed targets, and torch CUDA "
                "render-summary execution; it is not a full-resolution training throughput benchmark."
            ),
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/truck-pts129k-manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/torch_backend_validation_2026-06-24.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument("--max-regions", type=int, default=2048)
    parser.add_argument("--pixel-stride", type=int, default=64)
    parser.add_argument("--max-targets-per-frame", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--max-targets-per-batch", type=int, default=256)
    parser.add_argument("--min-manifest-regions", type=int, default=100000)
    parser.add_argument("--min-packed-targets", type=int, default=32)
    args = parser.parse_args()

    payload = run_validation(
        manifest_path=args.manifest,
        device=args.device,
        max_frames=args.max_frames,
        max_regions=args.max_regions,
        pixel_stride=args.pixel_stride,
        max_targets_per_frame=args.max_targets_per_frame,
        tile_size=args.tile_size,
        max_targets_per_batch=args.max_targets_per_batch,
        min_manifest_regions=args.min_manifest_regions,
        min_packed_targets=args.min_packed_targets,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
