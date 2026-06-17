from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from math import pi
from typing import Any, Sequence

from aura.ingest.capture import CaptureFrameTensors, CaptureTensor
from aura.optimize import RenderTarget
from aura.scene import AuraScene


@dataclass(frozen=True)
class TorchRendererStatus:
    available: bool
    cuda_available: bool
    default_device: str | None
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "cudaAvailable": self.cuda_available,
            "defaultDevice": self.default_device,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TorchRenderBatch:
    device: str
    frame_ids: tuple[str, ...]
    element_ids: tuple[str | None, ...]
    carrier_ids: tuple[str | None, ...]
    predicted_color: tuple[tuple[float, float, float], ...]
    predicted_depth: tuple[float | None, ...]
    transmittance: tuple[float, ...]
    confidence: tuple[float, ...]
    residual: tuple[bool, ...]
    semantic_ids: tuple[str | None, ...]
    target_color: tuple[tuple[float, float, float], ...]
    target_depth: tuple[float, ...]
    image_loss: tuple[float, ...]
    depth_loss: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "frameIds": list(self.frame_ids),
            "elementIds": list(self.element_ids),
            "carrierIds": list(self.carrier_ids),
            "predictedColor": [list(color) for color in self.predicted_color],
            "predictedDepth": list(self.predicted_depth),
            "transmittance": list(self.transmittance),
            "confidence": list(self.confidence),
            "residual": list(self.residual),
            "semanticIds": list(self.semantic_ids),
            "targetColor": [list(color) for color in self.target_color],
            "targetDepth": list(self.target_depth),
            "imageLoss": list(self.image_loss),
            "depthLoss": list(self.depth_loss),
        }


@dataclass(frozen=True)
class TorchCaptureAssetBatch:
    device: str
    frame_ids: tuple[str, ...]
    image: Any
    depth: Any | None
    depth_present: Any | None
    mask: Any | None
    mask_present: Any | None
    normal: Any | None
    normal_present: Any | None

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "frameIds": list(self.frame_ids),
            "image": _torch_tensor_metadata(self.image),
            "depth": _torch_tensor_metadata(self.depth),
            "depthPresent": _torch_tensor_metadata(self.depth_present),
            "mask": _torch_tensor_metadata(self.mask),
            "maskPresent": _torch_tensor_metadata(self.mask_present),
            "normal": _torch_tensor_metadata(self.normal),
            "normalPresent": _torch_tensor_metadata(self.normal_present),
        }


def torch_renderer_status() -> TorchRendererStatus:
    if find_spec("torch") is None:
        return TorchRendererStatus(
            available=False,
            cuda_available=False,
            default_device=None,
            reason="Install aura-core[gpu] or torch to enable the optional PyTorch renderer.",
        )
    torch = _import_torch()
    cuda_available = bool(torch.cuda.is_available())
    return TorchRendererStatus(
        available=True,
        cuda_available=cuda_available,
        default_device="cuda" if cuda_available else "cpu",
    )


def require_torch() -> Any:
    status = torch_renderer_status()
    if not status.available:
        raise RuntimeError(status.reason or "PyTorch renderer is unavailable")
    return _import_torch()


def torch_capture_asset_batch(
    frames: Sequence[CaptureFrameTensors],
    *,
    device: str | None = None,
) -> TorchCaptureAssetBatch:
    """Move capture-manifest image/depth/mask/normal tensors into torch batches."""

    if not frames:
        raise ValueError("torch capture asset batching requires at least one frame")
    torch = require_torch()
    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"
    image = _stack_required_capture_tensors(torch, tuple(frame.image for frame in frames), device=resolved_device, name="image")
    depth, depth_present = _stack_optional_capture_tensors(
        torch,
        tuple(frame.depth for frame in frames),
        device=resolved_device,
        name="depth",
    )
    mask, mask_present = _stack_optional_capture_tensors(
        torch,
        tuple(frame.mask for frame in frames),
        device=resolved_device,
        name="mask",
    )
    normal, normal_present = _stack_optional_capture_tensors(
        torch,
        tuple(frame.normal for frame in frames),
        device=resolved_device,
        name="normal",
    )
    return TorchCaptureAssetBatch(
        device=str(resolved_device),
        frame_ids=tuple(frame.frame_id for frame in frames),
        image=image,
        depth=depth,
        depth_present=depth_present,
        mask=mask,
        mask_present=mask_present,
        normal=normal,
        normal_present=normal_present,
    )


def torch_render_targets(
    scene: AuraScene,
    targets: Sequence[RenderTarget],
    *,
    device: str | None = None,
) -> TorchRenderBatch:
    """Vectorized PyTorch reference renderer over native AURA bounds.

    This prototype keeps the same `AuraScene` and `RenderTarget` contracts as
    the CPU differentiable renderer. It performs batched first-hit AABB queries
    and loss computation over native carriers; it is not a 3DGS render path.
    """

    if not targets:
        raise ValueError("torch renderer requires at least one target")
    if not scene.elements:
        raise ValueError("torch renderer requires at least one scene element")

    torch = require_torch()
    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"

    mins = torch.tensor([element.bounds.min_corner for element in scene.elements], dtype=torch.float32, device=resolved_device)
    maxs = torch.tensor([element.bounds.max_corner for element in scene.elements], dtype=torch.float32, device=resolved_device)
    colors = torch.tensor([element.color for element in scene.elements], dtype=torch.float32, device=resolved_device)
    opacities = torch.tensor([element.opacity for element in scene.elements], dtype=torch.float32, device=resolved_device)
    confidences = torch.tensor([element.confidence for element in scene.elements], dtype=torch.float32, device=resolved_device)

    origins = torch.tensor([target.ray.origin for target in targets], dtype=torch.float32, device=resolved_device)
    directions = torch.tensor([target.ray.direction for target in targets], dtype=torch.float32, device=resolved_device)
    target_colors = torch.tensor([target.target_color for target in targets], dtype=torch.float32, device=resolved_device)
    target_depths = torch.tensor([target.target_depth for target in targets], dtype=torch.float32, device=resolved_device)

    safe_directions = torch.where(directions.abs() < 1e-8, torch.full_like(directions, 1e-8), directions)
    t0 = (mins[None, :, :] - origins[:, None, :]) / safe_directions[:, None, :]
    t1 = (maxs[None, :, :] - origins[:, None, :]) / safe_directions[:, None, :]
    lower = torch.minimum(t0, t1)
    upper = torch.maximum(t0, t1)
    parallel_outside = (directions.abs()[:, None, :] < 1e-8) & (
        (origins[:, None, :] < mins[None, :, :]) | (origins[:, None, :] > maxs[None, :, :])
    )
    entry = torch.clamp(torch.max(lower, dim=2).values, min=0.0)
    exit_depth = torch.min(upper, dim=2).values
    hits = (exit_depth >= entry) & (~torch.any(parallel_outside, dim=2))
    hit_depths = torch.where(hits, entry, torch.full_like(entry, float("inf")))
    best_depth, best_index = torch.min(hit_depths, dim=1)
    has_hit = torch.isfinite(best_depth)

    hit_points = origins + directions * best_depth.unsqueeze(1)
    carrier_colors, transmittance, confidence, residual_flags = _carrier_response_tensors(
        torch,
        tuple(scene.elements),
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        resolved_device,
    )
    gathered_colors = carrier_colors * (1.0 - transmittance).unsqueeze(1)
    predicted_colors = torch.where(has_hit.unsqueeze(1), gathered_colors, torch.zeros_like(gathered_colors))
    predicted_depths = torch.where(has_hit, best_depth, torch.zeros_like(best_depth))
    transmittance = torch.where(has_hit, transmittance, torch.ones_like(transmittance))
    confidence = torch.where(has_hit, confidence, torch.zeros_like(confidence))
    residual_flags = torch.where(has_hit, residual_flags, torch.zeros_like(residual_flags))
    image_loss = torch.mean((predicted_colors - target_colors) ** 2, dim=1)
    depth_loss = torch.where(has_hit, torch.abs(predicted_depths - target_depths), target_depths)

    best_indices = best_index.detach().cpu().tolist()
    hit_flags = has_hit.detach().cpu().tolist()
    elements = tuple(scene.elements)
    return TorchRenderBatch(
        device=str(resolved_device),
        frame_ids=tuple(target.frame_id for target in targets),
        element_ids=tuple(elements[index].id if hit else None for index, hit in zip(best_indices, hit_flags)),
        carrier_ids=tuple(elements[index].carrier_id if hit else None for index, hit in zip(best_indices, hit_flags)),
        predicted_color=_tensor_vec3_tuple(predicted_colors.detach().cpu().tolist()),
        predicted_depth=tuple(float(value) if hit else None for value, hit in zip(predicted_depths.detach().cpu().tolist(), hit_flags)),
        transmittance=tuple(float(value) for value in transmittance.detach().cpu().tolist()),
        confidence=tuple(float(value) for value in confidence.detach().cpu().tolist()),
        residual=tuple(bool(value) for value in residual_flags.detach().cpu().tolist()),
        semantic_ids=tuple(_semantic_id_for(elements[index]) if hit else None for index, hit in zip(best_indices, hit_flags)),
        target_color=tuple(tuple(float(channel) for channel in target.target_color) for target in targets),  # type: ignore[return-value]
        target_depth=tuple(float(target.target_depth) for target in targets),
        image_loss=tuple(float(value) for value in image_loss.detach().cpu().tolist()),
        depth_loss=tuple(float(value) for value in depth_loss.detach().cpu().tolist()),
    )


def _import_torch() -> Any:
    import torch

    return torch


def _stack_required_capture_tensors(torch: Any, tensors: Sequence[CaptureTensor], *, device: str, name: str) -> Any:
    if any(tensor is None for tensor in tensors):
        raise ValueError(f"{name} tensors are required for every frame")
    shape = _shared_capture_tensor_shape(tensors, name=name)
    values = [list(tensor.values) for tensor in tensors]
    return torch.tensor(values, dtype=torch.float32, device=device).reshape((len(tensors), *shape))


def _stack_optional_capture_tensors(
    torch: Any,
    tensors: Sequence[CaptureTensor | None],
    *,
    device: str,
    name: str,
) -> tuple[Any | None, Any | None]:
    if not any(tensor is not None for tensor in tensors):
        return None, None
    present_items = tuple(tensor for tensor in tensors if tensor is not None)
    shape = _shared_capture_tensor_shape(present_items, name=name)
    values = []
    present = []
    zero_values = [0.0] * (shape[0] * shape[1] * shape[2])
    for tensor in tensors:
        if tensor is None:
            values.append(zero_values)
            present.append(False)
            continue
        if tensor.shape != shape:
            raise ValueError(f"{name} tensor shapes must match within a batch")
        values.append(list(tensor.values))
        present.append(True)
    batch = torch.tensor(values, dtype=torch.float32, device=device).reshape((len(tensors), *shape))
    present_tensor = torch.tensor(present, dtype=torch.bool, device=device)
    return batch, present_tensor


def _shared_capture_tensor_shape(tensors: Sequence[CaptureTensor], *, name: str) -> tuple[int, int, int]:
    if not tensors:
        raise ValueError(f"{name} tensor batch is empty")
    shape = tensors[0].shape
    mismatched = [tensor.shape for tensor in tensors if tensor.shape != shape]
    if mismatched:
        raise ValueError(f"{name} tensor shapes must match within a batch")
    return shape


def _torch_tensor_metadata(tensor: Any | None) -> dict | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _carrier_response_tensors(
    torch: Any,
    elements: Sequence,
    best_index: Any,
    best_depth: Any,
    exit_depth: Any,
    hit_points: Any,
    colors: Any,
    opacities: Any,
    confidences: Any,
    mins: Any,
    maxs: Any,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    carrier_colors = colors[best_index]
    transmittance = 1.0 - opacities[best_index]
    confidence = confidences[best_index]
    residual = torch.tensor([elements[index].residual for index in best_index.detach().cpu().tolist()], dtype=torch.bool, device=device)

    for element_index, element in enumerate(elements):
        mask = best_index == element_index
        if not bool(torch.any(mask)):
            continue
        payload_type = element.payload.get("type")
        if payload_type == "volume_cell":
            density = float(element.payload.get("density", element.opacity))
            path_length = torch.clamp(exit_depth[mask, element_index] - best_depth[mask], min=0.0)
            transmittance[mask] = torch.clamp(torch.exp(-density * path_length), min=0.0, max=1.0)
        elif payload_type == "beta_kernel":
            weight = _torch_beta_weight(torch, hit_points[mask], mins[element_index], maxs[element_index], element.payload)
            transmittance[mask] = torch.clamp(1.0 - opacities[element_index] * weight, min=0.0, max=1.0)
        elif payload_type == "gabor_frequency":
            frequency = torch.tensor(element.payload.get("frequency", (0.0, 0.0, 0.0)), dtype=torch.float32, device=device)
            phase = float(element.payload.get("phase", 0.0))
            bandwidth = max(0.0, min(1.0, float(element.payload.get("bandwidth", 1.0))))
            wave = 0.5 + 0.5 * torch.sin(2.0 * pi * torch.sum(hit_points[mask] * frequency, dim=1) + phase)
            modulation = 1.0 - bandwidth + bandwidth * wave
            carrier_colors[mask] = torch.clamp(carrier_colors[mask] * modulation.unsqueeze(1), min=0.0, max=1.0)
            confidence[mask] = torch.clamp(confidence[mask] * bandwidth, min=0.0, max=1.0)
        elif payload_type == "neural_residual":
            residual_scale = float(element.payload.get("residual_scale", 0.0))
            confidence[mask] = torch.clamp(confidence[mask] * (1.0 - residual_scale * 0.25), min=0.0, max=1.0)
            residual[mask] = True
        elif payload_type == "semantic_feature":
            confidence[mask] = torch.clamp(float(element.payload.get("confidence", element.confidence)), min=0.0, max=1.0)

    return carrier_colors, transmittance, confidence, residual


def _torch_beta_weight(torch: Any, points: Any, mins: Any, maxs: Any, payload: dict) -> Any:
    extent = torch.clamp(maxs - mins, min=1e-6)
    coordinates = torch.clamp((points - mins) / extent, min=0.0, max=1.0)
    u = torch.mean(coordinates, dim=1)
    alpha = max(1e-6, float(payload.get("alpha", 1.0)))
    beta = max(1e-6, float(payload.get("beta", 1.0)))
    raw = (u ** (alpha - 1.0)) * ((1.0 - u) ** (beta - 1.0))
    if alpha > 1.0 and beta > 1.0:
        mode = (alpha - 1.0) / (alpha + beta - 2.0)
        peak = (mode ** (alpha - 1.0)) * ((1.0 - mode) ** (beta - 1.0))
        if peak > 0.0:
            raw = raw / peak
    return torch.clamp(raw, min=0.0, max=1.0)


def _semantic_id_for(element: Any) -> str | None:
    if element.semantic_id is not None:
        return element.semantic_id
    if element.payload.get("type") == "semantic_feature":
        return str(element.payload.get("label", "")) or None
    return None


def _tensor_vec3_tuple(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(channel) for channel in row) for row in values)  # type: ignore[return-value]
