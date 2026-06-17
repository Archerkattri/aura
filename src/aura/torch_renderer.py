from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Sequence

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
            "targetColor": [list(color) for color in self.target_color],
            "targetDepth": list(self.target_depth),
            "imageLoss": list(self.image_loss),
            "depthLoss": list(self.depth_loss),
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

    gathered_colors = colors[best_index] * opacities[best_index].unsqueeze(1)
    predicted_colors = torch.where(has_hit.unsqueeze(1), gathered_colors, torch.zeros_like(gathered_colors))
    predicted_depths = torch.where(has_hit, best_depth, torch.zeros_like(best_depth))
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
        target_color=tuple(tuple(float(channel) for channel in target.target_color) for target in targets),  # type: ignore[return-value]
        target_depth=tuple(float(target.target_depth) for target in targets),
        image_loss=tuple(float(value) for value in image_loss.detach().cpu().tolist()),
        depth_loss=tuple(float(value) for value in depth_loss.detach().cpu().tolist()),
    )


def _import_torch() -> Any:
    import torch

    return torch


def _tensor_vec3_tuple(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(channel) for channel in row) for row in values)  # type: ignore[return-value]
