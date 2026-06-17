from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Sequence

from aura.ingest.capture import CaptureFrameTensors, CaptureTensor
from aura.optimize import RenderTarget
from aura.core import TrainingFrame
from aura.scene import AuraScene
from aura.torch_kernels import torch_carrier_parameter_tensors, torch_carrier_response_tensors


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
    opacity: tuple[float, ...]
    confidence: tuple[float, ...]
    normal: tuple[tuple[float, float, float] | None, ...]
    material_ids: tuple[str | None, ...]
    residual: tuple[bool, ...]
    semantic_ids: tuple[str | None, ...]
    provenance: tuple[str | None, ...]
    target_color: tuple[tuple[float, float, float], ...]
    target_depth: tuple[float, ...]
    target_normal: tuple[tuple[float, float, float] | None, ...]
    target_semantic_ids: tuple[str | None, ...]
    target_material_ids: tuple[str | None, ...]
    image_loss: tuple[float, ...]
    depth_loss: tuple[float, ...]
    normal_loss: tuple[float, ...]
    query_loss: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "frameIds": list(self.frame_ids),
            "elementIds": list(self.element_ids),
            "carrierIds": list(self.carrier_ids),
            "predictedColor": [list(color) for color in self.predicted_color],
            "predictedDepth": list(self.predicted_depth),
            "transmittance": list(self.transmittance),
            "opacity": list(self.opacity),
            "confidence": list(self.confidence),
            "normal": [list(normal) if normal is not None else None for normal in self.normal],
            "materialIds": list(self.material_ids),
            "residual": list(self.residual),
            "semanticIds": list(self.semantic_ids),
            "provenance": list(self.provenance),
            "targetColor": [list(color) for color in self.target_color],
            "targetDepth": list(self.target_depth),
            "targetNormal": [list(normal) if normal is not None else None for normal in self.target_normal],
            "targetSemanticIds": list(self.target_semantic_ids),
            "targetMaterialIds": list(self.target_material_ids),
            "imageLoss": list(self.image_loss),
            "depthLoss": list(self.depth_loss),
            "normalLoss": list(self.normal_loss),
            "queryLoss": list(self.query_loss),
        }


@dataclass(frozen=True)
class TorchRenderObjective:
    device: str
    frame_ids: tuple[str, ...]
    carrier_parameters: dict[str, dict[str, Any]]
    total_loss: Any
    image_loss: Any
    depth_loss: Any
    normal_loss: Any
    mask_loss: Any

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "frameIds": list(self.frame_ids),
            "totalLoss": float(self.total_loss.detach().cpu().item()),
            "imageLoss": float(self.image_loss.detach().cpu().item()),
            "depthLoss": float(self.depth_loss.detach().cpu().item()),
            "normalLoss": float(self.normal_loss.detach().cpu().item()),
            "maskLoss": float(self.mask_loss.detach().cpu().item()),
            "carrierParameterIds": sorted(self.carrier_parameters),
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


@dataclass(frozen=True)
class TorchCaptureTrainingBatch:
    device: str
    frame_ids: tuple[str, ...]
    frame_indices: Any
    pixel_xy: Any
    ray_origins: Any
    ray_directions: Any
    target_color: Any
    target_depth: Any
    target_mask: Any | None
    target_normal: Any | None
    target_normal_present: Any | None

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "frameIds": list(self.frame_ids),
            "frameIndices": _torch_tensor_metadata(self.frame_indices),
            "pixelXY": _torch_tensor_metadata(self.pixel_xy),
            "rayOrigins": _torch_tensor_metadata(self.ray_origins),
            "rayDirections": _torch_tensor_metadata(self.ray_directions),
            "targetColor": _torch_tensor_metadata(self.target_color),
            "targetDepth": _torch_tensor_metadata(self.target_depth),
            "targetMask": _torch_tensor_metadata(self.target_mask),
            "targetNormal": _torch_tensor_metadata(self.target_normal),
            "targetNormalPresent": _torch_tensor_metadata(self.target_normal_present),
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


def torch_capture_training_batch(
    frames: Sequence[TrainingFrame],
    assets: TorchCaptureAssetBatch,
    *,
    pixel_stride: int = 1,
    max_targets_per_frame: int | None = None,
) -> TorchCaptureTrainingBatch:
    """Create per-pixel torch training rays and targets from capture assets."""

    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    if not assets.frame_ids:
        raise ValueError("capture asset batch is empty")
    by_frame = {frame.id: frame for frame in frames}
    missing = [frame_id for frame_id in assets.frame_ids if frame_id not in by_frame]
    if missing:
        raise ValueError(f"capture asset batch references unknown training frames: {', '.join(missing)}")
    torch = require_torch()
    device = assets.image.device
    height = int(assets.image.shape[1])
    width = int(assets.image.shape[2])
    frame_indices: list[int] = []
    pixels: list[tuple[int, int]] = []
    origins: list[tuple[float, float, float]] = []
    directions: list[tuple[float, float, float]] = []
    for frame_index, frame_id in enumerate(assets.frame_ids):
        frame = by_frame[frame_id]
        produced = 0
        for y in range(0, height, pixel_stride):
            for x in range(0, width, pixel_stride):
                frame_indices.append(frame_index)
                pixels.append((x, y))
                origins.append(frame.camera_origin)
                directions.append(_pixel_ray_direction(frame, x, y))
                produced += 1
                if max_targets_per_frame is not None and produced >= max_targets_per_frame:
                    break
            if max_targets_per_frame is not None and produced >= max_targets_per_frame:
                break
    index_tensor = torch.tensor(frame_indices, dtype=torch.long, device=device)
    pixel_tensor = torch.tensor(pixels, dtype=torch.long, device=device)
    y_index = pixel_tensor[:, 1]
    x_index = pixel_tensor[:, 0]
    target_color = assets.image[index_tensor, y_index, x_index, :3]
    if assets.depth is not None:
        sampled_depth = assets.depth[index_tensor, y_index, x_index, 0]
        frame_depths = torch.tensor([by_frame[frame_id].target_depth for frame_id in assets.frame_ids], dtype=torch.float32, device=device)
        fallback_depth = frame_depths[index_tensor]
        target_depth = torch.where(sampled_depth > 0.0, sampled_depth, fallback_depth)
    else:
        frame_depths = torch.tensor([by_frame[frame_id].target_depth for frame_id in assets.frame_ids], dtype=torch.float32, device=device)
        target_depth = frame_depths[index_tensor]
    target_mask = assets.mask[index_tensor, y_index, x_index, 0] if assets.mask is not None else None
    target_normal = assets.normal[index_tensor, y_index, x_index, :3] if assets.normal is not None else None
    target_normal_present = assets.normal_present[index_tensor] if assets.normal_present is not None else None
    return TorchCaptureTrainingBatch(
        device=str(device),
        frame_ids=tuple(assets.frame_ids),
        frame_indices=index_tensor,
        pixel_xy=pixel_tensor,
        ray_origins=torch.tensor(origins, dtype=torch.float32, device=device),
        ray_directions=torch.tensor(directions, dtype=torch.float32, device=device),
        target_color=target_color,
        target_depth=target_depth,
        target_mask=target_mask,
        target_normal=target_normal,
        target_normal_present=target_normal_present,
    )


def torch_render_capture_training_batch(
    scene: AuraScene,
    batch: TorchCaptureTrainingBatch,
    *,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
) -> TorchRenderBatch:
    """Render sampled capture tensor targets through the torch AURA contract."""

    require_torch()
    if int(batch.frame_indices.numel()) == 0:
        raise ValueError("torch capture training batch requires at least one target")
    frame_indices = batch.frame_indices.detach().cpu().tolist()
    sample_frame_ids = tuple(batch.frame_ids[index] for index in frame_indices)
    return _torch_render_tensor_targets(
        scene,
        frame_ids=sample_frame_ids,
        origins=batch.ray_origins,
        directions=batch.ray_directions,
        target_colors=batch.target_color,
        target_depths=batch.target_depth,
        target_normals=batch.target_normal,
        target_normal_present=batch.target_normal_present,
        target_semantic_ids=(None,) * len(sample_frame_ids),
        target_material_ids=(None,) * len(sample_frame_ids),
        device=str(batch.ray_origins.device),
        carrier_parameters=carrier_parameters,
    )


def torch_render_capture_training_objective(
    scene: AuraScene,
    batch: TorchCaptureTrainingBatch,
    *,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
) -> TorchRenderObjective:
    """Return a live torch loss for sampled capture tensor targets."""

    require_torch()
    if int(batch.frame_indices.numel()) == 0:
        raise ValueError("torch capture training batch requires at least one target")
    frame_indices = batch.frame_indices.detach().cpu().tolist()
    sample_frame_ids = tuple(batch.frame_ids[index] for index in frame_indices)
    return _torch_render_objective_tensor_targets(
        scene,
        frame_ids=sample_frame_ids,
        origins=batch.ray_origins,
        directions=batch.ray_directions,
        target_colors=batch.target_color,
        target_depths=batch.target_depth,
        target_mask=batch.target_mask,
        target_normals=batch.target_normal,
        target_normal_present=batch.target_normal_present,
        device=str(batch.ray_origins.device),
        carrier_parameters=carrier_parameters,
    )


def torch_render_targets(
    scene: AuraScene,
    targets: Sequence[RenderTarget],
    *,
    device: str | None = None,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
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

    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"
    torch = require_torch()
    origins = torch.tensor([target.ray.origin for target in targets], dtype=torch.float32, device=resolved_device)
    directions = torch.tensor([target.ray.direction for target in targets], dtype=torch.float32, device=resolved_device)
    target_colors = torch.tensor([target.target_color for target in targets], dtype=torch.float32, device=resolved_device)
    target_depths = torch.tensor([target.target_depth for target in targets], dtype=torch.float32, device=resolved_device)
    target_normals = torch.tensor(
        [target.target_normal if target.target_normal is not None else (0.0, 0.0, 0.0) for target in targets],
        dtype=torch.float32,
        device=resolved_device,
    )
    target_normal_present = torch.tensor([target.target_normal is not None for target in targets], dtype=torch.bool, device=resolved_device)
    return _torch_render_tensor_targets(
        scene,
        frame_ids=tuple(target.frame_id for target in targets),
        origins=origins,
        directions=directions,
        target_colors=target_colors,
        target_depths=target_depths,
        target_normals=target_normals,
        target_normal_present=target_normal_present,
        target_semantic_ids=tuple(target.target_semantic_id for target in targets),
        target_material_ids=tuple(target.target_material_id for target in targets),
        device=str(resolved_device),
        carrier_parameters=carrier_parameters,
    )


def torch_render_target_objective(
    scene: AuraScene,
    targets: Sequence[RenderTarget],
    *,
    device: str | None = None,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
) -> TorchRenderObjective:
    """Return a live torch loss over native AURA carrier parameters."""

    if not targets:
        raise ValueError("torch renderer requires at least one target")
    if not scene.elements:
        raise ValueError("torch renderer requires at least one scene element")

    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"
    torch = require_torch()
    return _torch_render_objective_tensor_targets(
        scene,
        frame_ids=tuple(target.frame_id for target in targets),
        origins=torch.tensor([target.ray.origin for target in targets], dtype=torch.float32, device=resolved_device),
        directions=torch.tensor([target.ray.direction for target in targets], dtype=torch.float32, device=resolved_device),
        target_colors=torch.tensor([target.target_color for target in targets], dtype=torch.float32, device=resolved_device),
        target_depths=torch.tensor([target.target_depth for target in targets], dtype=torch.float32, device=resolved_device),
        target_normals=torch.tensor(
            [target.target_normal if target.target_normal is not None else (0.0, 0.0, 0.0) for target in targets],
            dtype=torch.float32,
            device=resolved_device,
        ),
        target_normal_present=torch.tensor([target.target_normal is not None for target in targets], dtype=torch.bool, device=resolved_device),
        device=str(resolved_device),
        carrier_parameters=carrier_parameters,
    )


def _torch_render_tensor_targets(
    scene: AuraScene,
    *,
    frame_ids: Sequence[str],
    origins: Any,
    directions: Any,
    target_colors: Any,
    target_depths: Any,
    target_normals: Any | None,
    target_normal_present: Any | None,
    target_semantic_ids: Sequence[str | None],
    target_material_ids: Sequence[str | None],
    device: str,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
) -> TorchRenderBatch:
    torch = require_torch()
    if not scene.elements:
        raise ValueError("torch renderer requires at least one scene element")
    if len(frame_ids) == 0:
        raise ValueError("torch renderer requires at least one target")
    if int(origins.shape[0]) != len(frame_ids):
        raise ValueError("torch tensor target count must match frame ids")
    if len(target_semantic_ids) != len(frame_ids) or len(target_material_ids) != len(frame_ids):
        raise ValueError("torch query target counts must match frame ids")
    if target_normals is not None and int(target_normals.shape[0]) != len(frame_ids):
        raise ValueError("torch target normal count must match frame ids")
    if target_normal_present is not None and int(target_normal_present.shape[0]) != len(frame_ids):
        raise ValueError("torch target normal presence count must match frame ids")
    if target_normals is None:
        target_normals = torch.zeros((len(frame_ids), 3), dtype=torch.float32, device=device)
    if target_normal_present is None:
        target_normal_present = torch.zeros((len(frame_ids),), dtype=torch.bool, device=device)

    mins = torch.tensor([element.bounds.min_corner for element in scene.elements], dtype=torch.float32, device=device)
    maxs = torch.tensor([element.bounds.max_corner for element in scene.elements], dtype=torch.float32, device=device)
    colors = torch.tensor([element.color for element in scene.elements], dtype=torch.float32, device=device)
    opacities = torch.tensor([element.opacity for element in scene.elements], dtype=torch.float32, device=device)
    confidences = torch.tensor([element.confidence for element in scene.elements], dtype=torch.float32, device=device)
    carrier_parameters = carrier_parameters or torch_carrier_parameter_tensors(torch, tuple(scene.elements), device=device)

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

    safe_best_depth = torch.where(has_hit, best_depth, torch.zeros_like(best_depth))
    hit_points = origins + directions * safe_best_depth.unsqueeze(1)
    carrier_colors, transmittance, confidence, residual_flags = torch_carrier_response_tensors(
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
        device,
        carrier_parameters=carrier_parameters,
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
    normals = tuple(_normal_for(elements[index]) if hit else None for index, hit in zip(best_indices, hit_flags))
    predicted_normals, predicted_normal_present = _predicted_normal_tensors(torch, normals, device=device)
    normal_loss = _torch_normal_loss(torch, predicted_normals, predicted_normal_present, target_normals, target_normal_present)
    semantic_ids = tuple(_semantic_id_for(elements[index]) if hit else None for index, hit in zip(best_indices, hit_flags))
    material_ids = tuple(elements[index].material_id if hit else None for index, hit in zip(best_indices, hit_flags))
    query_loss = tuple(
        _query_contract_loss(predicted_semantic, target_semantic, predicted_material, target_material)
        for predicted_semantic, target_semantic, predicted_material, target_material in zip(
            semantic_ids,
            target_semantic_ids,
            material_ids,
            target_material_ids,
        )
    )
    return TorchRenderBatch(
        device=device,
        frame_ids=tuple(frame_ids),
        element_ids=tuple(elements[index].id if hit else None for index, hit in zip(best_indices, hit_flags)),
        carrier_ids=tuple(elements[index].carrier_id if hit else None for index, hit in zip(best_indices, hit_flags)),
        predicted_color=_tensor_vec3_tuple(predicted_colors.detach().cpu().tolist()),
        predicted_depth=tuple(float(value) if hit else None for value, hit in zip(predicted_depths.detach().cpu().tolist(), hit_flags)),
        transmittance=tuple(float(value) for value in transmittance.detach().cpu().tolist()),
        opacity=tuple(1.0 - float(value) for value in transmittance.detach().cpu().tolist()),
        confidence=tuple(float(value) for value in confidence.detach().cpu().tolist()),
        normal=normals,
        material_ids=material_ids,
        residual=tuple(bool(value) for value in residual_flags.detach().cpu().tolist()),
        semantic_ids=semantic_ids,
        provenance=tuple(elements[index].id if hit else "miss" for index, hit in zip(best_indices, hit_flags)),
        target_color=_tensor_vec3_tuple(target_colors.detach().cpu().tolist()),
        target_depth=tuple(float(value) for value in target_depths.detach().cpu().tolist()),
        target_normal=_optional_target_normal_tuple(target_normals, target_normal_present),
        target_semantic_ids=tuple(target_semantic_ids),
        target_material_ids=tuple(target_material_ids),
        image_loss=tuple(float(value) for value in image_loss.detach().cpu().tolist()),
        depth_loss=tuple(float(value) for value in depth_loss.detach().cpu().tolist()),
        normal_loss=tuple(float(value) for value in normal_loss.detach().cpu().tolist()),
        query_loss=query_loss,
    )


def _torch_render_objective_tensor_targets(
    scene: AuraScene,
    *,
    frame_ids: Sequence[str],
    origins: Any,
    directions: Any,
    target_colors: Any,
    target_depths: Any,
    device: str,
    target_mask: Any | None = None,
    target_normals: Any | None = None,
    target_normal_present: Any | None = None,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
) -> TorchRenderObjective:
    torch = require_torch()
    if int(origins.shape[0]) != len(frame_ids):
        raise ValueError("torch tensor target count must match frame ids")

    mins = torch.tensor([element.bounds.min_corner for element in scene.elements], dtype=torch.float32, device=device)
    maxs = torch.tensor([element.bounds.max_corner for element in scene.elements], dtype=torch.float32, device=device)
    colors = torch.tensor([element.color for element in scene.elements], dtype=torch.float32, device=device)
    opacities = torch.tensor([element.opacity for element in scene.elements], dtype=torch.float32, device=device)
    confidences = torch.tensor([element.confidence for element in scene.elements], dtype=torch.float32, device=device)
    carrier_parameters = carrier_parameters or torch_carrier_parameter_tensors(torch, tuple(scene.elements), device=device)

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

    safe_best_depth = torch.where(has_hit, best_depth, torch.zeros_like(best_depth))
    hit_points = origins + directions * safe_best_depth.unsqueeze(1)
    carrier_colors, transmittance, _confidence, _residual_flags = torch_carrier_response_tensors(
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
        device,
        carrier_parameters=carrier_parameters,
    )
    predicted_colors = torch.where(
        has_hit.unsqueeze(1),
        carrier_colors * (1.0 - transmittance).unsqueeze(1),
        torch.zeros_like(carrier_colors),
    )
    predicted_depths = torch.where(has_hit, best_depth, torch.zeros_like(best_depth))
    predicted_opacity = torch.where(has_hit, 1.0 - transmittance, torch.zeros_like(transmittance))
    image_loss = torch.mean((predicted_colors - target_colors) ** 2)
    depth_loss = torch.mean(torch.where(has_hit, torch.abs(predicted_depths - target_depths), target_depths))
    mask_loss = _torch_mask_loss(torch, predicted_opacity, target_mask)
    elements = tuple(scene.elements)
    best_indices = best_index.detach().cpu().tolist()
    hit_flags = has_hit.detach().cpu().tolist()
    normals = tuple(_normal_for(elements[index]) if hit else None for index, hit in zip(best_indices, hit_flags))
    predicted_normals, predicted_normal_present = _predicted_normal_tensors(torch, normals, device=device)
    normal_loss = torch.mean(_torch_normal_loss(torch, predicted_normals, predicted_normal_present, target_normals, target_normal_present))
    return TorchRenderObjective(
        device=device,
        frame_ids=tuple(frame_ids),
        carrier_parameters=carrier_parameters,
        total_loss=image_loss + depth_loss + normal_loss + mask_loss,
        image_loss=image_loss,
        depth_loss=depth_loss,
        normal_loss=normal_loss,
        mask_loss=mask_loss,
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


def _pixel_ray_direction(frame: TrainingFrame, x: int, y: int) -> tuple[float, float, float]:
    forward = _normalize(tuple(frame.look_at[index] - frame.camera_origin[index] for index in range(3)))
    if frame.intrinsics is None:
        return forward
    fx = max(frame.intrinsics["fx"], 1e-6)
    fy = max(frame.intrinsics["fy"], 1e-6)
    cx = frame.intrinsics["cx"]
    cy = frame.intrinsics["cy"]
    right_raw = _cross(forward, (0.0, 1.0, 0.0))
    if _norm(right_raw) <= 1e-12:
        right_raw = _cross(forward, (1.0, 0.0, 0.0))
    right = _normalize(right_raw)
    up = _normalize(_cross(right, forward))
    px = ((x + 0.5) - cx) / fx
    py = ((y + 0.5) - cy) / fy
    return _normalize(
        (
            forward[0] + right[0] * px - up[0] * py,
            forward[1] + right[1] * px - up[1] * py,
            forward[2] + right[2] * px - up[2] * py,
        )
    )


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _norm(vector)
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _norm(vector: tuple[float, float, float]) -> float:
    return sum(axis * axis for axis in vector) ** 0.5


def _semantic_id_for(element: Any) -> str | None:
    if element.semantic_id is not None:
        return element.semantic_id
    if element.payload.get("type") == "semantic_feature":
        return str(element.payload.get("label", "")) or None
    return None


def _normal_for(element: Any) -> tuple[float, float, float] | None:
    if element.normal is not None:
        return tuple(float(channel) for channel in element.normal)  # type: ignore[return-value]
    payload_normal = element.payload.get("normal")
    if isinstance(payload_normal, list | tuple) and len(payload_normal) == 3:
        return tuple(float(channel) for channel in payload_normal)  # type: ignore[return-value]
    return None


def _predicted_normal_tensors(torch: Any, normals: Sequence[tuple[float, float, float] | None], *, device: str) -> tuple[Any, Any]:
    values = [normal if normal is not None else (0.0, 0.0, 0.0) for normal in normals]
    present = [normal is not None for normal in normals]
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(present, dtype=torch.bool, device=device),
    )


def _torch_normal_loss(
    torch: Any,
    predicted_normals: Any,
    predicted_normal_present: Any,
    target_normals: Any | None,
    target_normal_present: Any | None,
) -> Any:
    sample_count = int(predicted_normals.shape[0])
    if target_normals is None or target_normal_present is None:
        return torch.zeros(sample_count, dtype=torch.float32, device=predicted_normals.device)
    predicted_norm = torch.linalg.norm(predicted_normals, dim=1)
    target_norm = torch.linalg.norm(target_normals, dim=1)
    valid = target_normal_present & predicted_normal_present & (predicted_norm > 1e-8) & (target_norm > 1e-8)
    cosine = torch.sum(predicted_normals * target_normals, dim=1) / torch.clamp(predicted_norm * target_norm, min=1e-8)
    cosine_loss = torch.clamp((1.0 - cosine) * 0.5, min=0.0, max=1.0)
    missing_loss = torch.ones_like(cosine_loss)
    supervised_loss = torch.where(valid, cosine_loss, missing_loss)
    return torch.where(target_normal_present, supervised_loss, torch.zeros_like(supervised_loss))


def _torch_mask_loss(torch: Any, predicted_opacity: Any, target_mask: Any | None) -> Any:
    if target_mask is None:
        return torch.zeros((), dtype=torch.float32, device=predicted_opacity.device)
    return torch.mean((predicted_opacity - torch.clamp(target_mask, min=0.0, max=1.0)) ** 2)


def _optional_target_normal_tuple(target_normals: Any | None, target_normal_present: Any | None) -> tuple[tuple[float, float, float] | None, ...]:
    if target_normals is None or target_normal_present is None:
        return ()
    values = target_normals.detach().cpu().tolist()
    present = target_normal_present.detach().cpu().tolist()
    return tuple(tuple(float(channel) for channel in value) if is_present else None for value, is_present in zip(values, present))  # type: ignore[return-value]


def _query_contract_loss(
    predicted_semantic_id: str | None,
    target_semantic_id: str | None,
    predicted_material_id: str | None,
    target_material_id: str | None,
) -> float:
    misses = 0
    total = 0
    if target_semantic_id is not None:
        total += 1
        misses += predicted_semantic_id != target_semantic_id
    if target_material_id is not None:
        total += 1
        misses += predicted_material_id != target_material_id
    return 0.0 if total == 0 else misses / total


def _tensor_vec3_tuple(values: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(channel) for channel in row) for row in values)  # type: ignore[return-value]
