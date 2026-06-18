from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Sequence

from aura.ingest.capture import CaptureFrameTensors, CaptureTensor
from aura.optimize import RenderTarget
from aura.core import TrainingFrame
from aura.scene import AuraScene
from aura.training_targets import CapturePackedRenderBatch
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
    ray_origins: tuple[tuple[float, float, float], ...]
    ray_directions: tuple[tuple[float, float, float], ...]
    element_ids: tuple[str | None, ...]
    carrier_ids: tuple[str | None, ...]
    ordered_hits: tuple[tuple[dict[str, object], ...], ...]
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
    target_confidence: tuple[float | None, ...]
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
            "rayOrigins": [list(origin) for origin in self.ray_origins],
            "rayDirections": [list(direction) for direction in self.ray_directions],
            "elementIds": list(self.element_ids),
            "carrierIds": list(self.carrier_ids),
            "orderedHits": [[dict(hit) for hit in ray_hits] for ray_hits in self.ordered_hits],
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
            "targetConfidence": list(self.target_confidence),
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
    query_loss: Any
    normal_loss: Any
    mask_loss: Any
    confidence_loss: Any

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "frameIds": list(self.frame_ids),
            "totalLoss": float(self.total_loss.detach().cpu().item()),
            "imageLoss": float(self.image_loss.detach().cpu().item()),
            "depthLoss": float(self.depth_loss.detach().cpu().item()),
            "queryLoss": float(self.query_loss.detach().cpu().item()),
            "normalLoss": float(self.normal_loss.detach().cpu().item()),
            "maskLoss": float(self.mask_loss.detach().cpu().item()),
            "confidenceLoss": float(self.confidence_loss.detach().cpu().item()),
            "carrierParameterIds": sorted(self.carrier_parameters),
        }


@dataclass(frozen=True)
class TorchSceneTensors:
    device: str
    element_ids: tuple[str, ...]
    carrier_ids: tuple[str, ...]
    carrier_group_indices: dict[str, Any]
    chunk_ids: tuple[str, ...]
    mins: Any
    maxs: Any
    chunk_mins: Any | None
    chunk_maxs: Any | None
    element_chunk_indices: Any | None
    colors: Any
    opacities: Any
    confidences: Any
    surface_plane_points: Any
    surface_normals: Any
    gabor_plane_points: Any
    gabor_normals: Any
    element_normals: Any
    element_normal_present: Any
    gaussian_means: Any
    gaussian_inverse_covariances: Any
    gaussian_support_radius_sq: Any
    beta_support_radii: Any
    carrier_parameters: dict[str, dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "elementIds": list(self.element_ids),
            "carrierIds": list(self.carrier_ids),
            "carrierGroupIndices": {
                carrier_id: _torch_index_tensor_values(indices) for carrier_id, indices in self.carrier_group_indices.items()
            },
            "chunkIds": list(self.chunk_ids),
            "mins": _torch_tensor_metadata(self.mins),
            "maxs": _torch_tensor_metadata(self.maxs),
            "chunkMins": _torch_tensor_metadata(self.chunk_mins),
            "chunkMaxs": _torch_tensor_metadata(self.chunk_maxs),
            "elementChunkIndices": _torch_tensor_metadata(self.element_chunk_indices),
            "supportsChunkCulling": self.element_chunk_indices is not None and self.chunk_mins is not None and self.chunk_maxs is not None,
            "colors": _torch_tensor_metadata(self.colors),
            "opacities": _torch_tensor_metadata(self.opacities),
            "confidences": _torch_tensor_metadata(self.confidences),
            "surfacePlanePoints": _torch_tensor_metadata(self.surface_plane_points),
            "surfaceNormals": _torch_tensor_metadata(self.surface_normals),
            "gaborPlanePoints": _torch_tensor_metadata(self.gabor_plane_points),
            "gaborNormals": _torch_tensor_metadata(self.gabor_normals),
            "elementNormals": _torch_tensor_metadata(self.element_normals),
            "elementNormalPresent": _torch_tensor_metadata(self.element_normal_present),
            "gaussianMeans": _torch_tensor_metadata(self.gaussian_means),
            "gaussianInverseCovariances": _torch_tensor_metadata(self.gaussian_inverse_covariances),
            "gaussianSupportRadiusSq": _torch_tensor_metadata(self.gaussian_support_radius_sq),
            "betaSupportRadii": _torch_tensor_metadata(self.beta_support_radii),
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
    target_confidence: Any | None = None
    target_confidence_present: Any | None = None
    target_semantic_ids: tuple[str | None, ...] = ()
    target_material_ids: tuple[str | None, ...] = ()

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
            "targetConfidence": _torch_tensor_metadata(self.target_confidence),
            "targetConfidencePresent": _torch_tensor_metadata(self.target_confidence_present),
            "targetSemanticIds": list(self.target_semantic_ids),
            "targetMaterialIds": list(self.target_material_ids),
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


def torch_scene_tensors(
    scene: AuraScene,
    *,
    device: str | None = None,
    requires_grad: bool = True,
) -> TorchSceneTensors:
    """Materialize reusable native AURA scene tensors on the selected torch device."""

    if not scene.elements:
        raise ValueError("torch scene tensor cache requires at least one scene element")
    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"
    torch = require_torch()
    elements = tuple(scene.elements)
    chunk_ids = tuple(chunk.id for chunk in scene.chunks)
    chunk_index_by_id = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    element_chunk_indices = tuple(chunk_index_by_id.get(element.chunk_id, -1) for element in elements)
    supports_chunk_culling = bool(chunk_ids) and all(index >= 0 for index in element_chunk_indices)
    return TorchSceneTensors(
        device=str(resolved_device),
        element_ids=tuple(element.id for element in elements),
        carrier_ids=tuple(element.carrier_id for element in elements),
        carrier_group_indices=_torch_carrier_group_indices(torch, elements, device=resolved_device),
        chunk_ids=chunk_ids,
        mins=torch.tensor([element.bounds.min_corner for element in elements], dtype=torch.float32, device=resolved_device),
        maxs=torch.tensor([element.bounds.max_corner for element in elements], dtype=torch.float32, device=resolved_device),
        chunk_mins=torch.tensor([chunk.bounds.min_corner for chunk in scene.chunks], dtype=torch.float32, device=resolved_device)
        if supports_chunk_culling
        else None,
        chunk_maxs=torch.tensor([chunk.bounds.max_corner for chunk in scene.chunks], dtype=torch.float32, device=resolved_device)
        if supports_chunk_culling
        else None,
        element_chunk_indices=torch.tensor(element_chunk_indices, dtype=torch.long, device=resolved_device) if supports_chunk_culling else None,
        colors=torch.tensor([element.color for element in elements], dtype=torch.float32, device=resolved_device),
        opacities=torch.tensor([element.opacity for element in elements], dtype=torch.float32, device=resolved_device),
        confidences=torch.tensor([element.confidence for element in elements], dtype=torch.float32, device=resolved_device),
        surface_plane_points=torch.tensor([_surface_plane_point_or_nan(element) for element in elements], dtype=torch.float32, device=resolved_device),
        surface_normals=torch.tensor([_surface_normal_or_nan(element) for element in elements], dtype=torch.float32, device=resolved_device),
        gabor_plane_points=torch.tensor(
            [_gabor_plane_point_or_nan(element) for element in elements],
            dtype=torch.float32,
            device=resolved_device,
        ),
        gabor_normals=torch.tensor(
            [_gabor_normal_or_nan(element) for element in elements],
            dtype=torch.float32,
            device=resolved_device,
        ),
        element_normals=torch.tensor([_normal_for(element) or _default_trainable_normal_for(element) for element in elements], dtype=torch.float32, device=resolved_device),
        element_normal_present=torch.tensor([_normal_present_for(element) for element in elements], dtype=torch.bool, device=resolved_device),
        gaussian_means=torch.tensor([_gaussian_mean_or_nan(element) for element in elements], dtype=torch.float32, device=resolved_device),
        gaussian_inverse_covariances=torch.tensor(
            [_gaussian_inverse_covariance_or_nan(element) for element in elements],
            dtype=torch.float32,
            device=resolved_device,
        ),
        gaussian_support_radius_sq=torch.tensor(
            [_gaussian_support_radius_sq(element) for element in elements],
            dtype=torch.float32,
            device=resolved_device,
        ),
        beta_support_radii=torch.tensor(
            [_beta_support_radius_or_nan(element) for element in elements],
            dtype=torch.float32,
            device=resolved_device,
        ),
        carrier_parameters=torch_carrier_parameter_tensors(torch, elements, device=str(resolved_device), requires_grad=requires_grad),
    )


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
    include_masked_targets: bool = False,
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
    frame_index_tensors = []
    pixel_tensors = []
    origin_tensors = []
    direction_tensors = []
    target_semantic_ids: list[str | None] = []
    for frame_index, frame_id in enumerate(assets.frame_ids):
        frame = by_frame[frame_id]
        sampled_pixels = _sampled_training_pixels_for_frame(
            torch,
            assets,
            frame_index=frame_index,
            height=height,
            width=width,
            pixel_stride=pixel_stride,
            max_targets_per_frame=max_targets_per_frame,
            include_masked_targets=include_masked_targets,
        )
        sample_count = int(sampled_pixels.shape[0])
        if sample_count == 0:
            continue
        frame_index_tensors.append(torch.full((sample_count,), frame_index, dtype=torch.long, device=device))
        pixel_tensors.append(sampled_pixels)
        origin_tensors.append(
            torch.as_tensor(frame.camera_origin, dtype=torch.float32, device=device).reshape(1, 3).expand(sample_count, 3)
        )
        direction_tensors.append(_pixel_ray_directions_tensor(torch, frame, sampled_pixels, device=device))
        target_semantic_ids.extend([frame.semantic_label] * sample_count)
    if not frame_index_tensors:
        raise ValueError("torch capture training batch produced no sampled pixels")
    index_tensor = torch.cat(tuple(frame_index_tensors), dim=0)
    pixel_tensor = torch.cat(tuple(pixel_tensors), dim=0)
    ray_origins = torch.cat(tuple(origin_tensors), dim=0)
    ray_directions = torch.cat(tuple(direction_tensors), dim=0)
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
    if target_mask is not None and assets.mask_present is not None:
        target_mask = torch.where(assets.mask_present[index_tensor], target_mask, torch.ones_like(target_mask))
    target_normal = assets.normal[index_tensor, y_index, x_index, :3] if assets.normal is not None else None
    target_normal_present = assets.normal_present[index_tensor] if assets.normal_present is not None else None
    target_count = int(index_tensor.shape[0])
    target_confidence = torch.clamp(target_mask, min=0.0, max=1.0) if target_mask is not None else torch.ones((target_count,), dtype=torch.float32, device=device)
    target_confidence_present = torch.ones((target_count,), dtype=torch.bool, device=device)
    return TorchCaptureTrainingBatch(
        device=str(device),
        frame_ids=tuple(assets.frame_ids),
        frame_indices=index_tensor,
        pixel_xy=pixel_tensor,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        target_color=target_color,
        target_depth=target_depth,
        target_mask=target_mask,
        target_normal=target_normal,
        target_normal_present=target_normal_present,
        target_confidence=target_confidence,
        target_confidence_present=target_confidence_present,
        target_semantic_ids=tuple(target_semantic_ids),
        target_material_ids=(None,) * target_count,
    )


def _sampled_training_pixels_for_frame(
    torch: Any,
    assets: TorchCaptureAssetBatch,
    *,
    frame_index: int,
    height: int,
    width: int,
    pixel_stride: int,
    max_targets_per_frame: int | None,
    include_masked_targets: bool,
) -> Any:
    device = assets.image.device
    y_values = torch.arange(0, height, pixel_stride, dtype=torch.long, device=device)
    x_values = torch.arange(0, width, pixel_stride, dtype=torch.long, device=device)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    pixel_xy = torch.stack((x_grid.reshape(-1), y_grid.reshape(-1)), dim=1)
    if assets.mask is not None and not include_masked_targets:
        mask_values = assets.mask[frame_index, pixel_xy[:, 1], pixel_xy[:, 0], 0]
        if assets.mask_present is not None:
            mask_values = torch.where(assets.mask_present[frame_index], mask_values, torch.ones_like(mask_values))
        pixel_xy = pixel_xy[mask_values > 0.0]
    if max_targets_per_frame is not None:
        pixel_xy = pixel_xy[:max_targets_per_frame]
    return pixel_xy


def _pixel_ray_directions_tensor(torch: Any, frame: TrainingFrame, pixel_xy: Any, *, device: str) -> Any:
    forward = _normalize(tuple(frame.look_at[index] - frame.camera_origin[index] for index in range(3)))
    if frame.intrinsics is None:
        return torch.as_tensor(forward, dtype=torch.float32, device=device).reshape(1, 3).expand(int(pixel_xy.shape[0]), 3)
    right_raw = _cross(forward, (0.0, 1.0, 0.0))
    if _norm(right_raw) <= 1e-12:
        right_raw = _cross(forward, (1.0, 0.0, 0.0))
    right = _normalize(right_raw)
    up = _normalize(_cross(right, forward))
    fx = float(frame.intrinsics["fx"])
    fy = float(frame.intrinsics["fy"])
    cx = float(frame.intrinsics["cx"])
    cy = float(frame.intrinsics["cy"])
    pixels = pixel_xy.to(dtype=torch.float32)
    px = ((pixels[:, 0] + 0.5) - cx) / fx
    py = ((pixels[:, 1] + 0.5) - cy) / fy
    forward_tensor = torch.as_tensor(forward, dtype=torch.float32, device=device).reshape(1, 3)
    right_tensor = torch.as_tensor(right, dtype=torch.float32, device=device).reshape(1, 3)
    up_tensor = torch.as_tensor(up, dtype=torch.float32, device=device).reshape(1, 3)
    directions = forward_tensor + right_tensor * px.unsqueeze(1) - up_tensor * py.unsqueeze(1)
    return directions / torch.clamp(torch.linalg.norm(directions, dim=1, keepdim=True), min=1e-8)


def torch_capture_training_batch_from_packed(
    batch: CapturePackedRenderBatch,
    *,
    device: str | None = None,
) -> TorchCaptureTrainingBatch:
    """Move one packed capture render-target descriptor into torch tensors."""

    if batch.target_count <= 0:
        raise ValueError("packed torch capture training batch requires at least one target")
    torch = require_torch()
    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"
    target_count = batch.target_count
    frame_indices = torch.as_tensor(batch.frame_indices, dtype=torch.long, device=resolved_device).reshape(target_count)
    pixel_xy = torch.as_tensor(batch.pixel_xy, dtype=torch.long, device=resolved_device).reshape(target_count, 2)
    target_mask = (
        torch.as_tensor(batch.target_mask, dtype=torch.float32, device=resolved_device).reshape(target_count)
        if batch.target_mask is not None
        else None
    )
    target_normal = (
        torch.as_tensor(batch.target_normal, dtype=torch.float32, device=resolved_device).reshape(target_count, 3)
        if batch.target_normal is not None
        else None
    )
    target_normal_present = (
        torch.as_tensor(batch.target_normal_present, dtype=torch.bool, device=resolved_device).reshape(target_count)
        if batch.target_normal_present is not None
        else None
    )
    target_confidence = torch.clamp(target_mask, min=0.0, max=1.0) if target_mask is not None else torch.ones((target_count,), dtype=torch.float32, device=resolved_device)
    target_confidence_present = torch.ones((target_count,), dtype=torch.bool, device=resolved_device)
    return TorchCaptureTrainingBatch(
        device=str(resolved_device),
        frame_ids=batch.frame_ids,
        frame_indices=frame_indices,
        pixel_xy=pixel_xy,
        ray_origins=torch.as_tensor(batch.ray_origins, dtype=torch.float32, device=resolved_device).reshape(target_count, 3),
        ray_directions=torch.as_tensor(batch.ray_directions, dtype=torch.float32, device=resolved_device).reshape(target_count, 3),
        target_color=torch.as_tensor(batch.target_color, dtype=torch.float32, device=resolved_device).reshape(target_count, 3),
        target_depth=torch.as_tensor(batch.target_depth, dtype=torch.float32, device=resolved_device).reshape(target_count),
        target_mask=target_mask,
        target_normal=target_normal,
        target_normal_present=target_normal_present,
        target_confidence=target_confidence,
        target_confidence_present=target_confidence_present,
        target_semantic_ids=tuple(batch.frame_semantic_ids[int(index)] for index in frame_indices.detach().cpu().tolist()),
        target_material_ids=(None,) * target_count,
    )


def torch_render_capture_training_batch(
    scene: AuraScene,
    batch: TorchCaptureTrainingBatch,
    *,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
    scene_tensors: TorchSceneTensors | None = None,
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
        target_confidence=batch.target_confidence,
        target_confidence_present=batch.target_confidence_present,
        target_semantic_ids=batch.target_semantic_ids,
        target_material_ids=batch.target_material_ids,
        device=str(batch.ray_origins.device),
        carrier_parameters=carrier_parameters,
        scene_tensors=scene_tensors,
    )


def torch_render_capture_training_objective(
    scene: AuraScene,
    batch: TorchCaptureTrainingBatch,
    *,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
    scene_tensors: TorchSceneTensors | None = None,
) -> TorchRenderObjective:
    """Return a live torch loss for sampled capture tensor targets."""

    require_torch()
    if int(batch.frame_indices.numel()) == 0:
        raise ValueError("torch capture training batch requires at least one target")
    sample_count = int(batch.frame_indices.numel())
    return _torch_render_objective_tensor_targets(
        scene,
        frame_ids=("<capture>",) * sample_count,
        origins=batch.ray_origins,
        directions=batch.ray_directions,
        target_colors=batch.target_color,
        target_depths=batch.target_depth,
        target_mask=batch.target_mask,
        target_normals=batch.target_normal,
        target_normal_present=batch.target_normal_present,
        target_confidence=batch.target_confidence,
        target_confidence_present=batch.target_confidence_present,
        target_semantic_ids=batch.target_semantic_ids,
        target_material_ids=batch.target_material_ids,
        device=str(batch.ray_origins.device),
        carrier_parameters=carrier_parameters,
        scene_tensors=scene_tensors,
    )


def torch_render_targets(
    scene: AuraScene,
    targets: Sequence[RenderTarget],
    *,
    device: str | None = None,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
    scene_tensors: TorchSceneTensors | None = None,
) -> TorchRenderBatch:
    """Vectorized PyTorch renderer over native AURA bounds.

    This keeps the same `AuraScene` and `RenderTarget` contracts as AURA
    training while performing batched ordered AABB queries, native carrier
    compositing, and loss computation; it is not a 3DGS render path.
    """

    if not targets:
        raise ValueError("torch renderer requires at least one target")
    if not scene.elements:
        raise ValueError("torch renderer requires at least one scene element")

    status = torch_renderer_status()
    resolved_device = device or (scene_tensors.device if scene_tensors is not None else None) or status.default_device or "cpu"
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
    target_confidence = torch.tensor(
        [target.target_confidence if target.target_confidence is not None else 0.0 for target in targets],
        dtype=torch.float32,
        device=resolved_device,
    )
    target_confidence_present = torch.tensor([target.target_confidence is not None for target in targets], dtype=torch.bool, device=resolved_device)
    return _torch_render_tensor_targets(
        scene,
        frame_ids=tuple(target.frame_id for target in targets),
        origins=origins,
        directions=directions,
        target_colors=target_colors,
        target_depths=target_depths,
        target_normals=target_normals,
        target_normal_present=target_normal_present,
        target_confidence=target_confidence,
        target_confidence_present=target_confidence_present,
        target_semantic_ids=tuple(target.target_semantic_id for target in targets),
        target_material_ids=tuple(target.target_material_id for target in targets),
        device=str(resolved_device),
        carrier_parameters=carrier_parameters,
        scene_tensors=scene_tensors,
    )


def torch_render_target_objective(
    scene: AuraScene,
    targets: Sequence[RenderTarget],
    *,
    device: str | None = None,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
    scene_tensors: TorchSceneTensors | None = None,
) -> TorchRenderObjective:
    """Return a live torch loss over native AURA carrier parameters."""

    if not targets:
        raise ValueError("torch renderer requires at least one target")
    if not scene.elements:
        raise ValueError("torch renderer requires at least one scene element")

    status = torch_renderer_status()
    resolved_device = device or (scene_tensors.device if scene_tensors is not None else None) or status.default_device or "cpu"
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
        target_confidence=torch.tensor(
            [target.target_confidence if target.target_confidence is not None else 0.0 for target in targets],
            dtype=torch.float32,
            device=resolved_device,
        ),
        target_confidence_present=torch.tensor([target.target_confidence is not None for target in targets], dtype=torch.bool, device=resolved_device),
        target_semantic_ids=tuple(target.target_semantic_id for target in targets),
        target_material_ids=tuple(target.target_material_id for target in targets),
        device=str(resolved_device),
        carrier_parameters=carrier_parameters,
        scene_tensors=scene_tensors,
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
    target_confidence: Any | None,
    target_confidence_present: Any | None,
    target_semantic_ids: Sequence[str | None],
    target_material_ids: Sequence[str | None],
    device: str,
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
    scene_tensors: TorchSceneTensors | None = None,
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
    if target_confidence is not None and int(target_confidence.shape[0]) != len(frame_ids):
        raise ValueError("torch target confidence count must match frame ids")
    if target_confidence_present is not None and int(target_confidence_present.shape[0]) != len(frame_ids):
        raise ValueError("torch target confidence presence count must match frame ids")
    if target_normals is None:
        target_normals = torch.zeros((len(frame_ids), 3), dtype=torch.float32, device=device)
    if target_normal_present is None:
        target_normal_present = torch.zeros((len(frame_ids),), dtype=torch.bool, device=device)
    if target_confidence is None:
        target_confidence = torch.zeros((len(frame_ids),), dtype=torch.float32, device=device)
    if target_confidence_present is None:
        target_confidence_present = torch.zeros((len(frame_ids),), dtype=torch.bool, device=device)

    scene_tensors = _resolve_scene_tensors(scene, scene_tensors=scene_tensors, device=device)
    colors = scene_tensors.colors
    opacities = scene_tensors.opacities
    confidences = scene_tensors.confidences
    carrier_parameters = carrier_parameters or scene_tensors.carrier_parameters
    (
        mins,
        maxs,
        surface_plane_points,
        gabor_plane_points,
        gaussian_means,
        gaussian_inverse_covariances,
        beta_support_radii,
        surface_normals,
        gabor_normals,
        element_normals,
    ) = _torch_geometry_from_carrier_parameters(
        torch,
        tuple(scene.elements),
        carrier_parameters,
        scene_tensors.mins,
        scene_tensors.maxs,
        scene_tensors.surface_plane_points,
        scene_tensors.gabor_plane_points,
        scene_tensors.gaussian_means,
        scene_tensors.gaussian_inverse_covariances,
        scene_tensors.beta_support_radii,
        scene_tensors.surface_normals,
        scene_tensors.gabor_normals,
        scene_tensors.element_normals,
    )

    composited = _torch_composite_carrier_hits(
        torch,
        tuple(scene.elements),
        origins,
        directions,
        mins,
        maxs,
        colors,
        opacities,
        confidences,
        scene_tensors.chunk_mins,
        scene_tensors.chunk_maxs,
        scene_tensors.element_chunk_indices,
        surface_plane_points,
        surface_normals,
        gabor_plane_points,
        gabor_normals,
        gaussian_means,
        gaussian_inverse_covariances,
        scene_tensors.gaussian_support_radius_sq,
        beta_support_radii,
        device=device,
        carrier_parameters=carrier_parameters,
        collect_traces=True,
    )
    has_hit = composited["has_hit"]
    first_index = composited["first_index"]
    first_depth = composited["first_depth"]
    predicted_colors = composited["color"]
    predicted_depths = torch.where(has_hit, first_depth, torch.zeros_like(first_depth))
    transmittance = composited["transmittance"]
    confidence = composited["confidence"]
    residual_flags = composited["residual"]
    image_loss = torch.mean((predicted_colors - target_colors) ** 2, dim=1)
    depth_loss = torch.where(has_hit, torch.abs(predicted_depths - target_depths), target_depths)

    best_indices = first_index.detach().cpu().tolist()
    hit_flags = has_hit.detach().cpu().tolist()
    elements = tuple(scene.elements)
    predicted_normals, predicted_normal_present = _predicted_normal_tensors_from_indices(
        torch,
        first_index,
        has_hit,
        element_normals,
        scene_tensors.element_normal_present,
    )
    output_normals = _optional_target_normal_tuple(predicted_normals, predicted_normal_present)
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
        ray_origins=_tensor_vec3_tuple(origins.detach().cpu().tolist()),
        ray_directions=_tensor_vec3_tuple(directions.detach().cpu().tolist()),
        element_ids=tuple(elements[index].id if hit else None for index, hit in zip(best_indices, hit_flags)),
        carrier_ids=tuple(elements[index].carrier_id if hit else None for index, hit in zip(best_indices, hit_flags)),
        ordered_hits=_torch_ordered_hit_traces(
            elements,
            composited["hit_indices"],
            composited["hit_depths"],
            composited["hit_transmittance"],
        ),
        predicted_color=_tensor_vec3_tuple(predicted_colors.detach().cpu().tolist()),
        predicted_depth=tuple(float(value) if hit else None for value, hit in zip(predicted_depths.detach().cpu().tolist(), hit_flags)),
        transmittance=tuple(float(value) for value in transmittance.detach().cpu().tolist()),
        opacity=tuple(1.0 - float(value) for value in transmittance.detach().cpu().tolist()),
        confidence=tuple(float(value) for value in confidence.detach().cpu().tolist()),
        normal=output_normals,
        material_ids=material_ids,
        residual=tuple(bool(value) for value in residual_flags.detach().cpu().tolist()),
        semantic_ids=semantic_ids,
        provenance=tuple(_torch_hit_provenance(elements, indices) for indices in composited["hit_indices"]),
        target_color=_tensor_vec3_tuple(target_colors.detach().cpu().tolist()),
        target_depth=tuple(float(value) for value in target_depths.detach().cpu().tolist()),
        target_normal=_optional_target_normal_tuple(target_normals, target_normal_present),
        target_confidence=_optional_target_confidence_tuple(target_confidence, target_confidence_present),
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
    target_confidence: Any | None = None,
    target_confidence_present: Any | None = None,
    target_semantic_ids: Sequence[str | None] = (),
    target_material_ids: Sequence[str | None] = (),
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
    scene_tensors: TorchSceneTensors | None = None,
) -> TorchRenderObjective:
    torch = require_torch()
    if int(origins.shape[0]) != len(frame_ids):
        raise ValueError("torch tensor target count must match frame ids")
    if target_semantic_ids and len(target_semantic_ids) != len(frame_ids):
        raise ValueError("torch target semantic id count must match frame ids")
    if target_material_ids and len(target_material_ids) != len(frame_ids):
        raise ValueError("torch target material id count must match frame ids")
    if target_confidence is not None and int(target_confidence.shape[0]) != len(frame_ids):
        raise ValueError("torch target confidence count must match frame ids")
    if target_confidence_present is not None and int(target_confidence_present.shape[0]) != len(frame_ids):
        raise ValueError("torch target confidence presence count must match frame ids")
    if target_confidence is None:
        target_confidence = torch.zeros((len(frame_ids),), dtype=torch.float32, device=device)
    if target_confidence_present is None:
        target_confidence_present = torch.zeros((len(frame_ids),), dtype=torch.bool, device=device)

    scene_tensors = _resolve_scene_tensors(scene, scene_tensors=scene_tensors, device=device)
    colors = scene_tensors.colors
    opacities = scene_tensors.opacities
    confidences = scene_tensors.confidences
    carrier_parameters = carrier_parameters or scene_tensors.carrier_parameters
    (
        mins,
        maxs,
        surface_plane_points,
        gabor_plane_points,
        gaussian_means,
        gaussian_inverse_covariances,
        beta_support_radii,
        surface_normals,
        gabor_normals,
        element_normals,
    ) = _torch_geometry_from_carrier_parameters(
        torch,
        tuple(scene.elements),
        carrier_parameters,
        scene_tensors.mins,
        scene_tensors.maxs,
        scene_tensors.surface_plane_points,
        scene_tensors.gabor_plane_points,
        scene_tensors.gaussian_means,
        scene_tensors.gaussian_inverse_covariances,
        scene_tensors.beta_support_radii,
        scene_tensors.surface_normals,
        scene_tensors.gabor_normals,
        scene_tensors.element_normals,
    )

    composited = _torch_composite_carrier_hits(
        torch,
        tuple(scene.elements),
        origins,
        directions,
        mins,
        maxs,
        colors,
        opacities,
        confidences,
        scene_tensors.chunk_mins,
        scene_tensors.chunk_maxs,
        scene_tensors.element_chunk_indices,
        surface_plane_points,
        surface_normals,
        gabor_plane_points,
        gabor_normals,
        gaussian_means,
        gaussian_inverse_covariances,
        scene_tensors.gaussian_support_radius_sq,
        beta_support_radii,
        device=device,
        carrier_parameters=carrier_parameters,
        collect_traces=False,
    )
    has_hit = composited["has_hit"]
    first_index = composited["first_index"]
    first_depth = composited["first_depth"]
    predicted_colors = composited["color"]
    predicted_depths = torch.where(has_hit, first_depth, torch.zeros_like(first_depth))
    predicted_opacity = 1.0 - composited["transmittance"]
    predicted_confidence = composited["confidence"]
    image_loss = torch.mean((predicted_colors - target_colors) ** 2)
    depth_loss = torch.mean(torch.where(has_hit, torch.abs(predicted_depths - target_depths), target_depths))
    mask_loss = _torch_mask_loss(torch, predicted_opacity, target_mask)
    confidence_loss = _torch_confidence_loss(torch, predicted_confidence, target_confidence, target_confidence_present)
    predicted_normals, predicted_normal_present = _predicted_normal_tensors_from_indices(
        torch,
        first_index,
        has_hit,
        element_normals,
        scene_tensors.element_normal_present,
    )
    normal_loss = torch.mean(_torch_normal_loss(torch, predicted_normals, predicted_normal_present, target_normals, target_normal_present))
    query_loss = _torch_query_contract_loss(
        torch,
        tuple(scene.elements),
        composited["element_weights"],
        target_semantic_ids=target_semantic_ids or (None,) * len(frame_ids),
        target_material_ids=target_material_ids or (None,) * len(frame_ids),
        device=device,
    )
    return TorchRenderObjective(
        device=device,
        frame_ids=tuple(frame_ids),
        carrier_parameters=carrier_parameters,
        total_loss=image_loss + depth_loss + query_loss + normal_loss + mask_loss + confidence_loss,
        image_loss=image_loss,
        depth_loss=depth_loss,
        query_loss=query_loss,
        normal_loss=normal_loss,
        mask_loss=mask_loss,
        confidence_loss=confidence_loss,
    )


def _resolve_scene_tensors(
    scene: AuraScene,
    *,
    scene_tensors: TorchSceneTensors | None,
    device: str,
) -> TorchSceneTensors:
    if scene_tensors is None:
        return torch_scene_tensors(scene, device=device)
    scene_element_ids = tuple(element.id for element in scene.elements)
    if scene_tensors.element_ids != scene_element_ids:
        raise ValueError("torch scene tensor cache does not match scene element ids")
    if not _torch_devices_match(scene_tensors.device, device):
        raise ValueError(f"torch scene tensor cache device {scene_tensors.device!r} does not match render device {device!r}")
    return scene_tensors


def _torch_devices_match(left: str, right: str) -> bool:
    if str(left) == str(right):
        return True
    torch = require_torch()
    left_device = torch.device(left)
    right_device = torch.device(right)
    if left_device.type != right_device.type:
        return False
    if left_device.type != "cuda":
        return left_device == right_device
    left_index = 0 if left_device.index is None else int(left_device.index)
    right_index = 0 if right_device.index is None else int(right_device.index)
    return left_index == right_index


def _torch_geometry_from_carrier_parameters(
    torch: Any,
    elements: Sequence[Any],
    carrier_parameters: dict[str, dict[str, Any]] | None,
    base_mins: Any,
    base_maxs: Any,
    base_surface_plane_points: Any,
    base_gabor_plane_points: Any,
    base_gaussian_means: Any,
    base_gaussian_inverse_covariances: Any,
    base_beta_support_radii: Any,
    base_surface_normals: Any,
    base_gabor_normals: Any,
    base_element_normals: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    if carrier_parameters is None:
        return (
            base_mins,
            base_maxs,
            base_surface_plane_points,
            base_gabor_plane_points,
            base_gaussian_means,
            base_gaussian_inverse_covariances,
            base_beta_support_radii,
            base_surface_normals,
            base_gabor_normals,
            base_element_normals,
        )
    mins = []
    maxs = []
    surface_plane_points = []
    gabor_plane_points = []
    gaussian_means = []
    gaussian_inverse_covariances = []
    beta_support_radii = []
    surface_normals = []
    gabor_normals = []
    element_normals = []
    for index, element in enumerate(elements):
        fields = carrier_parameters.get(element.id, {})
        mins.append(fields.get("min_corner", base_mins[index]))
        maxs.append(fields.get("max_corner", base_maxs[index]))
        normal = fields.get("normal")
        if element.payload.get("type") == "gabor_frequency" or element.carrier_id == "gabor":
            surface_plane_points.append(base_surface_plane_points[index])
            gabor_plane_points.append(fields.get("plane_point", base_gabor_plane_points[index]))
            surface_normals.append(base_surface_normals[index])
            gabor_normals.append(_torch_normalized_vector(torch, normal if normal is not None else base_gabor_normals[index]))
            element_normals.append(_torch_normalized_vector(torch, normal if normal is not None else base_element_normals[index]))
        else:
            surface_plane_points.append(fields.get("plane_point", base_surface_plane_points[index]))
            gabor_plane_points.append(base_gabor_plane_points[index])
            surface_normals.append(_torch_normalized_vector(torch, normal if normal is not None else base_surface_normals[index]))
            gabor_normals.append(base_gabor_normals[index])
            element_normals.append(_torch_normalized_vector(torch, normal if normal is not None else base_element_normals[index]))
        gaussian_means.append(fields.get("gaussian_mean", base_gaussian_means[index]))
        gaussian_inverse_covariances.append(
            _gaussian_inverse_covariance_from_fields(torch, fields, base_gaussian_inverse_covariances[index])
        )
        beta_support_radii.append(fields.get("support_radius", base_beta_support_radii[index]))
    return (
        torch.stack(tuple(mins), dim=0),
        torch.stack(tuple(maxs), dim=0),
        torch.stack(tuple(surface_plane_points), dim=0),
        torch.stack(tuple(gabor_plane_points), dim=0),
        torch.stack(tuple(gaussian_means), dim=0),
        torch.stack(tuple(gaussian_inverse_covariances), dim=0),
        torch.stack(tuple(beta_support_radii), dim=0),
        torch.stack(tuple(surface_normals), dim=0),
        torch.stack(tuple(gabor_normals), dim=0),
        torch.stack(tuple(element_normals), dim=0),
    )


def _torch_normalized_vector(torch: Any, value: Any) -> Any:
    if not torch.isfinite(value).all():
        return value
    norm = torch.linalg.norm(value)
    return torch.where(norm > 1e-8, value / torch.clamp(norm, min=1e-8), value)


def _gaussian_inverse_covariance_from_fields(torch: Any, fields: dict[str, Any], base_inverse_covariance: Any) -> Any:
    covariance_diag = fields.get("gaussian_covariance_diag")
    if covariance_diag is None:
        return base_inverse_covariance
    safe_diag = torch.clamp(covariance_diag, min=1e-6)
    return torch.diag(1.0 / safe_diag)


def _torch_composite_carrier_hits(
    torch: Any,
    elements: Sequence[Any],
    origins: Any,
    directions: Any,
    mins: Any,
    maxs: Any,
    colors: Any,
    opacities: Any,
    confidences: Any,
    chunk_mins: Any | None,
    chunk_maxs: Any | None,
    element_chunk_indices: Any | None,
    surface_plane_points: Any,
    surface_normals: Any,
    gabor_plane_points: Any,
    gabor_normals: Any,
    gaussian_means: Any,
    gaussian_inverse_covariances: Any,
    gaussian_support_radius_sq: Any,
    beta_support_radii: Any,
    *,
    device: str,
    carrier_parameters: dict[str, dict[str, Any]] | None,
    collect_traces: bool = True,
) -> dict[str, Any]:
    entry, exit_depth, hits = _torch_carrier_hits(
        torch,
        tuple(elements),
        origins,
        directions,
        mins,
        maxs,
        surface_plane_points,
        surface_normals,
        gabor_plane_points,
        gabor_normals,
        gaussian_means,
        gaussian_inverse_covariances,
        gaussian_support_radius_sq,
        beta_support_radii,
    )
    chunk_culling_active = chunk_mins is not None and chunk_maxs is not None and element_chunk_indices is not None
    if chunk_culling_active:
        _chunk_entry, _chunk_exit, chunk_hits = _torch_aabb_hits(torch, origins, directions, chunk_mins, chunk_maxs)
        hits = hits & chunk_hits[:, element_chunk_indices]
    hit_depths = torch.where(hits, entry, torch.full_like(entry, float("inf")))
    first_depth, first_index = torch.min(hit_depths, dim=1)
    has_hit = torch.isfinite(first_depth)
    sorted_depths, sorted_indices = torch.sort(hit_depths, dim=1)

    ray_count = int(origins.shape[0])
    color = torch.zeros((ray_count, 3), dtype=torch.float32, device=device)
    remaining = torch.ones((ray_count,), dtype=torch.float32, device=device)
    confidence_num = torch.zeros((ray_count,), dtype=torch.float32, device=device)
    confidence_den = torch.zeros((ray_count,), dtype=torch.float32, device=device)
    residual = torch.zeros((ray_count,), dtype=torch.bool, device=device)
    element_weights = torch.zeros((ray_count, len(elements)), dtype=torch.float32, device=device)
    ordered_transmittance: list[Any] = []

    for order in range(len(elements)):
        current_index = sorted_indices[:, order]
        current_depth = sorted_depths[:, order]
        active = torch.isfinite(current_depth)
        safe_depth = torch.where(active, current_depth, torch.zeros_like(current_depth))
        hit_points = _torch_carrier_sample_points(
            torch,
            current_index,
            safe_depth,
            exit_depth,
            origins,
            directions,
            gaussian_means,
            device=device,
        )
        carrier_colors, transmittance, confidence, residual_flags = torch_carrier_response_tensors(
            torch,
            elements,
            current_index,
            safe_depth,
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
        transmittance = torch.clamp(torch.where(active, transmittance, torch.ones_like(transmittance)), min=0.0, max=1.0)
        alpha = 1.0 - transmittance
        weight = torch.where(active, remaining * alpha, torch.zeros_like(remaining))
        color = color + weight.unsqueeze(1) * carrier_colors
        confidence_num = confidence_num + weight * confidence
        confidence_den = confidence_den + weight
        element_weights = element_weights.scatter_add(1, current_index.unsqueeze(1), weight.unsqueeze(1))
        remaining = torch.where(active, remaining * transmittance, remaining)
        residual = residual | (active & residual_flags)
        if collect_traces:
            ordered_transmittance.append(torch.where(active, transmittance, torch.ones_like(transmittance)))

    if collect_traces:
        hit_indices = []
        hit_depths = []
        for indices, depths in zip(sorted_indices.detach().cpu().tolist(), sorted_depths.detach().cpu().tolist()):
            hit_indices.append(tuple(int(index) for index, depth in zip(indices, depths) if depth != float("inf")))
            hit_depths.append(tuple(float(depth) for depth in depths if depth != float("inf")))
        hit_transmittance = _torch_hit_transmittance_traces(torch, sorted_depths, ordered_transmittance)
    else:
        hit_indices = tuple()
        hit_depths = tuple()
        hit_transmittance = tuple()

    confidence = torch.where(confidence_den > 0.0, confidence_num / torch.clamp(confidence_den, min=1e-8), torch.zeros_like(confidence_den))
    return {
        "color": color,
        "transmittance": torch.where(has_hit, remaining, torch.ones_like(remaining)),
        "confidence": torch.where(has_hit, confidence, torch.zeros_like(confidence)),
        "residual": torch.where(has_hit, residual, torch.zeros_like(residual)),
        "first_depth": first_depth,
        "first_index": first_index,
        "has_hit": has_hit,
        "element_weights": element_weights,
        "hit_indices": tuple(hit_indices),
        "hit_depths": tuple(hit_depths),
        "hit_transmittance": hit_transmittance,
        "chunk_culling": bool(chunk_culling_active),
    }


def _torch_aabb_hits(torch: Any, origins: Any, directions: Any, mins: Any, maxs: Any) -> tuple[Any, Any, Any]:
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
    return entry, exit_depth, hits


def _torch_carrier_hits(
    torch: Any,
    elements: Sequence[Any],
    origins: Any,
    directions: Any,
    mins: Any,
    maxs: Any,
    surface_plane_points: Any,
    surface_normals: Any,
    gabor_plane_points: Any,
    gabor_normals: Any,
    gaussian_means: Any,
    gaussian_inverse_covariances: Any,
    gaussian_support_radius_sq: Any,
    beta_support_radii: Any,
) -> tuple[Any, Any, Any]:
    entry, exit_depth, hits = _torch_aabb_hits(torch, origins, directions, mins, maxs)
    entry_columns = []
    exit_columns = []
    hit_columns = []
    for element_index, element in enumerate(elements):
        current_entry = entry[:, element_index]
        current_exit = exit_depth[:, element_index]
        current_hits = hits[:, element_index]
        payload_type = element.payload.get("type")
        if payload_type == "surface_cell" or element.carrier_id == "surface":
            surface_entry, surface_exit, surface_hits = _torch_surface_plane_hits(
                torch,
                origins,
                directions,
                mins[element_index],
                maxs[element_index],
                surface_plane_points[element_index],
                surface_normals[element_index],
            )
            valid = torch.isfinite(surface_entry) & torch.isfinite(surface_exit)
            current_hits = torch.where(valid, surface_hits, current_hits)
            current_entry = torch.where(valid, surface_entry, current_entry)
            current_exit = torch.where(valid, surface_exit, current_exit)
        elif payload_type == "gaussian_fallback":
            gaussian_entry, gaussian_exit, gaussian_hits = _torch_gaussian_ellipsoid_hits(
                torch,
                origins,
                directions,
                gaussian_means[element_index],
                gaussian_inverse_covariances[element_index],
                gaussian_support_radius_sq[element_index],
            )
            valid = (
                torch.isfinite(gaussian_means[element_index]).all()
                & torch.isfinite(gaussian_inverse_covariances[element_index]).all()
                & torch.isfinite(gaussian_support_radius_sq[element_index])
                & (gaussian_support_radius_sq[element_index] > 0.0)
            )
            bounded_entry = torch.maximum(entry[:, element_index], gaussian_entry)
            bounded_exit = torch.minimum(exit_depth[:, element_index], gaussian_exit)
            bounded_hits = hits[:, element_index] & gaussian_hits & (bounded_exit >= bounded_entry)
            current_hits = torch.where(valid, bounded_hits, current_hits)
            current_entry = torch.where(valid, bounded_entry, current_entry)
            current_exit = torch.where(valid, bounded_exit, current_exit)
        elif payload_type == "beta_kernel":
            center = (mins[element_index] + maxs[element_index]) * 0.5
            beta_entry, beta_exit, beta_hits = _torch_beta_ellipsoid_hits(
                torch,
                origins,
                directions,
                center,
                beta_support_radii[element_index],
            )
            valid = torch.isfinite(beta_support_radii[element_index]).all() & torch.all(beta_support_radii[element_index] > 0.0)
            bounded_entry = torch.maximum(entry[:, element_index], beta_entry)
            bounded_exit = torch.minimum(exit_depth[:, element_index], beta_exit)
            bounded_hits = hits[:, element_index] & beta_hits & (bounded_exit >= bounded_entry)
            current_hits = torch.where(valid, bounded_hits, current_hits)
            current_entry = torch.where(valid, bounded_entry, current_entry)
            current_exit = torch.where(valid, bounded_exit, current_exit)
        elif payload_type == "gabor_frequency":
            gabor_entry, gabor_exit, gabor_hits = _torch_surface_plane_hits(
                torch,
                origins,
                directions,
                mins[element_index],
                maxs[element_index],
                gabor_plane_points[element_index],
                gabor_normals[element_index],
            )
            valid = torch.isfinite(gabor_entry) & torch.isfinite(gabor_exit)
            current_hits = torch.where(valid, gabor_hits, current_hits)
            current_entry = torch.where(valid, gabor_entry, current_entry)
            current_exit = torch.where(valid, gabor_exit, current_exit)
        entry_columns.append(current_entry)
        exit_columns.append(current_exit)
        hit_columns.append(current_hits)
    return (
        torch.stack(tuple(entry_columns), dim=1),
        torch.stack(tuple(exit_columns), dim=1),
        torch.stack(tuple(hit_columns), dim=1),
    )


def _torch_surface_plane_hits(
    torch: Any,
    origins: Any,
    directions: Any,
    mins: Any,
    maxs: Any,
    plane_point: Any,
    normal: Any,
) -> tuple[Any, Any, Any]:
    valid_surface = torch.isfinite(plane_point).all() & torch.isfinite(normal).all()
    denom = torch.sum(directions * normal.unsqueeze(0), dim=1)
    numerator = torch.sum((plane_point.unsqueeze(0) - origins) * normal.unsqueeze(0), dim=1)
    parallel = torch.abs(denom) < 1e-8
    depth = numerator / torch.where(parallel, torch.ones_like(denom), denom)
    points = origins + directions * depth.unsqueeze(1)
    inside = torch.all((points >= mins.unsqueeze(0) - 1e-5) & (points <= maxs.unsqueeze(0) + 1e-5), dim=1)
    hits = valid_surface & (~parallel) & (depth >= 0.0) & inside
    safe_depth = torch.where(hits, depth, torch.full_like(depth, float("inf")))
    return safe_depth, safe_depth, hits


def _torch_gaussian_ellipsoid_hits(
    torch: Any,
    origins: Any,
    directions: Any,
    mean: Any,
    inverse_covariance: Any,
    support_radius_sq: Any,
) -> tuple[Any, Any, Any]:
    ray_count = int(origins.shape[0])
    invalid_entry = torch.full((ray_count,), float("inf"), dtype=origins.dtype, device=origins.device)
    invalid_hits = torch.zeros((ray_count,), dtype=torch.bool, device=origins.device)
    valid_gaussian = (
        torch.isfinite(mean).all()
        & torch.isfinite(inverse_covariance).all()
        & torch.isfinite(support_radius_sq)
        & (support_radius_sq > 0.0)
    )

    delta = origins - mean.unsqueeze(0)
    inv_directions = directions @ inverse_covariance
    inv_delta = delta @ inverse_covariance
    a = torch.sum(inv_directions * directions, dim=1)
    b = 2.0 * torch.sum(inv_delta * directions, dim=1)
    c = torch.sum(inv_delta * delta, dim=1) - support_radius_sq
    discriminant = b * b - 4.0 * a * c
    valid = (a > 1e-8) & (discriminant >= 0.0)
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
    near = (-b - sqrt_discriminant) / torch.clamp(2.0 * a, min=1e-8)
    far = (-b + sqrt_discriminant) / torch.clamp(2.0 * a, min=1e-8)
    entry = torch.where(near >= 0.0, near, torch.zeros_like(near))
    hits = valid_gaussian & valid & (far >= 0.0) & (entry >= 0.0)
    entry = torch.where(hits, entry, invalid_entry)
    exit_depth = torch.where(hits, torch.clamp(far, min=0.0), invalid_entry)
    return entry, exit_depth, hits


def _torch_beta_ellipsoid_hits(
    torch: Any,
    origins: Any,
    directions: Any,
    center: Any,
    support_radii: Any,
) -> tuple[Any, Any, Any]:
    ray_count = int(origins.shape[0])
    invalid_entry = torch.full((ray_count,), float("inf"), dtype=origins.dtype, device=origins.device)
    invalid_hits = torch.zeros((ray_count,), dtype=torch.bool, device=origins.device)
    valid_beta = torch.isfinite(center).all() & torch.isfinite(support_radii).all() & torch.all(support_radii > 0.0)

    scaled_origin = (origins - center.unsqueeze(0)) / support_radii.unsqueeze(0)
    scaled_direction = directions / support_radii.unsqueeze(0)
    a = torch.sum(scaled_direction * scaled_direction, dim=1)
    b = 2.0 * torch.sum(scaled_origin * scaled_direction, dim=1)
    c = torch.sum(scaled_origin * scaled_origin, dim=1) - 1.0
    discriminant = b * b - 4.0 * a * c
    valid = (a > 1e-8) & (discriminant >= 0.0)
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
    denom = torch.where(a.abs() > 1e-8, 2.0 * a, torch.ones_like(a))
    near = (-b - sqrt_discriminant) / denom
    far = (-b + sqrt_discriminant) / denom
    entry = torch.clamp(torch.minimum(near, far), min=0.0)
    exit_depth = torch.maximum(near, far)
    hits = valid_beta & valid & (exit_depth >= entry)
    return (
        torch.where(hits, entry, invalid_entry),
        torch.where(hits, exit_depth, invalid_entry),
        hits,
    )


def _torch_carrier_sample_points(
    torch: Any,
    current_index: Any,
    entry_depth: Any,
    exit_depth: Any,
    origins: Any,
    directions: Any,
    gaussian_means: Any,
    *,
    device: str,
) -> Any:
    points = origins + directions * entry_depth.unsqueeze(1)
    del device
    selected_means = gaussian_means[current_index]
    has_gaussian_mean = torch.isfinite(selected_means).all(dim=1)
    ray_to_mean = selected_means - origins
    direction_norm = torch.sum(directions * directions, dim=1)
    projected_depth = torch.sum(ray_to_mean * directions, dim=1) / torch.clamp(direction_norm, min=1e-8)
    selected_exit_depth = exit_depth.gather(1, current_index.unsqueeze(1)).squeeze(1)
    gaussian_depth = torch.maximum(entry_depth, torch.minimum(selected_exit_depth, projected_depth))
    gaussian_points = origins + directions * gaussian_depth.unsqueeze(1)
    return torch.where(has_gaussian_mean.unsqueeze(1), gaussian_points, points)


def _torch_hit_transmittance_traces(
    torch: Any,
    sorted_depths: Any,
    ordered_transmittance: Sequence[Any],
) -> tuple[tuple[float, ...], ...]:
    if not ordered_transmittance:
        return tuple(() for _index in range(int(sorted_depths.shape[0])))
    transmittance_by_order = torch.stack(tuple(ordered_transmittance), dim=1)
    active_by_order = torch.isfinite(sorted_depths)
    traces: list[tuple[float, ...]] = []
    for active, values in zip(active_by_order.detach().cpu().tolist(), transmittance_by_order.detach().cpu().tolist()):
        traces.append(tuple(float(value) for is_active, value in zip(active, values) if is_active))
    return tuple(tuple(ray_trace) for ray_trace in traces)


def _torch_ordered_hit_traces(
    elements: Sequence[Any],
    hit_indices: Sequence[Sequence[int]],
    hit_depths: Sequence[Sequence[float]],
    hit_transmittance: Sequence[Sequence[float]],
) -> tuple[tuple[dict[str, object], ...], ...]:
    traces = []
    for ray_indices, ray_depths, ray_transmittance in zip(hit_indices, hit_depths, hit_transmittance):
        ray_trace = []
        for index, depth, transmittance in zip(ray_indices, ray_depths, ray_transmittance):
            element = elements[index]
            ray_trace.append(
                {
                    "elementId": element.id,
                    "carrierId": element.carrier_id,
                    "depth": float(depth),
                    "transmittance": float(transmittance),
                    "opacity": 1.0 - float(transmittance),
                    "provenance": element.id,
                }
            )
        traces.append(tuple(ray_trace))
    return tuple(traces)


def _torch_hit_provenance(elements: Sequence[Any], indices: Sequence[int]) -> str:
    if not indices:
        return "miss"
    return ",".join(elements[index].id for index in indices)


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


def _torch_carrier_group_indices(torch: Any, elements: Sequence[Any], *, device: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, element in enumerate(elements):
        groups.setdefault(element.carrier_id, []).append(index)
    return {carrier_id: torch.tensor(indices, dtype=torch.long, device=device) for carrier_id, indices in sorted(groups.items())}


def _gaussian_mean_or_nan(element: Any) -> tuple[float, float, float]:
    if element.payload.get("type") != "gaussian_fallback":
        return (float("nan"), float("nan"), float("nan"))
    mean = element.payload.get("mean")
    if not isinstance(mean, (list, tuple)) or len(mean) != 3:
        return (float("nan"), float("nan"), float("nan"))
    return tuple(float(value) for value in mean)  # type: ignore[return-value]


def _surface_normal_or_nan(element: Any) -> tuple[float, float, float]:
    if element.payload.get("type") != "surface_cell" and element.carrier_id != "surface":
        return (float("nan"), float("nan"), float("nan"))
    normal = _normal_for(element)
    if normal is None:
        return (float("nan"), float("nan"), float("nan"))
    try:
        return _normalize(normal)
    except ValueError:
        return (float("nan"), float("nan"), float("nan"))


def _surface_plane_point_or_nan(element: Any) -> tuple[float, float, float]:
    normal = _surface_normal_or_nan(element)
    if any(value != value for value in normal):
        return (float("nan"), float("nan"), float("nan"))
    point = element.payload.get("plane_point") or element.payload.get("point")
    if isinstance(point, (list, tuple)) and len(point) == 3:
        return tuple(float(value) for value in point)  # type: ignore[return-value]
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    center = [(min_corner[index] + max_corner[index]) * 0.5 for index in range(3)]
    dominant_axis = max(range(3), key=lambda index: abs(normal[index]))
    center[dominant_axis] = min_corner[dominant_axis] if normal[dominant_axis] < 0.0 else max_corner[dominant_axis]
    return tuple(center)  # type: ignore[return-value]


def _gabor_normal_or_nan(element: Any) -> tuple[float, float, float]:
    if element.payload.get("type") != "gabor_frequency" and element.carrier_id != "gabor":
        return (float("nan"), float("nan"), float("nan"))
    normal = element.payload.get("normal")
    if isinstance(normal, (list, tuple)) and len(normal) == 3:
        try:
            return _normalize(tuple(float(value) for value in normal))
        except ValueError:
            return (float("nan"), float("nan"), float("nan"))

    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    extents = tuple(max_corner[index] - min_corner[index] for index in range(3))
    if any(value <= 0.0 for value in extents):
        return (float("nan"), float("nan"), float("nan"))
    axis = min(range(3), key=lambda index: extents[index])
    normal_values = [0.0, 0.0, 0.0]
    normal_values[axis] = 1.0
    return tuple(normal_values)  # type: ignore[return-value]


def _gabor_plane_point_or_nan(element: Any) -> tuple[float, float, float]:
    normal = _gabor_normal_or_nan(element)
    if any(value != value for value in normal):
        return (float("nan"), float("nan"), float("nan"))
    point = element.payload.get("plane_point") or element.payload.get("point")
    if isinstance(point, (list, tuple)) and len(point) == 3:
        return tuple(float(value) for value in point)  # type: ignore[return-value]
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    return tuple(
        (min_corner[index] + max_corner[index]) * 0.5 for index in range(3)
    )  # type: ignore[return-value]


def _gaussian_inverse_covariance_or_nan(element: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if element.payload.get("type") != "gaussian_fallback":
        return _nan_matrix3()
    covariance = element.payload.get("covariance")
    if not _is_matrix3(covariance):
        return _nan_matrix3()
    inverse = _inverse_matrix3(tuple(tuple(float(value) for value in row) for row in covariance))
    return inverse if inverse is not None else _nan_matrix3()


def _gaussian_support_radius_sq(element: Any) -> float:
    if element.payload.get("type") != "gaussian_fallback":
        return float("nan")
    explicit = element.payload.get("support_radius_sq")
    if explicit is not None:
        try:
            value = float(explicit)
        except (TypeError, ValueError):
            return float("nan")
        return value if value > 0.0 else float("nan")
    sigma_radius = element.payload.get("support_sigma", 3.0)
    try:
        sigma = float(sigma_radius)
    except (TypeError, ValueError):
        return float("nan")
    return sigma * sigma if sigma > 0.0 else float("nan")


def _beta_support_radius_or_nan(element: Any) -> tuple[float, float, float]:
    if element.payload.get("type") != "beta_kernel":
        return (float("nan"), float("nan"), float("nan"))
    support_radius = element.payload.get("support_radius")
    if not isinstance(support_radius, (list, tuple)) or len(support_radius) != 3:
        return (float("nan"), float("nan"), float("nan"))
    try:
        radii = tuple(float(value) for value in support_radius)
    except (TypeError, ValueError):
        return (float("nan"), float("nan"), float("nan"))
    if any(value <= 0.0 for value in radii):
        return (float("nan"), float("nan"), float("nan"))
    return radii  # type: ignore[return-value]


def _nan_matrix3() -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    nan = float("nan")
    return ((nan, nan, nan), (nan, nan, nan), (nan, nan, nan))


def _inverse_matrix3(matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    cofactor00 = e * i - f * h
    cofactor01 = -(d * i - f * g)
    cofactor02 = d * h - e * g
    cofactor10 = -(b * i - c * h)
    cofactor11 = a * i - c * g
    cofactor12 = -(a * h - b * g)
    cofactor20 = b * f - c * e
    cofactor21 = -(a * f - c * d)
    cofactor22 = a * e - b * d
    determinant = a * cofactor00 + b * cofactor01 + c * cofactor02
    if abs(determinant) <= 1e-12:
        return None
    inv_det = 1.0 / determinant
    return (
        (cofactor00 * inv_det, cofactor10 * inv_det, cofactor20 * inv_det),
        (cofactor01 * inv_det, cofactor11 * inv_det, cofactor21 * inv_det),
        (cofactor02 * inv_det, cofactor12 * inv_det, cofactor22 * inv_det),
    )


def _is_matrix3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value)


def _torch_index_tensor_values(tensor: Any) -> list[int]:
    return [int(index) for index in tensor.detach().cpu().tolist()]


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


def _normal_present_for(element: Any) -> bool:
    return _normal_for(element) is not None or element.carrier_id in {"surface", "gabor"} or element.payload.get("type") in {
        "surface_cell",
        "gabor_frequency",
    }


def _default_trainable_normal_for(element: Any) -> tuple[float, float, float]:
    if element.carrier_id == "surface" or element.payload.get("type") == "surface_cell":
        return (0.0, 0.0, -1.0)
    if element.carrier_id == "gabor" or element.payload.get("type") == "gabor_frequency":
        return (0.0, 0.0, 1.0)
    return (0.0, 0.0, 0.0)


def _predicted_normal_tensors(torch: Any, normals: Sequence[tuple[float, float, float] | None], *, device: str) -> tuple[Any, Any]:
    values = [normal if normal is not None else (0.0, 0.0, 0.0) for normal in normals]
    present = [normal is not None for normal in normals]
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(present, dtype=torch.bool, device=device),
    )


def _predicted_normal_tensors_from_indices(
    torch: Any,
    first_index: Any,
    has_hit: Any,
    element_normals: Any,
    element_normal_present: Any,
) -> tuple[Any, Any]:
    safe_indices = torch.clamp(first_index, min=0)
    predicted_normals = element_normals[safe_indices]
    predicted_present = element_normal_present[safe_indices] & has_hit
    predicted_normals = torch.where(
        predicted_present.unsqueeze(1),
        predicted_normals,
        torch.zeros_like(predicted_normals),
    )
    return predicted_normals, predicted_present


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


def _torch_confidence_loss(torch: Any, predicted_confidence: Any, target_confidence: Any | None, target_confidence_present: Any | None) -> Any:
    if target_confidence is None or target_confidence_present is None:
        return torch.zeros((), dtype=torch.float32, device=predicted_confidence.device)
    present = target_confidence_present.to(dtype=predicted_confidence.dtype)
    clamped_target = torch.clamp(target_confidence, min=0.0, max=1.0)
    squared_error = (predicted_confidence - clamped_target) ** 2
    return torch.sum(squared_error * present) / torch.clamp(torch.sum(present), min=1.0)


def _torch_query_contract_loss(
    torch: Any,
    elements: Sequence[Any],
    element_weights: Any,
    *,
    target_semantic_ids: Sequence[str | None],
    target_material_ids: Sequence[str | None],
    device: str,
) -> Any:
    losses = []
    for targets, element_values in (
        (target_semantic_ids, tuple(_semantic_id_for(element) for element in elements)),
        (target_material_ids, tuple(element.material_id for element in elements)),
    ):
        supervised = [index for index, target in enumerate(targets) if target is not None]
        if not supervised:
            continue
        match_rows = []
        for target in targets:
            match_rows.append([1.0 if target is not None and value == target else 0.0 for value in element_values])
        match_mask = torch.tensor(match_rows, dtype=torch.float32, device=device)
        matched = torch.sum(element_weights * match_mask, dim=1)
        supervised_indices = torch.tensor(supervised, dtype=torch.long, device=device)
        supervised_match = torch.clamp(matched[supervised_indices], min=0.0, max=1.0)
        losses.append(torch.mean(1.0 - supervised_match))
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=element_weights.device)
    return torch.mean(torch.stack(tuple(losses)))


def _optional_target_normal_tuple(target_normals: Any | None, target_normal_present: Any | None) -> tuple[tuple[float, float, float] | None, ...]:
    if target_normals is None or target_normal_present is None:
        return ()
    values = target_normals.detach().cpu().tolist()
    present = target_normal_present.detach().cpu().tolist()
    return tuple(tuple(float(channel) for channel in value) if is_present else None for value, is_present in zip(values, present))  # type: ignore[return-value]


def _optional_target_confidence_tuple(target_confidence: Any | None, target_confidence_present: Any | None) -> tuple[float | None, ...]:
    if target_confidence is None or target_confidence_present is None:
        return ()
    values = target_confidence.detach().cpu().tolist()
    present = target_confidence_present.detach().cpu().tolist()
    return tuple(float(value) if is_present else None for value, is_present in zip(values, present))


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
