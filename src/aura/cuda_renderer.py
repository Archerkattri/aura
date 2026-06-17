from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from aura.cuda_kernels import CudaExtensionStatus, cuda_kernel_extension_status
from aura.optimize import RenderTarget
from aura.ray import Ray
from aura.scene import AuraScene, RayTraversal


CudaFallbackBackend = Literal["cpu", "torch", "auto", "none"]


@dataclass(frozen=True)
class CudaRendererLaunchConfig:
    """Validated launch shape for the future CUDA renderer boundary."""

    ray_count: int
    threads_per_block: int = 128
    max_hits: int = 8
    fallback_backend: CudaFallbackBackend = "cpu"
    device: str | None = None
    require_cuda: bool = False

    def __post_init__(self) -> None:
        if int(self.ray_count) <= 0:
            raise ValueError("ray_count must be positive")
        if int(self.threads_per_block) <= 0:
            raise ValueError("threads_per_block must be positive")
        if int(self.threads_per_block) > 1024:
            raise ValueError("threads_per_block must be <= 1024")
        if int(self.max_hits) <= 0:
            raise ValueError("max_hits must be positive")
        if self.fallback_backend not in {"cpu", "torch", "auto", "none"}:
            raise ValueError("fallback_backend must be one of cpu, torch, auto, none")
        object.__setattr__(self, "ray_count", int(self.ray_count))
        object.__setattr__(self, "threads_per_block", int(self.threads_per_block))
        object.__setattr__(self, "max_hits", int(self.max_hits))

    @property
    def block_count(self) -> int:
        return (self.ray_count + self.threads_per_block - 1) // self.threads_per_block

    def to_dict(self) -> dict[str, object]:
        return {
            "rayCount": self.ray_count,
            "threadsPerBlock": self.threads_per_block,
            "blockCount": self.block_count,
            "maxHits": self.max_hits,
            "fallbackBackend": self.fallback_backend,
            "device": self.device,
            "requireCuda": self.require_cuda,
        }


@dataclass(frozen=True)
class CudaRendererBatch:
    """AURA ray-query outputs returned by CUDA boundary fallbacks."""

    launch_config: CudaRendererLaunchConfig
    backend: str
    device: str
    extension: CudaExtensionStatus
    reason: str
    element_ids: tuple[str | None, ...]
    carrier_ids: tuple[str | None, ...]
    color: tuple[tuple[float, float, float], ...]
    opacity: tuple[float, ...]
    transmittance: tuple[float, ...]
    depth: tuple[float | None, ...]
    normal: tuple[tuple[float, float, float] | None, ...]
    confidence: tuple[float, ...]
    residual: tuple[bool, ...]
    material_ids: tuple[str | None, ...]
    semantic_ids: tuple[str | None, ...]
    provenance: tuple[str | None, ...]
    ordered_hits: tuple[tuple[dict[str, object], ...], ...]
    ordered_hit_overflow: tuple[bool, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_BATCH",
            "apiName": "cuda_render_rays",
            "productionReady": False,
            "available": False,
            "backend": self.backend,
            "device": self.device,
            "reason": self.reason,
            "launchConfig": self.launch_config.to_dict(),
            "extension": self.extension.to_dict(),
            "elementIds": list(self.element_ids),
            "carrierIds": list(self.carrier_ids),
            "color": [list(value) for value in self.color],
            "opacity": list(self.opacity),
            "transmittance": list(self.transmittance),
            "depth": list(self.depth),
            "normal": [list(value) if value is not None else None for value in self.normal],
            "confidence": list(self.confidence),
            "residual": list(self.residual),
            "materialIds": list(self.material_ids),
            "semanticIds": list(self.semantic_ids),
            "provenance": list(self.provenance),
            "orderedHits": [[dict(hit) for hit in ray_hits] for ray_hits in self.ordered_hits],
            "orderedHitOverflow": list(self.ordered_hit_overflow),
        }


def cuda_renderer_launch_config(
    ray_count: int,
    *,
    threads_per_block: int = 128,
    max_hits: int = 8,
    fallback_backend: CudaFallbackBackend = "cpu",
    device: str | None = None,
    require_cuda: bool = False,
) -> CudaRendererLaunchConfig:
    return CudaRendererLaunchConfig(
        ray_count=ray_count,
        threads_per_block=threads_per_block,
        max_hits=max_hits,
        fallback_backend=fallback_backend,
        device=device,
        require_cuda=require_cuda,
    )


def cuda_render_rays(
    scene: AuraScene,
    ray_origins: Sequence[Sequence[float]] | Any,
    ray_directions: Sequence[Sequence[float]] | Any,
    *,
    threads_per_block: int = 128,
    max_hits: int = 8,
    fallback_backend: CudaFallbackBackend = "cpu",
    device: str | None = None,
    require_cuda: bool = False,
) -> CudaRendererBatch:
    """Render batched rays through the CUDA renderer boundary.

    No CUDA acceleration is claimed here. Until a compiled renderer extension is
    available, this function either raises when CUDA is required or returns an
    explicit CPU/torch fallback batch with the AURA ray-query contract fields.
    """

    rays = _validated_rays(ray_origins, ray_directions)
    launch_config = cuda_renderer_launch_config(
        len(rays),
        threads_per_block=threads_per_block,
        max_hits=max_hits,
        fallback_backend=fallback_backend,
        device=device,
        require_cuda=require_cuda,
    )
    extension = cuda_kernel_extension_status(build=False)
    if extension.available:
        raise NotImplementedError("compiled CUDA renderer dispatch is not implemented in this Python boundary")
    if require_cuda or fallback_backend == "none":
        raise RuntimeError(f"CUDA renderer extension is unavailable: {extension.reason or 'not_available'}")

    resolved_backend = _resolve_fallback_backend(fallback_backend, scene=scene)
    if resolved_backend == "torch":
        return _torch_fallback_batch(scene, rays, launch_config, extension, device=device)
    return _cpu_fallback_batch(scene, rays, launch_config, extension)


def _validated_rays(ray_origins: Sequence[Sequence[float]] | Any, ray_directions: Sequence[Sequence[float]] | Any) -> tuple[Ray, ...]:
    origins = _vec3_rows(ray_origins, "ray_origins")
    directions = _vec3_rows(ray_directions, "ray_directions")
    if len(origins) != len(directions):
        raise ValueError(f"ray_origins count {len(origins)} does not match ray_directions count {len(directions)}")
    if not origins:
        raise ValueError("ray_count must be positive")
    return tuple(Ray(origin=origin, direction=direction) for origin, direction in zip(origins, directions))


def _vec3_rows(values: Sequence[Sequence[float]] | Any, name: str) -> tuple[tuple[float, float, float], ...]:
    if values is None:
        raise ValueError(f"{name} is required")
    shape = getattr(values, "shape", None)
    if shape is not None:
        shape_tuple = tuple(int(dim) for dim in shape)
        if len(shape_tuple) != 2 or shape_tuple[1] != 3:
            raise ValueError(f"{name} must have shape rayCount x 3")
        try:
            values = values.detach().cpu().tolist()
        except AttributeError:
            try:
                values = values.tolist()
            except AttributeError:
                pass
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence or tensor-like object with shape")
    rows = []
    for row in values:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise ValueError(f"{name} must contain 3D ray vectors")
        rows.append((float(row[0]), float(row[1]), float(row[2])))
    return tuple(rows)


def _resolve_fallback_backend(fallback_backend: CudaFallbackBackend, *, scene: AuraScene) -> Literal["cpu", "torch"]:
    if fallback_backend != "auto":
        return fallback_backend  # type: ignore[return-value]
    try:
        from aura.torch_renderer import torch_renderer_status

        status = torch_renderer_status()
    except Exception:
        return "cpu"
    if status.available and scene.elements:
        return "torch"
    return "cpu"


def _cpu_fallback_batch(
    scene: AuraScene,
    rays: Sequence[Ray],
    launch_config: CudaRendererLaunchConfig,
    extension: CudaExtensionStatus,
) -> CudaRendererBatch:
    traversals = tuple(scene.traverse_ray(ray) for ray in rays)
    return _batch_from_traversals(
        launch_config,
        extension,
        backend="cpu",
        device="cpu",
        reason="cuda_extension_unavailable_cpu_fallback",
        traversals=traversals,
    )


def _torch_fallback_batch(
    scene: AuraScene,
    rays: Sequence[Ray],
    launch_config: CudaRendererLaunchConfig,
    extension: CudaExtensionStatus,
    *,
    device: str | None,
) -> CudaRendererBatch:
    from aura.torch_renderer import torch_render_targets

    targets = tuple(
        RenderTarget(
            frame_id=f"cuda_fallback_ray_{index}",
            ray=ray,
            target_color=(0.0, 0.0, 0.0),
            target_depth=1.0,
        )
        for index, ray in enumerate(rays)
    )
    torch_batch = torch_render_targets(scene, targets, device=device)
    ordered_hits, overflow = _trim_hits(torch_batch.ordered_hits, launch_config.max_hits)
    return CudaRendererBatch(
        launch_config=launch_config,
        backend="torch",
        device=torch_batch.device,
        extension=extension,
        reason="cuda_extension_unavailable_torch_fallback",
        element_ids=torch_batch.element_ids,
        carrier_ids=torch_batch.carrier_ids,
        color=torch_batch.predicted_color,
        opacity=torch_batch.opacity,
        transmittance=torch_batch.transmittance,
        depth=torch_batch.predicted_depth,
        normal=torch_batch.normal,
        confidence=torch_batch.confidence,
        residual=torch_batch.residual,
        material_ids=torch_batch.material_ids,
        semantic_ids=torch_batch.semantic_ids,
        provenance=torch_batch.provenance,
        ordered_hits=ordered_hits,
        ordered_hit_overflow=overflow,
    )


def _batch_from_traversals(
    launch_config: CudaRendererLaunchConfig,
    extension: CudaExtensionStatus,
    *,
    backend: str,
    device: str,
    reason: str,
    traversals: Sequence[RayTraversal],
) -> CudaRendererBatch:
    ordered_hits, overflow = _trim_hits(
        tuple(tuple(hit.to_dict() for hit in traversal.ordered_hits) for traversal in traversals),
        launch_config.max_hits,
    )
    first_hits = tuple(traversal.ordered_hits[0] if traversal.ordered_hits else None for traversal in traversals)
    return CudaRendererBatch(
        launch_config=launch_config,
        backend=backend,
        device=device,
        extension=extension,
        reason=reason,
        element_ids=tuple(hit.element_id if hit is not None else None for hit in first_hits),
        carrier_ids=tuple(hit.carrier_id if hit is not None else None for hit in first_hits),
        color=tuple(traversal.result.color for traversal in traversals),
        opacity=tuple(traversal.result.opacity for traversal in traversals),
        transmittance=tuple(traversal.result.transmittance for traversal in traversals),
        depth=tuple(traversal.result.depth for traversal in traversals),
        normal=tuple(traversal.result.normal for traversal in traversals),
        confidence=tuple(traversal.result.confidence for traversal in traversals),
        residual=tuple(traversal.result.residual for traversal in traversals),
        material_ids=tuple(traversal.result.material_id for traversal in traversals),
        semantic_ids=tuple(traversal.result.semantic_id for traversal in traversals),
        provenance=tuple(traversal.result.provenance or "miss" for traversal in traversals),
        ordered_hits=ordered_hits,
        ordered_hit_overflow=overflow,
    )


def _trim_hits(
    ordered_hits: Sequence[Sequence[dict[str, object]]],
    max_hits: int,
) -> tuple[tuple[tuple[dict[str, object], ...], ...], tuple[bool, ...]]:
    return (
        tuple(tuple(dict(hit) for hit in ray_hits[:max_hits]) for ray_hits in ordered_hits),
        tuple(len(ray_hits) > max_hits for ray_hits in ordered_hits),
    )
