from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from aura.cuda_kernels import CudaExtensionStatus, cuda_kernel_extension_status, cuda_renderer_source_report
from aura.optimize import RenderTarget
from aura.ray import Ray
from aura.scene import AuraScene, RayTraversal


CudaFallbackBackend = Literal["cpu", "torch", "auto", "none"]
CUDA_RENDERER_CARRIER_IDS = {
    "surface": 0,
    "volume": 1,
    "beta": 2,
    "gabor": 3,
    "neural": 4,
    "semantic": 5,
    "gaussian": 6,
}


@dataclass(frozen=True)
class CudaRendererSceneBuffers:
    """Flat host buffers matching the packaged CUDA renderer scene ABI."""

    element_ids: tuple[str, ...]
    carrier_ids: tuple[str, ...]
    carrier_kernel_ids: tuple[int, ...]
    material_id_table: tuple[str, ...]
    semantic_id_table: tuple[str, ...]
    material_ids: tuple[int, ...]
    semantic_ids: tuple[int, ...]
    element_mins: tuple[float, ...]
    element_maxs: tuple[float, ...]
    colors: tuple[float, ...]
    opacities: tuple[float, ...]
    confidences: tuple[float, ...]

    def __post_init__(self) -> None:
        element_count = len(self.element_ids)
        if len(self.carrier_ids) != element_count or len(self.carrier_kernel_ids) != element_count:
            raise ValueError("CUDA scene buffers require one carrier id per element")
        for name, values, expected in (
            ("material_ids", self.material_ids, element_count),
            ("semantic_ids", self.semantic_ids, element_count),
            ("opacities", self.opacities, element_count),
            ("confidences", self.confidences, element_count),
            ("element_mins", self.element_mins, element_count * 3),
            ("element_maxs", self.element_maxs, element_count * 3),
            ("colors", self.colors, element_count * 3),
        ):
            if len(values) != expected:
                raise ValueError(f"CUDA scene buffer {name} length {len(values)} does not match expected {expected}")

    @property
    def element_count(self) -> int:
        return len(self.element_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_SCENE_BUFFERS",
            "elementCount": self.element_count,
            "elementIds": list(self.element_ids),
            "carrierIds": list(self.carrier_ids),
            "carrierKernelIds": list(self.carrier_kernel_ids),
            "materialIdTable": list(self.material_id_table),
            "semanticIdTable": list(self.semantic_id_table),
            "materialIds": _flat_buffer_metadata(self.material_ids, "int32", (self.element_count,)),
            "semanticIds": _flat_buffer_metadata(self.semantic_ids, "int32", (self.element_count,)),
            "elementMins": _flat_buffer_metadata(self.element_mins, "float32", (self.element_count, 3)),
            "elementMaxs": _flat_buffer_metadata(self.element_maxs, "float32", (self.element_count, 3)),
            "colors": _flat_buffer_metadata(self.colors, "float32", (self.element_count, 3)),
            "opacities": _flat_buffer_metadata(self.opacities, "float32", (self.element_count,)),
            "confidences": _flat_buffer_metadata(self.confidences, "float32", (self.element_count,)),
        }


@dataclass(frozen=True)
class CudaRendererKernelInputBuffers:
    """Flat host buffers for the `aura_render_rays_kernel` launch ABI."""

    scene: CudaRendererSceneBuffers
    ray_origins: tuple[float, ...]
    ray_directions: tuple[float, ...]
    max_hits: int

    def __post_init__(self) -> None:
        if self.max_hits <= 0:
            raise ValueError("CUDA renderer kernel max_hits must be positive")
        if len(self.ray_origins) != len(self.ray_directions):
            raise ValueError("CUDA renderer ray origin/direction buffers must have matching lengths")
        if len(self.ray_origins) % 3 != 0:
            raise ValueError("CUDA renderer ray buffers must be flat rayCount x 3 arrays")

    @property
    def ray_count(self) -> int:
        return len(self.ray_origins) // 3

    @property
    def element_count(self) -> int:
        return self.scene.element_count

    def output_buffer_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "out_color": (self.ray_count, 3),
            "out_alpha": (self.ray_count,),
            "out_transmittance": (self.ray_count,),
            "out_depth": (self.ray_count,),
            "out_normal": (self.ray_count, 3),
            "out_confidence": (self.ray_count,),
            "out_residual": (self.ray_count,),
            "out_material_id": (self.ray_count,),
            "out_semantic_id": (self.ray_count,),
            "ordered_hits": (self.ray_count, self.max_hits),
        }

    def to_kernel_args(self) -> dict[str, object]:
        return {
            "ray_origins": self.ray_origins,
            "ray_directions": self.ray_directions,
            "element_mins": self.scene.element_mins,
            "element_maxs": self.scene.element_maxs,
            "carrier_ids": self.scene.carrier_kernel_ids,
            "colors": self.scene.colors,
            "opacities": self.scene.opacities,
            "confidences": self.scene.confidences,
            "material_ids": self.scene.material_ids,
            "semantic_ids": self.scene.semantic_ids,
            "ray_count": self.ray_count,
            "element_count": self.element_count,
            "max_hits": self.max_hits,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_KERNEL_INPUT_BUFFERS",
            "kernelSymbol": "aura_render_rays_kernel",
            "rayCount": self.ray_count,
            "elementCount": self.element_count,
            "maxHits": self.max_hits,
            "scene": self.scene.to_dict(),
            "rayOrigins": _flat_buffer_metadata(self.ray_origins, "float32", (self.ray_count, 3)),
            "rayDirections": _flat_buffer_metadata(self.ray_directions, "float32", (self.ray_count, 3)),
            "outputBufferShapes": {name: list(shape) for name, shape in self.output_buffer_shapes().items()},
            "kernelArgs": {
                name: _kernel_arg_summary(value)
                for name, value in self.to_kernel_args().items()
            },
        }


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


def cuda_renderer_scene_buffers(scene: AuraScene) -> CudaRendererSceneBuffers:
    """Pack an AURA scene into host buffers matching the CUDA renderer ABI."""

    material_table = _stable_id_table(element.material_id for element in scene.elements)
    semantic_table = _stable_id_table(element.semantic_id for element in scene.elements)
    return CudaRendererSceneBuffers(
        element_ids=tuple(element.id for element in scene.elements),
        carrier_ids=tuple(element.carrier_id for element in scene.elements),
        carrier_kernel_ids=tuple(_carrier_kernel_id(element.carrier_id) for element in scene.elements),
        material_id_table=material_table,
        semantic_id_table=semantic_table,
        material_ids=tuple(_lookup_table_id(material_table, element.material_id) for element in scene.elements),
        semantic_ids=tuple(_lookup_table_id(semantic_table, element.semantic_id) for element in scene.elements),
        element_mins=tuple(value for element in scene.elements for value in element.bounds.min_corner),
        element_maxs=tuple(value for element in scene.elements for value in element.bounds.max_corner),
        colors=tuple(value for element in scene.elements for value in element.color),
        opacities=tuple(float(element.opacity) for element in scene.elements),
        confidences=tuple(float(element.confidence) for element in scene.elements),
    )


def cuda_renderer_kernel_inputs(
    scene: AuraScene,
    ray_origins: Sequence[Sequence[float]] | Any,
    ray_directions: Sequence[Sequence[float]] | Any,
    *,
    max_hits: int = 8,
) -> CudaRendererKernelInputBuffers:
    rays = _validated_rays(ray_origins, ray_directions)
    return CudaRendererKernelInputBuffers(
        scene=cuda_renderer_scene_buffers(scene),
        ray_origins=tuple(value for ray in rays for value in ray.origin),
        ray_directions=tuple(value for ray in rays for value in ray.direction),
        max_hits=max_hits,
    )


def cuda_renderer_reference_first_hit_indices(scene: AuraScene, rays: Sequence[Ray]) -> tuple[int, ...]:
    element_index_by_id = {element.id: index for index, element in enumerate(scene.elements)}
    indices = []
    for ray in rays:
        traversal = scene.traverse_ray(ray)
        if traversal.ordered_hits:
            indices.append(element_index_by_id.get(traversal.ordered_hits[0].element_id, -1))
        else:
            indices.append(-1)
    return tuple(indices)


def cuda_renderer_boundary_report(
    scene: AuraScene | None = None,
    *,
    probe_ray_origin: Sequence[float] = (0.0, 0.0, -1.0),
    probe_ray_direction: Sequence[float] = (0.0, 0.0, 1.0),
    fallback_backend: CudaFallbackBackend = "cpu",
    max_hits: int = 8,
) -> dict[str, object]:
    """Report callable CUDA renderer boundary readiness without claiming CUDA.

    When a scene is supplied, the report executes one fallback ray through the
    same public ``cuda_render_rays`` function used by integration callers. This
    proves the Python boundary can return the AURA ray-query output fields even
    before the compiled CUDA dispatch exists.
    """

    extension = cuda_kernel_extension_status(build=False)
    report: dict[str, object] = {
        "format": "AURA_CUDA_RENDERER_BOUNDARY_REPORT",
        "apiName": "aura.cuda_renderer.cuda_render_rays",
        "callableBoundaryAvailable": True,
        "available": bool(extension.available),
        "productionReady": False,
        "extension": extension.to_dict(),
        "rendererSource": cuda_renderer_source_report(),
        "fallbackBackends": ["cpu", "torch", "auto", "none"],
        "fallbackContractFields": [
            "elementIds",
            "carrierIds",
            "color",
            "opacity",
            "transmittance",
            "depth",
            "normal",
            "confidence",
            "residual",
            "materialIds",
            "semanticIds",
            "provenance",
            "orderedHits",
            "orderedHitOverflow",
        ],
        "productionBlockers": [
            "compiled_cuda_renderer_dispatch_missing",
            "cuda_renderer_parity_benchmarks_missing",
            "cuda_renderer_speed_benchmarks_missing",
        ],
        "notes": (
            "This is the callable renderer boundary and fallback contract. It is "
            "not production CUDA acceleration until a compiled dispatch is "
            "available, parity-tested, and benchmarked."
        ),
    }
    if scene is None:
        report["fallbackProbe"] = None
        report["kernelInputProbe"] = None
        return report
    try:
        kernel_inputs = cuda_renderer_kernel_inputs(
            scene,
            ray_origins=(tuple(float(value) for value in probe_ray_origin),),
            ray_directions=(tuple(float(value) for value in probe_ray_direction),),
            max_hits=max_hits,
        )
        batch = cuda_render_rays(
            scene,
            ray_origins=(tuple(float(value) for value in probe_ray_origin),),
            ray_directions=(tuple(float(value) for value in probe_ray_direction),),
            fallback_backend=fallback_backend,
            max_hits=max_hits,
        )
    except Exception as exc:
        report["fallbackProbe"] = {
            "executed": False,
            "error": str(exc),
        }
        report["kernelInputProbe"] = None
        return report
    payload = batch.to_dict()
    report["fallbackProbe"] = {
        "executed": True,
        "backend": payload["backend"],
        "reason": payload["reason"],
        "rayCount": payload["launchConfig"]["rayCount"],
        "maxHits": payload["launchConfig"]["maxHits"],
        "outputFields": [
            key
            for key in (
                "elementIds",
                "carrierIds",
                "color",
                "opacity",
                "transmittance",
                "depth",
                "normal",
                "confidence",
                "residual",
                "materialIds",
                "semanticIds",
                "provenance",
                "orderedHits",
                "orderedHitOverflow",
            )
            if key in payload
        ],
        "orderedHitOverflow": payload["orderedHitOverflow"],
    }
    report["kernelInputProbe"] = {
        "format": "AURA_CUDA_RENDERER_KERNEL_INPUT_PROBE",
        "kernelSymbol": "aura_render_rays_kernel",
        "rayCount": kernel_inputs.ray_count,
        "elementCount": kernel_inputs.element_count,
        "maxHits": kernel_inputs.max_hits,
        "outputBufferShapes": {name: list(shape) for name, shape in kernel_inputs.output_buffer_shapes().items()},
    }
    return report


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


def _carrier_kernel_id(carrier_id: str) -> int:
    try:
        return CUDA_RENDERER_CARRIER_IDS[carrier_id]
    except KeyError as exc:
        raise ValueError(f"unsupported CUDA renderer carrier id: {carrier_id}") from exc


def _stable_id_table(values: Sequence[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _lookup_table_id(table: Sequence[str], value: str | None) -> int:
    if value is None:
        return -1
    try:
        return tuple(table).index(value)
    except ValueError as exc:
        raise ValueError(f"value {value!r} is missing from CUDA renderer id table") from exc


def _flat_buffer_metadata(values: Sequence[object], dtype: str, shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": list(shape),
        "length": len(values),
        "preview": list(values[: min(6, len(values))]),
    }


def _kernel_arg_summary(value: object) -> dict[str, object] | object:
    if isinstance(value, tuple):
        return {
            "length": len(value),
            "preview": list(value[: min(6, len(value))]),
        }
    return value
