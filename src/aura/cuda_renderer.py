from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import as_file, files
from typing import Any, Literal, Mapping, Sequence

from aura.cuda_kernels import (
    CUDA_EXTENSION_MODULE_NAME,
    CUDA_RENDERER_BINDING_SYMBOL,
    CUDA_RENDERER_KERNEL_SYMBOL,
    CUDA_RENDERER_LAUNCHER_SYMBOL,
    CudaExtensionStatus,
    cuda_kernel_extension_status,
    cuda_renderer_source_report,
)
from aura.ray import Ray
from aura.scene import AuraScene, RayTraversal


CudaFallbackBackend = Literal["cpu", "torch", "auto", "none"]
CUDA_RENDERER_BVH_BINDING_SYMBOL = "render_rays_bvh"
CUDA_RENDERER_CARRIER_IDS = {
    "surface": 0,
    "volume": 1,
    "beta": 2,
    "gabor": 3,
    "neural": 4,
    "semantic": 5,
    "gaussian": 6,
}
CUDA_RENDERER_INF_SENTINEL = 3.402823466e38


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
    plane_points: tuple[float, ...]
    plane_normals: tuple[float, ...]
    beta_support_radii: tuple[float, ...]
    gaussian_means: tuple[float, ...]
    gaussian_inverse_covariances: tuple[float, ...]
    gaussian_support_radius_sq: tuple[float, ...]
    colors: tuple[float, ...]
    opacities: tuple[float, ...]
    confidences: tuple[float, ...]
    payload_params: tuple[float, ...]

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
            ("plane_points", self.plane_points, element_count * 3),
            ("plane_normals", self.plane_normals, element_count * 3),
            ("beta_support_radii", self.beta_support_radii, element_count * 3),
            ("gaussian_means", self.gaussian_means, element_count * 3),
            ("gaussian_inverse_covariances", self.gaussian_inverse_covariances, element_count * 9),
            ("gaussian_support_radius_sq", self.gaussian_support_radius_sq, element_count),
            ("colors", self.colors, element_count * 3),
            ("payload_params", self.payload_params, element_count * 5),
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
            "planePoints": _flat_buffer_metadata(self.plane_points, "float32", (self.element_count, 3)),
            "planeNormals": _flat_buffer_metadata(self.plane_normals, "float32", (self.element_count, 3)),
            "betaSupportRadii": _flat_buffer_metadata(self.beta_support_radii, "float32", (self.element_count, 3)),
            "gaussianMeans": _flat_buffer_metadata(self.gaussian_means, "float32", (self.element_count, 3)),
            "gaussianInverseCovariances": _flat_buffer_metadata(
                self.gaussian_inverse_covariances,
                "float32",
                (self.element_count, 3, 3),
            ),
            "gaussianSupportRadiusSq": _flat_buffer_metadata(self.gaussian_support_radius_sq, "float32", (self.element_count,)),
            "colors": _flat_buffer_metadata(self.colors, "float32", (self.element_count, 3)),
            "opacities": _flat_buffer_metadata(self.opacities, "float32", (self.element_count,)),
            "confidences": _flat_buffer_metadata(self.confidences, "float32", (self.element_count,)),
            "payloadParams": _flat_buffer_metadata(self.payload_params, "float32", (self.element_count, 5)),
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
            "plane_points": self.scene.plane_points,
            "plane_normals": self.scene.plane_normals,
            "beta_support_radii": self.scene.beta_support_radii,
            "gaussian_means": self.scene.gaussian_means,
            "gaussian_inverse_covariances": self.scene.gaussian_inverse_covariances,
            "gaussian_support_radius_sq": self.scene.gaussian_support_radius_sq,
            "carrier_ids": self.scene.carrier_kernel_ids,
            "colors": self.scene.colors,
            "opacities": self.scene.opacities,
            "confidences": self.scene.confidences,
            "payload_params": self.scene.payload_params,
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
class CudaRendererKernelSimulation:
    """CPU oracle for the packaged `aura_render_rays_kernel` ABI."""

    inputs: CudaRendererKernelInputBuffers
    out_color: tuple[float, ...]
    out_alpha: tuple[float, ...]
    out_transmittance: tuple[float, ...]
    out_depth: tuple[float, ...]
    out_normal: tuple[float, ...]
    out_confidence: tuple[float, ...]
    out_residual: tuple[int, ...]
    out_material_id: tuple[int, ...]
    out_semantic_id: tuple[int, ...]
    ordered_hits: tuple[int, ...]

    def __post_init__(self) -> None:
        ray_count = self.inputs.ray_count
        max_hits = self.inputs.max_hits
        for name, values, expected in (
            ("out_color", self.out_color, ray_count * 3),
            ("out_alpha", self.out_alpha, ray_count),
            ("out_transmittance", self.out_transmittance, ray_count),
            ("out_depth", self.out_depth, ray_count),
            ("out_normal", self.out_normal, ray_count * 3),
            ("out_confidence", self.out_confidence, ray_count),
            ("out_residual", self.out_residual, ray_count),
            ("out_material_id", self.out_material_id, ray_count),
            ("out_semantic_id", self.out_semantic_id, ray_count),
            ("ordered_hits", self.ordered_hits, ray_count * max_hits),
        ):
            if len(values) != expected:
                raise ValueError(f"CUDA renderer simulation {name} length {len(values)} does not match expected {expected}")

    @property
    def ray_count(self) -> int:
        return self.inputs.ray_count

    @property
    def first_hit_indices(self) -> tuple[int, ...]:
        return tuple(self.ordered_hits[index * self.inputs.max_hits] for index in range(self.ray_count))

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_KERNEL_SIMULATION",
            "kernelSymbol": "aura_render_rays_kernel",
            "productionReady": False,
            "rayCount": self.ray_count,
            "elementCount": self.inputs.element_count,
            "maxHits": self.inputs.max_hits,
            "firstHitIndices": list(self.first_hit_indices),
            "outColor": _flat_buffer_metadata(self.out_color, "float32", (self.ray_count, 3)),
            "outAlpha": _flat_buffer_metadata(self.out_alpha, "float32", (self.ray_count,)),
            "outTransmittance": _flat_buffer_metadata(self.out_transmittance, "float32", (self.ray_count,)),
            "outDepth": _flat_buffer_metadata(self.out_depth, "float32", (self.ray_count,)),
            "outNormal": _flat_buffer_metadata(self.out_normal, "float32", (self.ray_count, 3)),
            "outConfidence": _flat_buffer_metadata(self.out_confidence, "float32", (self.ray_count,)),
            "outResidual": _flat_buffer_metadata(self.out_residual, "uint8", (self.ray_count,)),
            "outMaterialId": _flat_buffer_metadata(self.out_material_id, "int32", (self.ray_count,)),
            "outSemanticId": _flat_buffer_metadata(self.out_semantic_id, "int32", (self.ray_count,)),
            "orderedHits": _flat_buffer_metadata(self.ordered_hits, "int32", (self.ray_count, self.inputs.max_hits)),
            "notes": (
                "CPU oracle for the packaged CUDA renderer ABI. This validates flat buffer semantics "
                "without compiling or launching CUDA."
            ),
        }


@dataclass(frozen=True)
class CudaRendererSymbolProbe:
    """Verification state for the compiled renderer kernel/launcher symbols."""

    extension: CudaExtensionStatus
    module_name: str
    kernel_symbol: str = CUDA_RENDERER_KERNEL_SYMBOL
    launcher_symbol: str = CUDA_RENDERER_LAUNCHER_SYMBOL
    binding_symbol: str = CUDA_RENDERER_BINDING_SYMBOL
    kernel_symbol_available: bool = False
    launcher_symbol_available: bool = False
    binding_symbol_available: bool = False
    binding_callable: bool = False
    module_object_available: bool = False
    reason: str | None = None

    @property
    def dispatch_symbols_ready(self) -> bool:
        return (
            self.extension.available
            and self.module_object_available
            and self.kernel_symbol_available
            and self.launcher_symbol_available
            and self.binding_symbol_available
        )

    @property
    def production_ready(self) -> bool:
        return self.dispatch_symbols_ready and self.binding_callable

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_SYMBOL_PROBE",
            "productionReady": self.production_ready,
            "dispatchSymbolsReady": self.dispatch_symbols_ready,
            "moduleName": self.module_name,
            "moduleObjectAvailable": self.module_object_available,
            "kernelSymbol": self.kernel_symbol,
            "launcherSymbol": self.launcher_symbol,
            "bindingSymbol": self.binding_symbol,
            "kernelSymbolAvailable": self.kernel_symbol_available,
            "launcherSymbolAvailable": self.launcher_symbol_available,
            "bindingSymbolAvailable": self.binding_symbol_available,
            "bindingCallable": self.binding_callable,
            "extension": self.extension.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CudaRendererDispatchContract:
    """Host-side contract for the compiled renderer launch boundary."""

    inputs: CudaRendererKernelInputBuffers
    launch_config: CudaRendererLaunchConfig
    extension: CudaExtensionStatus
    symbol_probe: CudaRendererSymbolProbe

    @property
    def dispatch_ready(self) -> bool:
        return self.extension.available and self.symbol_probe.dispatch_symbols_ready and self.symbol_probe.binding_callable

    @property
    def reason(self) -> str:
        if not self.extension.available:
            return f"compiled_cuda_renderer_extension_unavailable: {self.extension.reason or 'not_available'}"
        if not self.symbol_probe.dispatch_symbols_ready:
            return f"compiled_cuda_renderer_binding_unavailable: {self.symbol_probe.reason or 'unknown'}"
        if not self.symbol_probe.binding_callable:
            return "python_cuda_renderer_binding_missing"
        return "compiled_cuda_renderer_python_binding_ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_DISPATCH_CONTRACT",
            "kernelSymbol": CUDA_RENDERER_KERNEL_SYMBOL,
            "launcherSymbol": CUDA_RENDERER_LAUNCHER_SYMBOL,
            "productionReady": self.dispatch_ready,
            "dispatchReady": self.dispatch_ready,
            "reason": self.reason,
            "compiledExtensionAvailable": self.extension.available,
            "rendererSymbolsReady": self.symbol_probe.dispatch_symbols_ready,
            "pythonBindingAvailable": self.symbol_probe.binding_callable,
            "launchConfig": self.launch_config.to_dict(),
            "extension": self.extension.to_dict(),
            "symbolProbe": self.symbol_probe.to_dict(),
            "rayCount": self.inputs.ray_count,
            "elementCount": self.inputs.element_count,
            "maxHits": self.inputs.max_hits,
            "kernelArgs": {
                name: _kernel_arg_summary(value)
                for name, value in self.inputs.to_kernel_args().items()
            },
            "outputBufferShapes": {name: list(shape) for name, shape in self.inputs.output_buffer_shapes().items()},
            "parityOracle": "simulate_cuda_renderer_kernel",
            "missingDispatchWork": [
                *(() if self.extension.available else ("build and import the Python CUDA extension module",)),
                *(() if self.symbol_probe.binding_symbol_available else ("verify render_rays binding in the loaded extension",)),
                *(() if self.symbol_probe.binding_callable else ("validate render_rays Python tensor dispatch on CUDA hardware",)),
                *(() if self.dispatch_ready else ("run compiled CUDA dispatch before production claims",)),
            ],
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

    @property
    def production_ready(self) -> bool:
        return self.backend == "cuda" and self.extension.available and self.extension.compiled and self.extension.loadable

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_BATCH",
            "apiName": "cuda_render_rays",
            "productionReady": self.production_ready,
            "available": self.backend == "cuda",
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


@dataclass(frozen=True)
class CudaRendererBvh:
    """Flattened median-split BVH over element AABBs for GPU traversal.

    Internal nodes store left/right child node indices; leaf nodes store the
    element index in ``node_element`` (``-1`` for internal nodes and ``node_left``
    ``-1`` for leaves). The arrays are laid out for direct upload as CUDA tensors
    and a stack-based device traversal.
    """

    node_mins: tuple[float, ...]
    node_maxs: tuple[float, ...]
    node_left: tuple[int, ...]
    node_right: tuple[int, ...]
    node_element: tuple[int, ...]
    node_count: int
    element_count: int
    max_depth: int

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "AURA_CUDA_RENDERER_BVH",
            "nodeCount": self.node_count,
            "elementCount": self.element_count,
            "maxDepth": self.max_depth,
            "leafCount": sum(1 for value in self.node_element if value >= 0),
        }


def cuda_renderer_build_bvh(scene: AuraScene) -> CudaRendererBvh:
    """Build a flattened median-split BVH over the scene element AABBs."""

    element_count = len(scene.elements)
    mins = [tuple(float(value) for value in element.bounds.min_corner) for element in scene.elements]
    maxs = [tuple(float(value) for value in element.bounds.max_corner) for element in scene.elements]

    node_mins: list[tuple[float, float, float]] = []
    node_maxs: list[tuple[float, float, float]] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_element: list[int] = []

    def _bounds_of(indices: list[int]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lo = [float("inf"), float("inf"), float("inf")]
        hi = [float("-inf"), float("-inf"), float("-inf")]
        for index in indices:
            for axis in range(3):
                lo[axis] = min(lo[axis], mins[index][axis])
                hi[axis] = max(hi[axis], maxs[index][axis])
        return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])

    max_depth = 0

    def _build(indices: list[int], depth: int) -> int:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        node_index = len(node_mins)
        node_mins.append((0.0, 0.0, 0.0))
        node_maxs.append((0.0, 0.0, 0.0))
        node_left.append(-1)
        node_right.append(-1)
        node_element.append(-1)
        lo, hi = _bounds_of(indices)
        node_mins[node_index] = lo
        node_maxs[node_index] = hi
        if len(indices) == 1:
            node_element[node_index] = indices[0]
            return node_index
        extent = tuple(hi[axis] - lo[axis] for axis in range(3))
        axis = max(range(3), key=lambda candidate: extent[candidate])
        centroid = lambda index: 0.5 * (mins[index][axis] + maxs[index][axis])  # noqa: E731
        ordered = sorted(indices, key=centroid)
        mid = len(ordered) // 2
        left_indices = ordered[:mid]
        right_indices = ordered[mid:]
        if not left_indices or not right_indices:
            # Degenerate split (coincident centroids); fall back to index split.
            mid = len(indices) // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]
        node_left[node_index] = _build(left_indices, depth + 1)
        node_right[node_index] = _build(right_indices, depth + 1)
        return node_index

    if element_count > 0:
        _build(list(range(element_count)), 1)

    return CudaRendererBvh(
        node_mins=tuple(value for node in node_mins for value in node),
        node_maxs=tuple(value for node in node_maxs for value in node),
        node_left=tuple(node_left),
        node_right=tuple(node_right),
        node_element=tuple(node_element),
        node_count=len(node_mins),
        element_count=element_count,
        max_depth=max_depth,
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
        plane_points=tuple(value for element in scene.elements for value in _cuda_renderer_plane_point(element)),
        plane_normals=tuple(value for element in scene.elements for value in _cuda_renderer_plane_normal(element)),
        beta_support_radii=tuple(value for element in scene.elements for value in _cuda_renderer_beta_support_radius(element)),
        gaussian_means=tuple(value for element in scene.elements for value in _cuda_renderer_gaussian_mean(element)),
        gaussian_inverse_covariances=tuple(
            value
            for element in scene.elements
            for row in _cuda_renderer_gaussian_inverse_covariance(element)
            for value in row
        ),
        gaussian_support_radius_sq=tuple(_cuda_renderer_gaussian_support_radius_sq(element) for element in scene.elements),
        colors=tuple(value for element in scene.elements for value in element.color),
        opacities=tuple(float(element.opacity) for element in scene.elements),
        confidences=tuple(float(element.confidence) for element in scene.elements),
        payload_params=tuple(value for element in scene.elements for value in _cuda_renderer_payload_params(element)),
    )


def _cuda_renderer_payload_params(element: Any) -> tuple[float, float, float, float, float]:
    payload_type = element.payload.get("type")
    if payload_type == "volume_cell" or element.carrier_id == "volume":
        return (float(element.payload.get("density", element.opacity)), float(element.payload.get("opacity", 1.0)), 0.0, 0.0, 0.0)
    if payload_type == "beta_kernel" or element.carrier_id == "beta":
        return (float(element.payload.get("alpha", 1.0)), float(element.payload.get("beta", 1.0)), 0.0, 0.0, 0.0)
    if payload_type == "gabor_frequency" or element.carrier_id == "gabor":
        frequency = element.payload.get("frequency", (0.0, 0.0, 0.0))
        if not isinstance(frequency, (list, tuple)) or len(frequency) != 3:
            frequency = (0.0, 0.0, 0.0)
        return (
            float(frequency[0]),
            float(frequency[1]),
            float(frequency[2]),
            float(element.payload.get("phase", 0.0)),
            float(element.payload.get("bandwidth", 1.0)),
        )
    if payload_type == "neural_residual" or element.carrier_id == "neural":
        return (float(element.payload.get("residual_scale", 0.0)), 0.0, 0.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0, 0.0, 0.0)


def _cuda_renderer_plane_normal(element: Any) -> tuple[float, float, float]:
    payload_type = element.payload.get("type")
    if payload_type == "surface_cell" or element.carrier_id == "surface":
        normal = _normal_for_element(element)
        if normal is None:
            return _nan_vec3()
        try:
            return _normalize_vec3(normal)
        except ValueError:
            return _nan_vec3()
    if payload_type == "gabor_frequency" or element.carrier_id == "gabor":
        normal = element.payload.get("normal")
        if isinstance(normal, (list, tuple)) and len(normal) == 3:
            try:
                return _normalize_vec3(tuple(float(value) for value in normal))
            except ValueError:
                return _nan_vec3()
        min_corner = tuple(float(value) for value in element.bounds.min_corner)
        max_corner = tuple(float(value) for value in element.bounds.max_corner)
        extents = tuple(max_corner[index] - min_corner[index] for index in range(3))
        if any(value <= 0.0 for value in extents):
            return _nan_vec3()
        axis = min(range(3), key=lambda index: extents[index])
        values = [0.0, 0.0, 0.0]
        values[axis] = 1.0
        return tuple(values)  # type: ignore[return-value]
    return _nan_vec3()


def _cuda_renderer_plane_point(element: Any) -> tuple[float, float, float]:
    payload_type = element.payload.get("type")
    normal = _cuda_renderer_plane_normal(element)
    if _is_nan_vec3(normal):
        return _nan_vec3()
    point = element.payload.get("plane_point") or element.payload.get("point")
    if isinstance(point, (list, tuple)) and len(point) == 3:
        return tuple(float(value) for value in point)  # type: ignore[return-value]
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    center = [(min_corner[index] + max_corner[index]) * 0.5 for index in range(3)]
    if payload_type == "surface_cell" or element.carrier_id == "surface":
        dominant_axis = max(range(3), key=lambda index: abs(normal[index]))
        center[dominant_axis] = min_corner[dominant_axis] if normal[dominant_axis] < 0.0 else max_corner[dominant_axis]
    return tuple(center)  # type: ignore[return-value]


def _cuda_renderer_beta_support_radius(element: Any) -> tuple[float, float, float]:
    payload_type = element.payload.get("type")
    if payload_type != "beta_kernel" and element.carrier_id != "beta":
        return _nan_vec3()
    support_radius = element.payload.get("support_radius")
    if isinstance(support_radius, (list, tuple)) and len(support_radius) == 3:
        try:
            radii = tuple(float(value) for value in support_radius)
        except (TypeError, ValueError):
            return _nan_vec3()
        return radii if all(value > 0.0 for value in radii) else _nan_vec3()  # type: ignore[return-value]
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    radii = tuple(max((max_corner[index] - min_corner[index]) * 0.5, 1.0e-4) for index in range(3))
    return radii  # type: ignore[return-value]


def _cuda_renderer_gaussian_mean(element: Any) -> tuple[float, float, float]:
    if element.payload.get("type") != "gaussian_fallback" and element.carrier_id != "gaussian":
        return _nan_vec3()
    mean = element.payload.get("mean")
    if not isinstance(mean, (list, tuple)) or len(mean) != 3:
        return _nan_vec3()
    try:
        return tuple(float(value) for value in mean)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return _nan_vec3()


def _cuda_renderer_gaussian_inverse_covariance(element: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if element.payload.get("type") != "gaussian_fallback" and element.carrier_id != "gaussian":
        return _nan_matrix3()
    covariance = element.payload.get("covariance")
    if not _is_matrix3(covariance):
        return _nan_matrix3()
    try:
        matrix = tuple(tuple(float(value) for value in row) for row in covariance)
    except (TypeError, ValueError):
        return _nan_matrix3()
    inverse = _inverse_matrix3(matrix)  # type: ignore[arg-type]
    return inverse if inverse is not None else _nan_matrix3()


def _cuda_renderer_gaussian_support_radius_sq(element: Any) -> float:
    if element.payload.get("type") != "gaussian_fallback" and element.carrier_id != "gaussian":
        return float("nan")
    explicit = element.payload.get("support_radius_sq")
    if explicit is not None:
        try:
            value = float(explicit)
        except (TypeError, ValueError):
            return float("nan")
        return value if value > 0.0 else float("nan")
    try:
        sigma = float(element.payload.get("support_sigma", 3.0))
    except (TypeError, ValueError):
        return float("nan")
    return sigma * sigma if sigma > 0.0 else float("nan")


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


def cuda_renderer_dispatch_contract(
    scene: AuraScene,
    ray_origins: Sequence[Sequence[float]] | Any,
    ray_directions: Sequence[Sequence[float]] | Any,
    *,
    threads_per_block: int = 128,
    max_hits: int = 8,
    fallback_backend: CudaFallbackBackend = "cpu",
    device: str | None = None,
    require_cuda: bool = False,
    extension: CudaExtensionStatus | None = None,
    extension_module: Any | None = None,
) -> CudaRendererDispatchContract:
    rays = _validated_rays(ray_origins, ray_directions)
    launch_config = cuda_renderer_launch_config(
        len(rays),
        threads_per_block=threads_per_block,
        max_hits=max_hits,
        fallback_backend=fallback_backend,
        device=device,
        require_cuda=require_cuda,
    )
    inputs = CudaRendererKernelInputBuffers(
        scene=cuda_renderer_scene_buffers(scene),
        ray_origins=tuple(value for ray in rays for value in ray.origin),
        ray_directions=tuple(value for ray in rays for value in ray.direction),
        max_hits=max_hits,
    )
    extension_status, resolved_module = _resolve_cuda_renderer_extension(
        extension=extension,
        extension_module=extension_module,
        build=require_cuda or fallback_backend == "none",
    )
    return CudaRendererDispatchContract(
        inputs=inputs,
        launch_config=launch_config,
        extension=extension_status,
        symbol_probe=cuda_renderer_symbol_probe(extension_status, extension_module=resolved_module),
    )


def cuda_renderer_symbol_probe(
    extension: CudaExtensionStatus | None = None,
    *,
    extension_module: Any | None = None,
) -> CudaRendererSymbolProbe:
    """Verify compiled renderer symbols without launching CUDA.

    The pybind11 module exposes ``render_rays`` to Python. The raw CUDA kernel
    and host launcher remain native symbols and are reported from the compiled
    extension status/source contract rather than required as Python attributes.
    """

    extension_status = extension or cuda_kernel_extension_status(build=False)
    module_name = extension_status.module_name
    if not extension_status.available:
        return CudaRendererSymbolProbe(
            extension=extension_status,
            module_name=module_name,
            reason=f"extension_unavailable: {extension_status.reason or 'not_available'}",
        )
    if extension_module is None:
        return CudaRendererSymbolProbe(
            extension=extension_status,
            module_name=module_name,
            reason="extension_module_object_unavailable",
        )
    binding = getattr(extension_module, CUDA_RENDERER_BINDING_SYMBOL, None)
    binding_available = hasattr(extension_module, CUDA_RENDERER_BINDING_SYMBOL)
    binding_callable = callable(binding)
    kernel_available = hasattr(extension_module, CUDA_RENDERER_KERNEL_SYMBOL) or (
        binding_callable and CUDA_RENDERER_KERNEL_SYMBOL in extension_status.symbols
    )
    launcher_available = hasattr(extension_module, CUDA_RENDERER_LAUNCHER_SYMBOL) or (
        binding_callable and CUDA_RENDERER_LAUNCHER_SYMBOL in extension_status.symbols
    )
    missing = []
    if not kernel_available:
        missing.append(CUDA_RENDERER_KERNEL_SYMBOL)
    if not launcher_available:
        missing.append(CUDA_RENDERER_LAUNCHER_SYMBOL)
    if not binding_available:
        missing.append(CUDA_RENDERER_BINDING_SYMBOL)
    return CudaRendererSymbolProbe(
        extension=extension_status,
        module_name=module_name,
        kernel_symbol_available=kernel_available,
        launcher_symbol_available=launcher_available,
        binding_symbol_available=binding_available,
        binding_callable=binding_callable,
        module_object_available=True,
        reason=None if not missing else f"missing_symbols: {', '.join(missing)}",
    )


def simulate_cuda_renderer_kernel(inputs: CudaRendererKernelInputBuffers) -> CudaRendererKernelSimulation:
    """Run the packaged CUDA renderer ABI on CPU for parity tests."""

    out_color: list[float] = []
    out_alpha: list[float] = []
    out_transmittance: list[float] = []
    out_depth: list[float] = []
    out_normal: list[float] = []
    out_confidence: list[float] = []
    out_residual: list[int] = []
    out_material_id: list[int] = []
    out_semantic_id: list[int] = []
    ordered_hits: list[int] = []

    for ray_index in range(inputs.ray_count):
        origin = _flat_vec3(inputs.ray_origins, ray_index)
        direction = _flat_vec3(inputs.ray_directions, ray_index)
        ray_hits: list[tuple[float, float, tuple[float, float, float], int]] = []
        for element_index in range(inputs.element_count):
            carrier_id = inputs.scene.carrier_kernel_ids[element_index]
            hit = _simulate_ray_aabb_intersect(
                origin,
                direction,
                _flat_vec3(inputs.scene.element_mins, element_index),
                _flat_vec3(inputs.scene.element_maxs, element_index),
            )
            if carrier_id in (CUDA_RENDERER_CARRIER_IDS["surface"], CUDA_RENDERER_CARRIER_IDS["gabor"]):
                plane_hit = _simulate_ray_plane_intersect(
                    origin,
                    direction,
                    _flat_vec3(inputs.scene.element_mins, element_index),
                    _flat_vec3(inputs.scene.element_maxs, element_index),
                    _flat_vec3(inputs.scene.plane_points, element_index),
                    _flat_vec3(inputs.scene.plane_normals, element_index),
                )
                if plane_hit is not None:
                    hit = plane_hit
            elif carrier_id == CUDA_RENDERER_CARRIER_IDS["beta"]:
                beta_support_radii = _flat_vec3(inputs.scene.beta_support_radii, element_index)
                beta_hit = _simulate_ray_beta_ellipsoid_intersect(
                    origin,
                    direction,
                    _flat_vec3(inputs.scene.element_mins, element_index),
                    _flat_vec3(inputs.scene.element_maxs, element_index),
                    beta_support_radii,
                )
                if beta_hit is None and _valid_support_radii(beta_support_radii):
                    hit = None
                elif beta_hit is not None and hit is not None:
                    bounded_entry = max(hit[0], beta_hit[0])
                    bounded_exit = min(hit[1], beta_hit[1])
                    hit = (bounded_entry, bounded_exit, hit[2]) if bounded_exit >= bounded_entry else None
            elif carrier_id == CUDA_RENDERER_CARRIER_IDS["gaussian"]:
                gaussian_hit = _simulate_ray_gaussian_ellipsoid_intersect(
                    origin,
                    direction,
                    _flat_vec3(inputs.scene.gaussian_means, element_index),
                    _flat_matrix3(inputs.scene.gaussian_inverse_covariances, element_index),
                    inputs.scene.gaussian_support_radius_sq[element_index],
                )
                if gaussian_hit is not None and hit is not None:
                    bounded_entry = max(hit[0], gaussian_hit[0])
                    bounded_exit = min(hit[1], gaussian_hit[1])
                    hit = (bounded_entry, bounded_exit, gaussian_hit[2]) if bounded_exit >= bounded_entry else None
                elif _valid_gaussian_geometry(
                    _flat_vec3(inputs.scene.gaussian_means, element_index),
                    _flat_matrix3(inputs.scene.gaussian_inverse_covariances, element_index),
                    inputs.scene.gaussian_support_radius_sq[element_index],
                ):
                    hit = None
            if hit is None:
                continue
            depth, _exit_depth, normal = hit
            ray_hits.append((depth, _exit_depth, normal, element_index))
        ray_hits.sort(key=lambda item: item[0])
        stored_hits = ray_hits[: inputs.max_hits]

        if not stored_hits:
            out_color.extend((0.0, 0.0, 0.0))
            out_alpha.append(0.0)
            out_transmittance.append(1.0)
            out_depth.append(CUDA_RENDERER_INF_SENTINEL)
            out_normal.extend((0.0, 0.0, 0.0))
            out_confidence.append(0.0)
            out_residual.append(0)
            out_material_id.append(-1)
            out_semantic_id.append(-1)
            ordered_hits.extend((-1,) * inputs.max_hits)
            continue

        color = [0.0, 0.0, 0.0]
        remaining = 1.0
        confidence_num = 0.0
        confidence_den = 0.0
        residual = 0
        for _depth, _exit_depth, _normal, element_index in stored_hits:
            transmittance, carrier_confidence, carrier_color, carrier_residual = _simulate_cuda_carrier_response(
                inputs,
                origin=origin,
                direction=direction,
                depth=_depth,
                exit_depth=_exit_depth,
                element_index=element_index,
            )
            weight = remaining * (1.0 - transmittance)
            color[0] += weight * carrier_color[0]
            color[1] += weight * carrier_color[1]
            color[2] += weight * carrier_color[2]
            confidence_num += weight * carrier_confidence
            confidence_den += weight
            remaining = _clamp_unit(remaining * transmittance)
            if carrier_residual:
                residual = 1

        first_depth, _first_exit, first_normal, first_element = stored_hits[0]
        out_color.extend(tuple(color))
        out_alpha.append(_clamp_unit(1.0 - remaining))
        out_transmittance.append(remaining)
        out_depth.append(first_depth)
        out_normal.extend(first_normal)
        out_confidence.append(confidence_num / confidence_den if confidence_den > 1e-8 else 0.0)
        out_residual.append(residual)
        out_material_id.append(inputs.scene.material_ids[first_element])
        out_semantic_id.append(inputs.scene.semantic_ids[first_element])
        ordered_hits.extend(element_index for _depth, _exit_depth, _normal, element_index in stored_hits)
        ordered_hits.extend((-1,) * (inputs.max_hits - len(stored_hits)))

    return CudaRendererKernelSimulation(
        inputs=inputs,
        out_color=tuple(out_color),
        out_alpha=tuple(out_alpha),
        out_transmittance=tuple(out_transmittance),
        out_depth=tuple(out_depth),
        out_normal=tuple(out_normal),
        out_confidence=tuple(out_confidence),
        out_residual=tuple(out_residual),
        out_material_id=tuple(out_material_id),
        out_semantic_id=tuple(out_semantic_id),
        ordered_hits=tuple(ordered_hits),
    )


def _simulate_cuda_carrier_response(
    inputs: CudaRendererKernelInputBuffers,
    *,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    depth: float,
    exit_depth: float,
    element_index: int,
) -> tuple[float, float, tuple[float, float, float], bool]:
    carrier_id = inputs.scene.carrier_kernel_ids[element_index]
    opacity = _clamp_unit(inputs.scene.opacities[element_index])
    confidence = _clamp_unit(inputs.scene.confidences[element_index])
    color = _flat_vec3(inputs.scene.colors, element_index)
    payload = _flat_payload5(inputs.scene.payload_params, element_index)
    residual = carrier_id == CUDA_RENDERER_CARRIER_IDS["neural"]
    if carrier_id == CUDA_RENDERER_CARRIER_IDS["volume"]:
        density = max(payload[0], 0.0)
        volume_opacity = _clamp_unit(payload[1])
        alpha = volume_opacity * (1.0 - _exp_neg(density * max(exit_depth - depth, 0.0)))
        return (_clamp_unit(1.0 - alpha), confidence, color, residual)
    if carrier_id == CUDA_RENDERER_CARRIER_IDS["beta"]:
        alpha = max(payload[0], 1e-6)
        beta = max(payload[1], 1e-6)
        point = tuple(origin[axis] + direction[axis] * depth for axis in range(3))
        support = _cuda_beta_support(
            point,
            _flat_vec3(inputs.scene.element_mins, element_index),
            _flat_vec3(inputs.scene.element_maxs, element_index),
            _flat_vec3(inputs.scene.beta_support_radii, element_index),
            alpha,
            beta,
        )
        return (_clamp_unit(1.0 - opacity * support), confidence, color, residual)
    if carrier_id == CUDA_RENDERER_CARRIER_IDS["gabor"]:
        point = tuple(origin[axis] + direction[axis] * depth for axis in range(3))
        phase = payload[3]
        bandwidth = _clamp_unit(payload[4])
        dot = point[0] * payload[0] + point[1] * payload[1] + point[2] * payload[2]
        modulation = 1.0 - bandwidth + bandwidth * (0.5 + 0.5 * _sin_tau(dot + phase / (2.0 * 3.141592653589793)))
        return (
            _clamp_unit(1.0 - opacity),
            _clamp_unit(confidence * bandwidth),
            tuple(_clamp_unit(channel * modulation) for channel in color),  # type: ignore[return-value]
            residual,
        )
    if carrier_id == CUDA_RENDERER_CARRIER_IDS["neural"]:
        neural_strength = _clamp_unit(payload[0])
        return (
            _clamp_unit(1.0 - opacity * neural_strength),
            _clamp_unit(confidence * (1.0 - neural_strength * 0.25)),
            color,
            True,
        )
    if carrier_id == CUDA_RENDERER_CARRIER_IDS["gaussian"]:
        mean = _flat_vec3(inputs.scene.gaussian_means, element_index)
        inverse_covariance = _flat_matrix3(inputs.scene.gaussian_inverse_covariances, element_index)
        support_radius_sq = inputs.scene.gaussian_support_radius_sq[element_index]
        gaussian_weight = 1.0
        if _valid_gaussian_geometry(mean, inverse_covariance, support_radius_sq):
            ray_to_mean = tuple(mean[axis] - origin[axis] for axis in range(3))
            direction_norm = max(sum(direction[axis] * direction[axis] for axis in range(3)), 1e-8)
            projected_depth = sum(ray_to_mean[axis] * direction[axis] for axis in range(3)) / direction_norm
            gaussian_depth = max(depth, min(exit_depth, projected_depth))
            point = tuple(origin[axis] + direction[axis] * gaussian_depth for axis in range(3))
            delta = tuple(point[axis] - mean[axis] for axis in range(3))
            weighted_delta = tuple(
                inverse_covariance[axis][0] * delta[0]
                + inverse_covariance[axis][1] * delta[1]
                + inverse_covariance[axis][2] * delta[2]
                for axis in range(3)
            )
            mahalanobis = max(sum(delta[axis] * weighted_delta[axis] for axis in range(3)), 0.0)
            gaussian_weight = _clamp_unit(_exp_neg(0.5 * mahalanobis))
        return (
            _clamp_unit(1.0 - opacity * gaussian_weight),
            _clamp_unit(confidence * gaussian_weight),
            color,
            residual,
        )
    return (_clamp_unit(1.0 - opacity), confidence, color, residual)


def _flat_payload5(values: Sequence[float], index: int) -> tuple[float, float, float, float, float]:
    start = index * 5
    return (
        float(values[start]),
        float(values[start + 1]),
        float(values[start + 2]),
        float(values[start + 3]),
        float(values[start + 4]),
    )


def _cuda_beta_support(
    point: tuple[float, float, float],
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    support_radii: tuple[float, float, float],
    alpha: float,
    beta: float,
) -> float:
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    normalized = []
    for axis in range(3):
        radius = max(support_radii[axis], 1e-6)
        normalized.append(_clamp_unit(1.0 - abs(point[axis] - center[axis]) / radius))
    u = sum(normalized) / 3.0
    raw = (u ** (alpha - 1.0)) * ((1.0 - u) ** (beta - 1.0))
    if alpha > 1.0 and beta > 1.0:
        mode = _clamp_unit((alpha - 1.0) / max(alpha + beta - 2.0, 1e-6))
        peak = (mode ** (alpha - 1.0)) * ((1.0 - mode) ** (beta - 1.0))
        if peak > 0.0:
            raw /= peak
    return _clamp_unit(raw)


def _exp_neg(value: float) -> float:
    from math import exp

    return float(exp(-value))


def _sin_tau(value: float) -> float:
    from math import pi, sin

    return float(sin(2.0 * pi * value))


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
        "symbolProbe": cuda_renderer_symbol_probe(extension).to_dict(),
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
        report["dispatchContractProbe"] = None
        return report
    try:
        kernel_inputs = cuda_renderer_kernel_inputs(
            scene,
            ray_origins=(tuple(float(value) for value in probe_ray_origin),),
            ray_directions=(tuple(float(value) for value in probe_ray_direction),),
            max_hits=max_hits,
        )
        dispatch_contract = cuda_renderer_dispatch_contract(
            scene,
            ray_origins=(tuple(float(value) for value in probe_ray_origin),),
            ray_directions=(tuple(float(value) for value in probe_ray_direction),),
            fallback_backend=fallback_backend,
            max_hits=max_hits,
            extension=extension,
        )
        batch = cuda_render_rays(
            scene,
            ray_origins=(tuple(float(value) for value in probe_ray_origin),),
            ray_directions=(tuple(float(value) for value in probe_ray_direction),),
            fallback_backend=fallback_backend,
            max_hits=max_hits,
            extension=extension,
        )
    except Exception as exc:
        report["fallbackProbe"] = {
            "executed": False,
            "error": str(exc),
        }
        report["kernelInputProbe"] = None
        report["dispatchContractProbe"] = None
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
        "kernelSymbol": CUDA_RENDERER_KERNEL_SYMBOL,
        "rayCount": kernel_inputs.ray_count,
        "elementCount": kernel_inputs.element_count,
        "maxHits": kernel_inputs.max_hits,
        "outputBufferShapes": {name: list(shape) for name, shape in kernel_inputs.output_buffer_shapes().items()},
    }
    dispatch_payload = dispatch_contract.to_dict()
    report["dispatchContractProbe"] = {
        "format": dispatch_payload["format"],
        "kernelSymbol": dispatch_payload["kernelSymbol"],
        "launcherSymbol": dispatch_payload["launcherSymbol"],
        "dispatchReady": dispatch_payload["dispatchReady"],
        "productionReady": dispatch_payload["productionReady"],
        "reason": dispatch_payload["reason"],
        "compiledExtensionAvailable": dispatch_payload["compiledExtensionAvailable"],
        "rendererSymbolsReady": dispatch_payload["rendererSymbolsReady"],
        "pythonBindingAvailable": dispatch_payload["pythonBindingAvailable"],
        "symbolProbe": dispatch_payload["symbolProbe"],
        "outputBufferShapes": dispatch_payload["outputBufferShapes"],
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
    extension: CudaExtensionStatus | None = None,
    extension_module: Any | None = None,
) -> CudaRendererBatch:
    """Render batched rays through the CUDA renderer boundary.

    No CUDA acceleration is claimed here. Until a compiled renderer extension is
    available, this function either raises when CUDA is required or returns an
    explicit CPU/torch fallback batch with the AURA ray-query contract fields.
    """

    ray_count = _ray_count_from_inputs(ray_origins, ray_directions)
    launch_config = cuda_renderer_launch_config(
        ray_count,
        threads_per_block=threads_per_block,
        max_hits=max_hits,
        fallback_backend=fallback_backend,
        device=device,
        require_cuda=require_cuda,
    )
    extension, extension_module = _resolve_cuda_renderer_extension(
        extension=extension,
        extension_module=extension_module,
        build=require_cuda or fallback_backend == "none",
    )
    symbol_probe = cuda_renderer_symbol_probe(extension, extension_module=extension_module)
    if (
        _device_requests_cuda(device)
        and extension.available
        and symbol_probe.dispatch_symbols_ready
        and symbol_probe.binding_callable
        and extension_module is not None
    ):
        return _compiled_extension_batch(scene, ray_origins, ray_directions, launch_config, extension, extension_module, device=device)
    if extension.available and (require_cuda or fallback_backend == "none"):
        raise RuntimeError(f"CUDA renderer Python dispatch is unavailable: {symbol_probe.reason or 'binding_not_ready'}")
    if require_cuda or fallback_backend == "none":
        raise RuntimeError(f"CUDA renderer extension is unavailable: {extension.reason or 'not_available'}")

    resolved_backend = _resolve_fallback_backend(fallback_backend, scene=scene)
    if resolved_backend == "torch":
        return _torch_fallback_batch(scene, ray_origins, ray_directions, launch_config, extension, device=device, symbol_probe=symbol_probe)
    rays = _validated_rays(ray_origins, ray_directions)
    return _cpu_fallback_batch(scene, rays, launch_config, extension, symbol_probe=symbol_probe)


def _device_requests_cuda(device: str | None) -> bool:
    return device is None or str(device).startswith("cuda")


def _ray_count_from_inputs(ray_origins: Sequence[Sequence[float]] | Any, ray_directions: Sequence[Sequence[float]] | Any) -> int:
    origin_count = _ray_count_from_rows(ray_origins, "ray_origins")
    direction_count = _ray_count_from_rows(ray_directions, "ray_directions")
    if origin_count != direction_count:
        raise ValueError(f"ray_origins count {origin_count} does not match ray_directions count {direction_count}")
    if origin_count <= 0:
        raise ValueError("ray_count must be positive")
    return origin_count


def _ray_count_from_rows(values: Sequence[Sequence[float]] | Any, name: str) -> int:
    if values is None:
        raise ValueError(f"{name} is required")
    shape = getattr(values, "shape", None)
    if shape is not None:
        shape_tuple = tuple(int(dim) for dim in shape)
        if len(shape_tuple) != 2 or shape_tuple[1] != 3:
            raise ValueError(f"{name} must have shape rayCount x 3")
        return shape_tuple[0]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence or tensor-like object with shape")
    for row in values:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise ValueError(f"{name} must contain 3D ray vectors")
    return len(values)


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


def _resolve_cuda_renderer_extension(
    *,
    extension: CudaExtensionStatus | None,
    extension_module: Any | None,
    build: bool,
) -> tuple[CudaExtensionStatus, Any | None]:
    if extension_module is not None:
        status = extension or _available_extension_status(build_attempted=False)
        return status, extension_module
    if extension is not None and not extension.available:
        return extension, None

    imported_module = _import_cuda_renderer_extension_module()
    if imported_module is not None:
        return extension or _available_extension_status(build_attempted=False), imported_module
    if extension is not None:
        return extension, None
    if not build:
        return cuda_kernel_extension_status(build=False), None
    return _build_cuda_renderer_extension_module()


def _import_cuda_renderer_extension_module() -> Any | None:
    try:
        return import_module(CUDA_EXTENSION_MODULE_NAME)
    except Exception:
        return None


def _build_cuda_renderer_extension_module() -> tuple[CudaExtensionStatus, Any | None]:
    source_paths = ("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu")
    symbols = (
        "aura_surface_forward_kernel",
        "aura_volume_forward_kernel",
        "aura_beta_forward_kernel",
        "aura_gabor_forward_kernel",
        "aura_neural_forward_kernel",
        "aura_semantic_forward_kernel",
        "aura_gaussian_forward_kernel",
        CUDA_RENDERER_KERNEL_SYMBOL,
        CUDA_RENDERER_LAUNCHER_SYMBOL,
        CUDA_RENDERER_BINDING_SYMBOL,
    )
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
    except Exception as exc:  # pragma: no cover - depends on optional torch install state.
        return _extension_status_failure(source_paths, symbols, f"torch_extension_unavailable: {exc}", build_attempted=True), None
    if CUDA_HOME is None:
        return _extension_status_failure(source_paths, symbols, "cuda_home_unavailable", build_attempted=True), None
    if not bool(torch.cuda.is_available()):
        return _extension_status_failure(source_paths, symbols, "torch_cuda_unavailable", build_attempted=True), None
    try:
        with ExitStack() as stack:
            resolved_sources = [
                str(stack.enter_context(as_file(files("aura").joinpath(path))))
                for path in source_paths
            ]
            module = load(
                name=CUDA_EXTENSION_MODULE_NAME,
                sources=resolved_sources,
                with_cuda=True,
                is_python_module=True,
                verbose=False,
            )
    except Exception as exc:  # pragma: no cover - requires CUDA compiler/runtime matrix.
        return _extension_status_failure(source_paths, symbols, f"build_or_load_failed: {exc}", build_attempted=True), None
    return _available_extension_status(build_attempted=True), module


def _available_extension_status(*, build_attempted: bool) -> CudaExtensionStatus:
    return CudaExtensionStatus(
        available=True,
        build_attempted=build_attempted,
        compiled=True,
        loadable=True,
        module_name=CUDA_EXTENSION_MODULE_NAME,
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=(
            "aura_surface_forward_kernel",
            "aura_volume_forward_kernel",
            "aura_beta_forward_kernel",
            "aura_gabor_forward_kernel",
            "aura_neural_forward_kernel",
            "aura_semantic_forward_kernel",
            "aura_gaussian_forward_kernel",
            CUDA_RENDERER_KERNEL_SYMBOL,
            CUDA_RENDERER_LAUNCHER_SYMBOL,
            CUDA_RENDERER_BINDING_SYMBOL,
        ),
    )


def _extension_status_failure(
    source_paths: tuple[str, ...],
    symbols: tuple[str, ...],
    reason: str,
    *,
    build_attempted: bool,
) -> CudaExtensionStatus:
    return CudaExtensionStatus(
        available=False,
        build_attempted=build_attempted,
        compiled=False,
        loadable=False,
        module_name=CUDA_EXTENSION_MODULE_NAME,
        source_paths=source_paths,
        symbols=symbols,
        reason=reason,
    )


def _cpu_fallback_batch(
    scene: AuraScene,
    rays: Sequence[Ray],
    launch_config: CudaRendererLaunchConfig,
    extension: CudaExtensionStatus,
    *,
    symbol_probe: CudaRendererSymbolProbe | None = None,
) -> CudaRendererBatch:
    traversals = tuple(scene.traverse_ray(ray) for ray in rays)
    reason = "cuda_extension_unavailable_cpu_fallback"
    if extension.available and symbol_probe is not None and not symbol_probe.dispatch_symbols_ready:
        reason = f"cuda_extension_available_python_binding_unavailable_cpu_fallback: {symbol_probe.reason or 'binding_not_ready'}"
    return _batch_from_traversals(
        launch_config,
        extension,
        backend="cpu",
        device="cpu",
        reason=reason,
        traversals=traversals,
    )


def _torch_fallback_batch(
    scene: AuraScene,
    ray_origins: Sequence[Sequence[float]] | Any,
    ray_directions: Sequence[Sequence[float]] | Any,
    launch_config: CudaRendererLaunchConfig,
    extension: CudaExtensionStatus,
    *,
    device: str | None,
    symbol_probe: CudaRendererSymbolProbe | None = None,
) -> CudaRendererBatch:
    from aura.torch_renderer import torch_render_rays

    torch_batch = torch_render_rays(
        scene,
        ray_origins,
        ray_directions,
        device=device,
        frame_id_prefix="cuda_fallback_ray",
    )
    ordered_hits, overflow = _trim_hits(torch_batch.ordered_hits, launch_config.max_hits)
    reason = "cuda_extension_unavailable_torch_fallback"
    if extension.available and symbol_probe is not None and not symbol_probe.dispatch_symbols_ready:
        reason = f"cuda_extension_available_python_binding_unavailable_torch_fallback: {symbol_probe.reason or 'binding_not_ready'}"
    return CudaRendererBatch(
        launch_config=launch_config,
        backend="torch",
        device=torch_batch.device,
        extension=extension,
        reason=reason,
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


def _compiled_extension_batch(
    scene: AuraScene,
    ray_origins: Sequence[Sequence[float]] | Any,
    ray_directions: Sequence[Sequence[float]] | Any,
    launch_config: CudaRendererLaunchConfig,
    extension: CudaExtensionStatus,
    extension_module: Any,
    *,
    device: str | None,
) -> CudaRendererBatch:
    torch = _require_torch_for_cuda_dispatch()
    resolved_device = device or "cuda"
    scene_buffers = cuda_renderer_scene_buffers(scene)
    element_count = scene_buffers.element_count
    ray_origin_tensor = _cuda_float_ray_tensor(torch, ray_origins, "ray_origins", resolved_device)
    ray_direction_tensor = _cuda_float_ray_tensor(torch, ray_directions, "ray_directions", resolved_device)
    if int(ray_origin_tensor.shape[0]) != launch_config.ray_count:
        raise ValueError(f"ray_origins count {int(ray_origin_tensor.shape[0])} does not match launch config ray count")
    if int(ray_direction_tensor.shape[0]) != launch_config.ray_count:
        raise ValueError(f"ray_directions count {int(ray_direction_tensor.shape[0])} does not match launch config ray count")
    scene_args = (
        torch.tensor(scene_buffers.element_mins, dtype=torch.float32, device=resolved_device).reshape(element_count, 3).contiguous(),
        torch.tensor(scene_buffers.element_maxs, dtype=torch.float32, device=resolved_device).reshape(element_count, 3).contiguous(),
        torch.tensor(scene_buffers.plane_points, dtype=torch.float32, device=resolved_device).reshape(element_count, 3).contiguous(),
        torch.tensor(scene_buffers.plane_normals, dtype=torch.float32, device=resolved_device).reshape(element_count, 3).contiguous(),
        torch.tensor(scene_buffers.beta_support_radii, dtype=torch.float32, device=resolved_device).reshape(element_count, 3).contiguous(),
        torch.tensor(scene_buffers.gaussian_means, dtype=torch.float32, device=resolved_device).reshape(element_count, 3).contiguous(),
        torch.tensor(scene_buffers.gaussian_inverse_covariances, dtype=torch.float32, device=resolved_device).reshape(element_count, 3, 3).contiguous(),
        torch.tensor(scene_buffers.gaussian_support_radius_sq, dtype=torch.float32, device=resolved_device).contiguous(),
        torch.tensor(scene_buffers.carrier_kernel_ids, dtype=torch.int32, device=resolved_device).contiguous(),
        torch.tensor(scene_buffers.colors, dtype=torch.float32, device=resolved_device).reshape(element_count, 3).contiguous(),
        torch.tensor(scene_buffers.opacities, dtype=torch.float32, device=resolved_device).contiguous(),
        torch.tensor(scene_buffers.confidences, dtype=torch.float32, device=resolved_device).contiguous(),
        torch.tensor(scene_buffers.payload_params, dtype=torch.float32, device=resolved_device).reshape(element_count, 5).contiguous(),
        torch.tensor(scene_buffers.material_ids, dtype=torch.int32, device=resolved_device).contiguous(),
        torch.tensor(scene_buffers.semantic_ids, dtype=torch.int32, device=resolved_device).contiguous(),
    )
    bvh_binding = getattr(extension_module, CUDA_RENDERER_BVH_BINDING_SYMBOL, None)
    if element_count > 0 and callable(bvh_binding):
        bvh = cuda_renderer_build_bvh(scene)
        outputs = bvh_binding(
            ray_origin_tensor,
            ray_direction_tensor,
            *scene_args,
            torch.tensor(bvh.node_mins, dtype=torch.float32, device=resolved_device).reshape(bvh.node_count, 3).contiguous(),
            torch.tensor(bvh.node_maxs, dtype=torch.float32, device=resolved_device).reshape(bvh.node_count, 3).contiguous(),
            torch.tensor(bvh.node_left, dtype=torch.int32, device=resolved_device).contiguous(),
            torch.tensor(bvh.node_right, dtype=torch.int32, device=resolved_device).contiguous(),
            torch.tensor(bvh.node_element, dtype=torch.int32, device=resolved_device).contiguous(),
            int(launch_config.max_hits),
            int(launch_config.threads_per_block),
        )
        return _batch_from_compiled_outputs(
            launch_config.ray_count,
            outputs,
            launch_config,
            extension,
            scene_buffers,
            device=str(resolved_device),
        )
    outputs = getattr(extension_module, CUDA_RENDERER_BINDING_SYMBOL)(
        ray_origin_tensor,
        ray_direction_tensor,
        *scene_args,
        int(launch_config.max_hits),
        int(launch_config.threads_per_block),
    )
    return _batch_from_compiled_outputs(
        launch_config.ray_count,
        outputs,
        launch_config,
        extension,
        scene_buffers,
        device=str(resolved_device),
    )


def _cuda_float_ray_tensor(torch: Any, values: Sequence[Sequence[float]] | Any, name: str, device: str) -> Any:
    if values is None:
        raise ValueError(f"{name} is required")
    if hasattr(values, "shape") and hasattr(values, "to"):
        tensor = values.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if len(tuple(int(dim) for dim in tensor.shape)) != 2 or int(tensor.shape[1]) != 3:
        raise ValueError(f"{name} must have shape rayCount x 3")
    if int(tensor.shape[0]) <= 0:
        raise ValueError("ray_count must be positive")
    return tensor.contiguous()


def _batch_from_compiled_outputs(
    ray_count: int,
    outputs: Mapping[str, Any],
    launch_config: CudaRendererLaunchConfig,
    extension: CudaExtensionStatus,
    scene_buffers: CudaRendererSceneBuffers,
    *,
    device: str,
) -> CudaRendererBatch:
    out_color = _tensor_to_nested_float_tuple(outputs["out_color"], width=3)
    out_alpha = _tensor_to_float_tuple(outputs["out_alpha"])
    out_transmittance = _tensor_to_float_tuple(outputs["out_transmittance"])
    out_depth_raw = _tensor_to_float_tuple(outputs["out_depth"])
    out_normal_raw = _tensor_to_nested_float_tuple(outputs["out_normal"], width=3)
    out_confidence = _tensor_to_float_tuple(outputs["out_confidence"])
    out_residual = tuple(bool(value) for value in _tensor_to_int_tuple(outputs["out_residual"]))
    out_material = _tensor_to_int_tuple(outputs["out_material_id"])
    out_semantic = _tensor_to_int_tuple(outputs["out_semantic_id"])
    ordered_indices = _tensor_to_int_tuple(outputs["ordered_hits"])
    ordered_hits, overflow = _compiled_ordered_hits(scene_buffers, ordered_indices, launch_config.max_hits)
    first_indices = tuple(ordered_indices[index * launch_config.max_hits] for index in range(ray_count))
    return CudaRendererBatch(
        launch_config=launch_config,
        backend="cuda",
        device=device,
        extension=extension,
        reason="compiled_cuda_renderer_python_binding",
        element_ids=tuple(scene_buffers.element_ids[index] if index >= 0 else None for index in first_indices),
        carrier_ids=tuple(scene_buffers.carrier_ids[index] if index >= 0 else None for index in first_indices),
        color=out_color,
        opacity=out_alpha,
        transmittance=out_transmittance,
        depth=tuple(None if value >= CUDA_RENDERER_INF_SENTINEL * 0.5 else value for value in out_depth_raw),
        normal=tuple(None if first < 0 else out_normal_raw[index] for index, first in enumerate(first_indices)),
        confidence=out_confidence,
        residual=out_residual,
        material_ids=tuple(_table_value(scene_buffers.material_id_table, index) for index in out_material),
        semantic_ids=tuple(_table_value(scene_buffers.semantic_id_table, index) for index in out_semantic),
        provenance=tuple(scene_buffers.element_ids[index] if index >= 0 else "miss" for index in first_indices),
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


def _compiled_ordered_hits(
    scene_buffers: CudaRendererSceneBuffers,
    ordered_indices: Sequence[int],
    max_hits: int,
) -> tuple[tuple[tuple[dict[str, object], ...], ...], tuple[bool, ...]]:
    ray_count = len(ordered_indices) // max_hits
    traces: list[tuple[dict[str, object], ...]] = []
    overflow: list[bool] = []
    for ray_index in range(ray_count):
        ray_hits = []
        for hit_offset in range(max_hits):
            element_index = int(ordered_indices[ray_index * max_hits + hit_offset])
            if element_index < 0:
                continue
            ray_hits.append(
                {
                    "elementId": scene_buffers.element_ids[element_index],
                    "carrierId": scene_buffers.carrier_ids[element_index],
                    "kernelElementIndex": element_index,
                }
            )
        traces.append(tuple(ray_hits))
        overflow.append(False)
    return tuple(traces), tuple(overflow)


def _tensor_to_float_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in _tensor_to_flat_list(value))


def _tensor_to_int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in _tensor_to_flat_list(value))


def _tensor_to_nested_float_tuple(value: Any, *, width: int) -> tuple[tuple[float, ...], ...]:
    flat = _tensor_to_float_tuple(value)
    if len(flat) % width != 0:
        raise ValueError("compiled CUDA renderer output tensor has invalid flat length")
    return tuple(tuple(flat[index : index + width]) for index in range(0, len(flat), width))


def _tensor_to_flat_list(value: Any) -> list[Any]:
    try:
        value = value.detach().cpu()
    except AttributeError:
        pass
    try:
        value = value.reshape(-1)
    except AttributeError:
        pass
    try:
        return list(value.tolist())
    except AttributeError:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flattened: list[Any] = []
            for item in value:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
    raise ValueError("compiled CUDA renderer output is not tensor-like")


def _table_value(table: Sequence[str], index: int) -> str | None:
    if index < 0:
        return None
    if index >= len(table):
        raise ValueError("compiled CUDA renderer returned an out-of-range dictionary id")
    return table[index]


def _require_torch_for_cuda_dispatch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch install state.
        raise RuntimeError(f"PyTorch is required for compiled CUDA renderer dispatch: {exc}") from exc
    return torch


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


def _normal_for_element(element: Any) -> tuple[float, float, float] | None:
    if element.normal is not None:
        return tuple(float(value) for value in element.normal)  # type: ignore[return-value]
    normal = element.payload.get("normal")
    if isinstance(normal, (list, tuple)) and len(normal) == 3:
        return tuple(float(value) for value in normal)  # type: ignore[return-value]
    return None


def _normalize_vec3(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _nan_vec3() -> tuple[float, float, float]:
    return (float("nan"), float("nan"), float("nan"))


def _nan_matrix3() -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float("nan"), float("nan"), float("nan")),
        (float("nan"), float("nan"), float("nan")),
        (float("nan"), float("nan"), float("nan")),
    )


def _is_nan_vec3(vector: tuple[float, float, float]) -> bool:
    return any(value != value for value in vector)


def _is_matrix3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value)


def _inverse_matrix3(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) <= 1.0e-12:
        return None
    inv_det = 1.0 / det
    return (
        ((e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det),
        ((f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det),
        ((d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det),
    )


def _valid_support_radii(vector: tuple[float, float, float]) -> bool:
    return not _is_nan_vec3(vector) and all(value > 0.0 for value in vector)


def _flat_vec3(values: Sequence[float], index: int) -> tuple[float, float, float]:
    offset = index * 3
    return (float(values[offset]), float(values[offset + 1]), float(values[offset + 2]))


def _flat_matrix3(
    values: Sequence[float],
    index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    offset = index * 9
    return (
        (float(values[offset]), float(values[offset + 1]), float(values[offset + 2])),
        (float(values[offset + 3]), float(values[offset + 4]), float(values[offset + 5])),
        (float(values[offset + 6]), float(values[offset + 7]), float(values[offset + 8])),
    )


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _simulate_ray_aabb_intersect(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> tuple[float, float, tuple[float, float, float]] | None:
    t_min = 0.0
    t_max = CUDA_RENDERER_INF_SENTINEL
    normal = (0.0, 0.0, 0.0)
    for axis in range(3):
        ray_o = origin[axis]
        ray_d = direction[axis]
        lower = box_min[axis]
        upper = box_max[axis]
        if abs(ray_d) < 1.0e-8:
            if ray_o < lower or ray_o > upper:
                return None
            continue
        inv_d = 1.0 / ray_d
        t0 = (lower - ray_o) * inv_d
        t1 = (upper - ray_o) * inv_d
        sign = -1.0
        if t0 > t1:
            t0, t1 = t1, t0
            sign = 1.0
        if t0 > t_min:
            t_min = t0
            normal_values = [0.0, 0.0, 0.0]
            normal_values[axis] = sign
            normal = (normal_values[0], normal_values[1], normal_values[2])
        t_max = min(t_max, t1)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return (t_min, t_max, normal)


def _simulate_ray_plane_intersect(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    plane_point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> tuple[float, float, tuple[float, float, float]] | None:
    if _is_nan_vec3(plane_point) or _is_nan_vec3(normal):
        return None
    denom = sum(direction[axis] * normal[axis] for axis in range(3))
    if abs(denom) < 1.0e-8:
        return None
    depth = sum((plane_point[axis] - origin[axis]) * normal[axis] for axis in range(3)) / denom
    if depth < 0.0:
        return None
    point = tuple(origin[axis] + direction[axis] * depth for axis in range(3))
    if any(point[axis] < box_min[axis] - 1.0e-5 or point[axis] > box_max[axis] + 1.0e-5 for axis in range(3)):
        return None
    return (depth, depth, normal)


def _simulate_ray_beta_ellipsoid_intersect(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    support_radii: tuple[float, float, float],
) -> tuple[float, float, tuple[float, float, float]] | None:
    if _is_nan_vec3(support_radii) or any(value <= 0.0 for value in support_radii):
        return None
    center = tuple((box_min[axis] + box_max[axis]) * 0.5 for axis in range(3))
    scaled_origin = tuple((origin[axis] - center[axis]) / support_radii[axis] for axis in range(3))
    scaled_direction = tuple(direction[axis] / support_radii[axis] for axis in range(3))
    a = sum(value * value for value in scaled_direction)
    b = 2.0 * sum(scaled_origin[axis] * scaled_direction[axis] for axis in range(3))
    c = sum(value * value for value in scaled_origin) - 1.0
    discriminant = b * b - 4.0 * a * c
    if a <= 1.0e-8 or discriminant < 0.0:
        return None
    root = discriminant ** 0.5
    denom = 2.0 * a
    near = (-b - root) / denom
    far = (-b + root) / denom
    entry = max(min(near, far), 0.0)
    exit_depth = max(near, far)
    if exit_depth < entry:
        return None
    normal = _beta_ellipsoid_normal(origin, direction, entry, center, support_radii)
    return (entry, exit_depth, normal)


def _beta_ellipsoid_normal(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    depth: float,
    center: tuple[float, float, float],
    support_radii: tuple[float, float, float],
) -> tuple[float, float, float]:
    point = tuple(origin[axis] + direction[axis] * depth for axis in range(3))
    gradient = tuple((point[axis] - center[axis]) / max(support_radii[axis] * support_radii[axis], 1.0e-12) for axis in range(3))
    try:
        return _normalize_vec3(gradient)
    except ValueError:
        return (0.0, 0.0, 0.0)


def _simulate_ray_gaussian_ellipsoid_intersect(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    mean: tuple[float, float, float],
    inverse_covariance: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    support_radius_sq: float,
) -> tuple[float, float, tuple[float, float, float]] | None:
    if not _valid_gaussian_geometry(mean, inverse_covariance, support_radius_sq):
        return None
    delta = tuple(origin[axis] - mean[axis] for axis in range(3))
    inv_direction = _matvec3(inverse_covariance, direction)
    inv_delta = _matvec3(inverse_covariance, delta)
    a = sum(inv_direction[axis] * direction[axis] for axis in range(3))
    b = 2.0 * sum(inv_delta[axis] * direction[axis] for axis in range(3))
    c = sum(inv_delta[axis] * delta[axis] for axis in range(3)) - support_radius_sq
    discriminant = b * b - 4.0 * a * c
    if a <= 1.0e-8 or discriminant < 0.0:
        return None
    root = discriminant ** 0.5
    denom = 2.0 * a
    near = (-b - root) / denom
    far = (-b + root) / denom
    entry = near if near >= 0.0 else 0.0
    if far < 0.0 or entry < 0.0:
        return None
    normal = _gaussian_ellipsoid_normal(origin, direction, entry, mean, inverse_covariance)
    return (entry, max(far, 0.0), normal)


def _gaussian_ellipsoid_normal(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    depth: float,
    mean: tuple[float, float, float],
    inverse_covariance: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float]:
    point = tuple(origin[axis] + direction[axis] * depth for axis in range(3))
    delta = tuple(point[axis] - mean[axis] for axis in range(3))
    gradient = _matvec3(inverse_covariance, delta)
    try:
        return _normalize_vec3(gradient)
    except ValueError:
        return (0.0, 0.0, 0.0)


def _valid_gaussian_geometry(
    mean: tuple[float, float, float],
    inverse_covariance: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    support_radius_sq: float,
) -> bool:
    return (
        not _is_nan_vec3(mean)
        and not any(value != value for row in inverse_covariance for value in row)
        and support_radius_sq == support_radius_sq
        and support_radius_sq > 0.0
    )


def _matvec3(
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


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
