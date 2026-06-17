from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Sequence
from importlib.resources import files
from importlib.resources.abc import Traversable


@dataclass(frozen=True)
class CudaKernelArgument:
    name: str
    dtype: str
    role: str
    shape: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role,
            "shape": self.shape,
        }


@dataclass(frozen=True)
class CudaKernelSource:
    carrier_id: str
    payload_type: str
    symbol: str
    path: str
    arguments: tuple[CudaKernelArgument, ...]
    contract_outputs: tuple[str, ...]
    required: bool = True

    def to_dict(self) -> dict:
        missing_fragments = cuda_kernel_source_missing_fragments(self)
        return {
            "carrierId": self.carrier_id,
            "payloadType": self.payload_type,
            "symbol": self.symbol,
            "path": self.path,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "contractOutputs": list(self.contract_outputs),
            "required": self.required,
            "available": cuda_kernel_source_available(self.path),
            "sourceSymbolAvailable": not missing_fragments and cuda_kernel_source_available(self.path),
            "contractComplete": not missing_fragments,
            "missingSourceFragments": missing_fragments,
        }


@dataclass(frozen=True)
class CudaExtensionStatus:
    available: bool
    build_attempted: bool
    compiled: bool
    loadable: bool
    module_name: str
    source_paths: tuple[str, ...]
    symbols: tuple[str, ...]
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "buildAttempted": self.build_attempted,
            "compiled": self.compiled,
            "loadable": self.loadable,
            "moduleName": self.module_name,
            "sourcePaths": list(self.source_paths),
            "symbols": list(self.symbols),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CudaRendererLaunchReport:
    api_name: str
    ray_count: int | None
    validated_inputs: bool
    available: bool
    production_ready: bool
    reason: str
    extension: CudaExtensionStatus
    contract: dict

    def to_dict(self) -> dict:
        return {
            "format": "AURA_CUDA_RENDERER_LAUNCH_REPORT",
            "apiName": self.api_name,
            "rayCount": self.ray_count,
            "validatedInputs": self.validated_inputs,
            "available": self.available,
            "productionReady": self.production_ready,
            "reason": self.reason,
            "extension": self.extension.to_dict(),
            "contract": self.contract,
        }


def cuda_kernel_sources() -> tuple[CudaKernelSource, ...]:
    return (
        CudaKernelSource(
            "surface",
            "surface_cell",
            "aura_surface_forward_kernel",
            "cuda/aura_carriers.cu",
            _kernel_arguments("color", "opacity", "confidence"),
            _contract_outputs(),
        ),
        CudaKernelSource(
            "volume",
            "volume_cell",
            "aura_volume_forward_kernel",
            "cuda/aura_carriers.cu",
            _kernel_arguments("color", "density", "path_length", "confidence"),
            _contract_outputs(),
        ),
        CudaKernelSource(
            "beta",
            "beta_kernel",
            "aura_beta_forward_kernel",
            "cuda/aura_carriers.cu",
            _kernel_arguments("color", "opacity", "confidence", "alpha", "beta", "u"),
            _contract_outputs(),
        ),
        CudaKernelSource(
            "gabor",
            "gabor_frequency",
            "aura_gabor_forward_kernel",
            "cuda/aura_carriers.cu",
            _kernel_arguments("color", "opacity", "confidence", "frequency", "phase", "bandwidth", "hit_point"),
            _contract_outputs(),
        ),
        CudaKernelSource(
            "neural",
            "neural_residual",
            "aura_neural_forward_kernel",
            "cuda/aura_carriers.cu",
            _kernel_arguments("color", "opacity", "confidence", "residual_scale"),
            _contract_outputs(),
        ),
        CudaKernelSource(
            "semantic",
            "semantic_feature",
            "aura_semantic_forward_kernel",
            "cuda/aura_carriers.cu",
            _kernel_arguments("color", "opacity", "confidence"),
            _contract_outputs(),
        ),
        CudaKernelSource(
            "gaussian",
            "gaussian_fallback",
            "aura_gaussian_forward_kernel",
            "cuda/aura_carriers.cu",
            _kernel_arguments("color", "opacity", "confidence"),
            _contract_outputs(),
        ),
    )


def cuda_kernel_source_report() -> dict:
    sources = cuda_kernel_sources()
    source_payloads = [source.to_dict() for source in sources]
    return {
        "format": "AURA_CUDA_KERNEL_SOURCE_REPORT",
        "sourceCount": len(sources),
        "availableSourceCount": sum(1 for source in sources if cuda_kernel_source_available(source.path)),
        "contractCompleteSourceCount": sum(1 for source in source_payloads if source["contractComplete"]),
        "contractOutputs": list(_contract_outputs()),
        "sources": source_payloads,
    }


def cuda_kernel_extension_status(*, build: bool = False, verbose: bool = False) -> CudaExtensionStatus:
    sources = cuda_kernel_sources()
    source_paths = tuple(sorted({source.path for source in sources}))
    symbols = tuple(source.symbol for source in sources)
    module_name = "aura_cuda_carriers"
    if not build:
        return CudaExtensionStatus(
            available=False,
            build_attempted=False,
            compiled=False,
            loadable=False,
            module_name=module_name,
            source_paths=source_paths,
            symbols=symbols,
            reason="build_not_attempted",
        )
    if find_spec("torch") is None:
        return CudaExtensionStatus(
            available=False,
            build_attempted=True,
            compiled=False,
            loadable=False,
            module_name=module_name,
            source_paths=source_paths,
            symbols=symbols,
            reason="torch_unavailable",
        )
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
    except Exception as exc:  # pragma: no cover - depends on optional torch install state.
        return _cuda_extension_failure(module_name, source_paths, symbols, f"torch_extension_unavailable: {exc}", build_attempted=True)
    if CUDA_HOME is None:
        return _cuda_extension_failure(module_name, source_paths, symbols, "cuda_home_unavailable", build_attempted=True)
    if not bool(torch.cuda.is_available()):
        return _cuda_extension_failure(module_name, source_paths, symbols, "torch_cuda_unavailable", build_attempted=True)
    try:
        with ExitStack() as stack:
            resolved_sources = [
                str(stack.enter_context(_resource_as_file(files("aura").joinpath(path))))
                for path in source_paths
            ]
            load(
                name=module_name,
                sources=resolved_sources,
                with_cuda=True,
                is_python_module=False,
                verbose=verbose,
            )
    except Exception as exc:  # pragma: no cover - requires a CUDA compiler/runtime matrix.
        return _cuda_extension_failure(module_name, source_paths, symbols, f"build_or_load_failed: {exc}", build_attempted=True)
    return CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name=module_name,
        source_paths=source_paths,
        symbols=symbols,
    )


def cuda_kernel_extension_report(*, build: bool = False, verbose: bool = False) -> dict:
    status = cuda_kernel_extension_status(build=build, verbose=verbose)
    return {
        "format": "AURA_CUDA_EXTENSION_REPORT",
        "productionReady": status.available and status.compiled and status.loadable,
        **status.to_dict(),
    }


def cuda_renderer_api_contract() -> dict:
    return {
        "format": "AURA_CUDA_RENDERER_API_CONTRACT",
        "apiName": "cuda_render_rays",
        "productionReady": False,
        "description": "Callable CUDA renderer launch contract for batched AURA ray queries.",
        "batchDimension": "rayCount",
        "inputTensors": [
            _renderer_tensor("ray_origins", "float32", "rayCount x 3", "input", "World-space ray origins."),
            _renderer_tensor("ray_directions", "float32", "rayCount x 3", "input", "World-space ray directions."),
            _renderer_tensor("element_mins", "float32", "elementCount x 3", "scene", "Element AABB minimum corners."),
            _renderer_tensor("element_maxs", "float32", "elementCount x 3", "scene", "Element AABB maximum corners."),
            _renderer_tensor("carrier_ids", "int32", "elementCount", "scene", "Native carrier identifier per element."),
            _renderer_tensor("colors", "float32", "elementCount x 3", "scene", "Base carrier color parameters."),
            _renderer_tensor("opacities", "float32", "elementCount", "scene", "Base carrier opacity parameters."),
            _renderer_tensor("confidences", "float32", "elementCount", "scene", "Base carrier confidence parameters."),
            _renderer_tensor("carrier_parameters", "float32", "carrierParameterCount", "scene", "Packed carrier-specific parameter buffer."),
            _renderer_tensor("chunk_indices", "int32", "chunkIndexCount", "scene", "Packed traversal/culling index buffer."),
        ],
        "outputTensors": [
            _renderer_tensor("out_color", "float32", "rayCount x 3", "output", "Composited RGB color."),
            _renderer_tensor("out_alpha", "float32", "rayCount", "output", "Composited opacity."),
            _renderer_tensor("out_transmittance", "float32", "rayCount", "output", "Final ray transmittance."),
            _renderer_tensor("out_depth", "float32", "rayCount", "output", "First-hit depth or infinity sentinel."),
            _renderer_tensor("out_normal", "float32", "rayCount x 3", "output", "First-hit normal or zero vector."),
            _renderer_tensor("out_confidence", "float32", "rayCount", "output", "Composited confidence."),
            _renderer_tensor("out_residual", "bool", "rayCount", "output", "Residual/view-dependent carrier flag."),
            _renderer_tensor("out_material_id", "int32", "rayCount", "output", "First-hit material dictionary index."),
            _renderer_tensor("out_semantic_id", "int32", "rayCount", "output", "First-hit semantic dictionary index."),
            _renderer_tensor("ordered_hits", "int32", "rayCount x maxHits", "output", "Ordered carrier hit trace indices for CPU/torch parity checks."),
        ],
        "sourceReport": cuda_kernel_source_report(),
        "extension": cuda_kernel_extension_status(build=False).to_dict(),
        "unavailableUntil": [
            "CUDA extension is compiled from packaged sources",
            "compiled extension is loadable in the current process",
            "renderer binding dispatches cuda_render_rays over batched ray tensors",
            "CPU/torch/CUDA parity and benchmark tests pass",
        ],
    }


def cuda_renderer_report() -> dict:
    return cuda_render_rays().to_dict()


def cuda_render_rays(
    ray_origins: Sequence[Sequence[float]] | Any | None = None,
    ray_directions: Sequence[Sequence[float]] | Any | None = None,
    *,
    scene_tensors: Any | None = None,
) -> CudaRendererLaunchReport:
    """Validate the CUDA renderer call shape and report why launch is unavailable.

    This function intentionally does not compile or load CUDA. It is a stable
    callable contract for future extension bindings and remains safe on CPU-only
    machines.
    """

    del scene_tensors
    extension = cuda_kernel_extension_status(build=False)
    contract = cuda_renderer_api_contract()
    try:
        ray_count = _validate_batched_rays(ray_origins, ray_directions)
    except ValueError as exc:
        return CudaRendererLaunchReport(
            api_name="cuda_render_rays",
            ray_count=None,
            validated_inputs=False,
            available=False,
            production_ready=False,
            reason=f"invalid_batched_ray_inputs: {exc}",
            extension=extension,
            contract=contract,
        )
    reason = "extension_not_compiled_or_loadable" if extension.reason == "build_not_attempted" else extension.reason
    return CudaRendererLaunchReport(
        api_name="cuda_render_rays",
        ray_count=ray_count,
        validated_inputs=True,
        available=False,
        production_ready=False,
        reason=reason or "extension_not_compiled_or_loadable",
        extension=extension,
        contract=contract,
    )


def cuda_kernel_source_available(path: str) -> bool:
    try:
        resource = files("aura").joinpath(path)
        return bool(resource.is_file())
    except (FileNotFoundError, ModuleNotFoundError):
        return False


def cuda_kernel_source_missing_fragments(source: CudaKernelSource) -> list[str]:
    source_text = _cuda_kernel_source_text(source.path)
    if source_text is None:
        return [source.symbol, *source.contract_outputs]

    expected_fragments = [
        f"extern \"C\" __global__ void {source.symbol}",
        *source.contract_outputs,
    ]
    return [fragment for fragment in expected_fragments if fragment not in source_text]


def _kernel_arguments(*inputs: str) -> tuple[CudaKernelArgument, ...]:
    input_arguments = tuple(CudaKernelArgument(name, _input_dtype(name), "input", _input_shape(name)) for name in inputs)
    output_arguments = (
        CudaKernelArgument("out_color", "float*", "output", "count x 3"),
        CudaKernelArgument("out_transmittance", "float*", "output", "count"),
        CudaKernelArgument("out_confidence", "float*", "output", "count"),
        CudaKernelArgument("out_residual", "unsigned char*", "output", "count"),
        CudaKernelArgument("count", "int", "size", "scalar"),
    )
    return (*input_arguments, *output_arguments)


def _contract_outputs() -> tuple[str, ...]:
    return ("out_color", "out_transmittance", "out_confidence", "out_residual")


def _input_dtype(name: str) -> str:
    return "const float*"


def _input_shape(name: str) -> str:
    if name in {"color", "frequency", "hit_point"}:
        return "count x 3"
    return "count"


def _renderer_tensor(name: str, dtype: str, shape: str, role: str, description: str) -> dict:
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "role": role,
        "description": description,
    }


def _validate_batched_rays(ray_origins: Any | None, ray_directions: Any | None) -> int | None:
    origin_count = _batched_vec3_count(ray_origins, "ray_origins")
    direction_count = _batched_vec3_count(ray_directions, "ray_directions")
    if origin_count is None and direction_count is None:
        return None
    if origin_count is None or direction_count is None:
        raise ValueError("ray_origins and ray_directions must be provided together")
    if origin_count != direction_count:
        raise ValueError(f"ray_origins count {origin_count} does not match ray_directions count {direction_count}")
    return origin_count


def _batched_vec3_count(values: Any | None, name: str) -> int | None:
    if values is None:
        return None
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


def _cuda_kernel_source_text(path: str) -> str | None:
    try:
        resource = files("aura").joinpath(path)
        if not resource.is_file():
            return None
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def _resource_as_file(resource: Traversable) -> Any:
    from importlib.resources import as_file

    return as_file(resource)


def _cuda_extension_failure(
    module_name: str,
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
        module_name=module_name,
        source_paths=source_paths,
        symbols=symbols,
        reason=reason,
    )
