from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any
from importlib.resources import files
from importlib.resources.abc import Traversable


@dataclass(frozen=True)
class CudaKernelSource:
    carrier_id: str
    symbol: str
    path: str
    required: bool = True

    def to_dict(self) -> dict:
        return {
            "carrierId": self.carrier_id,
            "symbol": self.symbol,
            "path": self.path,
            "required": self.required,
            "available": cuda_kernel_source_available(self.path),
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


def cuda_kernel_sources() -> tuple[CudaKernelSource, ...]:
    return (
        CudaKernelSource("surface", "aura_surface_forward_kernel", "cuda/aura_carriers.cu"),
        CudaKernelSource("volume", "aura_volume_forward_kernel", "cuda/aura_carriers.cu"),
        CudaKernelSource("beta", "aura_beta_forward_kernel", "cuda/aura_carriers.cu"),
        CudaKernelSource("gabor", "aura_gabor_forward_kernel", "cuda/aura_carriers.cu"),
        CudaKernelSource("neural", "aura_neural_forward_kernel", "cuda/aura_carriers.cu"),
        CudaKernelSource("semantic", "aura_semantic_forward_kernel", "cuda/aura_carriers.cu"),
        CudaKernelSource("gaussian", "aura_gaussian_forward_kernel", "cuda/aura_carriers.cu"),
    )


def cuda_kernel_source_report() -> dict:
    sources = cuda_kernel_sources()
    return {
        "format": "AURA_CUDA_KERNEL_SOURCE_REPORT",
        "sourceCount": len(sources),
        "availableSourceCount": sum(1 for source in sources if cuda_kernel_source_available(source.path)),
        "sources": [source.to_dict() for source in sources],
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


def cuda_kernel_source_available(path: str) -> bool:
    try:
        resource = files("aura").joinpath(path)
        return bool(resource.is_file())
    except (FileNotFoundError, ModuleNotFoundError):
        return False


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
