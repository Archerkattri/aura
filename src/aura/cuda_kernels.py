from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files


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


def cuda_kernel_source_available(path: str) -> bool:
    try:
        resource = files("aura").joinpath(path)
        return bool(resource.is_file())
    except (FileNotFoundError, ModuleNotFoundError):
        return False
