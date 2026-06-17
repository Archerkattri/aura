from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Sequence


@dataclass(frozen=True)
class TorchCarrierKernelSpec:
    payload_type: str
    carrier_id: str
    differentiable_fields: tuple[str, ...]
    description: str
    implementation_stage: str = "reference_torch_payload_kernel"
    autograd_kernel: bool = False
    cuda_kernel: bool = False

    def to_dict(self) -> dict:
        return {
            "payloadType": self.payload_type,
            "carrierId": self.carrier_id,
            "differentiableFields": list(self.differentiable_fields),
            "description": self.description,
            "implementationStage": self.implementation_stage,
            "autogradKernel": self.autograd_kernel,
            "cudaKernel": self.cuda_kernel,
            "productionReady": self.production_ready,
            "blockers": list(self.blockers),
        }

    @property
    def production_ready(self) -> bool:
        return self.autograd_kernel and self.cuda_kernel

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers = []
        if not self.autograd_kernel:
            blockers.append("missing_autograd_kernel")
        if not self.cuda_kernel:
            blockers.append("missing_cuda_kernel")
        return tuple(blockers)


def torch_carrier_kernel_specs() -> tuple[TorchCarrierKernelSpec, ...]:
    return (
        TorchCarrierKernelSpec(
            payload_type="surface_cell",
            carrier_id="surface",
            differentiable_fields=("color", "opacity", "confidence"),
            description="Opaque bounded radiance surface cell torch autograd kernel; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_surface_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="volume_cell",
            carrier_id="volume",
            differentiable_fields=("color", "density", "confidence"),
            description="Density/path-length transmittance reference kernel.",
        ),
        TorchCarrierKernelSpec(
            payload_type="beta_kernel",
            carrier_id="beta",
            differentiable_fields=("color", "opacity", "alpha", "beta"),
            description="Bounded beta support kernel for compact detail.",
        ),
        TorchCarrierKernelSpec(
            payload_type="gabor_frequency",
            carrier_id="gabor",
            differentiable_fields=("color", "frequency", "phase", "bandwidth"),
            description="Frequency-modulated radiance kernel.",
        ),
        TorchCarrierKernelSpec(
            payload_type="neural_residual",
            carrier_id="neural",
            differentiable_fields=("color", "residual_scale"),
            description="Residual primitive confidence/residual-flag reference kernel.",
        ),
        TorchCarrierKernelSpec(
            payload_type="semantic_feature",
            carrier_id="semantic",
            differentiable_fields=("confidence",),
            description="Semantic/object feature confidence reference kernel.",
        ),
        TorchCarrierKernelSpec(
            payload_type="gaussian_fallback",
            carrier_id="gaussian",
            differentiable_fields=("color", "opacity", "confidence"),
            description="Gaussian fallback path for evidence that does not justify a native carrier.",
        ),
    )


def torch_carrier_kernel_report() -> dict:
    specs = torch_carrier_kernel_specs()
    return {
        "format": "AURA_TORCH_CARRIER_KERNEL_REPORT",
        "productionReady": all(spec.production_ready for spec in specs),
        "carrierCount": len(specs),
        "referenceOnlyCarrierCount": sum(1 for spec in specs if not spec.production_ready),
        "autogradCarrierCount": sum(1 for spec in specs if spec.autograd_kernel),
        "cudaCarrierCount": sum(1 for spec in specs if spec.cuda_kernel),
        "kernelSpecs": [spec.to_dict() for spec in specs],
        "requiredNextStep": "replace reference torch payload kernels with carrier-complete autograd/CUDA kernels",
    }


def torch_carrier_response_tensors(
    torch: Any,
    elements: Sequence[Any],
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
