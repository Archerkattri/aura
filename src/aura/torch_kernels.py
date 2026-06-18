from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Sequence

from aura.cuda_kernels import cuda_kernel_extension_status, cuda_kernel_sources


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
            description="Density/path-length transmittance torch autograd kernel; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_volume_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="beta_kernel",
            carrier_id="beta",
            differentiable_fields=("color", "opacity", "alpha", "beta", "support_radius"),
            description="Bounded beta support torch autograd kernel; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_beta_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="gabor_frequency",
            carrier_id="gabor",
            differentiable_fields=("color", "frequency", "phase", "bandwidth"),
            description="Frequency-modulated radiance torch autograd kernel; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_gabor_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="neural_residual",
            carrier_id="neural",
            differentiable_fields=("color", "residual_scale"),
            description="Residual primitive torch autograd kernel for residual confidence scaling; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_neural_residual_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="semantic_feature",
            carrier_id="semantic",
            differentiable_fields=("confidence",),
            description="Semantic/object feature torch autograd kernel for confidence scoring; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_semantic_feature_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="gaussian_fallback",
            carrier_id="gaussian",
            differentiable_fields=("color", "opacity", "confidence"),
            description="Gaussian fallback torch autograd kernel for evidence that does not justify a native carrier; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_gaussian_fallback_kernel",
            autograd_kernel=True,
        ),
    )


def torch_carrier_kernel_report() -> dict:
    specs = torch_carrier_kernel_specs()
    sources_by_carrier = {source.carrier_id: source for source in cuda_kernel_sources()}
    extension_status = cuda_kernel_extension_status(build=False)
    return {
        "format": "AURA_TORCH_CARRIER_KERNEL_REPORT",
        "productionReady": all(spec.production_ready for spec in specs),
        "carrierCount": len(specs),
        "nonProductionCarrierCount": sum(1 for spec in specs if not spec.production_ready),
        "referenceOnlyCarrierCount": sum(1 for spec in specs if not spec.autograd_kernel and not spec.cuda_kernel),
        "autogradCarrierCount": sum(1 for spec in specs if spec.autograd_kernel),
        "cudaCarrierCount": sum(1 for spec in specs if spec.cuda_kernel),
        "cudaSourceCount": len(sources_by_carrier),
        "availableCudaSourceCount": sum(1 for source in sources_by_carrier.values() if source.to_dict()["available"]),
        "cudaExtension": extension_status.to_dict(),
        "kernelSpecs": [_kernel_spec_report(spec, sources_by_carrier.get(spec.carrier_id)) for spec in specs],
        "requiredNextStep": "add carrier-complete CUDA kernels for every autograd-covered carrier",
    }


def _kernel_spec_report(spec: TorchCarrierKernelSpec, source: Any | None) -> dict:
    payload = spec.to_dict()
    payload["cudaSource"] = source.to_dict() if source is not None else None
    return payload


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
    carrier_parameters: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, Any, Any, Any]:
    ray_count = int(best_index.shape[0])
    carrier_colors = torch.zeros((ray_count, 3), dtype=colors.dtype, device=device)
    transmittance = torch.ones((ray_count,), dtype=opacities.dtype, device=device)
    confidence = torch.zeros((ray_count,), dtype=confidences.dtype, device=device)
    residual_by_element = torch.tensor([element.residual for element in elements], dtype=torch.bool, device=device)
    residual = residual_by_element[best_index]

    for element_index, element in enumerate(elements):
        mask = best_index == element_index
        payload_type = element.payload.get("type")
        if payload_type == "surface_cell" or element.carrier_id == "surface":
            surface_color = (
                carrier_parameters[element.id]["color"]
                if carrier_parameters is not None and "color" in carrier_parameters.get(element.id, {})
                else colors[element_index]
            )
            surface_opacity = torch.clamp(
                (
                    carrier_parameters[element.id]["opacity"]
                    if carrier_parameters is not None and "opacity" in carrier_parameters.get(element.id, {})
                    else opacities[element_index]
                ),
                min=0.0,
                max=1.0,
            )
            surface_confidence = torch.clamp(
                (
                    carrier_parameters[element.id]["confidence"]
                    if carrier_parameters is not None and "confidence" in carrier_parameters.get(element.id, {})
                    else confidences[element_index]
                ),
                min=0.0,
                max=1.0,
            )
            carrier_colors[mask] = torch.clamp(surface_color, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - surface_opacity, min=0.0, max=1.0)
            confidence[mask] = surface_confidence
        elif payload_type == "volume_cell":
            volume_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            density = _carrier_parameter(torch, element, "density", carrier_parameters, device, default=element.payload.get("density", element.opacity))
            volume_confidence = torch.clamp(
                _carrier_parameter(torch, element, "confidence", carrier_parameters, device, default=element.confidence),
                min=0.0,
                max=1.0,
            )
            path_length = torch.clamp(exit_depth[mask, element_index] - best_depth[mask], min=0.0)
            carrier_colors[mask] = torch.clamp(volume_color, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(torch.exp(-density * path_length), min=0.0, max=1.0)
            confidence[mask] = volume_confidence
        elif payload_type == "beta_kernel":
            beta_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            beta_opacity = torch.clamp(
                _carrier_parameter(torch, element, "opacity", carrier_parameters, device, default=element.opacity),
                min=0.0,
                max=1.0,
            )
            alpha = _carrier_parameter(torch, element, "alpha", carrier_parameters, device, default=element.payload.get("alpha", 1.0))
            beta_value = _carrier_parameter(torch, element, "beta", carrier_parameters, device, default=element.payload.get("beta", 1.0))
            support_radius = _carrier_vector_parameter(
                torch,
                element,
                "support_radius",
                carrier_parameters,
                device,
                default=element.payload.get("support_radius", _half_extent(element)),
            )
            weight = _torch_beta_weight(
                torch,
                hit_points[mask],
                mins[element_index],
                maxs[element_index],
                support_radius=support_radius,
                alpha=alpha,
                beta=beta_value,
            )
            carrier_colors[mask] = torch.clamp(beta_color, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - beta_opacity * weight, min=0.0, max=1.0)
            confidence[mask] = torch.clamp(confidences[element_index], min=0.0, max=1.0)
        elif payload_type == "gabor_frequency":
            gabor_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            frequency = _carrier_vector_parameter(
                torch,
                element,
                "frequency",
                carrier_parameters,
                device,
                default=element.payload.get("frequency", (0.0, 0.0, 0.0)),
            )
            phase = _carrier_parameter(torch, element, "phase", carrier_parameters, device, default=element.payload.get("phase", 0.0))
            bandwidth = torch.clamp(
                _carrier_parameter(torch, element, "bandwidth", carrier_parameters, device, default=element.payload.get("bandwidth", 1.0)),
                min=0.0,
                max=1.0,
            )
            wave = 0.5 + 0.5 * torch.sin(2.0 * pi * torch.sum(hit_points[mask] * frequency, dim=1) + phase)
            modulation = 1.0 - bandwidth + bandwidth * wave
            carrier_colors[mask] = torch.clamp(gabor_color * modulation.unsqueeze(1), min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - opacities[element_index], min=0.0, max=1.0)
            confidence[mask] = torch.clamp(confidences[element_index] * bandwidth, min=0.0, max=1.0)
        elif payload_type == "neural_residual":
            neural_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            residual_scale = _carrier_parameter(
                torch,
                element,
                "residual_scale",
                carrier_parameters,
                device,
                default=element.payload.get("residual_scale", 0.0),
            )
            residual_strength = torch.clamp(residual_scale, min=0.0, max=1.0)
            carrier_colors[mask] = torch.clamp(neural_color, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - opacities[element_index] * residual_strength, min=0.0, max=1.0)
            confidence[mask] = torch.clamp(confidences[element_index] * (1.0 - residual_strength * 0.25), min=0.0, max=1.0)
            residual[mask] = True
        elif payload_type == "semantic_feature":
            semantic_confidence = _carrier_parameter(
                torch,
                element,
                "confidence",
                carrier_parameters,
                device,
                default=element.payload.get("confidence", element.confidence),
            )
            carrier_colors[mask] = torch.clamp(colors[element_index], min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - opacities[element_index], min=0.0, max=1.0)
            confidence[mask] = torch.clamp(semantic_confidence, min=0.0, max=1.0)
        elif payload_type == "gaussian_fallback":
            gaussian_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            gaussian_opacity = torch.clamp(
                _carrier_parameter(torch, element, "opacity", carrier_parameters, device, default=element.opacity),
                min=0.0,
                max=1.0,
            )
            gaussian_confidence = torch.clamp(
                _carrier_parameter(torch, element, "confidence", carrier_parameters, device, default=element.confidence),
                min=0.0,
                max=1.0,
            )
            covariance_diag = _carrier_vector_parameter(
                torch,
                element,
                "gaussian_covariance_diag",
                carrier_parameters,
                device,
                default=_gaussian_covariance_diag(element),
            )
            gaussian_weight = _torch_gaussian_weight(torch, hit_points[mask], element, device, covariance_diag=covariance_diag)
            carrier_colors[mask] = torch.clamp(gaussian_color, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - gaussian_opacity * gaussian_weight, min=0.0, max=1.0)
            confidence[mask] = torch.clamp(gaussian_confidence * gaussian_weight, min=0.0, max=1.0)

    return carrier_colors, transmittance, confidence, residual


def torch_carrier_parameter_tensors(
    torch: Any,
    elements: Sequence[Any],
    *,
    device: str,
    requires_grad: bool = True,
) -> dict[str, dict[str, Any]]:
    parameters: dict[str, dict[str, Any]] = {}
    for element in elements:
        payload_type = element.payload.get("type")
        geometry_parameters = _carrier_geometry_parameter_tensors(torch, element, device=device, requires_grad=requires_grad)
        if payload_type == "surface_cell" or element.carrier_id == "surface":
            parameters[element.id] = {
                **geometry_parameters,
                "color": torch.tensor(
                    tuple(element.payload.get("color", element.color)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "opacity": torch.tensor(
                    float(element.payload.get("opacity", element.opacity)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "confidence": torch.tensor(
                    float(element.payload.get("confidence", element.confidence)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
            }
        elif payload_type == "volume_cell":
            parameters[element.id] = {
                **geometry_parameters,
                "color": torch.tensor(
                    tuple(element.payload.get("color", element.color)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "density": torch.tensor(
                    float(element.payload.get("density", element.opacity)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "confidence": torch.tensor(
                    float(element.payload.get("confidence", element.confidence)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
            }
        elif payload_type == "beta_kernel":
            parameters[element.id] = {
                **geometry_parameters,
                "color": torch.tensor(
                    tuple(element.payload.get("color", element.color)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "opacity": torch.tensor(
                    float(element.payload.get("opacity", element.opacity)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "alpha": torch.tensor(
                    float(element.payload.get("alpha", 1.0)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "beta": torch.tensor(
                    float(element.payload.get("beta", 1.0)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "support_radius": torch.tensor(
                    tuple(element.payload.get("support_radius", _half_extent(element))),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
            }
        elif payload_type == "gabor_frequency":
            parameters[element.id] = {
                **geometry_parameters,
                "color": torch.tensor(
                    tuple(element.payload.get("color", element.color)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "frequency": torch.tensor(
                    tuple(element.payload.get("frequency", (0.0, 0.0, 0.0))),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "phase": torch.tensor(
                    float(element.payload.get("phase", 0.0)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "bandwidth": torch.tensor(
                    float(element.payload.get("bandwidth", 1.0)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
            }
        elif payload_type == "neural_residual":
            parameters[element.id] = {
                **geometry_parameters,
                "color": torch.tensor(
                    tuple(element.payload.get("color", element.color)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "residual_scale": torch.tensor(
                    float(element.payload.get("residual_scale", 0.0)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                )
            }
        elif payload_type == "semantic_feature":
            parameters[element.id] = {
                **geometry_parameters,
                "confidence": torch.tensor(
                    float(element.payload.get("confidence", element.confidence)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                )
            }
        elif payload_type == "gaussian_fallback":
            parameters[element.id] = {
                **geometry_parameters,
                "color": torch.tensor(
                    tuple(element.payload.get("color", element.color)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "opacity": torch.tensor(
                    float(element.payload.get("opacity", element.opacity)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "confidence": torch.tensor(
                    float(element.payload.get("confidence", element.confidence)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                "gaussian_covariance_diag": torch.tensor(
                    _gaussian_covariance_diag(element),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
            }
    return parameters


def _carrier_geometry_parameter_tensors(
    torch: Any,
    element: Any,
    *,
    device: str,
    requires_grad: bool,
) -> dict[str, Any]:
    parameters = {
        "min_corner": torch.tensor(
            tuple(element.bounds.min_corner),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        ),
        "max_corner": torch.tensor(
            tuple(element.bounds.max_corner),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        ),
    }
    if element.payload.get("type") == "surface_cell" or element.carrier_id == "surface":
        point = element.payload.get("plane_point") or element.payload.get("point")
        if isinstance(point, (list, tuple)) and len(point) == 3:
            plane_point = tuple(float(value) for value in point)
        else:
            plane_point = _surface_plane_point(element)
        parameters["plane_point"] = torch.tensor(
            plane_point,
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
    if element.payload.get("type") == "gabor_frequency" or element.carrier_id == "gabor":
        point = element.payload.get("plane_point") or element.payload.get("point")
        if isinstance(point, (list, tuple)) and len(point) == 3:
            plane_point = tuple(float(value) for value in point)
        else:
            plane_point = _center_point(element)
        parameters["plane_point"] = torch.tensor(
            plane_point,
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
    if element.payload.get("type") == "gaussian_fallback":
        mean = element.payload.get("mean")
        if isinstance(mean, (list, tuple)) and len(mean) == 3:
            parameters["gaussian_mean"] = torch.tensor(
                tuple(float(value) for value in mean),
                dtype=torch.float32,
                device=device,
                requires_grad=requires_grad,
            )
    return parameters


def _surface_plane_point(element: Any) -> tuple[float, float, float]:
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    center = [(lo + hi) * 0.5 for lo, hi in zip(min_corner, max_corner)]
    normal = element.normal or element.payload.get("normal")
    if isinstance(normal, (list, tuple)) and len(normal) == 3:
        normal = tuple(float(value) for value in normal)
        dominant_axis = max(range(3), key=lambda index: abs(normal[index]))
        center[dominant_axis] = min_corner[dominant_axis] if normal[dominant_axis] < 0.0 else max_corner[dominant_axis]
    return tuple(center)  # type: ignore[return-value]


def _center_point(element: Any) -> tuple[float, float, float]:
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    return tuple((lo + hi) * 0.5 for lo, hi in zip(min_corner, max_corner))  # type: ignore[return-value]


def _half_extent(element: Any) -> tuple[float, float, float]:
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    return tuple(
        max((max_corner[index] - min_corner[index]) * 0.5, 1e-4) for index in range(3)
    )  # type: ignore[return-value]


def _carrier_parameter(
    torch: Any,
    element: Any,
    name: str,
    carrier_parameters: dict[str, dict[str, Any]] | None,
    device: str,
    *,
    default: object,
) -> Any:
    if carrier_parameters is not None:
        parameter = carrier_parameters.get(element.id, {}).get(name)
        if parameter is not None:
            return parameter
    return torch.tensor(float(default), dtype=torch.float32, device=device)


def _carrier_vector_parameter(
    torch: Any,
    element: Any,
    name: str,
    carrier_parameters: dict[str, dict[str, Any]] | None,
    device: str,
    *,
    default: object,
) -> Any:
    if carrier_parameters is not None:
        parameter = carrier_parameters.get(element.id, {}).get(name)
        if parameter is not None:
            return parameter
    return torch.tensor(tuple(default), dtype=torch.float32, device=device)


def _torch_beta_weight(torch: Any, points: Any, mins: Any, maxs: Any, *, support_radius: Any, alpha: Any, beta: Any) -> Any:
    center = (mins + maxs) * 0.5
    radius = torch.clamp(support_radius, min=1e-6)
    normalized_distance = torch.clamp(torch.abs(points - center) / radius, min=0.0, max=1.0)
    u = torch.mean(1.0 - normalized_distance, dim=1)
    alpha_safe = torch.clamp(alpha, min=1e-6)
    beta_safe = torch.clamp(beta, min=1e-6)
    raw = (u ** (alpha_safe - 1.0)) * ((1.0 - u) ** (beta_safe - 1.0))
    normalize = (alpha_safe > 1.0) & (beta_safe > 1.0)
    raw_mode = (alpha_safe - 1.0) / torch.clamp(alpha_safe + beta_safe - 2.0, min=1e-6)
    mode = torch.where(normalize, raw_mode, torch.full_like(raw_mode, 0.5))
    peak = (mode ** (alpha_safe - 1.0)) * ((1.0 - mode) ** (beta_safe - 1.0))
    safe_peak = torch.where((peak > 0.0) & normalize, peak, torch.ones_like(peak))
    raw = torch.where(normalize, raw / safe_peak, raw)
    return torch.clamp(raw, min=0.0, max=1.0)


def _torch_gaussian_weight(torch: Any, points: Any, element: Any, device: str, *, covariance_diag: Any | None = None) -> Any:
    mean = element.payload.get("mean")
    if covariance_diag is None:
        covariance_diag = torch.tensor(_gaussian_covariance_diag(element), dtype=torch.float32, device=device)
    if not isinstance(mean, (list, tuple)) or len(mean) != 3:
        return torch.ones((int(points.shape[0]),), dtype=torch.float32, device=device)
    mean_tensor = torch.tensor(tuple(float(item) for item in mean), dtype=torch.float32, device=device)
    safe_covariance_diag = torch.clamp(covariance_diag, min=1e-6)
    delta = points - mean_tensor
    mahalanobis = torch.sum((delta * delta) / safe_covariance_diag.unsqueeze(0), dim=1)
    return torch.clamp(torch.exp(-0.5 * torch.clamp(mahalanobis, min=0.0)), min=0.0, max=1.0)


def _gaussian_covariance_diag(element: Any) -> tuple[float, float, float]:
    covariance = element.payload.get("covariance")
    if _is_matrix3(covariance):
        return tuple(max(float(covariance[index][index]), 1e-6) for index in range(3))  # type: ignore[return-value]
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    return tuple(max(((max_corner[index] - min_corner[index]) * 0.25) ** 2, 1e-6) for index in range(3))  # type: ignore[return-value]


def _is_matrix3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value)
