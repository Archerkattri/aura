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
            differentiable_fields=("color", "opacity", "confidence", "alpha", "beta", "support_radius"),
            description="Bounded beta support torch autograd kernel; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_beta_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="gabor_frequency",
            carrier_id="gabor",
            differentiable_fields=("color", "opacity", "confidence", "frequency", "phase", "bandwidth"),
            description="Frequency-modulated radiance torch autograd kernel; CUDA production kernel is still required.",
            implementation_stage="torch_autograd_gabor_kernel",
            autograd_kernel=True,
        ),
        TorchCarrierKernelSpec(
            payload_type="neural_residual",
            carrier_id="neural",
            differentiable_fields=("color", "opacity", "confidence", "residual_scale"),
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
            differentiable_fields=("color", "opacity", "confidence", "gaussian_mean", "gaussian_covariance_diag"),
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


def torch_carrier_response_tensors_batched(
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
    """Evaluate native carrier responses with tensor gather/scatter math."""

    element_count = len(elements)
    if element_count == 0:
        raise ValueError("batched carrier response requires at least one element")
    payload_types = tuple(str(element.payload.get("type", "")) for element in elements)
    carrier_ids = tuple(str(element.carrier_id) for element in elements)
    surface_mask = torch.tensor(
        [payload_type == "surface_cell" or carrier_id == "surface" for payload_type, carrier_id in zip(payload_types, carrier_ids)],
        dtype=torch.bool,
        device=device,
    )
    volume_mask = torch.tensor([payload_type == "volume_cell" for payload_type in payload_types], dtype=torch.bool, device=device)
    beta_mask = torch.tensor([payload_type == "beta_kernel" for payload_type in payload_types], dtype=torch.bool, device=device)
    gabor_mask = torch.tensor([payload_type == "gabor_frequency" for payload_type in payload_types], dtype=torch.bool, device=device)
    neural_mask = torch.tensor([payload_type == "neural_residual" for payload_type in payload_types], dtype=torch.bool, device=device)
    semantic_mask = torch.tensor([payload_type == "semantic_feature" for payload_type in payload_types], dtype=torch.bool, device=device)
    gaussian_mask = torch.tensor([payload_type == "gaussian_fallback" for payload_type in payload_types], dtype=torch.bool, device=device)

    all_colors = torch.clamp(_stack_vector_parameter(torch, elements, "color", carrier_parameters, device, defaults=tuple(element.color for element in elements)), min=0.0, max=1.0)
    all_opacities = torch.clamp(_stack_scalar_parameter(torch, elements, "opacity", carrier_parameters, device, defaults=tuple(element.opacity for element in elements)), min=0.0, max=1.0)
    all_confidences = torch.clamp(
        _stack_scalar_parameter(torch, elements, "confidence", carrier_parameters, device, defaults=tuple(element.confidence for element in elements)),
        min=0.0,
        max=1.0,
    )
    selected_color = all_colors[best_index]
    selected_opacity = all_opacities[best_index]
    selected_confidence = all_confidences[best_index]
    selected_payload_volume = volume_mask[best_index]
    selected_payload_beta = beta_mask[best_index]
    selected_payload_gabor = gabor_mask[best_index]
    selected_payload_neural = neural_mask[best_index]
    selected_payload_semantic = semantic_mask[best_index]
    selected_payload_gaussian = gaussian_mask[best_index]

    carrier_colors = selected_color
    transmittance = torch.clamp(1.0 - selected_opacity, min=0.0, max=1.0)
    confidence = selected_confidence
    residual_by_element = torch.tensor([element.residual for element in elements], dtype=torch.bool, device=device)
    residual = residual_by_element[best_index]

    if any(payload_type == "volume_cell" for payload_type in payload_types):
        densities = _stack_scalar_parameter(
            torch,
            elements,
            "density",
            carrier_parameters,
            device,
            defaults=tuple(element.payload.get("density", element.opacity) for element in elements),
        )
        volume_opacities = torch.clamp(
            _stack_scalar_parameter(
                torch,
                elements,
                "opacity",
                carrier_parameters,
                device,
                defaults=tuple(element.payload.get("opacity", 1.0) for element in elements),
            ),
            min=0.0,
            max=1.0,
        )
        selected_exit_depth = (
            exit_depth if exit_depth.dim() == 1 else exit_depth.gather(1, best_index.unsqueeze(1)).squeeze(1)
        )
        path_length = torch.clamp(selected_exit_depth - best_depth, min=0.0)
        alpha = volume_opacities[best_index] * (1.0 - torch.exp(-densities[best_index] * path_length))
        transmittance = torch.where(selected_payload_volume, torch.clamp(1.0 - alpha, min=0.0, max=1.0), transmittance)

    if any(payload_type == "beta_kernel" for payload_type in payload_types):
        beta_opacities = torch.clamp(
            _stack_scalar_parameter(
                torch,
                elements,
                "opacity",
                carrier_parameters,
                device,
                defaults=tuple(element.opacity for element in elements),
            ),
            min=0.0,
            max=1.0,
        )
        alpha = _stack_scalar_parameter(torch, elements, "alpha", carrier_parameters, device, defaults=tuple(element.payload.get("alpha", 1.0) for element in elements))
        beta_value = _stack_scalar_parameter(torch, elements, "beta", carrier_parameters, device, defaults=tuple(element.payload.get("beta", 1.0) for element in elements))
        support_radius = _stack_vector_parameter(
            torch,
            elements,
            "support_radius",
            carrier_parameters,
            device,
            defaults=tuple(element.payload.get("support_radius", _half_extent(element)) for element in elements),
        )
        beta_weight = _torch_beta_weight_batched(
            torch,
            hit_points,
            mins[best_index],
            maxs[best_index],
            support_radius=support_radius[best_index],
            alpha=alpha[best_index],
            beta=beta_value[best_index],
        )
        transmittance = torch.where(
            selected_payload_beta,
            torch.clamp(1.0 - beta_opacities[best_index] * beta_weight, min=0.0, max=1.0),
            transmittance,
        )

    if any(payload_type == "gabor_frequency" for payload_type in payload_types):
        gabor_opacities = torch.clamp(
            _stack_scalar_parameter(
                torch,
                elements,
                "opacity",
                carrier_parameters,
                device,
                defaults=tuple(element.opacity for element in elements),
            ),
            min=0.0,
            max=1.0,
        )
        frequencies = _stack_vector_parameter(
            torch,
            elements,
            "frequency",
            carrier_parameters,
            device,
            defaults=tuple(element.payload.get("frequency", (0.0, 0.0, 0.0)) for element in elements),
        )
        phases = _stack_scalar_parameter(torch, elements, "phase", carrier_parameters, device, defaults=tuple(element.payload.get("phase", 0.0) for element in elements))
        bandwidths = torch.clamp(
            _stack_scalar_parameter(torch, elements, "bandwidth", carrier_parameters, device, defaults=tuple(element.payload.get("bandwidth", 1.0) for element in elements)),
            min=0.0,
            max=1.0,
        )
        selected_bandwidth = bandwidths[best_index]
        wave = 0.5 + 0.5 * torch.sin(2.0 * pi * torch.sum(hit_points * frequencies[best_index], dim=1) + phases[best_index])
        modulation = 1.0 - selected_bandwidth + selected_bandwidth * wave
        carrier_colors = torch.where(
            selected_payload_gabor.unsqueeze(1),
            torch.clamp(all_colors[best_index] * modulation.unsqueeze(1), min=0.0, max=1.0),
            carrier_colors,
        )
        transmittance = torch.where(selected_payload_gabor, torch.clamp(1.0 - gabor_opacities[best_index], min=0.0, max=1.0), transmittance)
        confidence = torch.where(selected_payload_gabor, torch.clamp(all_confidences[best_index] * selected_bandwidth, min=0.0, max=1.0), confidence)

    if any(payload_type == "neural_residual" for payload_type in payload_types):
        residual_scales = torch.clamp(
            _stack_scalar_parameter(
                torch,
                elements,
                "residual_scale",
                carrier_parameters,
                device,
                defaults=tuple(element.payload.get("residual_scale", 0.0) for element in elements),
            ),
            min=0.0,
            max=1.0,
        )
        neural_strength = residual_scales[best_index]
        transmittance = torch.where(
            selected_payload_neural,
            torch.clamp(1.0 - all_opacities[best_index] * neural_strength, min=0.0, max=1.0),
            transmittance,
        )
        confidence = torch.where(
            selected_payload_neural,
            torch.clamp(all_confidences[best_index] * (1.0 - neural_strength * 0.25), min=0.0, max=1.0),
            confidence,
        )
        residual = residual | selected_payload_neural

    if any(payload_type == "semantic_feature" for payload_type in payload_types):
        semantic_confidences = torch.clamp(
            _stack_scalar_parameter(
                torch,
                elements,
                "confidence",
                carrier_parameters,
                device,
                defaults=tuple(element.payload.get("confidence", element.confidence) for element in elements),
            ),
            min=0.0,
            max=1.0,
        )
        confidence = torch.where(selected_payload_semantic, semantic_confidences[best_index], confidence)

    if any(payload_type == "gaussian_fallback" for payload_type in payload_types):
        gaussian_opacities = torch.clamp(
            _stack_scalar_parameter(
                torch,
                elements,
                "opacity",
                carrier_parameters,
                device,
                defaults=tuple(element.opacity for element in elements),
            ),
            min=0.0,
            max=1.0,
        )
        covariance_diag = _stack_vector_parameter(
            torch,
            elements,
            "gaussian_covariance_diag",
            carrier_parameters,
            device,
            defaults=tuple(_gaussian_covariance_diag(element) for element in elements),
        )
        gaussian_means, gaussian_has_mean = _stack_gaussian_mean_parameter(torch, elements, carrier_parameters, device)
        gaussian_weight = _torch_gaussian_weight_batched(
            torch,
            hit_points,
            mean=gaussian_means[best_index],
            covariance_diag=covariance_diag[best_index],
            has_mean=gaussian_has_mean[best_index],
        )
        transmittance = torch.where(
            selected_payload_gaussian,
            torch.clamp(1.0 - gaussian_opacities[best_index] * gaussian_weight, min=0.0, max=1.0),
            transmittance,
        )
        confidence = torch.where(
            selected_payload_gaussian,
            torch.clamp(all_confidences[best_index] * gaussian_weight, min=0.0, max=1.0),
            confidence,
        )

    return torch.clamp(carrier_colors, min=0.0, max=1.0), torch.clamp(transmittance, min=0.0, max=1.0), torch.clamp(confidence, min=0.0, max=1.0), residual


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
            volume_opacity = torch.clamp(
                _carrier_parameter(torch, element, "opacity", carrier_parameters, device, default=element.payload.get("opacity", 1.0)),
                min=0.0,
                max=1.0,
            )
            volume_confidence = torch.clamp(
                _carrier_parameter(torch, element, "confidence", carrier_parameters, device, default=element.confidence),
                min=0.0,
                max=1.0,
            )
            path_length = torch.clamp(exit_depth[mask, element_index] - best_depth[mask], min=0.0)
            alpha = volume_opacity * (1.0 - torch.exp(-density * path_length))
            carrier_colors[mask] = torch.clamp(volume_color, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - alpha, min=0.0, max=1.0)
            confidence[mask] = volume_confidence
        elif payload_type == "beta_kernel":
            beta_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            beta_opacity = torch.clamp(
                _carrier_parameter(torch, element, "opacity", carrier_parameters, device, default=element.opacity),
                min=0.0,
                max=1.0,
            )
            beta_confidence = torch.clamp(
                _carrier_parameter(torch, element, "confidence", carrier_parameters, device, default=element.payload.get("confidence", element.confidence)),
                min=0.0,
                max=1.0,
            )
            # DBS: adaptive_alpha/adaptive_beta override base alpha/beta if set
            _base_alpha = element.payload.get("alpha", 1.0)
            _base_beta = element.payload.get("beta", 1.0)
            _eff_alpha = element.payload["adaptive_alpha"] if element.payload.get("adaptive_alpha") is not None else _base_alpha
            _eff_beta = element.payload["adaptive_beta"] if element.payload.get("adaptive_beta") is not None else _base_beta
            alpha = _carrier_parameter(torch, element, "alpha", carrier_parameters, device, default=_eff_alpha)
            beta_value = _carrier_parameter(torch, element, "beta", carrier_parameters, device, default=_eff_beta)
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
            # DBS: apply frequency_scale and appearance_shift
            _freq_scale = float(element.payload.get("frequency_scale", 1.0))
            _app_shift = float(element.payload.get("appearance_shift", 0.0))
            scaled_weight = weight * _freq_scale
            carrier_colors[mask] = torch.clamp(beta_color + _app_shift, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - beta_opacity * scaled_weight, min=0.0, max=1.0)
            confidence[mask] = beta_confidence
        elif payload_type == "gabor_frequency":
            gabor_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            gabor_opacity = torch.clamp(
                _carrier_parameter(torch, element, "opacity", carrier_parameters, device, default=element.opacity),
                min=0.0,
                max=1.0,
            )
            gabor_confidence = torch.clamp(
                _carrier_parameter(torch, element, "confidence", carrier_parameters, device, default=element.confidence),
                min=0.0,
                max=1.0,
            )
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
            # Gabor filter bank: num_filters > 1 computes weighted sum of multiple Gabor kernels
            _num_filters = int(element.payload.get("num_filters", 1))
            if _num_filters <= 1:
                # Default single-filter path: identical to prior behavior
                wave = 0.5 + 0.5 * torch.sin(2.0 * pi * torch.sum(hit_points[mask] * frequency, dim=1) + phase)
                modulation = 1.0 - bandwidth + bandwidth * wave
            else:
                # Multi-filter bank path: weighted sum
                _freqs = element.payload.get("frequencies")
                _oris = element.payload.get("orientations")
                _phases_list = element.payload.get("phases")
                _fweights = element.payload.get("filter_weights")
                _base_freq = element.payload.get("frequency", (0.0, 0.0, 0.0))
                _base_phase = element.payload.get("phase", 0.0)
                _eq_weight = 1.0 / _num_filters
                modulation = torch.zeros(hit_points[mask].shape[0], dtype=torch.float32, device=device)
                for _fi in range(_num_filters):
                    _f_freq = _freqs[_fi] if _freqs is not None else _base_freq
                    _f_phase = _phases_list[_fi] if _phases_list is not None else _base_phase
                    _f_weight = _fweights[_fi] if _fweights is not None else _eq_weight
                    _ori = _oris[_fi] if _oris is not None else 0.0
                    # Apply orientation as rotation in xy-plane if frequency is a scalar
                    if isinstance(_f_freq, (int, float)):
                        _f_freq_vec = torch.tensor(
                            [_f_freq * float(__import__("math").cos(_ori)), _f_freq * float(__import__("math").sin(_ori)), 0.0],
                            dtype=torch.float32, device=device
                        )
                    else:
                        _f_freq_vec = torch.tensor(tuple(_f_freq), dtype=torch.float32, device=device)
                    _wave_i = 0.5 + 0.5 * torch.sin(2.0 * pi * torch.sum(hit_points[mask] * _f_freq_vec, dim=1) + _f_phase)
                    modulation = modulation + _f_weight * _wave_i
                modulation = 1.0 - bandwidth + bandwidth * modulation
            carrier_colors[mask] = torch.clamp(gabor_color * modulation.unsqueeze(1), min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - gabor_opacity, min=0.0, max=1.0)
            confidence[mask] = torch.clamp(gabor_confidence * bandwidth, min=0.0, max=1.0)
        elif payload_type == "neural_residual":
            neural_color = _carrier_vector_parameter(torch, element, "color", carrier_parameters, device, default=element.color)
            neural_opacity = torch.clamp(
                _carrier_parameter(torch, element, "opacity", carrier_parameters, device, default=element.payload.get("opacity", element.opacity)),
                min=0.0,
                max=1.0,
            )
            neural_confidence = torch.clamp(
                _carrier_parameter(torch, element, "confidence", carrier_parameters, device, default=element.payload.get("confidence", element.confidence)),
                min=0.0,
                max=1.0,
            )
            residual_scale = _carrier_parameter(
                torch,
                element,
                "residual_scale",
                carrier_parameters,
                device,
                default=element.payload.get("residual_scale", 0.0),
            )
            residual_strength = torch.clamp(residual_scale, min=0.0, max=1.0)
            # Scaffold-GS: anchor_feature_dim splits latent into anchor + residual portions
            # use_anchor_conditioning enables cross-carrier conditioning hook (no-op if no neighbors)
            _anchor_dim = element.payload.get("anchor_feature_dim")
            _use_anchor = bool(element.payload.get("use_anchor_conditioning", False))
            if _anchor_dim is not None:
                # Anchor features drive MLP decode; residual is added on top
                # In this reference path the anchor features modulate residual_strength
                _latent_dim = int(element.payload.get("latent_dim", 1))
                _anchor_ratio = min(float(_anchor_dim) / max(float(_latent_dim), 1.0), 1.0)
                residual_strength = residual_strength * _anchor_ratio
            if _use_anchor:
                # Real cross-carrier neural-residual MLP (Scaffold-GS arXiv:2312.00109).
                # The MLP is keyed under "cross_carrier_mlp" in the carrier_parameters dict.
                # When neighbors are present AND the MLP weights exist, we run a real
                # differentiable forward pass that conditions residual_strength on the
                # neighboring carriers' features (color, opacity, residual, centroid).
                # When no neighbors / no MLP weights are configured the correction is 0.
                _neighbor_elements = element.payload.get("neighbor_elements", None)
                _mlp_module = (
                    (carrier_parameters or {}).get(element.id, {}).get("cross_carrier_mlp", None)
                )
                if _neighbor_elements is not None and _mlp_module is not None:
                    from aura.cross_carrier import (
                        cross_carrier_residual_correction,
                        neighbor_features_from_carrier_parameters,
                    )
                    _neighbor_feats = neighbor_features_from_carrier_parameters(
                        torch, _neighbor_elements, carrier_parameters, device
                    )
                    if _neighbor_feats is not None:
                        _nb_colors, _nb_opacities, _nb_residuals, _nb_centroids = _neighbor_feats
                        _anchor_contribution = cross_carrier_residual_correction(
                            torch, _mlp_module,
                            _nb_colors, _nb_opacities, _nb_residuals, _nb_centroids,
                            device,
                        )
                    else:
                        _anchor_contribution = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    _anchor_contribution = torch.zeros((), dtype=torch.float32, device=device)
                residual_strength = torch.clamp(residual_strength + _anchor_contribution, min=0.0, max=1.0)
            carrier_colors[mask] = torch.clamp(neural_color, min=0.0, max=1.0)
            transmittance[mask] = torch.clamp(1.0 - neural_opacity * residual_strength, min=0.0, max=1.0)
            confidence[mask] = torch.clamp(neural_confidence * (1.0 - residual_strength * 0.25), min=0.0, max=1.0)
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
            gaussian_mean = _carrier_gaussian_mean_parameter(torch, element, carrier_parameters, device)
            gaussian_weight = _torch_gaussian_weight(
                torch,
                hit_points[mask],
                element,
                device,
                mean=gaussian_mean,
                covariance_diag=covariance_diag,
            )
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
                "opacity": torch.tensor(
                    float(element.payload.get("opacity", 1.0)),
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
                "confidence": torch.tensor(
                    float(element.payload.get("confidence", element.confidence)),
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
            _neural_params: dict[str, Any] = {
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
                "residual_scale": torch.tensor(
                    float(element.payload.get("residual_scale", 0.0)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
            }
            # When use_anchor_conditioning is enabled, attach a real cross-carrier MLP.
            # The MLP weights are real trainable nn.Parameters; the module is stored under
            # "cross_carrier_mlp" so the optimizer can discover and train them via
            # mlp_parameter_tensors_from_module().
            if bool(element.payload.get("use_anchor_conditioning", False)) and requires_grad:
                from aura.cross_carrier import build_cross_carrier_mlp, mlp_parameter_tensors_from_module
                _mlp = build_cross_carrier_mlp(torch, device)
                _neural_params["cross_carrier_mlp"] = _mlp
                # Also surface individual weight/bias tensors so existing SGD/Adam can train them
                _neural_params.update(mlp_parameter_tensors_from_module(torch, _mlp, requires_grad=requires_grad))
            parameters[element.id] = _neural_params
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
        parameters["normal"] = torch.tensor(
            _normal_parameter(element, fallback=(0.0, 0.0, -1.0)),
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
        parameters["normal"] = torch.tensor(
            _normal_parameter(element, fallback=(0.0, 0.0, 1.0)),
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


def _stack_scalar_parameter(
    torch: Any,
    elements: Sequence[Any],
    name: str,
    carrier_parameters: dict[str, dict[str, Any]] | None,
    device: str,
    *,
    defaults: Sequence[Any],
) -> Any:
    values = []
    for element, default in zip(elements, defaults):
        if carrier_parameters is not None:
            parameter = carrier_parameters.get(element.id, {}).get(name)
            if parameter is not None:
                values.append(parameter)
                continue
        values.append(torch.tensor(float(default), dtype=torch.float32, device=device))
    return torch.stack(tuple(values), dim=0)


def _stack_vector_parameter(
    torch: Any,
    elements: Sequence[Any],
    name: str,
    carrier_parameters: dict[str, dict[str, Any]] | None,
    device: str,
    *,
    defaults: Sequence[Any],
) -> Any:
    values = []
    for element, default in zip(elements, defaults):
        if carrier_parameters is not None:
            parameter = carrier_parameters.get(element.id, {}).get(name)
            if parameter is not None:
                values.append(parameter)
                continue
        values.append(torch.tensor(tuple(default), dtype=torch.float32, device=device))
    return torch.stack(tuple(values), dim=0)


def _stack_gaussian_mean_parameter(
    torch: Any,
    elements: Sequence[Any],
    carrier_parameters: dict[str, dict[str, Any]] | None,
    device: str,
) -> tuple[Any, Any]:
    values = []
    present = []
    for element in elements:
        parameter = carrier_parameters.get(element.id, {}).get("gaussian_mean") if carrier_parameters is not None else None
        if parameter is not None:
            values.append(parameter)
            present.append(True)
        elif _has_gaussian_mean(element):
            values.append(torch.tensor(_gaussian_mean(element), dtype=torch.float32, device=device))
            present.append(True)
        else:
            values.append(torch.zeros((3,), dtype=torch.float32, device=device))
            present.append(False)
    return torch.stack(tuple(values), dim=0), torch.tensor(present, dtype=torch.bool, device=device)


def _normal_parameter(element: Any, *, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    value = element.normal or element.payload.get("normal")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    return fallback


def _surface_plane_point(element: Any) -> tuple[float, float, float]:
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    center = [(lo + hi) * 0.5 for lo, hi in zip(min_corner, max_corner)]
    normal = element.normal or element.payload.get("normal") or (0.0, 0.0, -1.0)
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


def _carrier_gaussian_mean_parameter(
    torch: Any,
    element: Any,
    carrier_parameters: dict[str, dict[str, Any]] | None,
    device: str,
) -> Any | None:
    if carrier_parameters is not None:
        parameter = carrier_parameters.get(element.id, {}).get("gaussian_mean")
        if parameter is not None:
            return parameter
    if _has_gaussian_mean(element):
        return torch.tensor(_gaussian_mean(element), dtype=torch.float32, device=device)
    return None


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


def _torch_beta_weight_batched(torch: Any, points: Any, mins: Any, maxs: Any, *, support_radius: Any, alpha: Any, beta: Any) -> Any:
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


def _torch_gaussian_weight(
    torch: Any,
    points: Any,
    element: Any,
    device: str,
    *,
    mean: Any | None = None,
    covariance_diag: Any | None = None,
) -> Any:
    if mean is None:
        return torch.ones((int(points.shape[0]),), dtype=torch.float32, device=device)
    if covariance_diag is None:
        covariance_diag = torch.tensor(_gaussian_covariance_diag(element), dtype=torch.float32, device=device)
    safe_covariance_diag = torch.clamp(covariance_diag, min=1e-6)
    delta = points - mean.unsqueeze(0)
    mahalanobis = torch.sum((delta * delta) / safe_covariance_diag.unsqueeze(0), dim=1)
    return torch.clamp(torch.exp(-0.5 * torch.clamp(mahalanobis, min=0.0)), min=0.0, max=1.0)


def _torch_gaussian_weight_batched(torch: Any, points: Any, *, mean: Any, covariance_diag: Any, has_mean: Any) -> Any:
    safe_covariance_diag = torch.clamp(covariance_diag, min=1e-6)
    delta = points - mean
    mahalanobis = torch.sum((delta * delta) / safe_covariance_diag, dim=1)
    weighted = torch.clamp(torch.exp(-0.5 * torch.clamp(mahalanobis, min=0.0)), min=0.0, max=1.0)
    return torch.where(has_mean, weighted, torch.ones_like(weighted))


def _gaussian_mean(element: Any) -> tuple[float, float, float]:
    mean = element.payload.get("mean")
    if isinstance(mean, (list, tuple)) and len(mean) == 3:
        return tuple(float(item) for item in mean)  # type: ignore[return-value]
    return _center_point(element)


def _has_gaussian_mean(element: Any) -> bool:
    mean = element.payload.get("mean")
    return isinstance(mean, (list, tuple)) and len(mean) == 3


def _gaussian_covariance_diag(element: Any) -> tuple[float, float, float]:
    covariance = element.payload.get("covariance")
    if _is_matrix3(covariance):
        return tuple(max(float(covariance[index][index]), 1e-6) for index in range(3))  # type: ignore[return-value]
    min_corner = tuple(float(value) for value in element.bounds.min_corner)
    max_corner = tuple(float(value) for value in element.bounds.max_corner)
    return tuple(max(((max_corner[index] - min_corner[index]) * 0.25) ** 2, 1e-6) for index in range(3))  # type: ignore[return-value]


def _is_matrix3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in value)
