import importlib.util
import json
import subprocess
import sys

import pytest

from aura import AuraElement, Bounds, torch_carrier_kernel_report, torch_carrier_kernel_specs, torch_carrier_response_tensors


def test_torch_carrier_kernel_specs_cover_native_payloads_without_torch():
    specs = torch_carrier_kernel_specs()
    by_payload = {spec.payload_type: spec for spec in specs}

    assert set(by_payload) == {
        "surface_cell",
        "volume_cell",
        "beta_kernel",
        "gabor_frequency",
        "neural_residual",
        "semantic_feature",
        "gaussian_fallback",
    }
    assert by_payload["gaussian_fallback"].carrier_id == "gaussian"
    assert "opacity" in by_payload["surface_cell"].differentiable_fields
    assert by_payload["volume_cell"].to_dict()["payloadType"] == "volume_cell"
    assert by_payload["surface_cell"].to_dict()["implementationStage"] == "torch_autograd_surface_kernel"
    assert by_payload["surface_cell"].autograd_kernel is True
    assert by_payload["surface_cell"].cuda_kernel is False
    assert by_payload["surface_cell"].production_ready is False
    assert by_payload["surface_cell"].blockers == ("missing_cuda_kernel",)
    assert by_payload["volume_cell"].blockers == ("missing_autograd_kernel", "missing_cuda_kernel")


def test_torch_carrier_kernel_report_is_a_production_readiness_gate():
    report = torch_carrier_kernel_report()

    assert report["format"] == "AURA_TORCH_CARRIER_KERNEL_REPORT"
    assert report["productionReady"] is False
    assert report["carrierCount"] == 7
    assert report["referenceOnlyCarrierCount"] == 7
    assert report["autogradCarrierCount"] == 1
    assert report["cudaCarrierCount"] == 0
    by_carrier = {item["carrierId"]: item for item in report["kernelSpecs"]}
    assert set(by_carrier) == {"surface", "volume", "beta", "gabor", "neural", "semantic", "gaussian"}
    assert by_carrier["surface"]["autogradKernel"] is True
    assert by_carrier["surface"]["productionReady"] is False
    assert by_carrier["surface"]["blockers"] == ["missing_cuda_kernel"]
    assert by_carrier["gaussian"]["productionReady"] is False
    assert by_carrier["gaussian"]["blockers"] == ["missing_autograd_kernel", "missing_cuda_kernel"]
    assert "autograd/CUDA" in report["requiredNextStep"]


def test_torch_kernel_report_cli_prints_readiness_json():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "torch-kernel-report"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_TORCH_CARRIER_KERNEL_REPORT"
    assert payload["productionReady"] is False
    assert payload["carrierCount"] == 7
    assert payload["autogradCarrierCount"] == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_kernel_keeps_color_opacity_confidence_differentiable():
    import torch

    elements = (
        AuraElement(
            id="surface",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "surface_cell"},
        ),
    )
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.2, 0.4, 0.6]], requires_grad=True)
    opacities = torch.tensor([0.5], requires_grad=True)
    confidences = torch.tensor([0.75], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert colors.grad is not None
    assert opacities.grad is not None
    assert confidences.grad is not None
    assert colors.grad.tolist() == [[1.0, 1.0, 1.0]]
    assert opacities.grad.tolist() == pytest.approx([-1.0])
    assert confidences.grad.tolist() == pytest.approx([1.0])
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_carrier_response_tensors_apply_payload_kernels():
    import torch

    elements = (
        AuraElement(
            id="volume",
            carrier_id="volume",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=1.0,
            payload={"type": "volume_cell", "density": 2.0},
        ),
        AuraElement(
            id="semantic",
            carrier_id="semantic",
            bounds=Bounds((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.5,
            payload={"type": "semantic_feature", "label": "object", "confidence": 0.9},
        ),
        AuraElement(
            id="neural",
            carrier_id="neural",
            bounds=Bounds((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=1.0,
            payload={"type": "neural_residual", "latent_dim": 8, "residual_scale": 0.4},
        ),
    )
    best_index = torch.tensor([0, 1, 2], dtype=torch.long)
    best_depth = torch.tensor([1.0, 1.0, 1.0])
    exit_depth = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    hit_points = torch.tensor([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
    colors = torch.ones((3, 3))
    opacities = torch.tensor([0.5, 0.5, 0.5])
    confidences = torch.tensor([1.0, 0.5, 1.0])
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    _colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
    )

    assert transmittance[0].item() == pytest.approx(torch.exp(torch.tensor(-2.0)).item())
    assert confidence[1].item() == pytest.approx(0.9)
    assert confidence[2].item() == pytest.approx(0.9)
    assert residual.tolist() == [False, False, True]
