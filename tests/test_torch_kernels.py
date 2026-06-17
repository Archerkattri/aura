import importlib.util

import pytest

from aura import AuraElement, Bounds, torch_carrier_kernel_specs, torch_carrier_response_tensors


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
