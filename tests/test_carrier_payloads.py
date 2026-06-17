import pytest

from aura import (
    BetaKernelPayload,
    GaborFrequencyPayload,
    GaussianFallbackPayload,
    NeuralResidualPayload,
    SemanticFeaturePayload,
    SurfaceCellPayload,
    VolumeCellPayload,
)


def test_native_carrier_payloads_expose_typed_contracts():
    payloads = [
        SurfaceCellPayload(normal=(0.0, 0.0, 1.0), thickness=0.02).to_dict(),
        VolumeCellPayload(density=0.4, phase_anisotropy=0.1).to_dict(),
        BetaKernelPayload(alpha=2.0, beta=3.0, support_radius=(0.1, 0.2, 0.3)).to_dict(),
        GaborFrequencyPayload(frequency=(1.0, 0.0, 0.0), bandwidth=0.5).to_dict(),
        NeuralResidualPayload(latent_dim=16, residual_scale=0.25, model_ref="models/local.pt").to_dict(),
        GaussianFallbackPayload(
            mean=(0.0, 0.0, 0.0),
            covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.0025)),
            source="test",
        ).to_dict(),
        SemanticFeaturePayload(label="chair", confidence=0.9, feature_refs=("clip:1",)).to_dict(),
    ]

    assert {payload["type"] for payload in payloads} == {
        "surface_cell",
        "volume_cell",
        "beta_kernel",
        "gabor_frequency",
        "neural_residual",
        "gaussian_fallback",
        "semantic_feature",
    }


def test_native_carrier_payloads_round_trip_from_dict():
    payloads = [
        (SurfaceCellPayload, SurfaceCellPayload(normal=(0.0, 0.0, 1.0), thickness=0.02).to_dict()),
        (VolumeCellPayload, VolumeCellPayload(density=0.4, phase_anisotropy=0.1).to_dict()),
        (BetaKernelPayload, BetaKernelPayload(alpha=2.0, beta=3.0, support_radius=(0.1, 0.2, 0.3)).to_dict()),
        (GaborFrequencyPayload, GaborFrequencyPayload(frequency=(1.0, 0.0, 0.0), bandwidth=0.5).to_dict()),
        (NeuralResidualPayload, NeuralResidualPayload(latent_dim=16, residual_scale=0.25, model_ref="models/local.pt").to_dict()),
        (
            GaussianFallbackPayload,
            GaussianFallbackPayload(
                mean=(0.0, 0.0, 0.0),
                covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.0025)),
                source="test",
            ).to_dict(),
        ),
        (SemanticFeaturePayload, SemanticFeaturePayload(label="chair", confidence=0.9, feature_refs=("clip:1",)).to_dict()),
    ]

    for payload_type, payload in payloads:
        assert payload_type.from_dict(payload).to_dict() == payload


def test_payload_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="thickness"):
        SurfaceCellPayload(normal=(0.0, 0.0, 1.0), thickness=0.0).to_dict()
    with pytest.raises(ValueError, match="density"):
        VolumeCellPayload(density=1.5).to_dict()
    with pytest.raises(ValueError, match="latent_dim"):
        NeuralResidualPayload(latent_dim=0, residual_scale=0.2).to_dict()
    with pytest.raises(ValueError, match="payload type"):
        SurfaceCellPayload.from_dict({"type": "volume_cell", "normal": [0.0, 0.0, 1.0], "thickness": 0.1, "roughness": 0.5})
