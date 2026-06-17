import pytest

from aura import AuraAsset, CarrierKind, Ray, RayQueryResult, default_registry


def test_default_registry_exposes_required_carriers():
    kinds = {spec.kind for spec in default_registry().values()}

    assert CarrierKind.SURFACE_CELL in kinds
    assert CarrierKind.VOLUME_CELL in kinds
    assert CarrierKind.BETA_KERNEL in kinds
    assert CarrierKind.GABOR_FREQUENCY in kinds
    assert CarrierKind.NEURAL_RESIDUAL in kinds
    assert CarrierKind.GAUSSIAN_FALLBACK in kinds
    assert CarrierKind.SEMANTIC_FEATURE in kinds


def test_ray_normalizes_direction_and_rejects_zero_vector():
    ray = Ray(origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 2.0))

    assert ray.direction == (0.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="non-zero"):
        Ray(origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 0.0))


def test_ray_query_result_validates_confidence_and_transmittance():
    result = RayQueryResult(
        color=(0.1, 0.2, 0.3),
        transmittance=0.8,
        confidence=0.7,
    )

    assert result.opacity == pytest.approx(0.2)

    with pytest.raises(ValueError, match="transmittance"):
        RayQueryResult(color=(0.1, 0.2, 0.3), transmittance=1.5, confidence=0.7)

    with pytest.raises(ValueError, match="confidence"):
        RayQueryResult(color=(0.1, 0.2, 0.3), transmittance=0.8, confidence=-0.1)


def test_asset_manifest_reports_capabilities_from_carriers():
    asset = AuraAsset(
        name="room",
        carrier_ids=["surface", "volume", "semantic"],
    )

    capabilities = asset.capabilities(default_registry())

    assert capabilities["primaryRender"] is True
    assert capabilities["rayQuery"] is True
    assert capabilities["collisionProxy"] is True
    assert capabilities["semanticQuery"] is True
    assert capabilities["neuralResidual"] is False

