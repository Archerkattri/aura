from math import exp

import pytest

from aura import AuraElement, Bounds, Ray


def test_surface_carrier_uses_payload_normal_for_ray_query():
    element = AuraElement(
        id="surface",
        carrier_id="surface",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        payload={"type": "surface_cell", "normal": [0.0, 0.0, -1.0]},
    )

    result = element.ray_query(_center_ray())

    assert result is not None
    assert result.normal == (0.0, 0.0, -1.0)
    assert result.depth == pytest.approx(1.0)


def test_volume_carrier_integrates_density_over_path_length():
    element = AuraElement(
        id="volume",
        carrier_id="volume",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5)),
        opacity=1.0,
        payload={"type": "volume_cell", "density": 0.4},
    )

    result = element.ray_query(_center_ray())

    assert result is not None
    assert result.transmittance == pytest.approx(exp(-0.4 * 0.5))
    assert result.opacity == pytest.approx(1.0 - exp(-0.4 * 0.5))


def test_beta_carrier_uses_bounded_support_weight():
    element = AuraElement(
        id="beta",
        carrier_id="beta",
        bounds=Bounds((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        opacity=0.8,
        payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0},
    )

    center = element.ray_query(_center_ray())
    edge = element.ray_query(Ray(origin=(-0.49, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert center is not None
    assert edge is not None
    assert center.opacity > edge.opacity
    assert center.opacity < 0.8


def test_gabor_carrier_modulates_color_and_confidence():
    element = AuraElement(
        id="gabor",
        carrier_id="gabor",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        color=(1.0, 0.5, 0.0),
        confidence=0.9,
        payload={"type": "gabor_frequency", "frequency": [0.0, 0.0, 0.0], "phase": 0.0, "bandwidth": 0.5},
    )

    result = element.ray_query(_center_ray())

    assert result is not None
    assert result.color == pytest.approx((0.75, 0.375, 0.0))
    assert result.confidence == pytest.approx(0.45)


def test_neural_residual_carrier_marks_residual_and_scales_confidence():
    element = AuraElement(
        id="neural",
        carrier_id="neural",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        confidence=0.8,
        payload={"type": "neural_residual", "latent_dim": 16, "residual_scale": 0.5},
    )

    result = element.ray_query(_center_ray())

    assert result is not None
    assert result.residual is True
    assert result.confidence == pytest.approx(0.8 * (1.0 - 0.5 * 0.25))


def test_semantic_carrier_reports_payload_label_and_confidence():
    element = AuraElement(
        id="semantic",
        carrier_id="semantic",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        confidence=0.2,
        payload={"type": "semantic_feature", "label": "fixture_object", "confidence": 0.9},
    )

    result = element.ray_query(_center_ray())

    assert result is not None
    assert result.semantic_id == "fixture_object"
    assert result.confidence == pytest.approx(0.9)


def test_gaussian_fallback_carrier_uses_covariance_weighted_support():
    element = AuraElement(
        id="gaussian",
        carrier_id="gaussian",
        bounds=Bounds((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
        opacity=0.8,
        confidence=0.75,
        payload={
            "type": "gaussian_fallback",
            "mean": [0.0, 0.0, 0.0],
            "covariance": [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
            "source": "test",
        },
    )

    center = element.ray_query(_center_ray())
    offset = element.ray_query(Ray(origin=(0.4, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert center is not None
    assert offset is not None
    assert center.opacity == pytest.approx(0.8)
    assert center.confidence == pytest.approx(0.75)
    assert offset.opacity < center.opacity
    assert offset.confidence < center.confidence


def _center_ray() -> Ray:
    return Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0))
