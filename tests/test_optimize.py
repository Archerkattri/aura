import pytest

from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    Ray,
    RenderTarget,
    differentiate_scene_rays,
    gradient_descent_color_step,
    precondition_color_gradient,
)


def test_differentiable_scene_rays_report_loss_and_gradients():
    scene = AuraScene(
        name="gradient_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.2, 0.2, 0.2),
                opacity=1.0,
                confidence=0.85,
                normal=(0.0, 0.0, -1.0),
                material_id="mat_surface",
                semantic_id="panel",
            ),
        ),
    )
    target = RenderTarget(
        frame_id="frame",
        ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        target_color=(0.8, 0.2, 0.2),
        target_depth=2.0,
        target_semantic_id="panel",
        target_material_id="mat_surface",
    )

    sample = differentiate_scene_rays(scene, (target,))[0]

    assert sample.element_id == "surface"
    assert sample.carrier_id == "surface"
    assert sample.predicted_color == (0.2, 0.2, 0.2)
    assert sample.predicted_transmittance == 0.0
    assert sample.predicted_opacity == 1.0
    assert sample.predicted_confidence == 0.85
    assert sample.predicted_normal == (0.0, 0.0, -1.0)
    assert sample.predicted_material_id == "mat_surface"
    assert sample.predicted_semantic_id == "panel"
    assert sample.predicted_residual is False
    assert sample.predicted_provenance == "surface"
    assert sample.target_semantic_id == "panel"
    assert sample.target_material_id == "mat_surface"
    assert sample.query_loss == 0.0
    assert sample.image_loss == pytest.approx(0.12)
    assert sample.depth_loss == 0.0
    assert sample.color_jacobian == 1.0
    assert sample.color_gradient[0] < 0.0
    assert sample.color_gradient[1] == pytest.approx(0.0)
    assert sample.gradient_norm > 0.0


def test_gradient_descent_color_step_reduces_simple_color_loss():
    before = (0.2, 0.2, 0.2)
    gradient = (-0.4, 0.0, 0.0)

    after = gradient_descent_color_step(before, gradient, learning_rate=0.5)

    assert after == (0.4, 0.2, 0.2)
    assert (0.8 - after[0]) ** 2 < (0.8 - before[0]) ** 2


def test_differentiable_scene_rays_reports_query_contract_mismatch_loss():
    scene = AuraScene(
        name="query_loss_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                semantic_id="panel",
                material_id="mat_surface",
            ),
        ),
    )
    target = RenderTarget(
        frame_id="frame",
        ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        target_color=(1.0, 1.0, 1.0),
        target_depth=2.0,
        target_semantic_id="other_panel",
        target_material_id="other_material",
    )

    sample = differentiate_scene_rays(scene, (target,))[0]

    assert sample.query_loss == 1.0


def test_precondition_color_gradient_maps_attenuated_gradient_to_carrier_space():
    gradient = (-0.04, 0.0, 0.0)

    preconditioned = precondition_color_gradient(gradient, color_jacobian=0.1)

    assert preconditioned == pytest.approx((-6.0, 0.0, 0.0))


def test_differentiable_scene_rays_require_targets():
    with pytest.raises(ValueError, match="at least one target"):
        differentiate_scene_rays(AuraScene(name="empty", elements=()), ())
