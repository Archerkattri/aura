import importlib.util

import pytest

from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    Ray,
    RenderTarget,
    require_torch,
    torch_render_targets,
    torch_renderer_status,
)


def test_torch_renderer_status_reports_optional_backend():
    status = torch_renderer_status()

    assert status.available is (importlib.util.find_spec("torch") is not None)
    assert status.to_dict()["available"] is status.available
    if not status.available:
        assert status.cuda_available is False
        assert status.default_device is None
        assert "torch" in status.reason.lower()


def test_require_torch_reports_install_hint_when_unavailable():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    with pytest.raises(RuntimeError, match="torch"):
        require_torch()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_matches_native_first_hit_contract():
    scene = AuraScene(
        name="torch_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.2, 0.4, 0.6),
                opacity=1.0,
            ),
        ),
    )
    target = RenderTarget(
        frame_id="frame",
        ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        target_color=(0.2, 0.4, 0.6),
        target_depth=2.0,
    )

    batch = torch_render_targets(scene, (target,), device="cpu")

    assert batch.element_ids == ("surface",)
    assert batch.carrier_ids == ("surface",)
    assert batch.predicted_depth == (2.0,)
    assert batch.image_loss[0] == pytest.approx(0.0)
    assert batch.depth_loss[0] == pytest.approx(0.0)
