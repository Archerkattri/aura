import importlib.util

import pytest

from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    CaptureFrameTensors,
    CaptureTensor,
    Ray,
    RenderTarget,
    TrainingFrame,
    require_torch,
    torch_capture_asset_batch,
    torch_capture_training_batch,
    torch_carrier_parameter_tensors,
    torch_render_capture_training_batch,
    torch_render_capture_training_objective,
    torch_render_target_objective,
    torch_render_targets,
    torch_renderer_status,
    torch_scene_tensors,
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


def test_torch_capture_asset_batch_reports_install_hint_when_unavailable():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    with pytest.raises(RuntimeError, match="torch"):
        torch_capture_asset_batch((_capture_tensor_frame(),), device="cpu")


def test_torch_render_capture_training_batch_reports_install_hint_when_unavailable():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    scene = AuraScene(
        name="empty",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    with pytest.raises(RuntimeError, match="torch"):
        torch_render_capture_training_batch(scene, _fake_capture_training_batch())


def test_torch_render_target_objective_reports_install_hint_when_unavailable():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    scene = AuraScene(
        name="objective_unavailable_scene",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    target = RenderTarget(
        frame_id="frame",
        ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        target_color=(1.0, 1.0, 1.0),
        target_depth=2.0,
    )

    with pytest.raises(RuntimeError, match="torch"):
        torch_render_target_objective(scene, (target,), device="cpu")


def test_torch_render_capture_training_objective_reports_install_hint_when_unavailable():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    scene = AuraScene(
        name="capture_objective_unavailable_scene",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(RuntimeError, match="torch"):
        torch_render_capture_training_objective(scene, _fake_capture_training_batch())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_asset_batch_stacks_manifest_tensors_on_device():
    batch = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                image_values=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                depth_values=(0.25, 0.75),
                mask_values=(1.0, 0.0),
                normal_values=(0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
            ),
            _capture_tensor_frame(
                frame_id="frame_b",
                image_values=(0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
                depth_values=None,
                mask_values=(0.0, 1.0),
                normal_values=None,
            ),
        ),
        device="cpu",
    )
    payload = batch.to_dict()

    assert batch.frame_ids == ("frame_a", "frame_b")
    assert tuple(batch.image.shape) == (2, 1, 2, 3)
    assert tuple(batch.depth.shape) == (2, 1, 2, 1)
    assert tuple(batch.depth_present.tolist()) == (True, False)
    assert tuple(batch.mask_present.tolist()) == (True, True)
    assert tuple(batch.normal_present.tolist()) == (True, False)
    assert payload["image"]["shape"] == [2, 1, 2, 3]
    assert payload["depthPresent"]["dtype"] == "torch.bool"
    assert payload["normal"]["shape"] == [2, 1, 2, 3]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_samples_per_pixel_targets():
    frame = TrainingFrame(
        id="frame_a",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                image_values=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                depth_values=(0.25, 0.0),
                mask_values=(1.0, 0.0),
                normal_values=(0.0, 0.0, -1.0, 0.0, 1.0, 0.0),
            ),
        ),
        device="cpu",
    )

    batch = torch_capture_training_batch((frame,), assets)
    payload = batch.to_dict()

    assert tuple(batch.frame_indices.tolist()) == (0,)
    assert batch.pixel_xy.tolist() == [[0, 0]]
    assert batch.target_color.tolist() == [[1.0, 0.0, 0.0]]
    assert batch.target_depth.tolist() == [0.25]
    assert batch.target_mask.tolist() == [1.0]
    assert batch.target_normal.tolist() == [[0.0, 0.0, -1.0]]
    assert batch.target_normal_present.tolist() == [True]
    assert batch.ray_directions.tolist()[0] == [0.0, 0.0, 1.0]
    assert payload["targetColor"]["shape"] == [1, 3]
    assert payload["targetNormalPresent"]["shape"] == [1]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_rejects_fully_masked_targets():
    frame = TrainingFrame(
        id="frame_a",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                image_values=(1.0, 0.0, 0.0),
                mask_values=(0.0,),
                width=1,
                height=1,
            ),
        ),
        device="cpu",
    )

    with pytest.raises(ValueError, match="no sampled pixels"):
        torch_capture_training_batch((frame,), assets)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_batch_matches_render_target_path():
    scene = AuraScene(
        name="torch_capture_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame_a",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                image_values=(1.0, 0.0, 0.0),
                depth_values=(2.0,),
                mask_values=None,
                normal_values=(0.0, 0.0, -1.0),
                width=1,
                height=1,
            ),
        ),
        device="cpu",
    )
    capture_batch = torch_capture_training_batch((frame,), assets)
    direct_batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame_a",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
                target_normal=(0.0, 0.0, -1.0),
            ),
        ),
        device="cpu",
    )

    rendered = torch_render_capture_training_batch(scene, capture_batch)

    assert rendered.frame_ids == direct_batch.frame_ids
    assert rendered.element_ids == direct_batch.element_ids
    assert rendered.predicted_color == direct_batch.predicted_color
    assert rendered.predicted_depth == direct_batch.predicted_depth
    assert rendered.image_loss == direct_batch.image_loss
    assert rendered.depth_loss == direct_batch.depth_loss
    assert rendered.target_normal == direct_batch.target_normal
    assert rendered.normal_loss == direct_batch.normal_loss


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_asset_batch_rejects_mismatched_shapes():
    bad_frame = CaptureFrameTensors(
        frame_id="bad",
        image=CaptureTensor(
            path="bad.ppm",
            format="Netpbm",
            backend="stdlib",
            width=1,
            height=1,
            channels=3,
            values=(1.0, 0.0, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="image tensor shapes"):
        torch_capture_asset_batch((_capture_tensor_frame(), bad_frame), device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_carrier_parameter_tensors():
    import torch

    scene = AuraScene(
        name="torch_parameter_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.1, 0.1, 0.1),
                opacity=0.1,
                confidence=0.2,
                payload={"type": "gaussian_fallback"},
            ),
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")
    carrier_parameters["gaussian"]["color"] = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32, requires_grad=True)
    carrier_parameters["gaussian"]["opacity"] = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    carrier_parameters["gaussian"]["confidence"] = torch.tensor(0.75, dtype=torch.float32, requires_grad=True)

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(0.1, 0.2, 0.3),
                target_depth=2.0,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )

    assert batch.carrier_ids == ("gaussian",)
    assert batch.predicted_color[0] == pytest.approx((0.1, 0.2, 0.3))
    assert batch.opacity == pytest.approx((0.5,))
    assert batch.confidence == pytest.approx((0.75,))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_scene_tensors_cache_native_scene_on_device():
    scene = AuraScene(
        name="tensor_cache_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.1, 0.2, 0.3),
                opacity=0.4,
                confidence=0.8,
                payload={"type": "surface_cell"},
            ),
        ),
    )

    scene_tensors = torch_scene_tensors(scene, device="cpu")
    payload = scene_tensors.to_dict()

    assert scene_tensors.element_ids == ("surface",)
    assert scene_tensors.carrier_ids == ("surface",)
    assert payload["device"] == "cpu"
    assert payload["mins"]["shape"] == [1, 3]
    assert payload["colors"]["device"] == "cpu"
    assert payload["carrierParameterIds"] == ["surface"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_reuses_scene_tensor_cache():
    import torch

    scene = AuraScene(
        name="cached_render_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=0.2,
                confidence=0.3,
                payload={"type": "surface_cell"},
            ),
        ),
    )
    scene_tensors = torch_scene_tensors(scene, device="cpu")
    scene_tensors.carrier_parameters["surface"]["color"] = torch.tensor([0.8, 0.4, 0.2], dtype=torch.float32, requires_grad=True)
    scene_tensors.carrier_parameters["surface"]["opacity"] = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(0.4, 0.2, 0.1),
                target_depth=2.0,
            ),
        ),
        scene_tensors=scene_tensors,
    )

    assert batch.predicted_color[0] == pytest.approx((0.4, 0.2, 0.1))
    assert batch.opacity == pytest.approx((0.5,))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_rejects_mismatched_scene_tensor_cache():
    scene = AuraScene(
        name="cached_render_scene",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    other_scene = AuraScene(
        name="other_scene",
        elements=(AuraElement(id="other", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    scene_tensors = torch_scene_tensors(other_scene, device="cpu")

    with pytest.raises(ValueError, match="does not match scene element ids"):
        torch_render_targets(
            scene,
            (
                RenderTarget(
                    frame_id="frame",
                    ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                    target_color=(1.0, 1.0, 1.0),
                    target_depth=2.0,
                ),
            ),
            scene_tensors=scene_tensors,
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_target_objective_backpropagates_carrier_parameters():
    import torch

    scene = AuraScene(
        name="torch_objective_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.1, 0.1, 0.1),
                opacity=0.1,
                confidence=0.2,
                payload={"type": "gaussian_fallback"},
            ),
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")
    carrier_parameters["gaussian"]["color"] = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32, requires_grad=True)
    carrier_parameters["gaussian"]["opacity"] = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    carrier_parameters["gaussian"]["confidence"] = torch.tensor(0.75, dtype=torch.float32, requires_grad=True)

    objective = torch_render_target_objective(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(0.2, 0.2, 0.3),
                target_depth=2.0,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )
    objective.total_loss.backward()

    assert objective.frame_ids == ("frame",)
    assert objective.to_dict()["carrierParameterIds"] == ["gaussian"]
    assert objective.to_dict()["totalLoss"] > 0.0
    assert carrier_parameters["gaussian"]["color"].grad is not None
    assert carrier_parameters["gaussian"]["opacity"].grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_composites_ordered_carrier_hits():
    scene = AuraScene(
        name="torch_composite_scene",
        elements=(
            AuraElement(
                id="front_surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.5,
                confidence=0.8,
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="back_surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.2), (0.5, 0.5, 0.3)),
                color=(0.0, 0.0, 1.0),
                opacity=0.5,
                confidence=0.4,
                payload={"type": "surface_cell"},
            ),
        ),
    )

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(0.5, 0.0, 0.25),
                target_depth=2.0,
            ),
        ),
        device="cpu",
    )
    payload = batch.to_dict()

    assert batch.element_ids == ("front_surface",)
    assert batch.provenance == ("front_surface,back_surface",)
    assert batch.ordered_hits[0][0]["elementId"] == "front_surface"
    assert batch.ordered_hits[0][0]["carrierId"] == "surface"
    assert batch.ordered_hits[0][0]["depth"] == pytest.approx(2.0)
    assert batch.ordered_hits[0][0]["transmittance"] == pytest.approx(0.5)
    assert batch.ordered_hits[0][0]["opacity"] == pytest.approx(0.5)
    assert batch.ordered_hits[0][1]["elementId"] == "back_surface"
    assert batch.ordered_hits[0][1]["carrierId"] == "surface"
    assert batch.ordered_hits[0][1]["depth"] == pytest.approx(2.2)
    assert batch.ordered_hits[0][1]["transmittance"] == pytest.approx(0.5)
    assert batch.ordered_hits[0][1]["opacity"] == pytest.approx(0.5)
    assert payload["orderedHits"][0][0]["elementId"] == "front_surface"
    assert payload["orderedHits"][0][1]["carrierId"] == "surface"
    assert batch.predicted_depth == pytest.approx((2.0,))
    assert batch.predicted_color[0] == pytest.approx((0.5, 0.0, 0.25))
    assert batch.transmittance == pytest.approx((0.25,))
    assert batch.opacity == pytest.approx((0.75,))
    assert batch.confidence == pytest.approx(((0.5 * 0.8 + 0.25 * 0.4) / 0.75,))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_target_objective_backpropagates_ordered_carrier_hits():
    scene = AuraScene(
        name="torch_composite_objective_scene",
        elements=(
            AuraElement(
                id="front_surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.5,
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="back_surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.2), (0.5, 0.5, 0.3)),
                color=(0.0, 0.0, 1.0),
                opacity=0.5,
                payload={"type": "surface_cell"},
            ),
        ),
    )
    torch = require_torch()
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")

    objective = torch_render_target_objective(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(0.1, 0.0, 0.9),
                target_depth=2.0,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )
    objective.total_loss.backward()

    assert objective.to_dict()["carrierParameterIds"] == ["back_surface", "front_surface"]
    assert carrier_parameters["front_surface"]["color"].grad is not None
    assert carrier_parameters["front_surface"]["opacity"].grad is not None
    assert carrier_parameters["back_surface"]["color"].grad is not None
    assert carrier_parameters["back_surface"]["opacity"].grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_objective_backpropagates_native_surface_parameters():
    import torch

    scene = AuraScene(
        name="torch_capture_objective_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={"type": "surface_cell"},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame",
                image_values=(1.0, 0.0, 0.0),
                depth_values=(2.0,),
                mask_values=None,
                width=1,
                height=1,
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")

    objective = torch_render_capture_training_objective(scene, batch, carrier_parameters=carrier_parameters)
    objective.total_loss.backward()

    assert objective.to_dict()["carrierParameterIds"] == ["surface"]
    assert carrier_parameters["surface"]["color"].grad is not None
    assert carrier_parameters["surface"]["opacity"].grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_objective_includes_normal_loss():
    import torch

    scene = AuraScene(
        name="torch_capture_normal_objective_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame",
                image_values=(1.0, 0.0, 0.0),
                depth_values=(2.0,),
                mask_values=None,
                normal_values=(0.0, 0.0, 1.0),
                width=1,
                height=1,
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")

    objective = torch_render_capture_training_objective(scene, batch, carrier_parameters=carrier_parameters)
    payload = objective.to_dict()

    assert payload["imageLoss"] == pytest.approx(0.0)
    assert payload["depthLoss"] == pytest.approx(0.0)
    assert payload["normalLoss"] == pytest.approx(1.0)
    assert payload["totalLoss"] == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_objective_includes_mask_loss():
    import torch

    scene = AuraScene(
        name="torch_capture_mask_objective_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame",
                image_values=(1.0, 0.0, 0.0),
                depth_values=(2.0,),
                mask_values=(0.0,),
                normal_values=(0.0, 0.0, -1.0),
                width=1,
                height=1,
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")

    objective = torch_render_capture_training_objective(scene, batch, carrier_parameters=carrier_parameters)
    objective.total_loss.backward()
    payload = objective.to_dict()

    assert payload["imageLoss"] == pytest.approx(0.0)
    assert payload["depthLoss"] == pytest.approx(0.0)
    assert payload["normalLoss"] == pytest.approx(0.0)
    assert payload["maskLoss"] == pytest.approx(1.0)
    assert payload["totalLoss"] == pytest.approx(1.0)
    assert carrier_parameters["surface"]["opacity"].grad is not None


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
                normal=(0.0, 0.0, -1.0),
                material_id="mat_surface",
                semantic_id="panel",
            ),
        ),
    )
    target = RenderTarget(
        frame_id="frame",
        ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
        target_color=(0.2, 0.4, 0.6),
        target_depth=2.0,
        target_semantic_id="panel",
        target_material_id="mat_surface",
        target_normal=(0.0, 0.0, -1.0),
    )

    batch = torch_render_targets(scene, (target,), device="cpu")

    assert batch.element_ids == ("surface",)
    assert batch.carrier_ids == ("surface",)
    assert batch.ordered_hits[0][0]["elementId"] == "surface"
    assert batch.to_dict()["orderedHits"][0][0]["depth"] == 2.0
    assert batch.predicted_depth == (2.0,)
    assert batch.transmittance == (0.0,)
    assert batch.opacity == (1.0,)
    assert batch.confidence == (1.0,)
    assert batch.normal == ((0.0, 0.0, -1.0),)
    assert batch.material_ids == ("mat_surface",)
    assert batch.residual == (False,)
    assert batch.semantic_ids == ("panel",)
    assert batch.provenance == ("surface",)
    assert batch.target_semantic_ids == ("panel",)
    assert batch.target_material_ids == ("mat_surface",)
    assert batch.target_normal == ((0.0, 0.0, -1.0),)
    assert batch.query_loss == (0.0,)
    assert batch.normal_loss == (0.0,)
    assert batch.image_loss[0] == pytest.approx(0.0)
    assert batch.depth_loss[0] == pytest.approx(0.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_reports_native_payload_semantics():
    scene = AuraScene(
        name="torch_payload_scene",
        elements=(
            AuraElement(
                id="semantic",
                carrier_id="semantic",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.3, 0.3, 0.7),
                opacity=0.5,
                payload={"type": "semantic_feature", "label": "object", "confidence": 0.9},
            ),
            AuraElement(
                id="neural",
                carrier_id="neural",
                bounds=Bounds((1.0, -0.5, 0.0), (2.0, 0.5, 0.1)),
                color=(0.7, 0.2, 0.2),
                opacity=0.5,
                payload={"type": "neural_residual", "latent_dim": 16, "residual_scale": 0.4},
            ),
        ),
    )
    targets = (
        RenderTarget(
            frame_id="semantic_frame",
            ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(0.15, 0.15, 0.35),
            target_depth=2.0,
        ),
        RenderTarget(
            frame_id="neural_frame",
            ray=Ray(origin=(1.5, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(0.35, 0.1, 0.1),
            target_depth=2.0,
        ),
    )

    batch = torch_render_targets(scene, targets, device="cpu")

    assert batch.semantic_ids == ("object", None)
    assert batch.confidence[0] == pytest.approx(0.9)
    assert batch.residual == (False, True)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_reports_query_contract_loss():
    scene = AuraScene(
        name="torch_query_loss_scene",
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

    batch = torch_render_targets(scene, (target,), device="cpu")

    assert batch.semantic_ids == ("panel",)
    assert batch.material_ids == ("mat_surface",)
    assert batch.query_loss == (1.0,)
    assert batch.to_dict()["queryLoss"] == [1.0]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_reports_normal_target_loss():
    scene = AuraScene(
        name="torch_normal_loss_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                normal=(0.0, 0.0, -1.0),
            ),
            AuraElement(
                id="missing_normal",
                carrier_id="volume",
                bounds=Bounds((1.0, -0.5, 0.0), (2.0, 0.5, 0.1)),
            ),
        ),
    )
    targets = (
        RenderTarget(
            frame_id="aligned",
            ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(1.0, 1.0, 1.0),
            target_depth=2.0,
            target_normal=(0.0, 0.0, -1.0),
        ),
        RenderTarget(
            frame_id="opposed",
            ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(1.0, 1.0, 1.0),
            target_depth=2.0,
            target_normal=(0.0, 0.0, 1.0),
        ),
        RenderTarget(
            frame_id="unsupervised",
            ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(1.0, 1.0, 1.0),
            target_depth=2.0,
        ),
        RenderTarget(
            frame_id="missing_prediction",
            ray=Ray(origin=(1.5, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
            target_color=(1.0, 1.0, 1.0),
            target_depth=2.0,
            target_normal=(0.0, 0.0, -1.0),
        ),
    )

    batch = torch_render_targets(scene, targets, device="cpu")

    assert batch.target_normal == (
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        None,
        (0.0, 0.0, -1.0),
    )
    assert batch.normal_loss[0] == pytest.approx(0.0)
    assert batch.normal_loss[1] == pytest.approx(1.0)
    assert batch.normal_loss[2] == pytest.approx(0.0)
    assert batch.normal_loss[3] == pytest.approx(1.0)
    assert batch.to_dict()["normalLoss"][1] == pytest.approx(1.0)


def _capture_tensor_frame(
    *,
    frame_id: str = "frame",
    image_values=(1.0, 0.0, 0.0, 0.0, 0.5, 0.5),
    depth_values=(0.5, 1.0),
    mask_values=(1.0, 0.0),
    normal_values=None,
    width: int = 2,
    height: int = 1,
) -> CaptureFrameTensors:
    return CaptureFrameTensors(
        frame_id=frame_id,
        image=CaptureTensor(
            path=f"{frame_id}.ppm",
            format="Netpbm",
            backend="stdlib",
            width=width,
            height=height,
            channels=3,
            values=tuple(image_values),
        ),
        depth=CaptureTensor(
            path=f"{frame_id}.pgm",
            format="Netpbm",
            backend="stdlib",
            width=width,
            height=height,
            channels=1,
            values=tuple(depth_values),
        )
        if depth_values is not None
        else None,
        mask=CaptureTensor(
            path=f"{frame_id}_mask.pgm",
            format="Netpbm",
            backend="stdlib",
            width=width,
            height=height,
            channels=1,
            values=tuple(mask_values),
        )
        if mask_values is not None
        else None,
        normal=CaptureTensor(
            path=f"{frame_id}_normal.ppm",
            format="Netpbm",
            backend="stdlib",
            width=width,
            height=height,
            channels=3,
            values=tuple(normal_values),
        )
        if normal_values is not None
        else None,
    )


def _fake_capture_training_batch():
    class _FakeTensor:
        def numel(self):
            return 1

    return type(
        "FakeCaptureTrainingBatch",
        (),
        {
            "frame_indices": _FakeTensor(),
            "frame_ids": ("frame",),
            "ray_origins": None,
            "ray_directions": None,
            "target_color": None,
            "target_depth": None,
        },
    )()
