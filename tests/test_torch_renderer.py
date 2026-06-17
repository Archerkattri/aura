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
    torch_render_capture_training_batch,
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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_asset_batch_stacks_manifest_tensors_on_device():
    batch = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                image_values=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                depth_values=(0.25, 0.75),
                mask_values=(1.0, 0.0),
            ),
            _capture_tensor_frame(
                frame_id="frame_b",
                image_values=(0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
                depth_values=None,
                mask_values=(0.0, 1.0),
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
    assert payload["image"]["shape"] == [2, 1, 2, 3]
    assert payload["depthPresent"]["dtype"] == "torch.bool"


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
            ),
        ),
        device="cpu",
    )

    batch = torch_capture_training_batch((frame,), assets)
    payload = batch.to_dict()

    assert tuple(batch.frame_indices.tolist()) == (0, 0)
    assert batch.pixel_xy.tolist() == [[0, 0], [1, 0]]
    assert batch.target_color.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert batch.target_depth.tolist() == [0.25, 2.0]
    assert batch.target_mask.tolist() == [1.0, 0.0]
    assert batch.ray_directions.tolist()[0] == [0.0, 0.0, 1.0]
    assert payload["targetColor"]["shape"] == [2, 3]


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
    )

    batch = torch_render_targets(scene, (target,), device="cpu")

    assert batch.element_ids == ("surface",)
    assert batch.carrier_ids == ("surface",)
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
    assert batch.query_loss == (0.0,)
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


def _capture_tensor_frame(
    *,
    frame_id: str = "frame",
    image_values=(1.0, 0.0, 0.0, 0.0, 0.5, 0.5),
    depth_values=(0.5, 1.0),
    mask_values=(1.0, 0.0),
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
