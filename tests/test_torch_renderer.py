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
    assert batch.transmittance == (0.0,)
    assert batch.confidence == (1.0,)
    assert batch.residual == (False,)
    assert batch.semantic_ids == (None,)
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


def _capture_tensor_frame(
    *,
    frame_id: str = "frame",
    image_values=(1.0, 0.0, 0.0, 0.0, 0.5, 0.5),
    depth_values=(0.5, 1.0),
    mask_values=(1.0, 0.0),
) -> CaptureFrameTensors:
    return CaptureFrameTensors(
        frame_id=frame_id,
        image=CaptureTensor(
            path=f"{frame_id}.ppm",
            format="Netpbm",
            backend="stdlib",
            width=2,
            height=1,
            channels=3,
            values=tuple(image_values),
        ),
        depth=CaptureTensor(
            path=f"{frame_id}.pgm",
            format="Netpbm",
            backend="stdlib",
            width=2,
            height=1,
            channels=1,
            values=tuple(depth_values),
        )
        if depth_values is not None
        else None,
        mask=CaptureTensor(
            path=f"{frame_id}_mask.pgm",
            format="Netpbm",
            backend="stdlib",
            width=2,
            height=1,
            channels=1,
            values=tuple(mask_values),
        )
        if mask_values is not None
        else None,
    )
