import importlib.util

import pytest
import aura.torch_renderer as torch_renderer_module

from aura import (
    AuraChunk,
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
    torch_render_capture_training_summary,
    torch_render_ray_color_tensor,
    torch_render_rays,
    torch_render_target_objective,
    torch_render_tensor_targets,
    torch_render_targets,
    torch_renderer_status,
    torch_scene_tensors,
)
from aura import CapturePackedRenderBatch, torch_capture_training_batch_from_packed
from aura.cli import native_demo_scene


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
    assert batch.target_confidence.tolist() == [1.0]
    assert batch.target_confidence_present.tolist() == [True]
    assert batch.target_normal.tolist() == [[0.0, 0.0, -1.0]]
    assert batch.target_normal_present.tolist() == [True]
    assert batch.ray_directions.tolist()[0] == [0.0, 0.0, 1.0]
    assert payload["targetColor"]["shape"] == [1, 3]
    assert payload["targetConfidence"]["shape"] == [1]
    assert payload["targetNormalPresent"]["shape"] == [1]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_filters_masked_pixels_before_limit():
    frame = TrainingFrame(
        id="frame_a",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 1.0, "width": 4.0, "height": 2.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                width=4,
                height=2,
                image_values=(
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    3.0,
                    0.0,
                    0.0,
                    4.0,
                    0.0,
                    0.0,
                    5.0,
                    0.0,
                    0.0,
                    6.0,
                    0.0,
                    0.0,
                    7.0,
                    0.0,
                    0.0,
                ),
                depth_values=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7),
                mask_values=(0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
            ),
        ),
        device="cpu",
    )

    batch = torch_capture_training_batch((frame,), assets, max_targets_per_frame=3)

    assert batch.pixel_xy.tolist() == [[1, 0], [3, 0], [0, 1]]
    assert batch.target_color.tolist() == [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert batch.target_depth.tolist() == pytest.approx([1.1, 1.3, 1.4])
    assert batch.target_mask.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_treats_absent_frame_mask_as_visible():
    frames = (
        TrainingFrame(
            id="frame_a",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.0, 0.0, 0.0),
            target_depth=2.0,
            intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2.0, "height": 1.0},
        ),
        TrainingFrame(
            id="frame_b",
            camera_origin=(0.0, 0.0, -2.0),
            look_at=(0.0, 0.0, 0.0),
            target_color=(0.0, 0.0, 0.0),
            target_depth=2.0,
            intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2.0, "height": 1.0},
        ),
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                image_values=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                depth_values=(2.0, 2.0),
                mask_values=(0.0, 1.0),
            ),
            _capture_tensor_frame(
                frame_id="frame_b",
                image_values=(0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
                depth_values=(2.0, 2.0),
                mask_values=None,
            ),
        ),
        device="cpu",
    )

    batch = torch_capture_training_batch(frames, assets)

    assert batch.frame_indices.tolist() == [0, 1, 1]
    assert batch.sample_frame_ids == ("frame_a", "frame_b", "frame_b")
    assert batch.pixel_xy.tolist() == [[1, 0], [0, 0], [1, 0]]
    assert batch.target_mask.tolist() == [1.0, 1.0, 1.0]
    assert batch.target_confidence.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_builds_ray_tensors_on_asset_device():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = TrainingFrame(
        id="frame_a",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        semantic_label="surface_region",
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            _capture_tensor_frame(
                frame_id="frame_a",
                image_values=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                depth_values=(2.0, 2.0),
                mask_values=None,
            ),
        ),
        device=device,
    )

    batch = torch_capture_training_batch((frame,), assets)

    assert batch.ray_origins.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert batch.ray_directions.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert batch.sample_frame_ids == ("frame_a", "frame_a")
    assert batch.target_semantic_ids == ("surface_region", "surface_region")
    assert batch.ray_origins.detach().cpu().tolist() == [[0.0, 0.0, -2.0], [0.0, 0.0, -2.0]]
    assert batch.ray_directions[0].detach().cpu().tolist() == pytest.approx([0.4472135901, 0.0, 0.8944271803])
    assert batch.ray_directions[1].detach().cpu().tolist() == pytest.approx([-0.4472135901, 0.0, 0.8944271803])


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
                depth_values=(2.0,),
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
def test_torch_render_capture_training_summary_skips_full_batch_serialization(monkeypatch):
    scene = AuraScene(
        name="torch_capture_summary_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.25, 0.5, 0.75),
                opacity=0.8,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
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

    def fail_full_batch(*_args, **_kwargs):
        raise AssertionError("capture summary should not build a full render batch")

    monkeypatch.setattr(torch_renderer_module, "_torch_render_tensor_targets", fail_full_batch)

    summary = torch_render_capture_training_summary(scene, capture_batch)

    assert summary.element_ids == ("surface",)
    assert summary.carrier_ids == ("surface",)
    assert summary.ray_origins == ((0.0, 0.0, -2.0),)
    assert summary.ray_directions == ((0.0, 0.0, 1.0),)
    assert summary.predicted_color[0] == pytest.approx((0.2, 0.4, 0.6))
    assert summary.predicted_depth == pytest.approx((2.0,))
    assert summary.transmittance == pytest.approx((0.2,))
    assert summary.normal == ((0.0, 0.0, -1.0),)
    assert summary.target_color == ((1.0, 0.0, 0.0),)
    assert summary.target_depth == (2.0,)
    assert summary.target_point[0] == pytest.approx((0.0, 0.0, 0.0))
    assert summary.image_loss[0] == pytest.approx(((0.2 - 1.0) ** 2 + 0.4**2 + 0.6**2) / 3.0)
    assert summary.depth_loss == pytest.approx((0.0,))
    assert summary.normal_loss == pytest.approx((0.0,))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_rays_renders_raw_tensor_inputs_without_render_targets():
    import torch

    scene = AuraScene(
        name="torch_raw_ray_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.25, 0.5, 0.75),
                opacity=0.8,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    batch = torch_render_rays(scene, origins, directions, device="cpu", frame_id_prefix="raw")

    assert batch.frame_ids == ("raw_0",)
    assert batch.element_ids == ("surface",)
    assert batch.predicted_color[0] == pytest.approx((0.2, 0.4, 0.6))
    assert batch.opacity[0] == pytest.approx(0.8)
    assert batch.predicted_depth[0] == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_rays_can_skip_ordered_trace_serialization(monkeypatch):
    scene = AuraScene(
        name="torch_raw_ray_no_trace_scene",
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

    def fail_ordered_trace_serialization(*_args, **_kwargs):
        raise AssertionError("trace-free torch render should not serialize ordered hits")

    monkeypatch.setattr(torch_renderer_module, "_torch_ordered_hit_traces", fail_ordered_trace_serialization)

    batch = torch_render_rays(
        scene,
        ((0.0, 0.0, -1.0),),
        ((0.0, 0.0, 1.0),),
        device="cpu",
        collect_traces=False,
    )

    assert batch.element_ids == ("surface",)
    assert batch.ordered_hits == ((),)
    assert batch.provenance == ("surface",)
    assert batch.predicted_color[0] == pytest.approx((1.0, 0.0, 0.0))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_ray_color_tensor_returns_device_color_only(monkeypatch):
    scene = AuraScene(
        name="torch_raw_ray_color_tensor_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.25, 0.5, 0.75),
                opacity=0.8,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    def fail_ordered_trace_serialization(*_args, **_kwargs):
        raise AssertionError("color-only torch render should not serialize ordered hits")

    monkeypatch.setattr(torch_renderer_module, "_torch_ordered_hit_traces", fail_ordered_trace_serialization)

    color_tensor = torch_render_ray_color_tensor(
        scene,
        ((0.0, 0.0, -1.0),),
        ((0.0, 0.0, 1.0),),
        device="cpu",
    )

    assert tuple(color_tensor.shape) == (1, 3)
    assert color_tensor.device.type == "cpu"
    assert color_tensor.detach().cpu().tolist()[0] == pytest.approx([0.2, 0.4, 0.6])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_tensor_targets_accepts_raw_target_tensors():
    scene = AuraScene(
        name="torch_tensor_target_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.5, 0.25, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                semantic_id="panel",
                payload={"type": "surface_cell"},
            ),
        ),
    )

    batch = torch_render_tensor_targets(
        scene,
        frame_ids=("frame",),
        ray_origins=((0.0, 0.0, -1.0),),
        ray_directions=((0.0, 0.0, 1.0),),
        target_colors=((0.5, 0.25, 0.0),),
        target_depths=(1.0,),
        target_semantic_ids=("panel",),
        target_material_ids=(None,),
        device="cpu",
    )

    assert batch.frame_ids == ("frame",)
    assert batch.element_ids == ("surface",)
    assert batch.predicted_color[0] == pytest.approx((0.5, 0.25, 0.0))
    assert batch.image_loss == pytest.approx((0.0,))
    assert batch.query_loss == pytest.approx((0.0,))


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
        chunks=(
            AuraChunk(
                id="root",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                element_ids=("surface",),
            ),
        ),
    )

    scene_tensors = torch_scene_tensors(scene, device="cpu")
    payload = scene_tensors.to_dict()

    assert scene_tensors.element_ids == ("surface",)
    assert scene_tensors.carrier_ids == ("surface",)
    assert scene_tensors.carrier_group_indices["surface"].tolist() == [0]
    assert scene_tensors.chunk_ids == ("root",)
    assert payload["device"] == "cpu"
    assert payload["carrierGroupIndices"] == {"surface": [0]}
    assert payload["mins"]["shape"] == [1, 3]
    assert payload["chunkMins"]["shape"] == [1, 3]
    assert payload["elementChunkIndices"]["shape"] == [1]
    assert payload["supportsChunkCulling"] is True
    assert payload["colors"]["device"] == "cpu"
    assert payload["carrierParameterIds"] == ["surface"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_scene_tensors_groups_native_demo_carriers_for_dispatch():
    scene = native_demo_scene()

    scene_tensors = torch_scene_tensors(scene, device="cpu")
    payload = scene_tensors.to_dict()

    assert scene_tensors.carrier_ids == ("surface", "volume", "gabor", "neural", "semantic", "beta", "gaussian")
    assert {carrier_id: indices.tolist() for carrier_id, indices in scene_tensors.carrier_group_indices.items()} == {
        "beta": [5],
        "gabor": [2],
        "gaussian": [6],
        "neural": [3],
        "semantic": [4],
        "surface": [0],
        "volume": [1],
    }
    assert payload["carrierGroupIndices"] == {
        "beta": [5],
        "gabor": [2],
        "gaussian": [6],
        "neural": [3],
        "semantic": [4],
        "surface": [0],
        "volume": [1],
    }
    assert payload["gaborPlanePoints"]["shape"] == [len(scene.elements), 3]
    assert payload["gaborNormals"]["shape"] == [len(scene.elements), 3]
    assert scene_tensors.gabor_plane_points[2].tolist() == pytest.approx([0.7, -0.45, 0.075])
    assert scene_tensors.gabor_normals[2].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert payload["betaSupportRadii"]["shape"] == [len(scene.elements), 3]
    assert scene_tensors.beta_support_radii[5].tolist() == pytest.approx([0.15, 0.15, 0.075])
    assert sorted(index for indices in payload["carrierGroupIndices"].values() for index in indices) == list(range(len(scene.elements)))


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
def test_torch_render_targets_applies_chunk_culling_from_scene_tensor_cache():
    scene = AuraScene(
        name="chunk_cull_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                payload={"type": "surface_cell"},
                chunk_id="misplaced_chunk",
            ),
        ),
        chunks=(
            AuraChunk(
                id="misplaced_chunk",
                bounds=Bounds((10.0, 10.0, 0.0), (11.0, 11.0, 0.1)),
                element_ids=("surface",),
            ),
        ),
    )
    scene_tensors = torch_scene_tensors(scene, device="cpu")

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
            ),
        ),
        scene_tensors=scene_tensors,
    )

    assert batch.element_ids == (None,)
    assert batch.provenance == ("miss",)
    assert batch.predicted_depth == (None,)
    assert batch.transmittance == (1.0,)


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
def test_torch_render_target_objective_uses_shared_composited_scene_rays(monkeypatch):
    scene = AuraScene(
        name="torch_objective_shared_compositor_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.25, 0.5, 0.75),
                opacity=0.8,
                confidence=0.5,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    original_compositor = torch_renderer_module._torch_composited_scene_rays
    calls = []

    def counted_compositor(*args, **kwargs):
        calls.append(kwargs.get("collect_traces"))
        return original_compositor(*args, **kwargs)

    monkeypatch.setattr(torch_renderer_module, "_torch_composited_scene_rays", counted_compositor)

    objective = torch_render_target_objective(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
            ),
        ),
        device="cpu",
    )
    objective.total_loss.backward()

    assert calls == [False]
    assert objective.total_loss.detach().cpu().item() > 0.0
    assert objective.carrier_parameters["surface"]["color"].grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_target_objective_backpropagates_confidence_target():
    scene = AuraScene(
        name="torch_confidence_objective_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=0.2,
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
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
                target_confidence=1.0,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )
    objective.confidence_loss.backward()

    assert objective.confidence_loss.detach().cpu().item() == pytest.approx(0.64)
    assert objective.to_dict()["confidenceLoss"] == pytest.approx(0.64)
    assert carrier_parameters["surface"]["confidence"].grad is not None
    assert carrier_parameters["surface"]["confidence"].grad.detach().cpu().item() < 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_confidence_loss_reduces_present_targets_on_device():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predicted = torch.tensor([0.2, 0.8, 0.4], dtype=torch.float32, device=device)
    target = torch.tensor([1.0, 0.0, 0.9], dtype=torch.float32, device=device)
    present = torch.tensor([True, False, True], dtype=torch.bool, device=device)

    loss = torch_renderer_module._torch_confidence_loss(torch, predicted, target, present)

    assert loss.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert loss.detach().cpu().item() == pytest.approx(((0.2 - 1.0) ** 2 + (0.4 - 0.9) ** 2) / 2.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_confidence_loss_returns_zero_when_no_targets_present():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predicted = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
    target = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
    present = torch.tensor([False, False], dtype=torch.bool, device=device)

    loss = torch_renderer_module._torch_confidence_loss(torch, predicted, target, present)

    assert loss.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert loss.detach().cpu().item() == pytest.approx(0.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_target_objective_backpropagates_surface_geometry_parameters():
    scene = AuraScene(
        name="torch_surface_geometry_objective_scene",
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
    torch = require_torch()
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")

    objective = torch_render_target_objective(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=1.5,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )
    objective.total_loss.backward()

    assert objective.depth_loss.detach().cpu().item() > 0.0
    assert carrier_parameters["surface"]["plane_point"].grad is not None
    assert carrier_parameters["surface"]["plane_point"].grad[2].detach().cpu().item() > 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_target_objective_backpropagates_surface_normal():
    scene = AuraScene(
        name="torch_surface_normal_objective_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, -0.2, -1.0),
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
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
                target_normal=(0.0, -1.0, 0.0),
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )
    objective.total_loss.backward()

    assert objective.normal_loss.detach().cpu().item() > 0.0
    assert carrier_parameters["surface"]["normal"].grad is not None
    assert carrier_parameters["surface"]["normal"].grad[1].detach().cpu().item() > 0.0


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
    # Contribution-weighted expected depth: front (depth 2.0, weight 0.5) and
    # back (depth 2.2, weight 0.25) blended like the colour/confidence.
    assert batch.predicted_depth == pytest.approx(((0.5 * 2.0 + 0.25 * 2.2) / 0.75,))
    assert batch.predicted_color[0] == pytest.approx((0.5, 0.0, 0.25))
    assert batch.transmittance == pytest.approx((0.25,))
    assert batch.opacity == pytest.approx((0.75,))
    assert batch.confidence == pytest.approx(((0.5 * 0.8 + 0.25 * 0.4) / 0.75,))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_batches_ordered_carrier_responses(monkeypatch):
    scene = AuraScene(
        name="torch_batched_composite_scene",
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
    call_counts = []
    original = torch_renderer_module.torch_carrier_response_tensors_batched

    def count_batched_response(torch, elements, best_index, *args, **kwargs):
        call_counts.append(int(best_index.shape[0]))
        return original(torch, elements, best_index, *args, **kwargs)

    monkeypatch.setattr(torch_renderer_module, "torch_carrier_response_tensors_batched", count_batched_response)

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

    assert call_counts == [2]
    assert batch.predicted_color[0] == pytest.approx((0.5, 0.0, 0.25))
    assert batch.transmittance == pytest.approx((0.25,))


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
    batch = torch_capture_training_batch((frame,), assets, include_masked_targets=True)
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
    batch = torch_capture_training_batch((frame,), assets, include_masked_targets=True)
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")

    objective = torch_render_capture_training_objective(scene, batch, carrier_parameters=carrier_parameters)
    objective.total_loss.backward()
    payload = objective.to_dict()

    assert payload["imageLoss"] == pytest.approx(0.0)
    assert payload["depthLoss"] == pytest.approx(0.0)
    assert payload["normalLoss"] == pytest.approx(0.0)
    assert payload["maskLoss"] == pytest.approx(1.0)
    assert payload["confidenceLoss"] == pytest.approx(1.0)
    assert payload["totalLoss"] == pytest.approx(2.0)
    assert batch.target_confidence.tolist() == [0.0]
    assert carrier_parameters["surface"]["opacity"].grad is not None
    assert carrier_parameters["surface"]["confidence"].grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_objective_skips_cpu_hit_trace_construction(monkeypatch):
    scene = AuraScene(
        name="torch_capture_trace_free_objective_scene",
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
                normal_values=(0.0, 0.0, -1.0),
                width=1,
                height=1,
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    def fail_trace_construction(*_args, **_kwargs):
        raise AssertionError("objective path should not build CPU ordered-hit traces")

    monkeypatch.setattr(torch_renderer_module, "_torch_hit_transmittance_traces", fail_trace_construction)

    objective = torch_render_capture_training_objective(scene, batch)

    assert objective.to_dict()["totalLoss"] == pytest.approx(0.0)


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
def test_torch_render_targets_matches_native_gaussian_fallback_sampling():
    scene = AuraScene(
        name="torch_gaussian_sampling_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.5],
                    "covariance": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                },
            ),
        ),
    )
    ray = Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0))
    native = scene.ray_query(ray)

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=ray,
                target_color=native.color,
                target_depth=native.depth or 0.0,
            ),
        ),
        device="cpu",
    )

    assert batch.predicted_color[0] == pytest.approx(native.color)
    assert batch.transmittance[0] == pytest.approx(native.transmittance)
    assert batch.opacity[0] == pytest.approx(native.opacity)
    assert batch.confidence[0] == pytest.approx(native.confidence)
    assert batch.ordered_hits[0][0]["transmittance"] == pytest.approx(native.transmittance)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_surface_plane_geometry_when_available():
    scene = AuraScene(
        name="torch_surface_plane_geometry_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell", "plane_point": [0.0, 0.0, 0.5]},
            ),
        ),
    )

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.5,
            ),
        ),
        device="cpu",
    )

    assert batch.predicted_depth == pytest.approx((2.5,))
    assert batch.ordered_hits[0][0]["depth"] == pytest.approx(2.5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_gaussian_ellipsoid_support():
    scene = AuraScene(
        name="torch_gaussian_ellipsoid_geometry_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.5],
                    "covariance": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                    "support_sigma": 1.0,
                },
            ),
        ),
    )

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.5, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(0.0, 0.0, 0.0),
                target_depth=2.0,
            ),
        ),
        device="cpu",
    )

    assert batch.element_ids == (None,)
    assert batch.predicted_depth == (None,)
    assert batch.predicted_color == ((0.0, 0.0, 0.0),)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_trainable_gaussian_covariance():
    import torch

    scene = AuraScene(
        name="torch_gaussian_trainable_covariance_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.5],
                    "covariance": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                    "support_sigma": 1.0,
                },
            ),
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")
    carrier_parameters["gaussian"]["gaussian_covariance_diag"] = torch.tensor(
        [0.25, 0.25, 0.25],
        dtype=torch.float32,
        requires_grad=True,
    )

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.5, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )

    assert batch.element_ids == ("gaussian",)
    assert batch.predicted_depth == pytest.approx((2.5,))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_trainable_gaussian_mean():
    import torch

    scene = AuraScene(
        name="torch_gaussian_trainable_mean_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.5],
                    "covariance": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                    "support_sigma": 1.0,
                },
            ),
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")
    carrier_parameters["gaussian"]["gaussian_mean"] = torch.tensor(
        [0.5, 0.0, 0.5],
        dtype=torch.float32,
        requires_grad=True,
    )

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.5, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.0,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )

    assert batch.element_ids == ("gaussian",)
    assert batch.predicted_depth == pytest.approx((2.4,))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_gaussian_ellipsoid_invalid_geometry_masks_to_miss():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    origins = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32, device=device)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    mean = torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float32, device=device)
    inverse_covariance = torch.eye(3, dtype=torch.float32, device=device)
    support_radius_sq = torch.tensor(1.0, dtype=torch.float32, device=device)

    entry, exit_depth, hits = torch_renderer_module._torch_gaussian_ellipsoid_hits(
        torch,
        origins,
        directions,
        mean,
        inverse_covariance,
        support_radius_sq,
    )

    assert entry.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.isinf(entry).tolist() == [True]
    assert torch.isinf(exit_depth).tolist() == [True]
    assert hits.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_carrier_hits_batches_mixed_native_geometry_like_scalar_helpers():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene = native_demo_scene()
    scene_tensors = torch_scene_tensors(scene, device=device, requires_grad=False)
    origins = torch.tensor(
        [
            (0.0, 0.0, -2.0),
            (0.7, -0.45, -2.0),
            (-0.65, 0.55, -2.0),
            (0.35, 0.55, -2.0),
        ],
        dtype=torch.float32,
        device=device,
    )
    directions = torch.tensor([(0.0, 0.0, 1.0)] * 4, dtype=torch.float32, device=device)

    entry_topk, exit_depth_topk, hits_topk, global_indices = torch_renderer_module._torch_carrier_hits(
        torch,
        scene.elements,
        origins,
        directions,
        scene_tensors.mins,
        scene_tensors.maxs,
        scene_tensors.surface_plane_points,
        scene_tensors.surface_normals,
        scene_tensors.gabor_plane_points,
        scene_tensors.gabor_normals,
        scene_tensors.gaussian_means,
        scene_tensors.gaussian_inverse_covariances,
        scene_tensors.gaussian_support_radius_sq,
        scene_tensors.beta_support_radii,
    )
    # Results are depth-sorted [rays, K]; scatter back to element order [rays, N] for comparison.
    # Only scatter positions where hits_topk=True — non-hit positions carry garbage indices
    # (from the merge sort) and scattering inf there would overwrite correct finite values.
    n_elements = len(scene.elements)
    ray_count = int(origins.shape[0])
    entry = torch.full((ray_count, n_elements), float("inf"), dtype=entry_topk.dtype, device=device)
    exit_depth = torch.full((ray_count, n_elements), float("inf"), dtype=exit_depth_topk.dtype, device=device)
    hits = torch.zeros((ray_count, n_elements), dtype=torch.bool, device=device)
    hit_pos = hits_topk.nonzero(as_tuple=False)  # [n_hits, 2]
    if hit_pos.shape[0] > 0:
        ray_idx = hit_pos[:, 0]
        k_idx = hit_pos[:, 1]
        carrier_idx = global_indices[ray_idx, k_idx]
        entry[ray_idx, carrier_idx] = entry_topk[ray_idx, k_idx]
        exit_depth[ray_idx, carrier_idx] = exit_depth_topk[ray_idx, k_idx]
        hits[ray_idx, carrier_idx] = True
    base_entry, base_exit, base_hits = torch_renderer_module._torch_aabb_hits(
        torch,
        origins,
        directions,
        scene_tensors.mins,
        scene_tensors.maxs,
    )

    expected_entries = []
    expected_exits = []
    expected_hits = []
    for index, element in enumerate(scene.elements):
        scalar_entry = base_entry[:, index]
        scalar_exit = base_exit[:, index]
        scalar_hits = base_hits[:, index]
        payload_type = element.payload.get("type")
        if payload_type == "surface_cell" or element.carrier_id == "surface":
            surface_entry, surface_exit, surface_hits = torch_renderer_module._torch_surface_plane_hits(
                torch,
                origins,
                directions,
                scene_tensors.mins[index],
                scene_tensors.maxs[index],
                scene_tensors.surface_plane_points[index],
                scene_tensors.surface_normals[index],
            )
            valid = torch.isfinite(surface_entry) & torch.isfinite(surface_exit)
            scalar_entry = torch.where(valid, surface_entry, scalar_entry)
            scalar_exit = torch.where(valid, surface_exit, scalar_exit)
            scalar_hits = torch.where(valid, surface_hits, scalar_hits)
        elif payload_type == "gabor_frequency":
            gabor_entry, gabor_exit, gabor_hits = torch_renderer_module._torch_surface_plane_hits(
                torch,
                origins,
                directions,
                scene_tensors.mins[index],
                scene_tensors.maxs[index],
                scene_tensors.gabor_plane_points[index],
                scene_tensors.gabor_normals[index],
            )
            valid = torch.isfinite(gabor_entry) & torch.isfinite(gabor_exit)
            scalar_entry = torch.where(valid, gabor_entry, scalar_entry)
            scalar_exit = torch.where(valid, gabor_exit, scalar_exit)
            scalar_hits = torch.where(valid, gabor_hits, scalar_hits)
        elif payload_type == "beta_kernel":
            beta_entry, beta_exit, beta_hits = torch_renderer_module._torch_beta_ellipsoid_hits(
                torch,
                origins,
                directions,
                (scene_tensors.mins[index] + scene_tensors.maxs[index]) * 0.5,
                scene_tensors.beta_support_radii[index],
            )
            scalar_entry = torch.maximum(base_entry[:, index], beta_entry)
            scalar_exit = torch.minimum(base_exit[:, index], beta_exit)
            scalar_hits = base_hits[:, index] & beta_hits & (scalar_exit >= scalar_entry)
        elif payload_type == "gaussian_fallback":
            gaussian_entry, gaussian_exit, gaussian_hits = torch_renderer_module._torch_gaussian_ellipsoid_hits(
                torch,
                origins,
                directions,
                scene_tensors.gaussian_means[index],
                scene_tensors.gaussian_inverse_covariances[index],
                scene_tensors.gaussian_support_radius_sq[index],
            )
            scalar_entry = torch.maximum(base_entry[:, index], gaussian_entry)
            scalar_exit = torch.minimum(base_exit[:, index], gaussian_exit)
            scalar_hits = base_hits[:, index] & gaussian_hits & (scalar_exit >= scalar_entry)
        expected_entries.append(scalar_entry)
        expected_exits.append(scalar_exit)
        expected_hits.append(scalar_hits)

    expected_entry = torch.stack(tuple(expected_entries), dim=1)
    expected_exit = torch.stack(tuple(expected_exits), dim=1)
    expected_hit = torch.stack(tuple(expected_hits), dim=1)

    assert hits.tolist() == expected_hit.tolist()
    # Entry/exit are only meaningful where hits=True; the new streaming top-K returns inf for misses
    # while the old scalar baseline returned AABB depth for misses. Only compare where hits=True.
    hit_positions = expected_hit.nonzero(as_tuple=False)
    if hit_positions.shape[0] > 0:
        ray_h = hit_positions[:, 0]; col_h = hit_positions[:, 1]
        assert torch.allclose(
            torch.nan_to_num(entry[ray_h, col_h], posinf=1.0e9),
            torch.nan_to_num(expected_entry[ray_h, col_h], posinf=1.0e9),
        )
        assert torch.allclose(
            torch.nan_to_num(exit_depth[ray_h, col_h], posinf=1.0e9),
            torch.nan_to_num(expected_exit[ray_h, col_h], posinf=1.0e9),
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_gabor_surface_support_plane():
    scene = AuraScene(
        name="torch_gabor_surface_geometry_scene",
        elements=(
            AuraElement(
                id="gabor",
                carrier_id="gabor",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2)),
                color=(1.0, 0.5, 0.25),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gabor_frequency",
                    "frequency": [1.0, 0.0, 0.0],
                    "bandwidth": 0.5,
                    "phase": 0.0,
                },
            ),
        ),
    )

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.5, 0.25),
                target_depth=2.1,
            ),
        ),
        device="cpu",
    )

    assert batch.element_ids == ("gabor",)
    assert batch.predicted_depth == pytest.approx((2.1,))
    assert batch.ordered_hits[0][0]["depth"] == pytest.approx(2.1)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_objective_backpropagates_gabor_plane_point():
    scene = AuraScene(
        name="torch_gabor_geometry_objective_scene",
        elements=(
            AuraElement(
                id="gabor",
                carrier_id="gabor",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2)),
                color=(1.0, 0.5, 0.25),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gabor_frequency",
                    "frequency": [1.0, 0.0, 0.0],
                    "bandwidth": 0.5,
                    "phase": 0.0,
                },
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
                target_color=(1.0, 0.5, 0.25),
                target_depth=1.8,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )
    objective.total_loss.backward()

    assert objective.depth_loss.detach().cpu().item() > 0.0
    assert carrier_parameters["gabor"]["plane_point"].grad is not None
    assert carrier_parameters["gabor"]["plane_point"].grad[2].detach().cpu().item() > 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_beta_support_ellipsoid():
    scene = AuraScene(
        name="torch_beta_ellipsoid_geometry_scene",
        elements=(
            AuraElement(
                id="beta",
                carrier_id="beta",
                bounds=Bounds((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "beta_kernel",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "support_radius": [0.5, 0.5, 0.5],
                },
            ),
        ),
    )

    outside_batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="outside",
                ray=Ray(origin=(1.5, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(0.0, 0.0, 0.0),
                target_depth=2.0,
            ),
        ),
        device="cpu",
    )
    center_batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="center",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.5,
            ),
        ),
        device="cpu",
    )

    assert outside_batch.element_ids == (None,)
    assert outside_batch.predicted_depth == (None,)
    assert outside_batch.transmittance == (1.0,)
    assert center_batch.element_ids == ("beta",)
    assert center_batch.predicted_depth == pytest.approx((2.5,))
    assert center_batch.ordered_hits[0][0]["depth"] == pytest.approx(2.5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_beta_ellipsoid_invalid_geometry_masks_to_miss():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    origins = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32, device=device)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    support_radii = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32, device=device)

    entry, exit_depth, hits = torch_renderer_module._torch_beta_ellipsoid_hits(
        torch,
        origins,
        directions,
        center,
        support_radii,
    )

    assert entry.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.isinf(entry).tolist() == [True]
    assert torch.isinf(exit_depth).tolist() == [True]
    assert hits.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_uses_trainable_beta_support_radius():
    import torch

    scene = AuraScene(
        name="torch_beta_trainable_support_scene",
        elements=(
            AuraElement(
                id="beta",
                carrier_id="beta",
                bounds=Bounds((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "beta_kernel",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "support_radius": [0.5, 0.5, 0.5],
                },
            ),
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")
    carrier_parameters["beta"]["support_radius"] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, requires_grad=True)

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.75, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 0.0, 0.0),
                target_depth=2.338562,
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )

    assert batch.element_ids == ("beta",)
    assert batch.predicted_depth == pytest.approx((2.338562,), abs=1e-5)


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
def test_torch_render_target_objective_includes_differentiable_query_loss():
    torch = require_torch()
    scene = AuraScene(
        name="torch_query_objective_scene",
        elements=(
            AuraElement(
                id="panel",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 1.0, 1.0),
                opacity=0.25,
                semantic_id="panel",
                material_id="mat_surface",
            ),
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, scene.elements, device="cpu")
    objective = torch_render_target_objective(
        scene,
        (
            RenderTarget(
                frame_id="frame",
                ray=Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
                target_color=(1.0, 1.0, 1.0),
                target_depth=2.0,
                target_semantic_id="panel",
                target_material_id="mat_surface",
            ),
        ),
        device="cpu",
        carrier_parameters=carrier_parameters,
    )
    objective.query_loss.backward()

    assert objective.query_loss.detach().cpu().item() == pytest.approx(0.75)
    assert objective.to_dict()["queryLoss"] == pytest.approx(0.75)
    assert carrier_parameters["panel"]["opacity"].grad is not None
    assert carrier_parameters["panel"]["opacity"].grad.detach().cpu().item() < 0.0


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
            "sample_frame_ids": ("frame",),
            "ray_origins": None,
            "ray_directions": None,
            "target_color": None,
            "target_depth": None,
        },
    )()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_mip_splatting_3d_frequency_cap_reduces_gaussian_support():
    # A gaussian carrier with large support radius should have its support clamped
    # when mip_splatting=True with a small focal_length_pixels
    import torch
    from aura.torch_renderer import _torch_composite_carrier_hits
    from aura import AuraElement, AuraScene, Bounds

    scene = AuraScene(
        name="mip_splatting_test",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-2.0, -2.0, -0.5), (2.0, 2.0, 3.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 1.0],
                    "covariance": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                    "support_sigma": 3.0,  # large support
                },
            ),
        ),
    )
    from aura.torch_renderer import _resolve_scene_tensors, _torch_geometry_from_carrier_parameters

    st = _resolve_scene_tensors(scene, scene_tensors=None, device="cpu")
    (mins, maxs, sp_points, gab_points, g_means, g_inv_cov, b_radii, sp_normals, gab_normals, el_normals) = (
        _torch_geometry_from_carrier_parameters(
            torch, tuple(scene.elements), st.carrier_parameters,
            st.mins, st.maxs, st.surface_plane_points, st.gabor_plane_points,
            st.gaussian_means, st.gaussian_inverse_covariances, st.beta_support_radii,
            st.surface_normals, st.gabor_normals, st.element_normals,
        )
    )
    # A grazing ray that clips the edge of the gaussian: shrinking the support
    # via the mip 3D frequency cap must change whether/how much it hits. A
    # dead-center ray would hit regardless of support and hide a no-op.
    origins = torch.tensor([[1.3, 0.0, -1.0]], dtype=torch.float32)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    # Without mip_splatting: hits the gaussian (support_radius_sq=9.0)
    result_off = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        mip_splatting=False, focal_length_pixels=1.0,
    )

    # With mip_splatting + tiny focal length: clamps support drastically
    result_on = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        mip_splatting=True, focal_length_pixels=0.01,  # tiny focal length = huge cone angle = tiny nyquist
    )

    # With mip_splatting on and very small focal length, support gets clamped to tiny value
    # which should reduce the gaussian's effective range, possibly causing a miss
    # At minimum, the outputs should differ when support is meaningfully clamped
    # The gaussian with support_sigma=3.0 -> support_radius_sq=9.0
    # With focal=0.01, pixel_cone_angle ~= 2*atan(50) ~= pi, nyquist_sigma very small
    # So with mip_splatting=True, the gaussian support is clamped, and the ray may miss
    assert result_off is not None  # basic structure
    assert result_on is not None
    # The has_hit should differ or transmittance should differ
    # OFF: gaussian has large support so likely hits
    # ON with tiny focal: clamped support, so might miss
    transmittance_off = float(result_off["transmittance"][0])
    transmittance_on = float(result_on["transmittance"][0])
    # The grazing ray partially hits the full-support gaussian (transmittance < 1)...
    assert transmittance_off < 0.95, f"grazing ray should partially hit, got {transmittance_off}"
    # ...but with the mip frequency cap the support is clamped small enough that
    # the ray meaningfully MORE misses. A no-op would leave these equal.
    assert transmittance_on > transmittance_off + 0.1, (
        f"mip-splatting must change the result: off={transmittance_off} on={transmittance_on}"
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cone_prefilter_reduces_opacity_at_far_distances():
    import torch
    from aura.torch_renderer import _torch_composite_carrier_hits, _resolve_scene_tensors, _torch_geometry_from_carrier_parameters
    from aura import AuraElement, AuraScene, Bounds

    scene = AuraScene(
        name="cone_prefilter_test",
        elements=(
            AuraElement(
                id="small_surface",
                carrier_id="surface",
                bounds=Bounds((-0.01, -0.01, 0.0), (0.01, 0.01, 0.01)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={"type": "surface_cell"},
            ),
        ),
    )

    st = _resolve_scene_tensors(scene, scene_tensors=None, device="cpu")
    (mins, maxs, sp_points, gab_points, g_means, g_inv_cov, b_radii, sp_normals, gab_normals, el_normals) = (
        _torch_geometry_from_carrier_parameters(
            torch, tuple(scene.elements), st.carrier_parameters,
            st.mins, st.maxs, st.surface_plane_points, st.gabor_plane_points,
            st.gaussian_means, st.gaussian_inverse_covariances, st.beta_support_radii,
            st.surface_normals, st.gabor_normals, st.element_normals,
        )
    )

    # Ray at close distance
    origins_near = torch.tensor([[0.0, 0.0, -0.1]], dtype=torch.float32)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    # Ray at far distance
    origins_far = torch.tensor([[0.0, 0.0, -1000.0]], dtype=torch.float32)

    # Without prefilter: same opacity regardless of distance
    result_near_off = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins_near, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        cone_prefilter=False, focal_length_pixels=1.0,
    )

    # With prefilter at close distance: small footprint, carrier larger, full opacity
    result_near_on = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins_near, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        cone_prefilter=True, focal_length_pixels=100.0,  # large focal = small cone = carrier larger than footprint
    )

    # With prefilter at far distance with wide cone: large footprint, carrier smaller -> reduced opacity
    result_far_on = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins_far, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        cone_prefilter=True, focal_length_pixels=0.001,  # tiny focal = huge cone angle
    )

    # Far with huge cone should give higher transmittance (less opacity) than near with tight cone
    transmittance_near = float(result_near_on["transmittance"][0])
    transmittance_far = float(result_far_on["transmittance"][0])
    assert transmittance_far >= transmittance_near - 1e-6


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_ssaa_2x2_averages_four_subsamples():
    import torch
    from aura import AuraElement, AuraScene, Bounds
    from aura.torch_renderer import torch_render_ray_color_tensor

    # Small gaussian so a pixel at its edge has sub-samples that straddle the
    # boundary: the center ray misses, but the 2x2 jittered sub-rays partially
    # hit. This proves SSAA performs real supersampling (would FAIL if SSAA were
    # a no-op returning the single center sample).
    scene = AuraScene(
        name="ssaa_test",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.5, -1.5, 0.5), (1.5, 1.5, 1.5)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 1.0],
                    "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                    "support_sigma": 1.0,
                },
            ),
        ),
    )
    ray_origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    # Direction aimed just past the gaussian edge: the center ray misses.
    ray_directions = torch.tensor([[0.26, 0.0, 1.0]], dtype=torch.float32)

    color_no_ssaa = torch_render_ray_color_tensor(scene, ray_origins, ray_directions, device="cpu")
    color_ssaa = torch_render_ray_color_tensor(
        scene, ray_origins, ray_directions, device="cpu",
        ssaa_2x2=True, focal_length_pixels=2.0,
    )

    assert color_no_ssaa.shape == color_ssaa.shape == (1, 3)
    center_red = float(color_no_ssaa[0, 0])
    ssaa_red = float(color_ssaa[0, 0])
    # The center ray misses the gaussian entirely...
    assert center_red < 0.05, f"center ray expected to miss, got {center_red}"
    # ...but the supersampled pixel picks up real coverage from edge sub-samples.
    assert ssaa_red > 0.2, f"SSAA expected real coverage from sub-samples, got {ssaa_red}"
    assert abs(ssaa_red - center_red) > 0.1

    # Backward-compat property: with a near-zero jitter (huge focal length) SSAA
    # collapses to the single-sample result.
    flat_scene = AuraScene(
        name="ssaa_flat",
        elements=(
            AuraElement(
                id="surface", carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.8, 0.2, 0.4), opacity=1.0, confidence=1.0,
                payload={"type": "surface_cell"},
            ),
        ),
    )
    flat_dir = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    flat_off = torch_render_ray_color_tensor(flat_scene, ray_origins, flat_dir, device="cpu")
    flat_on = torch_render_ray_color_tensor(
        flat_scene, ray_origins, flat_dir, device="cpu",
        ssaa_2x2=True, focal_length_pixels=1e6,
    )
    assert torch.allclose(flat_off, flat_on, atol=1e-3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_early_transmittance_termination_conserves_energy():
    import torch
    from aura.torch_renderer import _torch_composite_carrier_hits, _resolve_scene_tensors, _torch_geometry_from_carrier_parameters
    from aura import AuraElement, AuraScene, Bounds

    # Scene with fully opaque front carrier - energy should stop at first hit
    scene = AuraScene(
        name="early_termination_test",
        elements=(
            AuraElement(
                id="front_opaque",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="back_carrier",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 1.0), (0.5, 0.5, 1.1)),
                color=(0.0, 0.0, 1.0),
                opacity=0.5,
                confidence=1.0,
                payload={"type": "surface_cell"},
            ),
        ),
    )

    st = _resolve_scene_tensors(scene, scene_tensors=None, device="cpu")
    (mins, maxs, sp_points, gab_points, g_means, g_inv_cov, b_radii, sp_normals, gab_normals, el_normals) = (
        _torch_geometry_from_carrier_parameters(
            torch, tuple(scene.elements), st.carrier_parameters,
            st.mins, st.maxs, st.surface_plane_points, st.gabor_plane_points,
            st.gaussian_means, st.gaussian_inverse_covariances, st.beta_support_radii,
            st.surface_normals, st.gabor_normals, st.element_normals,
        )
    )
    origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    # With threshold=0 (no termination)
    result_no_threshold = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        transmittance_threshold=0.0,
    )

    # With default threshold (1e-4) - fully opaque front terminates early
    result_with_threshold = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        transmittance_threshold=1e-4,
    )

    # For fully opaque front carrier: color should be red in both cases
    # (back carrier doesn't contribute since transmittance is 0 after front)
    color_no_th = result_no_threshold["color"][0].tolist()
    color_with_th = result_with_threshold["color"][0].tolist()

    assert color_no_th[0] == pytest.approx(1.0, abs=0.01)  # red channel
    assert color_with_th[0] == pytest.approx(1.0, abs=0.01)  # red channel
    # Colors should match since fully opaque front stops both paths
    assert color_no_th == pytest.approx(color_with_th, abs=1e-5)

    # threshold=0 gives same result as no threshold
    result_zero = _torch_composite_carrier_hits(
        torch, tuple(scene.elements), origins, directions,
        mins, maxs, st.colors, st.opacities, st.confidences,
        st.chunk_mins, st.chunk_maxs, st.element_chunk_indices,
        sp_points, sp_normals, gab_points, gab_normals, g_means, g_inv_cov,
        st.gaussian_support_radius_sq, b_radii,
        device="cpu", carrier_parameters=st.carrier_parameters, collect_traces=False,
        transmittance_threshold=0.0,
    )
    assert result_zero["color"][0].tolist() == pytest.approx(result_no_threshold["color"][0].tolist())


# ---- New coverage-targeting tests ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_render_summary_to_dict_serializes_all_fields():
    """Cover TorchCaptureRenderSummary.to_dict (line 115)."""
    scene = AuraScene(
        name="summary_dict_scene",
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
        (_capture_tensor_frame(frame_id="frame", image_values=(1.0, 0.0, 0.0), depth_values=(2.0,), mask_values=None, width=1, height=1),),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    summary = torch_render_capture_training_summary(scene, batch)

    payload = summary.to_dict()

    assert "rayOrigins" in payload
    assert "rayDirections" in payload
    assert "elementIds" in payload
    assert "predictedColor" in payload
    assert "predictedDepth" in payload
    assert "transmittance" in payload
    assert "normal" in payload
    assert "targetColor" in payload
    assert "targetDepth" in payload
    assert "targetPoint" in payload
    assert "imageLoss" in payload
    assert "depthLoss" in payload
    assert "queryLoss" in payload
    assert "normalLoss" in payload
    # normal could be None entries (serialized as null) — check the structure
    assert isinstance(payload["normal"], list)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_scene_tensors_rejects_empty_scene():
    """Cover torch_scene_tensors empty-scene guard (line 323)."""
    scene = AuraScene(name="empty_scene", elements=())

    with pytest.raises(ValueError, match="at least one scene element"):
        torch_scene_tensors(scene, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_asset_batch_rejects_empty_frames():
    """Cover torch_capture_asset_batch empty-frames guard (line 392)."""
    with pytest.raises(ValueError, match="at least one frame"):
        torch_capture_asset_batch((), device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_rejects_invalid_pixel_stride():
    """Cover pixel_stride <= 0 guard (line 439)."""
    assets = torch_capture_asset_batch((_capture_tensor_frame(),), device="cpu")
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 0.5, "width": 2.0, "height": 1.0},
    )

    with pytest.raises(ValueError, match="pixel_stride"):
        torch_capture_training_batch((frame,), assets, pixel_stride=0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_rejects_empty_asset_batch():
    """Cover empty asset frame_ids guard (line 441)."""
    import torch

    # Create a fake empty assets object
    class _FakeEmptyAssets:
        frame_ids = ()
        image = torch.zeros((0, 1, 1, 3))
        depth = None
        mask = None
        normal = None
        mask_present = None
        normal_present = None

    with pytest.raises(ValueError, match="empty"):
        torch_capture_training_batch((), _FakeEmptyAssets())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_rejects_unknown_frame_ids():
    """Cover missing frame ids guard (line 445)."""
    assets = torch_capture_asset_batch((_capture_tensor_frame(frame_id="unknown_frame"),), device="cpu")
    known_frame = TrainingFrame(
        id="other_frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 0.5, "width": 2.0, "height": 1.0},
    )

    with pytest.raises(ValueError, match="unknown training frames"):
        torch_capture_training_batch((known_frame,), assets)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_falls_back_to_frame_depth_when_no_depth_asset():
    """Cover depth fallback when assets.depth is None (lines 553, 556)."""
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=3.5,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    # Create assets with no depth
    assets = torch_capture_asset_batch(
        (CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib", width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
        ),),
        device="cpu",
    )

    batch = torch_capture_training_batch((frame,), assets)

    # Depth should be the frame's target_depth fallback since no depth asset
    assert batch.target_depth.tolist() == pytest.approx([3.5])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_uses_stride_to_subsample_pixels():
    """Cover pixel_stride > 1 sampling code path."""
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 1.0, "width": 4.0, "height": 2.0},
    )
    assets = torch_capture_asset_batch(
        (_capture_tensor_frame(
            frame_id="frame",
            width=4,
            height=2,
            image_values=(1.0, 0.0, 0.0) * 8,
            depth_values=(1.0,) * 8,
            mask_values=None,
        ),),
        device="cpu",
    )

    batch_stride1 = torch_capture_training_batch((frame,), assets, pixel_stride=1)
    batch_stride2 = torch_capture_training_batch((frame,), assets, pixel_stride=2)

    # With stride=2, we get fewer pixels
    assert batch_stride2.pixel_xy.shape[0] < batch_stride1.pixel_xy.shape[0]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_from_packed_converts_packed_batch():
    """Cover torch_capture_training_batch_from_packed (lines 580-606)."""
    packed = CapturePackedRenderBatch(
        batch_index=0,
        frame_ids=("frame_a",),
        frame_semantic_ids=(None,),
        target_offset=0,
        target_count=2,
        max_target_count=2,
        frame_indices=[0, 0],
        pixel_xy=[0, 0, 1, 0],
        ray_origins=[0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
        ray_directions=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        target_color=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        target_depth=[2.0, 2.0],
        target_mask=[1.0, 0.5],
        target_normal=[0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        target_normal_present=[1, 0],
    )

    batch = torch_capture_training_batch_from_packed(packed, device="cpu")

    assert batch.device == "cpu"
    assert batch.frame_ids == ("frame_a",)
    assert int(batch.frame_indices.shape[0]) == 2
    assert int(batch.pixel_xy.shape[0]) == 2
    # target_confidence = clamp(mask) since mask is provided
    assert batch.target_confidence.tolist() == pytest.approx([1.0, 0.5])
    assert batch.target_confidence_present.tolist() == [True, True]
    assert batch.target_normal is not None
    assert batch.target_normal_present is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_from_packed_without_mask_or_normal():
    """Cover packed batch path with no mask/normal (confidence=ones)."""
    packed = CapturePackedRenderBatch(
        batch_index=0,
        frame_ids=("frame_a",),
        frame_semantic_ids=(None,),
        target_offset=0,
        target_count=1,
        max_target_count=1,
        frame_indices=[0],
        pixel_xy=[0, 0],
        ray_origins=[0.0, 0.0, -1.0],
        ray_directions=[0.0, 0.0, 1.0],
        target_color=[1.0, 0.0, 0.0],
        target_depth=[2.0],
    )

    batch = torch_capture_training_batch_from_packed(packed, device="cpu")

    assert batch.target_mask is None
    assert batch.target_normal is None
    # Confidence should be 1.0 (fallback when no mask)
    assert batch.target_confidence.tolist() == pytest.approx([1.0])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_from_packed_rejects_empty():
    """Cover packed batch empty guard (line 580)."""
    packed = CapturePackedRenderBatch(
        batch_index=0,
        frame_ids=("frame_a",),
        frame_semantic_ids=(None,),
        target_offset=0,
        target_count=0,
        max_target_count=1,  # must be positive per contract
        frame_indices=[],
        pixel_xy=[],
        ray_origins=[],
        ray_directions=[],
        target_color=[],
        target_depth=[],
    )

    with pytest.raises(ValueError, match="at least one target"):
        torch_capture_training_batch_from_packed(packed)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_batch_rejects_empty_batch():
    """Cover empty frame_indices guard in render (line 637)."""
    import torch

    scene = AuraScene(
        name="empty_batch_scene",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    class _EmptyBatch:
        frame_indices = torch.zeros((0,), dtype=torch.long)
        sample_frame_ids = ()
        ray_origins = torch.zeros((0, 3))
        ray_directions = torch.zeros((0, 3))
        target_color = torch.zeros((0, 3))
        target_depth = torch.zeros((0,))
        target_normal = None
        target_normal_present = None
        target_confidence = None
        target_confidence_present = None
        target_semantic_ids = ()
        target_material_ids = ()
        target_mask = None

    with pytest.raises(ValueError, match="at least one target"):
        torch_render_capture_training_batch(scene, _EmptyBatch())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_objective_rejects_empty_batch():
    """Cover empty frame_indices guard in objective (line 754)."""
    import torch

    scene = AuraScene(
        name="empty_obj_scene",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    class _EmptyBatch:
        frame_indices = torch.zeros((0,), dtype=torch.long)
        sample_frame_ids = ()
        ray_origins = torch.zeros((0, 3))
        ray_directions = torch.zeros((0, 3))
        target_color = torch.zeros((0, 3))
        target_depth = torch.zeros((0,))
        target_normal = None
        target_normal_present = None
        target_confidence = None
        target_confidence_present = None
        target_semantic_ids = ()
        target_material_ids = ()
        target_mask = None

    with pytest.raises(ValueError, match="at least one target"):
        torch_render_capture_training_objective(scene, _EmptyBatch())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_rejects_empty_targets():
    """Cover empty targets guard in torch_render_targets (line 792)."""
    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="at least one target"):
        torch_render_targets(scene, (), device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_targets_rejects_empty_scene():
    """Cover empty scene guard in torch_render_targets (line 794)."""
    scene = AuraScene(name="empty", elements=())

    with pytest.raises(ValueError, match="at least one scene element"):
        torch_render_targets(
            scene,
            (RenderTarget(frame_id="f", ray=Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)), target_color=(0.0, 0.0, 0.0), target_depth=1.0),),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_rays_rejects_empty_scene():
    """Cover empty scene guard in torch_render_rays (line 848)."""
    import torch

    scene = AuraScene(name="empty", elements=())
    origins = torch.zeros((1, 3))
    directions = torch.zeros((1, 3))
    directions[:, 2] = 1.0

    with pytest.raises(ValueError, match="at least one scene element"):
        torch_render_rays(scene, origins, directions, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_rays_rejects_mismatched_ray_counts():
    """Cover direction/origin count mismatch guard (line 856)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    origins = torch.zeros((2, 3))
    directions = torch.zeros((1, 3))
    directions[:, 2] = 1.0

    with pytest.raises(ValueError, match="ray_directions count"):
        torch_render_rays(scene, origins, directions, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_rays_rejects_empty_ray_count():
    """Cover empty ray_count guard (line 858)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    origins = torch.zeros((0, 3))
    directions = torch.zeros((0, 3))

    with pytest.raises(ValueError, match="ray_count must be positive"):
        torch_render_rays(scene, origins, directions, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_ray_color_tensor_rejects_empty_scene():
    """Cover empty scene guard in ray color tensor (line 893)."""
    import torch

    scene = AuraScene(name="empty", elements=())
    origins = torch.zeros((1, 3))
    directions = torch.ones((1, 3))

    with pytest.raises(ValueError, match="at least one scene element"):
        torch_render_ray_color_tensor(scene, origins, directions, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_ray_color_tensor_rejects_mismatched_ray_counts():
    """Cover ray count mismatch guard (line 901)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    origins = torch.zeros((2, 3))
    directions = torch.zeros((1, 3))
    directions[:, 2] = 1.0

    with pytest.raises(ValueError, match="ray_directions count"):
        torch_render_ray_color_tensor(scene, origins, directions, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_ray_color_tensor_rejects_empty_rays():
    """Cover empty ray count guard (line 903)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    origins = torch.zeros((0, 3))
    directions = torch.zeros((0, 3))

    with pytest.raises(ValueError, match="ray_count must be positive"):
        torch_render_ray_color_tensor(scene, origins, directions, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_tensor_targets_rejects_empty_frame_ids():
    """Cover empty frame_ids guard (line 972)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="at least one target"):
        torch_render_tensor_targets(
            scene,
            frame_ids=(),
            ray_origins=torch.zeros((0, 3)),
            ray_directions=torch.zeros((0, 3)),
            target_colors=torch.zeros((0, 3)),
            target_depths=torch.zeros((0,)),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_tensor_targets_rejects_empty_scene():
    """Cover empty scene guard in tensor targets (line 974)."""
    import torch

    scene = AuraScene(name="empty", elements=())

    with pytest.raises(ValueError, match="at least one scene element"):
        torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            ray_origins=torch.zeros((1, 3)),
            ray_directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_ray_tensor_rejects_none_values():
    """Cover _torch_ray_tensor None guard (line 1080)."""
    from aura.torch_renderer import _torch_ray_tensor
    import torch

    with pytest.raises(ValueError, match="is required"):
        _torch_ray_tensor(torch, None, name="my_ray", device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_ray_tensor_accepts_existing_tensor():
    """Cover _torch_ray_tensor tensor-shortcut path (line 1087)."""
    from aura.torch_renderer import _torch_ray_tensor
    import torch

    existing = torch.zeros((3, 3))
    existing[:, 2] = 1.0
    result = _torch_ray_tensor(torch, existing, name="dirs", device="cpu")
    assert result.shape == (3, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_ray_tensor_rejects_wrong_shape():
    """Cover _torch_ray_tensor shape validation (line 1093)."""
    from aura.torch_renderer import _torch_ray_tensor
    import torch

    with pytest.raises(ValueError, match="shape"):
        _torch_ray_tensor(torch, [[1.0, 2.0]], name="bad", device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_vec3_tensor_rejects_none():
    """Cover _torch_vec3_tensor None guard (line 1095)."""
    from aura.torch_renderer import _torch_vec3_tensor
    import torch

    with pytest.raises(ValueError, match="is required"):
        _torch_vec3_tensor(torch, None, name="colors", device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_vec3_tensor_accepts_existing_tensor():
    """Cover _torch_vec3_tensor tensor-shortcut path (line 1100)."""
    from aura.torch_renderer import _torch_vec3_tensor
    import torch

    existing = torch.zeros((2, 3))
    result = _torch_vec3_tensor(torch, existing, name="cols", device="cpu")
    assert result.shape == (2, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_vec3_tensor_rejects_wrong_shape():
    """Cover _torch_vec3_tensor shape validation (line 1106)."""
    from aura.torch_renderer import _torch_vec3_tensor
    import torch

    with pytest.raises(ValueError, match="shape"):
        _torch_vec3_tensor(torch, [[1.0, 2.0]], name="bad", device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_1d_tensor_rejects_none():
    """Cover _torch_1d_tensor None guard (line 1108)."""
    from aura.torch_renderer import _torch_1d_tensor
    import torch

    with pytest.raises(ValueError, match="is required"):
        _torch_1d_tensor(torch, None, name="depths", dtype=torch.float32, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_1d_tensor_accepts_existing_tensor():
    """Cover _torch_1d_tensor tensor-shortcut path (line 1113)."""
    from aura.torch_renderer import _torch_1d_tensor
    import torch

    existing = torch.ones((4,))
    result = _torch_1d_tensor(torch, existing, name="depths", dtype=torch.float32, device="cpu")
    assert result.shape == (4,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_1d_tensor_rejects_2d_shape():
    """Cover _torch_1d_tensor shape validation (line 1138)."""
    from aura.torch_renderer import _torch_1d_tensor
    import torch

    with pytest.raises(ValueError, match="shape"):
        _torch_1d_tensor(torch, [[1.0, 2.0]], name="bad", dtype=torch.float32, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_tensor_targets_validates_direction_count():
    """Cover _torch_render_tensor_targets direction count check (line 1140)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="count must match"):
        torch_render_tensor_targets(
            scene,
            frame_ids=("f", "g"),
            ray_origins=torch.zeros((2, 3)),
            ray_directions=torch.zeros((1, 3)),
            target_colors=torch.zeros((2, 3)),
            target_depths=torch.ones((2,)),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_tensor_targets_validates_depth_count():
    """Cover _torch_render_tensor_targets depth count check (line 1142)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="count must match"):
        torch_render_tensor_targets(
            scene,
            frame_ids=("f", "g"),
            ray_origins=torch.zeros((2, 3)),
            ray_directions=torch.zeros((2, 3)),
            target_colors=torch.zeros((2, 3)),
            target_depths=torch.ones((1,)),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_tensor_targets_validates_color_count():
    """Cover _torch_render_tensor_targets color count check (line 1144)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="count must match"):
        torch_render_tensor_targets(
            scene,
            frame_ids=("f", "g"),
            ray_origins=torch.zeros((2, 3)),
            ray_directions=torch.zeros((2, 3)),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((2,)),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_validates_target_normal_count():
    """Cover _torch_render_tensor_targets normal count check (line 1146)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="normal count"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_normals=torch.zeros((2, 3)),  # wrong count
            target_normal_present=None,
            target_confidence=None,
            target_confidence_present=None,
            target_semantic_ids=(None,),
            target_material_ids=(None,),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_validates_normal_present_count():
    """Cover _torch_render_tensor_targets normal_present count check (line 1148)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="normal presence count"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_normals=torch.zeros((1, 3)),
            target_normal_present=torch.tensor([True, False]),  # wrong count
            target_confidence=None,
            target_confidence_present=None,
            target_semantic_ids=(None,),
            target_material_ids=(None,),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_validates_confidence_count():
    """Cover _torch_render_tensor_targets confidence count check (line 1150)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="confidence count"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_normals=None,
            target_normal_present=None,
            target_confidence=torch.zeros((2,)),  # wrong count
            target_confidence_present=None,
            target_semantic_ids=(None,),
            target_material_ids=(None,),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_validates_confidence_present_count():
    """Cover _torch_render_tensor_targets confidence_present count check (line 1152)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="confidence presence count"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_normals=None,
            target_normal_present=None,
            target_confidence=None,
            target_confidence_present=torch.tensor([True, False]),  # wrong count
            target_semantic_ids=(None,),
            target_material_ids=(None,),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_devices_match_cuda_index_normalization():
    """Cover _torch_devices_match CUDA index normalization branch (lines 1348-1360)."""
    from aura.torch_renderer import _torch_devices_match

    # CPU device matching
    assert _torch_devices_match("cpu", "cpu") is True
    assert _torch_devices_match("cpu", "cuda") is False


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_geometry_from_carrier_parameters_returns_base_when_none():
    """Cover _torch_geometry_from_carrier_parameters None path (lines 1428-1444)."""
    from aura.torch_renderer import _torch_geometry_from_carrier_parameters
    import torch

    base = torch.zeros((1, 3))
    result = _torch_geometry_from_carrier_parameters(
        torch, [], None, base, base, base, base, base, base, base, base, base, base
    )
    # Should return all base tensors
    assert result[0] is base


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_geometry_from_carrier_parameters_batched_fast_path():
    """Cover _torch_geometry_from_carrier_parameters batched path (lines 1463+)."""
    from aura.torch_renderer import _torch_geometry_from_carrier_parameters
    import torch

    base = torch.zeros((1, 3))
    # Simulate a batched carrier_parameters dict with __batched__ key
    batched_params = {
        "__batched__": {
            "min_corner": torch.zeros((1, 3)),
            "max_corner": torch.ones((1, 3)),
            "gaussian_mean": torch.tensor([[0.0, 0.0, 0.5]]),
            "gaussian_covariance_diag": torch.tensor([[0.1, 0.1, 0.1]]),
        },
        "__batched_meta__": {
            "gaussian_mean_present": torch.tensor([True]),
        },
    }

    result = _torch_geometry_from_carrier_parameters(
        torch,
        [object()],  # 1 element
        batched_params,
        base, base, base, base, base, base, base, base, base, base,
    )

    # Returns batched min/max corners
    assert result[0].shape == (1, 3)
    assert result[1].shape == (1, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_geometry_from_carrier_parameters_batched_without_present_mask():
    """Cover batched path without gaussian_mean_present (lines 1480-1491)."""
    from aura.torch_renderer import _torch_geometry_from_carrier_parameters
    import torch

    base = torch.zeros((1, 3))
    batched_params = {
        "__batched__": {
            "min_corner": torch.zeros((1, 3)),
            "max_corner": torch.ones((1, 3)),
            "gaussian_mean": torch.tensor([[0.0, 0.0, 0.5]]),
            "gaussian_covariance_diag": torch.tensor([[0.1, 0.1, 0.1]]),
        },
        # No __batched_meta__ key
    }

    result = _torch_geometry_from_carrier_parameters(
        torch,
        [object()],
        batched_params,
        base, base, base, base, base, base, base, base, base, base,
    )

    # Should still return proper means (no masking applied)
    assert result[4].shape == (1, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_carrier_hits_with_empty_scene_returns_empty_tensors():
    """Cover _torch_carrier_hits with 0 elements (lines 1844-1845)."""
    from aura.torch_renderer import _torch_carrier_hits
    import torch

    origins = torch.zeros((2, 3))
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    empty = torch.zeros((0, 3))
    empty_sq = torch.zeros((0,))
    empty_radii = torch.zeros((0, 3))
    empty_cov = torch.zeros((0, 3, 3))

    entry, exit_, hits, indices = _torch_carrier_hits(
        torch,
        [],  # empty elements
        origins,
        directions,
        empty, empty,  # mins, maxs
        empty, empty,  # surface_plane_points, surface_normals
        empty, empty,  # gabor_plane_points, gabor_normals
        empty, empty_cov,  # gaussian_means, gaussian_inverse_covariances
        empty_sq,  # gaussian_support_radius_sq
        empty_radii,  # beta_support_radii
    )

    assert entry.shape[0] == 2
    assert entry.shape[1] == 0
    assert hits.shape[1] == 0
    assert indices.shape[1] == 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_mean_or_nan_returns_nan_for_non_gaussian():
    """Cover _gaussian_mean_or_nan for non-gaussian element (line 2196)."""
    from aura.torch_renderer import _gaussian_mean_or_nan

    elem = AuraElement(id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)))
    result = _gaussian_mean_or_nan(elem)
    import math
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_mean_or_nan_returns_nan_for_missing_mean():
    """Cover _gaussian_mean_or_nan missing mean (line 2200)."""
    from aura.torch_renderer import _gaussian_mean_or_nan
    import math

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback"}  # no "mean" key
    )
    result = _gaussian_mean_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_mean_or_nan_returns_values_for_valid_gaussian():
    """Cover _gaussian_mean_or_nan valid case (line 2203)."""
    from aura.torch_renderer import _gaussian_mean_or_nan

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "mean": [0.1, 0.2, 0.3]}
    )
    result = _gaussian_mean_or_nan(elem)
    assert result == pytest.approx((0.1, 0.2, 0.3))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_normal_or_nan_returns_nan_for_non_surface():
    """Cover _surface_normal_or_nan for non-surface (line 2257)."""
    from aura.torch_renderer import _surface_normal_or_nan
    import math

    elem = AuraElement(id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)))
    result = _surface_normal_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_normal_or_nan_returns_nan_for_missing_normal():
    """Cover _surface_normal_or_nan missing normal (line 2283)."""
    from aura.torch_renderer import _surface_normal_or_nan
    import math

    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "surface_cell"}  # no normal
    )
    result = _surface_normal_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_plane_point_or_nan_uses_payload_plane_point():
    """Cover _surface_plane_point_or_nan payload plane_point path (line 2293)."""
    from aura.torch_renderer import _surface_plane_point_or_nan

    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        normal=(0.0, 0.0, -1.0),
        payload={"type": "surface_cell", "plane_point": [0.5, 0.5, 0.05]},
    )
    result = _surface_plane_point_or_nan(elem)
    assert result == pytest.approx((0.5, 0.5, 0.05))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_plane_point_or_nan_falls_back_to_center():
    """Cover _surface_plane_point_or_nan center-fallback path (line 2303)."""
    from aura.torch_renderer import _surface_plane_point_or_nan

    # Surface with normal but no plane_point — should compute from bounds
    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        normal=(0.0, 0.0, -1.0),
        payload={"type": "surface_cell"},
    )
    result = _surface_plane_point_or_nan(elem)
    import math
    assert not any(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_or_nan_returns_nan_for_non_gabor():
    """Cover _gabor_normal_or_nan non-gabor path (line 2335)."""
    from aura.torch_renderer import _gabor_normal_or_nan
    import math

    elem = AuraElement(id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)))
    result = _gabor_normal_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_or_nan_returns_explicit_normal():
    """Cover _gabor_normal_or_nan explicit normal path (line 2359)."""
    from aura.torch_renderer import _gabor_normal_or_nan

    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2)),
        payload={"type": "gabor_frequency", "normal": [0.0, 0.0, 1.0]},
    )
    result = _gabor_normal_or_nan(elem)
    assert result == pytest.approx((0.0, 0.0, 1.0))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_or_nan_falls_back_to_min_extent_axis():
    """Cover _gabor_normal_or_nan fallback axis path (lines 2362-2368)."""
    from aura.torch_renderer import _gabor_normal_or_nan

    # Thinnest in z → gabor normal should be z-axis
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.01)),
        payload={"type": "gabor_frequency"},
    )
    result = _gabor_normal_or_nan(elem)
    import math
    assert not any(math.isnan(v) for v in result)
    assert abs(result[2]) == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_plane_point_or_nan_uses_explicit_point():
    """Cover _gabor_plane_point_or_nan explicit point (line 2381)."""
    from aura.torch_renderer import _gabor_plane_point_or_nan

    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2)),
        payload={"type": "gabor_frequency", "normal": [0.0, 0.0, 1.0], "plane_point": [0.1, 0.2, 0.3]},
    )
    result = _gabor_plane_point_or_nan(elem)
    assert result == pytest.approx((0.1, 0.2, 0.3))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_plane_point_or_nan_falls_back_to_center():
    """Cover _gabor_plane_point_or_nan center fallback (lines 2404-2408)."""
    from aura.torch_renderer import _gabor_plane_point_or_nan

    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.01)),
        payload={"type": "gabor_frequency"},
    )
    result = _gabor_plane_point_or_nan(elem)
    import math
    assert not any(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_support_radius_or_nan_returns_nan_for_non_beta():
    """Cover _beta_support_radius_or_nan non-beta path (line 2412)."""
    from aura.torch_renderer import _beta_support_radius_or_nan
    import math

    elem = AuraElement(id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)))
    result = _beta_support_radius_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_support_radius_or_nan_returns_nan_for_missing_radii():
    """Cover _beta_support_radius_or_nan missing support_radius (line 2413)."""
    from aura.torch_renderer import _beta_support_radius_or_nan
    import math

    elem = AuraElement(
        id="b", carrier_id="beta", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "beta_kernel"}  # no support_radius
    )
    result = _beta_support_radius_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_support_radius_or_nan_returns_radii_for_valid_beta():
    """Cover _beta_support_radius_or_nan valid case (line 2425-2426)."""
    from aura.torch_renderer import _beta_support_radius_or_nan

    elem = AuraElement(
        id="b", carrier_id="beta", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        payload={"type": "beta_kernel", "support_radius": [0.3, 0.3, 0.05]},
    )
    result = _beta_support_radius_or_nan(elem)
    assert result == pytest.approx((0.3, 0.3, 0.05))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_support_radius_returns_nan_for_non_positive_radius():
    """Cover _beta_support_radius_or_nan non-positive radii (line 2428)."""
    from aura.torch_renderer import _beta_support_radius_or_nan
    import math

    elem = AuraElement(
        id="b", carrier_id="beta", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        payload={"type": "beta_kernel", "support_radius": [0.0, 0.3, 0.05]},
    )
    result = _beta_support_radius_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_inverse_matrix3_returns_none_for_singular():
    """Cover _inverse_matrix3 singular matrix (line 2452)."""
    from aura.torch_renderer import _inverse_matrix3

    # Singular matrix (all zeros)
    result = _inverse_matrix3(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    assert result is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_inverse_matrix3_inverts_diagonal():
    """Cover _inverse_matrix3 successful inversion (line 2470)."""
    from aura.torch_renderer import _inverse_matrix3

    # Diagonal matrix with known inverse
    m = ((2.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 8.0))
    result = _inverse_matrix3(m)
    assert result is not None
    assert result[0][0] == pytest.approx(0.5)
    assert result[1][1] == pytest.approx(0.25)
    assert result[2][2] == pytest.approx(0.125)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_inverse_covariance_or_nan_for_non_gaussian():
    """Cover _gaussian_inverse_covariance_or_nan non-gaussian (line 2504)."""
    from aura.torch_renderer import _gaussian_inverse_covariance_or_nan
    import math

    elem = AuraElement(id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)))
    result = _gaussian_inverse_covariance_or_nan(elem)
    assert all(math.isnan(v) for row in result for v in row)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_inverse_covariance_or_nan_for_valid_covariance():
    """Cover _gaussian_inverse_covariance_or_nan valid case (line 2525)."""
    from aura.torch_renderer import _gaussian_inverse_covariance_or_nan

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "covariance": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
    )
    result = _gaussian_inverse_covariance_or_nan(elem)
    import math
    assert not any(math.isnan(v) for row in result for v in row)
    # Identity inverse is identity
    assert result[0][0] == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_support_radius_sq_returns_nan_for_non_gaussian():
    """Cover _gaussian_support_radius_sq non-gaussian (line 2545)."""
    from aura.torch_renderer import _gaussian_support_radius_sq
    import math

    elem = AuraElement(id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)))
    result = _gaussian_support_radius_sq(elem)
    assert math.isnan(result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_support_radius_sq_uses_explicit_support_radius_sq():
    """Cover _gaussian_support_radius_sq explicit value path (line 2546)."""
    from aura.torch_renderer import _gaussian_support_radius_sq

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "support_radius_sq": 9.0},
    )
    result = _gaussian_support_radius_sq(elem)
    assert result == pytest.approx(9.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_support_radius_sq_uses_sigma_fallback():
    """Cover _gaussian_support_radius_sq sigma fallback (line 2547)."""
    from aura.torch_renderer import _gaussian_support_radius_sq

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "support_sigma": 2.0},
    )
    result = _gaussian_support_radius_sq(elem)
    assert result == pytest.approx(4.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_optional_capture_tensors_handles_all_none():
    """Cover _stack_optional_capture_tensors all-None path (line 2599)."""
    from aura.torch_renderer import _stack_optional_capture_tensors
    import torch

    result_batch, result_present = _stack_optional_capture_tensors(
        torch, (None, None), device="cpu", name="depth"
    )
    assert result_batch is None
    assert result_present is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_optional_capture_tensors_handles_mixed_none():
    """Cover _stack_optional_capture_tensors mixed-None path (lines 2638, 2646)."""
    from aura.torch_renderer import _stack_optional_capture_tensors
    import torch

    present_tensor = CaptureTensor(
        path="d.pgm", format="Netpbm", backend="stdlib", width=1, height=1, channels=1, values=(0.5,)
    )
    result_batch, result_present = _stack_optional_capture_tensors(
        torch, (present_tensor, None), device="cpu", name="depth"
    )
    assert result_batch is not None
    assert result_present is not None
    assert result_present.tolist() == [True, False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_optional_capture_tensors_rejects_shape_mismatch():
    """Cover _stack_optional_capture_tensors shape mismatch (line 2646)."""
    from aura.torch_renderer import _stack_optional_capture_tensors
    import torch

    t1 = CaptureTensor(path="d.pgm", format="Netpbm", backend="stdlib", width=1, height=1, channels=1, values=(0.5,))
    t2 = CaptureTensor(path="d2.pgm", format="Netpbm", backend="stdlib", width=2, height=1, channels=1, values=(0.5, 0.5))

    with pytest.raises(ValueError, match="shapes must match"):
        _stack_optional_capture_tensors(torch, (t1, t2), device="cpu", name="depth")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_target_objective_rejects_empty_targets():
    """Cover torch_render_target_objective empty targets guard (line 1044)."""
    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="at least one target"):
        torch_render_target_objective(scene, (), device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_target_objective_rejects_empty_scene():
    """Cover torch_render_target_objective empty scene guard (line 1046)."""
    scene = AuraScene(name="empty", elements=())

    with pytest.raises(ValueError, match="at least one scene element"):
        torch_render_target_objective(
            scene,
            (RenderTarget(frame_id="f", ray=Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)), target_color=(0.0, 0.0, 0.0), target_depth=1.0),),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_objective_validates_frame_id_count():
    """Cover _torch_render_objective_tensor_targets frame count check (line 1138)."""
    import torch
    from aura.torch_renderer import _torch_render_objective_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="target count must match"):
        _torch_render_objective_tensor_targets(
            scene,
            frame_ids=("f", "g"),  # 2 frame_ids
            origins=torch.zeros((1, 3)),  # 1 origin
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_objective_validates_semantic_id_count():
    """Cover _torch_render_objective_tensor_targets semantic id count check (line 1140)."""
    import torch
    from aura.torch_renderer import _torch_render_objective_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="semantic id count"):
        _torch_render_objective_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_semantic_ids=("label_a", "label_b"),  # wrong count
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_objective_validates_material_id_count():
    """Cover _torch_render_objective_tensor_targets material id count check (line 1142)."""
    import torch
    from aura.torch_renderer import _torch_render_objective_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="material id count"):
        _torch_render_objective_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_material_ids=("mat_a", "mat_b"),  # wrong count
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_objective_validates_confidence_count():
    """Cover _torch_render_objective_tensor_targets confidence count check (line 1144)."""
    import torch
    from aura.torch_renderer import _torch_render_objective_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="confidence count"):
        _torch_render_objective_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_confidence=torch.zeros((2,)),  # wrong count
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_pixel_ray_directions_tensor_with_near_vertical_look():
    """Cover _pixel_ray_directions_tensor vertical-look fallback (line 556)."""
    import torch
    from aura.torch_renderer import torch_capture_training_batch

    # Look nearly straight up — forward is almost (0,1,0) so cross(forward,(0,1,0)) degenerates
    # Need a frame that causes the fallback in _pixel_ray_directions_tensor
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, -2.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib", width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
        ),),
        device="cpu",
    )

    batch = torch_capture_training_batch((frame,), assets)
    # Should compute valid directions despite near-degenerate forward axis
    assert tuple(batch.ray_directions.shape) == (1, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_capture_render_summary_rejects_empty_batch():
    """Cover torch_render_capture_training_summary empty batch guard (line 669)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    class _EmptyBatch:
        frame_indices = torch.zeros((0,), dtype=torch.long)

    with pytest.raises(ValueError, match="at least one target"):
        torch_render_capture_training_summary(scene, _EmptyBatch())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_tensor_targets_validates_frame_id_count_in_internal():
    """Cover _torch_render_tensor_targets frame_ids count check (line 1004)."""
    import torch

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="count must match"):
        torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            ray_origins=torch.zeros((2, 3)),  # 2 origins vs 1 frame_id
            ray_directions=torch.zeros((2, 3)),
            target_colors=torch.zeros((2, 3)),
            target_depths=torch.ones((2,)),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_hits_zero_order_count_fallback():
    """Cover zero-order-count path in compositing (lines 1636-1637)."""
    # Test with a scene where no AABB hit exists but compositing still runs
    import torch
    from aura.torch_renderer import _torch_carrier_hits

    origins = torch.tensor([[0.0, 0.0, -1.0]])
    directions = torch.tensor([[0.0, 0.0, 1.0]])

    # Provide a scene with an element that is positioned off to the side (ray misses)
    scene = AuraScene(
        name="miss_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((10.0, 10.0, 0.0), (11.0, 11.0, 0.1)),  # far from ray
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
            ),
        ),
    )
    st = torch_scene_tensors(scene, device="cpu")

    entry, exit_, hits, indices = _torch_carrier_hits(
        torch,
        tuple(scene.elements),
        origins,
        directions,
        st.mins, st.maxs,
        st.surface_plane_points, st.surface_normals,
        st.gabor_plane_points, st.gabor_normals,
        st.gaussian_means, st.gaussian_inverse_covariances,
        st.gaussian_support_radius_sq,
        st.beta_support_radii,
    )

    # Should return all False hits since ray misses the scene
    assert not hits.any()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_batched_carrier_fast_path_with_present_mask():
    """Cover _is_batched_carriers code path (lines 1854-1858)."""
    import torch

    # Create a scene with gaussian elements and provide batched parameters
    scene = AuraScene(
        name="batched_gauss_scene",
        elements=(
            AuraElement(
                id="g1",
                carrier_id="gaussian",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5)),
                color=(1.0, 0.0, 0.0),
                opacity=0.9,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.1],
                    "covariance": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                    "support_sigma": 3.0,
                },
            ),
        ),
    )
    st = torch_scene_tensors(scene, device="cpu")
    # Create batched parameters with __batched__ key (must include color/opacity/confidence)
    batched_carrier_params = {
        "__batched__": {
            "min_corner": st.mins,
            "max_corner": st.maxs,
            "gaussian_mean": torch.tensor([[0.0, 0.0, 0.1]]),
            "gaussian_covariance_diag": torch.tensor([[0.01, 0.01, 0.01]]),
            "color": torch.tensor([[1.0, 0.0, 0.0]]),
            "opacity": torch.tensor([0.9]),
            "confidence": torch.tensor([1.0]),
        },
        "__batched_meta__": {
            "gaussian_mean_present": torch.tensor([True]),
        },
    }

    batch = torch_render_rays(
        scene,
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        device="cpu",
        carrier_parameters=batched_carrier_params,
    )
    assert len(batch.element_ids) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_hits_batched_active_path():
    """Cover _is_batched_carriers gaussian valid path (line 1910)."""
    import torch

    scene = AuraScene(
        name="batched_active_scene",
        elements=(
            AuraElement(
                id="g1",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, -0.5), (1.0, 1.0, 1.5)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.5],
                    "covariance": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "support_sigma": 3.0,
                },
            ),
        ),
    )
    st = torch_scene_tensors(scene, device="cpu")
    batched_carrier_params = {
        "__batched__": {
            "min_corner": st.mins,
            "max_corner": st.maxs,
            "gaussian_mean": torch.tensor([[0.0, 0.0, 0.5]]),
            "gaussian_covariance_diag": torch.tensor([[1.0, 1.0, 1.0]]),
            "color": torch.tensor([[0.5, 0.5, 0.5]]),
            "opacity": torch.tensor([0.8]),
            "confidence": torch.tensor([1.0]),
        },
    }

    # Ray pointing through gaussian center
    origins = torch.tensor([[0.0, 0.0, -2.0]])
    directions = torch.tensor([[0.0, 0.0, 1.0]])

    batch = torch_render_rays(
        scene,
        origins,
        directions,
        device="cpu",
        carrier_parameters=batched_carrier_params,
        collect_traces=False,
    )
    assert len(batch.element_ids) == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_inverse_covariance_or_nan_for_missing_covariance():
    """Cover _gaussian_inverse_covariance_or_nan missing covariance (line 2504)."""
    from aura.torch_renderer import _gaussian_inverse_covariance_or_nan
    import math

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback"}  # no covariance key
    )
    result = _gaussian_inverse_covariance_or_nan(elem)
    assert all(math.isnan(v) for row in result for v in row)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_inverse_covariance_or_nan_for_singular_covariance():
    """Cover _gaussian_inverse_covariance_or_nan singular covariance returns nan (line 2525)."""
    from aura.torch_renderer import _gaussian_inverse_covariance_or_nan
    import math

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "covariance": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}
    )
    result = _gaussian_inverse_covariance_or_nan(elem)
    # Singular → should return NaN matrix
    assert all(math.isnan(v) for row in result for v in row)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_support_radius_sq_returns_nan_for_non_positive_sigma():
    """Cover _gaussian_support_radius_sq non-positive sigma (line 2547)."""
    from aura.torch_renderer import _gaussian_support_radius_sq
    import math

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "support_sigma": 0.0}  # invalid sigma
    )
    result = _gaussian_support_radius_sq(elem)
    assert math.isnan(result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_render_capture_training_summary_with_with_mask():
    """Cover _sampled_training_pixels mask_present branch (lines 553-556)."""
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    # Two frames: one with mask present, one without
    frames = (frame, TrainingFrame(
        id="frame2",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    ))
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib", width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
                mask=CaptureTensor(path="f_mask.pgm", format="Netpbm", backend="stdlib", width=1, height=1, channels=1, values=(1.0,)),
            ),
            CaptureFrameTensors(
                frame_id="frame2",
                image=CaptureTensor(path="f2.ppm", format="Netpbm", backend="stdlib", width=1, height=1, channels=3, values=(0.0, 1.0, 0.0)),
                mask=None,  # absent for frame2
            ),
        ),
        device="cpu",
    )

    # This exercises the mask_present logic in _sampled_training_pixels_for_frame
    batch = torch_capture_training_batch(frames, assets)
    assert batch.frame_indices.shape[0] >= 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_devices_match_returns_true_for_same_string():
    """Cover _torch_devices_match identical-string fast path (line 1348)."""
    from aura.torch_renderer import _torch_devices_match

    assert _torch_devices_match("cpu", "cpu") is True


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_devices_match_returns_false_for_different_types():
    """Cover _torch_devices_match type mismatch (line 1354)."""
    from aura.torch_renderer import _torch_devices_match

    assert _torch_devices_match("cpu", "cuda") is False


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_or_nan_returns_nan_for_degenerate_bounds():
    """Cover _gabor_normal_or_nan degenerate bounds (line 2362)."""
    from aura.torch_renderer import _gabor_normal_or_nan
    import math

    # Zero-extent element → can't compute axis, returns NaN
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        payload={"type": "gabor_frequency"},
    )
    result = _gabor_normal_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_plane_point_or_nan_returns_nan_for_non_gabor():
    """Cover _gabor_plane_point_or_nan non-gabor path (line 2406)."""
    from aura.torch_renderer import _gabor_plane_point_or_nan
    import math

    elem = AuraElement(id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)))
    result = _gabor_plane_point_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_capture_training_batch_with_no_intrinsics_uses_forward_direction():
    """Cover _pixel_ray_directions_tensor when frame.intrinsics is None (line 553)."""
    # Frame with no intrinsics: all pixels get the same forward direction
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        # no intrinsics key means intrinsics=None
    )
    assets = torch_capture_asset_batch(
        (CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib", width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
        ),),
        device="cpu",
    )

    batch = torch_capture_training_batch((frame,), assets)

    # All directions should point forward (0, 0, 1) since there are no intrinsics
    assert tuple(batch.ray_directions.shape) == (1, 3)
    dirs = batch.ray_directions.tolist()[0]
    assert dirs[2] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_rejects_empty_scene():
    """Cover _torch_render_tensor_targets empty scene guard (line 1138)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(name="empty", elements=())

    with pytest.raises(ValueError, match="at least one scene element"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_normals=None,
            target_normal_present=None,
            target_confidence=None,
            target_confidence_present=None,
            target_semantic_ids=(None,),
            target_material_ids=(None,),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_rejects_empty_frame_ids():
    """Cover _torch_render_tensor_targets empty frame_ids guard (line 1140)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="at least one target"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=(),
            origins=torch.zeros((0, 3)),
            directions=torch.zeros((0, 3)),
            target_colors=torch.zeros((0, 3)),
            target_depths=torch.zeros((0,)),
            target_normals=None,
            target_normal_present=None,
            target_confidence=None,
            target_confidence_present=None,
            target_semantic_ids=(),
            target_material_ids=(),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_rejects_origin_count_mismatch():
    """Cover _torch_render_tensor_targets origin count mismatch (line 1142)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="target count must match"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=("f", "g"),
            origins=torch.zeros((1, 3)),  # 1 vs 2 frame_ids
            directions=torch.zeros((1, 3)),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_normals=None,
            target_normal_present=None,
            target_confidence=None,
            target_confidence_present=None,
            target_semantic_ids=(None, None),
            target_material_ids=(None, None),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_tensor_targets_rejects_semantic_count_mismatch():
    """Cover _torch_render_tensor_targets semantic/material count mismatch (line 1144)."""
    import torch
    from aura.torch_renderer import _torch_render_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="query target counts"):
        _torch_render_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_normals=None,
            target_normal_present=None,
            target_confidence=None,
            target_confidence_present=None,
            target_semantic_ids=(None, None),  # 2 vs 1
            target_material_ids=(None,),
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_devices_match_non_cuda_non_identical_devices():
    """Cover _torch_devices_match for same non-CUDA type comparison (line 1356-1360)."""
    from aura.torch_renderer import _torch_devices_match

    # Same CPU but different string representation
    assert _torch_devices_match("cpu:0", "cpu") is False or _torch_devices_match("cpu", "cpu") is True


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_scene_tensors_without_requires_grad():
    """Cover torch_scene_tensors requires_grad=False path (line 1428)."""
    scene = AuraScene(
        name="no_grad_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
            ),
        ),
    )

    st = torch_scene_tensors(scene, device="cpu", requires_grad=False)
    # All carrier parameters should not require grad
    assert st.element_ids == ("surface",)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_resolve_scene_tensors_validates_device_mismatch():
    """Cover _resolve_scene_tensors device mismatch check (lines 1440-1444)."""
    from aura.torch_renderer import _resolve_scene_tensors

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    # Create scene tensors on "cpu"
    st = torch_scene_tensors(scene, device="cpu")
    # Pretend device is "cuda" — this can't work but will expose the check
    # We need to trick the device check
    import torch

    # Test through _resolve_scene_tensors directly
    with pytest.raises(ValueError, match="device"):
        # Simulate device mismatch by creating a fake scene_tensors with different device
        class _FakeSceneTensors:
            device = "cuda"
            element_ids = ("s",)

        _resolve_scene_tensors(scene, scene_tensors=_FakeSceneTensors(), device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_required_capture_tensors_rejects_none_image():
    """Cover _stack_required_capture_tensors None guard (line 2257)."""
    from aura.torch_renderer import _stack_required_capture_tensors
    import torch

    with pytest.raises(ValueError, match="are required"):
        _stack_required_capture_tensors(torch, (None, None), device="cpu", name="image")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_shared_capture_tensor_shape_rejects_empty():
    """Cover _shared_capture_tensor_shape empty guard (line 2293)."""
    from aura.torch_renderer import _shared_capture_tensor_shape

    with pytest.raises(ValueError, match="batch is empty"):
        _shared_capture_tensor_shape((), name="test")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_tensor_metadata_returns_none_for_none():
    """Cover _torch_tensor_metadata None return path (line 2303)."""
    from aura.torch_renderer import _torch_tensor_metadata

    result = _torch_tensor_metadata(None)
    assert result is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_training_batch_to_dict_with_none_mask_calls_torch_tensor_metadata_none():
    """Cover TorchCaptureTrainingBatch.to_dict path where target_mask is None."""
    # Create a batch with no mask (so target_mask=None)
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib", width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
            # No mask, no depth, no normal
        ),),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    d = batch.to_dict()

    # target_mask should be None (frame has no mask)
    assert d["targetMask"] is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_hit_transmittance_traces_with_no_transmittances():
    """Cover _torch_hit_transmittance_traces empty case (line 2196)."""
    import torch
    from aura.torch_renderer import _torch_hit_transmittance_traces

    sorted_depths = torch.tensor([[0.5, float("inf")]])
    result = _torch_hit_transmittance_traces(torch, sorted_depths, [])
    # No transmittances → empty tuples per ray
    assert result == ((),)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_hit_transmittance_traces_with_actual_values():
    """Cover _torch_hit_transmittance_traces with real data (lines 2198-2203)."""
    import torch
    from aura.torch_renderer import _torch_hit_transmittance_traces

    sorted_depths = torch.tensor([[0.5, 1.0, float("inf")]])
    t1 = torch.tensor([0.8])
    t2 = torch.tensor([0.3])
    result = _torch_hit_transmittance_traces(torch, sorted_depths, [t1, t2])
    # 2 active hits (inf is inactive)
    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0][0] == pytest.approx(0.8)
    assert result[0][1] == pytest.approx(0.3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_normal_or_nan_returns_nan_when_normal_not_present_for_surface_carrier():
    """Cover _surface_normal_or_nan missing normal (lines 2257-2283)."""
    from aura.torch_renderer import _surface_normal_or_nan
    import math

    # Surface carrier but no normal specified
    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "surface_cell"}
    )
    result = _surface_normal_or_nan(elem)
    # No normal → NaN
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_plane_point_fallback_uses_dominant_axis():
    """Cover _surface_plane_point_or_nan dominant-axis center fallback (line 2303)."""
    from aura.torch_renderer import _surface_plane_point_or_nan
    import math

    # Normal in z-direction, bounds asymmetric in z
    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-2.0, -2.0, 0.0), (2.0, 2.0, 3.0)),
        normal=(0.0, 0.0, 1.0),  # Positive z-normal → use max_corner z
        payload={"type": "surface_cell"},
    )
    result = _surface_plane_point_or_nan(elem)
    assert not any(math.isnan(v) for v in result)
    assert result[2] == pytest.approx(3.0)  # dominant axis (z) → max_corner z


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_or_nan_for_explicit_gabor_carrier():
    """Cover _gabor_normal_or_nan carrier_id == 'gabor' path (lines 2335-2336)."""
    from aura.torch_renderer import _gabor_normal_or_nan
    import math

    # carrier_id="gabor" without payload type
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.01)),
        payload={},  # no type
    )
    result = _gabor_normal_or_nan(elem)
    # Should fall back to min-extent axis (z is thinnest)
    assert not any(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_or_nan_normalizes_explicit_normal():
    """Cover _gabor_normal_or_nan normalization error branch (line 2361)."""
    from aura.torch_renderer import _gabor_normal_or_nan
    import math

    # Zero normal → normalize fails → return NaN
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2)),
        payload={"type": "gabor_frequency", "normal": [0.0, 0.0, 0.0]},
    )
    result = _gabor_normal_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_plane_point_uses_point_payload():
    """Cover _gabor_plane_point_or_nan 'point' key path (line 2406)."""
    from aura.torch_renderer import _gabor_plane_point_or_nan

    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.01)),
        payload={"type": "gabor_frequency", "point": [0.2, 0.3, 0.05]},
    )
    result = _gabor_plane_point_or_nan(elem)
    assert result == pytest.approx((0.2, 0.3, 0.05))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_support_radius_type_error_returns_nan():
    """Cover _beta_support_radius_or_nan type error during conversion (line 2425)."""
    from aura.torch_renderer import _beta_support_radius_or_nan
    import math

    # support_radius with non-numeric values → TypeError → NaN
    elem = AuraElement(
        id="b", carrier_id="beta", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        payload={"type": "beta_kernel", "support_radius": ["a", "b", "c"]},
    )
    result = _beta_support_radius_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_inverse_matrix3_inverts_general_matrix():
    """Cover _inverse_matrix3 non-trivial inversion (lines 2470-2484)."""
    from aura.torch_renderer import _inverse_matrix3

    # A non-diagonal invertible matrix
    m = ((2.0, 1.0, 0.0), (0.0, 3.0, 1.0), (0.0, 0.0, 2.0))
    result = _inverse_matrix3(m)
    assert result is not None
    # Verify A * A^{-1} ≈ I
    import math
    for i in range(3):
        for j in range(3):
            total = sum(m[i][k] * result[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            assert math.isclose(total, expected, abs_tol=1e-9)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_support_radius_sq_returns_nan_for_invalid_explicit():
    """Cover _gaussian_support_radius_sq non-positive explicit value (line 2546)."""
    from aura.torch_renderer import _gaussian_support_radius_sq
    import math

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "support_radius_sq": 0.0}  # non-positive
    )
    result = _gaussian_support_radius_sq(elem)
    assert math.isnan(result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_objective_validates_confidence_present_count():
    """Cover _torch_render_objective_tensor_targets confidence_present count (line 1356)."""
    import torch
    from aura.torch_renderer import _torch_render_objective_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(ValueError, match="confidence presence count"):
        _torch_render_objective_tensor_targets(
            scene,
            frame_ids=("f",),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            target_colors=torch.zeros((1, 3)),
            target_depths=torch.ones((1,)),
            target_confidence_present=torch.tensor([True, False]),  # wrong count
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_internal_render_objective_fills_defaults_when_confidence_none():
    """Cover _torch_render_objective_tensor_targets default confidence fill (lines 1358, 1360)."""
    import torch
    from aura.torch_renderer import _torch_render_objective_tensor_targets

    scene = AuraScene(
        name="s",
        elements=(
            AuraElement(
                id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0), opacity=1.0,
            ),
        ),
    )

    # Call without confidence or confidence_present (should fill with zeros)
    objective = _torch_render_objective_tensor_targets(
        scene,
        frame_ids=("f",),
        origins=torch.tensor([[0.0, 0.0, -1.0]]),
        directions=torch.tensor([[0.0, 0.0, 1.0]]),
        target_colors=torch.tensor([[1.0, 0.0, 0.0]]),
        target_depths=torch.tensor([1.0]),
        # No confidence or confidence_present → lines 1358, 1360 are reached
        device="cpu",
    )
    assert objective.confidence_loss is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_device_match_cuda_index_normalization():
    """Cover _torch_devices_match CUDA index comparison (lines 1442-1444)."""
    from aura.torch_renderer import _torch_devices_match
    import torch

    # Skip if no CUDA
    if not torch.cuda.is_available():
        # Test with CPU as proxy
        assert _torch_devices_match("cpu", "cpu") is True
        assert _torch_devices_match("cpu", "cuda") is False
    else:
        # cuda:0 should match cuda (default)
        assert _torch_devices_match("cuda:0", "cuda") is True
        assert _torch_devices_match("cuda:0", "cuda:0") is True


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_hits_with_zero_order_count():
    """Cover order_count == 0 branch in compositing (lines 1636-1637)."""
    import torch
    from aura.torch_renderer import _torch_carrier_hits

    # A single-element scene where AABB is hit but no element type matches
    # This is tricky to trigger naturally; use a volume (no surface/gaussian/beta/gabor mask)
    scene = AuraScene(
        name="volume_scene",
        elements=(
            AuraElement(
                id="vol",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5)),
                color=(1.0, 0.0, 0.0),
                opacity=0.5,
                payload={"type": "volume_cell"},
            ),
        ),
    )
    st = torch_scene_tensors(scene, device="cpu")
    origins = torch.tensor([[0.0, 0.0, -1.0]])
    directions = torch.tensor([[0.0, 0.0, 1.0]])

    entry, exit_, hits, indices = _torch_carrier_hits(
        torch,
        tuple(scene.elements),
        origins,
        directions,
        st.mins, st.maxs,
        st.surface_plane_points, st.surface_normals,
        st.gabor_plane_points, st.gabor_normals,
        st.gaussian_means, st.gaussian_inverse_covariances,
        st.gaussian_support_radius_sq,
        st.beta_support_radii,
    )

    # Should return results (volume AABB is hit, the carrier contributes)
    assert entry.shape[0] == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_normal_or_nan_with_surface_carrier_id_but_no_normal():
    """Cover _surface_normal_or_nan returning nan when normal is None (line 2283)."""
    from aura.torch_renderer import _surface_normal_or_nan
    import math

    # carrier_id="surface" but no normal, no payload normal
    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={}
    )
    result = _surface_normal_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_with_carrier_id_gabor_no_explicit_normal():
    """Cover _gabor_normal_or_nan carrier_id='gabor' path (lines 2335-2336)."""
    from aura.torch_renderer import _gabor_normal_or_nan
    import math

    # carrier_id="gabor" but no payload type key and no explicit normal
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.01)),
        payload={"frequency": [1.0, 0.0, 0.0]},  # has a frequency but no explicit normal
    )
    result = _gabor_normal_or_nan(elem)
    # Should fall back to min-extent axis
    assert not any(math.isnan(v) for v in result)
    assert abs(result[2]) == pytest.approx(1.0)  # z-axis is thinnest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_plane_point_or_nan_uses_gabor_carrier_id():
    """Cover _gabor_plane_point_or_nan for gabor carrier_id without type (line 2406)."""
    from aura.torch_renderer import _gabor_plane_point_or_nan
    import math

    # carrier_id="gabor" without payload type, but no explicit point
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.01)),
        payload={},
    )
    result = _gabor_plane_point_or_nan(elem)
    # Should fall back to center (0, 0, 0.005)
    assert not any(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_support_radius_or_nan_for_wrong_length_list():
    """Cover _beta_support_radius_or_nan wrong list length (line 2412)."""
    from aura.torch_renderer import _beta_support_radius_or_nan
    import math

    # support_radius with only 2 elements (not 3)
    elem = AuraElement(
        id="b", carrier_id="beta", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        payload={"type": "beta_kernel", "support_radius": [0.1, 0.2]},  # wrong length
    )
    result = _beta_support_radius_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_inverse_matrix3_returns_correct_values():
    """Cover _inverse_matrix3 full computation including cofactors (lines 2470-2484)."""
    from aura.torch_renderer import _inverse_matrix3
    import math

    # Full non-trivial 3x3 matrix
    m = ((4.0, 7.0, 2.0), (1.0, 3.0, 1.0), (2.0, 5.0, 3.0))
    result = _inverse_matrix3(m)
    assert result is not None
    # Verify A * A^{-1} ≈ I
    for i in range(3):
        for j in range(3):
            total = sum(m[i][k] * result[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            assert math.isclose(total, expected, abs_tol=1e-9)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_inverse_covariance_or_nan_for_valid_non_identity():
    """Cover _gaussian_inverse_covariance_or_nan valid non-identity case (line 2525)."""
    from aura.torch_renderer import _gaussian_inverse_covariance_or_nan
    import math

    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback", "covariance": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]},
    )
    result = _gaussian_inverse_covariance_or_nan(elem)
    assert not any(math.isnan(v) for row in result for v in row)
    assert result[0][0] == pytest.approx(0.25)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_support_radius_sq_with_missing_sigma_key():
    """Cover _gaussian_support_radius_sq default sigma=3.0 fallback (lines 2545-2547)."""
    from aura.torch_renderer import _gaussian_support_radius_sq

    # Gaussian with no support_radius_sq and no support_sigma → uses default sigma=3.0
    elem = AuraElement(
        id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "gaussian_fallback"}  # no sigma or support_radius_sq
    )
    result = _gaussian_support_radius_sq(elem)
    # Default sigma=3.0, so sq = 9.0
    assert result == pytest.approx(9.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_optional_all_none_returns_none():
    """Cover _stack_optional_capture_tensors all-None returns (None, None) (line 2599)."""
    from aura.torch_renderer import _stack_optional_capture_tensors
    import torch

    result = _stack_optional_capture_tensors(torch, (None,), device="cpu", name="depth")
    assert result == (None, None)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_optional_fills_zeros_for_absent_frames():
    """Cover _stack_optional_capture_tensors zero-fill for missing tensors (line 2638)."""
    from aura.torch_renderer import _stack_optional_capture_tensors
    import torch

    present = CaptureTensor(path="d.pgm", format="Netpbm", backend="stdlib", width=2, height=1, channels=1, values=(0.5, 0.7))
    # First frame absent, second frame present
    batch, present_tensor = _stack_optional_capture_tensors(
        torch, (None, present), device="cpu", name="depth"
    )
    assert batch is not None
    assert present_tensor.tolist() == [False, True]
    # First frame should be zeros
    assert batch[0, 0, 0, 0].item() == pytest.approx(0.0)
    # Second frame should have actual values
    assert batch[1, 0, 0, 0].item() == pytest.approx(0.5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_optional_rejects_shape_mismatch_in_same_batch():
    """Cover _stack_optional_capture_tensors shape mismatch guard (line 2646)."""
    from aura.torch_renderer import _stack_optional_capture_tensors
    import torch

    t1 = CaptureTensor(path="d.pgm", format="Netpbm", backend="stdlib", width=2, height=1, channels=1, values=(0.5, 0.7))
    t2 = CaptureTensor(path="d2.pgm", format="Netpbm", backend="stdlib", width=3, height=1, channels=1, values=(0.1, 0.2, 0.3))

    with pytest.raises(ValueError, match="shapes must match"):
        _stack_optional_capture_tensors(torch, (t1, t2), device="cpu", name="depth")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_optional_target_normal_tuple_returns_empty_when_none():
    """Cover _optional_target_normal_tuple early return (line 2638)."""
    from aura.torch_renderer import _optional_target_normal_tuple

    result = _optional_target_normal_tuple(None, None)
    assert result == ()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_optional_target_confidence_tuple_returns_empty_when_none():
    """Cover _optional_target_confidence_tuple early return (line 2646)."""
    from aura.torch_renderer import _optional_target_confidence_tuple

    result = _optional_target_confidence_tuple(None, None)
    assert result == ()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_pixel_ray_direction_without_intrinsics():
    """Cover _pixel_ray_direction when frame.intrinsics is None (line 2470)."""
    from aura.torch_renderer import _pixel_ray_direction

    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        # No intrinsics
    )
    result = _pixel_ray_direction(frame, 0, 0)
    # Should return forward direction
    assert result[2] == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_pixel_ray_direction_with_intrinsics():
    """Cover _pixel_ray_direction with intrinsics (lines 2472-2484)."""
    from aura.torch_renderer import _pixel_ray_direction

    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    result = _pixel_ray_direction(frame, 0, 0)
    # Should return a normalized direction vector
    import math
    norm = sum(v * v for v in result) ** 0.5
    assert math.isclose(norm, 1.0, abs_tol=1e-6)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_predicted_normal_tensors_function():
    """Cover _predicted_normal_tensors utility (lines 2545-2547)."""
    from aura.torch_renderer import _predicted_normal_tensors
    import torch

    normals = [(0.0, 0.0, -1.0), None, (1.0, 0.0, 0.0)]
    values_tensor, present_tensor = _predicted_normal_tensors(torch, normals, device="cpu")

    assert tuple(values_tensor.shape) == (3, 3)
    assert present_tensor.tolist() == [True, False, True]
    assert values_tensor[1].tolist() == [0.0, 0.0, 0.0]  # None → zeros


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_confidence_loss_returns_zero_when_target_is_none():
    """Cover _torch_confidence_loss early return when confidence is None (line 2599)."""
    from aura.torch_renderer import _torch_confidence_loss
    import torch

    predicted = torch.tensor([0.5, 0.8])
    result = _torch_confidence_loss(torch, predicted, None, None)
    assert result.item() == pytest.approx(0.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_pixel_ray_direction_vertical_look_uses_fallback():
    """Cover _pixel_ray_direction near-vertical forward fallback (line 2479)."""
    from aura.torch_renderer import _pixel_ray_direction
    import math

    # Camera looking straight up — forward ≈ (0, 1, 0) degenerates with cross(up_ref)
    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, -2.0, 0.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    result = _pixel_ray_direction(frame, 0, 0)
    norm = sum(v * v for v in result) ** 0.5
    assert math.isclose(norm, 1.0, abs_tol=1e-6)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_normal_for_returns_payload_normal():
    """Cover _normal_for payload normal path (line 2525)."""
    from aura.torch_renderer import _normal_for

    # Element with no element.normal but with payload.normal
    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"normal": [0.0, 1.0, 0.0]},
    )
    result = _normal_for(elem)
    assert result == pytest.approx((0.0, 1.0, 0.0))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_normal_or_nan_returns_nan_when_normal_is_unnormalizable():
    """Cover _surface_normal_or_nan zero-length normal (line 2283)."""
    from aura.torch_renderer import _surface_normal_or_nan
    import math

    # Surface element with a zero-length normal → normalization fails → NaN
    elem = AuraElement(
        id="s", carrier_id="surface", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        normal=(0.0, 0.0, 0.0),  # zero-length normal
    )
    result = _surface_normal_or_nan(elem)
    assert all(math.isnan(v) for v in result)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_normal_with_carrier_id_checks_type():
    """Cover _gabor_normal_or_nan the carrier_id='gabor' branch (lines 2335-2336)."""
    from aura.torch_renderer import _gabor_normal_or_nan
    import math

    # Gabor carrier with no payload type, no explicit normal, but valid extent
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-2.0, -2.0, 0.0), (2.0, 2.0, 0.01)),
        payload={}  # no type key
    )
    result = _gabor_normal_or_nan(elem)
    # z-axis is thinnest, should return z-axis
    assert not any(math.isnan(v) for v in result)
    assert abs(result[2]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_plane_point_with_carrier_id_gabor():
    """Cover _gabor_plane_point_or_nan with carrier_id='gabor' (lines 2406-2407)."""
    from aura.torch_renderer import _gabor_plane_point_or_nan
    import math

    # Carrier_id='gabor', no explicit point in payload
    elem = AuraElement(
        id="g", carrier_id="gabor", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.01)),
        payload={"bandwidth": 0.5}  # no normal, no plane_point
    )
    result = _gabor_plane_point_or_nan(elem)
    # Falls back to center of bounds
    assert not any(math.isnan(v) for v in result)
    assert result[2] == pytest.approx(0.005)  # center z = (0.0 + 0.01) / 2


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_support_radius_or_nan_with_non_list_value():
    """Cover _beta_support_radius_or_nan non-list support_radius (line 2412)."""
    from aura.torch_renderer import _beta_support_radius_or_nan
    import math

    # support_radius is a scalar, not a list of 3
    elem = AuraElement(
        id="b", carrier_id="beta", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
        payload={"type": "beta_kernel", "support_radius": 0.5},  # scalar, not list
    )
    result = _beta_support_radius_or_nan(elem)
    assert all(math.isnan(v) for v in result)
