import importlib.util

import pytest
import aura.torch_optimizer as torch_optimizer_module

from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    CaptureFrameTensors,
    CaptureTensor,
    TorchOptimizationConfig,
    TrainingFrame,
    capture_tensors_to_packed_render_batches,
    torch_capture_asset_batch,
    torch_capture_training_batch,
    torch_capture_training_batch_from_packed,
    torch_optimize_capture_batch,
    torch_optimize_capture_batches,
)
from aura.evolution import CarrierEvolutionPolicy
from aura.optimize import TrainingLossWeights


def test_torch_optimize_capture_batch_reports_install_hint_when_unavailable():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    scene = AuraScene(
        name="torch_optimizer_unavailable",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    with pytest.raises(RuntimeError, match="torch"):
        torch_optimize_capture_batch(scene, _fake_capture_training_batch())


def test_torch_optimization_config_validates_bounds():
    with pytest.raises(ValueError, match="iterations"):
        TorchOptimizationConfig(iterations=0)

    with pytest.raises(ValueError, match="color_learning_rate"):
        TorchOptimizationConfig(color_learning_rate=0.0)

    with pytest.raises(ValueError, match="gradient_clip_norm"):
        TorchOptimizationConfig(gradient_clip_norm=0.0)

    with pytest.raises(ValueError, match="max_samples_per_batch"):
        TorchOptimizationConfig(max_samples_per_batch=0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_gradient_step_clips_with_device_side_norm():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parameter = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device, requires_grad=True)
    parameter.grad = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float32, device=device)

    update = torch_optimizer_module._gradient_step_carrier_parameters(
        torch,
        {"surface": {"color": parameter}},
        learning_rate=0.1,
        gradient_clip_norm=2.0,
    )

    assert update.gradient_norm_tensor is not None
    assert update.scale_tensor is not None
    assert str(update.gradient_norm_tensor.device).startswith(device)
    assert str(update.scale_tensor.device).startswith(device)
    assert update.gradient_norm == pytest.approx(5.0)
    assert update.applied_gradient_norm == pytest.approx(2.0)
    assert update.updated_parameter_count == 1
    assert str(parameter.device).startswith(device)
    assert parameter.detach().cpu().tolist() == pytest.approx([0.38, 0.34, 0.5])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_updates_native_carrier_color():
    scene = AuraScene(
        name="torch_optimizer_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(2.0,),
                ),
                normal=CaptureTensor(
                    path="frame_normal.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(0.0, 0.0, -1.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.5,
            loss_weights=TrainingLossWeights(image=1.0, depth=1.0, query=0.0, normal=1.0, mask=1.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    assert result.steps[0].sample_count == 1
    assert result.steps[0].device == "cpu"
    assert result.steps[0].carrier_counts == {"surface": 1}
    assert result.steps[0].loss_weights["image"] == 1.0
    assert result.steps[0].loss_weights["query"] == 0.0
    assert result.steps[0].optimizer == "sgd"
    assert result.steps[0].gradient_norm > 0.0
    assert result.steps[0].applied_gradient_norm <= result.steps[0].gradient_norm
    assert result.steps[0].gradient_clip_norm == 10.0
    assert result.steps[0].updated_parameter_count > 0
    assert result.steps[0].max_samples_per_batch == 1
    assert result.steps[0].mask_loss == pytest.approx(0.0)
    assert result.steps[0].image_loss > result.steps[1].image_loss
    assert result.steps[0].normal_loss == pytest.approx(0.0)
    assert result.scene.elements[0].color[0] > scene.elements[0].color[0]
    assert result.scene.elements[0].metadata["optimized_by"] == "aura-core-torch-autograd"
    assert result.to_dict()["finalLoss"] == result.steps[-1].total_loss
    assert result.to_dict()["lossCurve"][-1]["totalLoss"] == result.steps[-1].total_loss
    assert result.to_dict()["checkpoints"][-1]["loss"]["total"] == result.steps[-1].total_loss


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_avoids_per_step_render_serialization_without_evolution(monkeypatch):
    scene = AuraScene(
        name="torch_optimizer_objective_only_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(2.0,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    original_summary = torch_optimizer_module.torch_render_capture_training_summary
    original_scene_from_parameters = torch_optimizer_module._scene_from_carrier_parameters
    summary_calls = []
    scene_materialization_calls = []

    def counted_summary(*args, **kwargs):
        summary_calls.append(1)
        return original_summary(*args, **kwargs)

    def counted_scene_from_parameters(*args, **kwargs):
        scene_materialization_calls.append(1)
        return original_scene_from_parameters(*args, **kwargs)

    monkeypatch.setattr(torch_optimizer_module, "torch_render_capture_training_summary", counted_summary)
    monkeypatch.setattr(torch_optimizer_module, "_scene_from_carrier_parameters", counted_scene_from_parameters)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.5,
            loss_weights=TrainingLossWeights(image=1.0, depth=1.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )

    assert len(result.steps) == 3
    assert not hasattr(torch_optimizer_module, "torch_render_capture_training_batch")
    assert summary_calls == [1]
    assert scene_materialization_calls == [1]
    assert result.scene.elements[0].metadata["optimized_by"] == "aura-core-torch-autograd"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_uses_compact_summaries_for_evolution(monkeypatch):
    scene = AuraScene(
        name="torch_optimizer_summary_evolution_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(2.0,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    original_summary = torch_optimizer_module.torch_render_capture_training_summary
    summary_calls = []

    def counted_summary(*args, **kwargs):
        summary_calls.append(1)
        return original_summary(*args, **kwargs)

    monkeypatch.setattr(torch_optimizer_module, "torch_render_capture_training_summary", counted_summary)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.05,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            evolution_policy=CarrierEvolutionPolicy(split_image_loss_threshold=0.0),
            max_samples_per_batch=1,
        ),
    )

    assert summary_calls == [1, 1, 1, 1]
    assert not hasattr(torch_optimizer_module, "torch_render_capture_training_batch")
    assert any(step.carrier_evolution for step in result.steps)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_checkpoints_without_full_render_batches(monkeypatch):
    scene = AuraScene(
        name="torch_optimizer_summary_checkpoint_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(2.0,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.5,
            loss_weights=TrainingLossWeights(image=1.0, depth=1.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
            checkpoint_interval=1,
        ),
    )

    assert len(result.scene_checkpoints) == 2
    assert not hasattr(torch_optimizer_module, "torch_render_capture_training_batch")
    assert result.scene_checkpoints[-1].scene.elements[0].metadata["optimized_by"] == "aura-core-torch-autograd"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_trains_surface_confidence_from_targets():
    scene = AuraScene(
        name="torch_optimizer_surface_confidence_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=0.1,
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=0.0, query=0.0, normal=0.0, mask=0.0, confidence=1.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    assert result.steps[0].confidence_loss > result.steps[-1].confidence_loss
    assert result.steps[0].loss_weights["confidence"] == 1.0
    assert result.to_dict()["lossCurve"][0]["confidenceLoss"] == result.steps[0].confidence_loss
    assert result.scene.elements[0].confidence > scene.elements[0].confidence


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_updates_gabor_plane_point_from_depth_loss():
    scene = AuraScene(
        name="torch_optimizer_gabor_geometry_scene",
        elements=(
            AuraElement(
                id="gabor",
                carrier_id="gabor",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2)),
                color=(1.0, 0.5, 0.25),
                opacity=1.0,
                payload={"type": "gabor_frequency", "frequency": [1.0, 0.0, 0.0], "bandwidth": 0.5, "phase": 0.0},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.5, 0.25),
        target_depth=1.8,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.5, 0.25),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(1.8,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=1.0, query=0.0, normal=0.0, mask=0.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    assert result.steps[0].depth_loss > result.steps[-1].depth_loss
    assert result.scene.elements[0].payload["plane_point"][2] < 0.1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_trains_gabor_opacity_from_mask_loss():
    scene = AuraScene(
        name="torch_optimizer_gabor_opacity_scene",
        elements=(
            AuraElement(
                id="gabor",
                carrier_id="gabor",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.2)),
                color=(1.0, 0.5, 0.25),
                opacity=0.1,
                payload={"type": "gabor_frequency", "frequency": [0.0, 0.0, 0.0], "bandwidth": 0.5, "phase": 0.0},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.5, 0.25),
        target_depth=2.1,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.5, 0.25),
                ),
                mask=CaptureTensor(
                    path="frame-mask.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(1.0,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=0.0, query=0.0, normal=0.0, mask=1.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    assert result.steps[0].mask_loss > result.steps[-1].mask_loss
    assert result.scene.elements[0].opacity > scene.elements[0].opacity


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_updates_surface_geometry_from_depth_loss():
    scene = AuraScene(
        name="torch_optimizer_geometry_scene",
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
        target_depth=1.5,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(1.5,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=1.0, query=0.0, normal=0.0, mask=0.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    assert result.steps[0].depth_loss > result.steps[1].depth_loss
    assert result.steps[0].updated_parameter_count > 0
    assert result.scene.elements[0].payload["plane_point"][2] < 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_updates_surface_normal_from_normal_loss():
    scene = AuraScene(
        name="torch_optimizer_surface_normal_scene",
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
                normal=CaptureTensor(
                    path="frame-normal.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(0.0, -1.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=0.0, query=0.0, normal=1.0, mask=0.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    learned_normal = result.scene.elements[0].normal
    assert result.steps[0].normal_loss > result.steps[-1].normal_loss
    assert learned_normal is not None
    assert learned_normal[1] < -0.2
    assert result.scene.elements[0].payload["normal"] == pytest.approx(learned_normal)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_persists_beta_support_radius():
    scene = AuraScene(
        name="torch_optimizer_beta_support_scene",
        elements=(
            AuraElement(
                id="beta",
                carrier_id="beta",
                bounds=Bounds((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                payload={
                    "type": "beta_kernel",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "confidence": 0.7,
                    "support_radius": [0.5, 0.5, 0.5],
                },
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.2,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(2.2,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=1.0, query=0.0, normal=0.0, mask=0.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    support_radius = result.scene.elements[0].payload["support_radius"]
    assert result.steps[0].depth_loss > result.steps[-1].depth_loss
    assert support_radius[2] > 0.5
    assert result.scene.elements[0].payload["confidence"] == pytest.approx(result.scene.elements[0].confidence)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_trains_gaussian_covariance_from_image_loss():
    scene = AuraScene(
        name="torch_optimizer_gaussian_covariance_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.5],
                    "covariance": [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
                    "source": "test",
                },
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.4, 0.0, -2.0),
        look_at=(0.4, 0.0, 0.5),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.5,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    covariance = result.scene.elements[0].payload["covariance"]
    assert result.steps[0].image_loss > result.steps[-1].image_loss
    assert covariance[0][0] > 0.04


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_trains_gaussian_mean_from_image_loss():
    scene = AuraScene(
        name="torch_optimizer_gaussian_mean_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.5],
                    "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                    "source": "test",
                },
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.4, 0.0, -2.0),
        look_at=(0.4, 0.0, 0.5),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.5,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    trained_mean = result.scene.elements[0].payload["mean"]
    assert result.steps[0].image_loss > result.steps[-1].image_loss
    assert trained_mean[0] > 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_trains_neural_residual_scale_from_image_loss():
    scene = AuraScene(
        name="torch_optimizer_neural_residual_scene",
        elements=(
            AuraElement(
                id="neural",
                carrier_id="neural",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                payload={"type": "neural_residual", "latent_dim": 8, "residual_scale": 0.1},
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    payload = result.scene.elements[0].payload
    assert result.steps[0].image_loss > result.steps[-1].image_loss
    assert payload["residual_scale"] > 0.1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_trains_neural_opacity_from_mask_loss():
    scene = AuraScene(
        name="torch_optimizer_neural_opacity_scene",
        elements=(
            AuraElement(
                id="neural",
                carrier_id="neural",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.1, 0.2, 0.3),
                opacity=0.1,
                confidence=0.8,
                payload={"type": "neural_residual", "latent_dim": 8, "residual_scale": 1.0, "opacity": 0.1},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.1, 0.2, 0.3),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(0.1, 0.2, 0.3),
                ),
                mask=CaptureTensor(
                    path="frame-mask.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(1.0,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=0.0, query=0.0, normal=0.0, mask=1.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    payload = result.scene.elements[0].payload
    assert result.steps[0].mask_loss > result.steps[-1].mask_loss
    assert result.scene.elements[0].opacity > 0.1
    assert payload["opacity"] == pytest.approx(result.scene.elements[0].opacity)
    assert payload["confidence"] == pytest.approx(result.scene.elements[0].confidence)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_trains_volume_opacity_from_mask_loss():
    scene = AuraScene(
        name="torch_optimizer_volume_opacity_scene",
        elements=(
            AuraElement(
                id="volume",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0)),
                color=(0.2, 0.4, 0.8),
                opacity=0.1,
                confidence=0.8,
                payload={"type": "volume_cell", "density": 1.0, "opacity": 0.1},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.2, 0.4, 0.8),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=3,
                    values=(0.2, 0.4, 0.8),
                ),
                mask=CaptureTensor(
                    path="frame-mask.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=1,
                    height=1,
                    channels=1,
                    values=(1.0,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.25,
            loss_weights=TrainingLossWeights(image=0.0, depth=0.0, query=0.0, normal=0.0, mask=1.0),
            gradient_clip_norm=10.0,
            max_samples_per_batch=1,
        ),
    )

    payload = result.scene.elements[0].payload
    assert result.steps[0].mask_loss > result.steps[-1].mask_loss
    assert result.scene.elements[0].opacity > 0.1
    assert payload["opacity"] == pytest.approx(result.scene.elements[0].opacity)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_optimizes_semantic_query_loss():
    scene = AuraScene(
        name="torch_optimizer_query_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.2,
                semantic_id="tooth",
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
        semantic_label="tooth",
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    tensors = (
        CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(
                path="frame.ppm",
                format="Netpbm",
                backend="stdlib",
                width=1,
                height=1,
                channels=3,
                values=(1.0, 0.0, 0.0),
            ),
            depth=CaptureTensor(
                path="frame.pgm",
                format="Netpbm",
                backend="stdlib",
                width=1,
                height=1,
                channels=1,
                values=(2.0,),
            ),
        ),
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        (frame,),
        tensors,
        tile_size=1,
        max_targets_per_batch=1,
    )

    result = torch_optimize_capture_batches(
        scene,
        packed_batches,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.5,
            loss_weights=TrainingLossWeights(image=0.0, depth=0.0, query=1.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
        device="cpu",
    )

    assert result.steps[0].query_loss > result.steps[1].query_loss
    assert result.steps[0].loss_weights["query"] == 1.0
    assert result.scene.elements[0].opacity > scene.elements[0].opacity
    assert result.scene.elements[0].confidence_map["torch_query_loss"] < result.steps[0].query_loss


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_localizes_evolved_child_to_high_error_ray():
    scene = AuraScene(
        name="torch_optimizer_local_evolution_scene",
        elements=(
            AuraElement(
                id="soft_volume",
                carrier_id="volume",
                bounds=Bounds((-1.5, -1.0, -0.5), (1.5, 1.0, 0.5)),
                color=(0.0, 0.0, 0.0),
                opacity=0.5,
                confidence=0.8,
                payload={"type": "volume_cell", "density": 0.5},
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    tensors = (
        CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(
                path="frame.ppm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=3,
                values=(1.0, 0.0, 0.0, 0.05, 0.0, 0.0),
            ),
            depth=CaptureTensor(
                path="frame.pgm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=1,
                values=(2.0, 2.0),
            ),
        ),
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        (frame,),
        tensors,
        tile_size=1,
        max_targets_per_batch=1,
    )

    result = torch_optimize_capture_batches(
        scene,
        packed_batches,
        TorchOptimizationConfig(
            iterations=1,
            color_learning_rate=0.05,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            evolution_policy=CarrierEvolutionPolicy(split_image_loss_threshold=0.0),
            max_samples_per_batch=1,
        ),
        device="cpu",
    )
    by_id = {element.id: element for element in result.scene.elements}

    assert "soft_volume_beta_detail" in by_id
    assert result.steps[-1].carrier_evolution[0]["action"] == "split_beta_detail"
    assert by_id["soft_volume_beta_detail"].bounds.max_corner[0] < 0.0
    assert by_id["soft_volume_beta_detail"].metadata["parent"] == "soft_volume"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_splits_surface_residual_during_training():
    scene = AuraScene(
        name="torch_optimizer_surface_evolution_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-1.5, -1.0, 0.0), (1.5, 1.0, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                confidence=0.8,
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
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    tensors = (
        CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(
                path="frame.ppm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=3,
                values=(1.0, 0.0, 0.0, 0.05, 0.0, 0.0),
            ),
            depth=CaptureTensor(
                path="frame.pgm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=1,
                values=(2.0, 2.0),
            ),
        ),
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        (frame,),
        tensors,
        tile_size=1,
        max_targets_per_batch=1,
    )

    result = torch_optimize_capture_batches(
        scene,
        packed_batches,
        TorchOptimizationConfig(
            iterations=1,
            color_learning_rate=0.05,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            evolution_policy=CarrierEvolutionPolicy(split_image_loss_threshold=0.0),
            max_samples_per_batch=1,
        ),
        device="cpu",
    )
    by_id = {element.id: element for element in result.scene.elements}

    assert "surface_beta_detail" in by_id
    assert result.steps[-1].carrier_evolution[0]["action"] == "split_beta_detail"
    assert result.steps[-1].carrier_evolution[0]["reason"] == "surface evidence benefits from compact bounded support"
    assert by_id["surface_beta_detail"].carrier_id == "beta"
    assert by_id["surface_beta_detail"].metadata["parent"] == "surface"
    assert by_id["surface_beta_detail"].payload["type"] == "beta_kernel"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_stream_packed_source_windows():
    scene = AuraScene(
        name="torch_packed_optimizer_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-3.0, -1.0, 0.0), (3.0, 1.0, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    tensors = (
        CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(
                path="frame.ppm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=3,
                values=(1.0, 0.0, 0.0, 0.5, 0.0, 0.0),
            ),
            depth=CaptureTensor(
                path="frame.pgm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=1,
                values=(2.0, 2.0),
            ),
            mask=CaptureTensor(
                path="frame-mask.pgm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=1,
                values=(1.0, 0.5),
            ),
        ),
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        (frame,),
        tensors,
        tile_size=1,
        max_targets_per_batch=1,
    )

    result = torch_optimize_capture_batches(
        scene,
        packed_batches,
        TorchOptimizationConfig(iterations=1, color_learning_rate=0.5, max_samples_per_batch=1),
        device="cpu",
    )

    assert len(result.steps) == 2
    assert [step.batch_index for step in result.steps] == [0, 1]
    assert [step.target_offset for step in result.steps] == [0, 1]
    assert all(step.sample_count == 1 for step in result.steps)
    assert result.steps[0].source_windows[0]["tileIndex"] == 0
    assert result.steps[1].source_windows[0]["tileIndex"] == 1
    assert result.steps[0].max_samples_per_batch == 1
    assert result.scene.elements[0].color[0] > scene.elements[0].color[0]
    first_batch = torch_capture_training_batch_from_packed(packed_batches[0], device="cpu")
    second_batch = torch_capture_training_batch_from_packed(packed_batches[1], device="cpu")
    assert first_batch.sample_frame_ids == ("frame",)
    assert second_batch.sample_frame_ids == ("frame",)
    assert first_batch.target_confidence.tolist() == [1.0]
    assert second_batch.target_confidence.tolist() == [0.5]
    assert result.to_dict()["steps"][0]["source_windows"][0]["targetCount"] == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_reuses_prepared_packed_batches(monkeypatch):
    scene = AuraScene(
        name="torch_packed_lazy_optimizer_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-3.0, -1.0, 0.0), (3.0, 1.0, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    tensors = (
        CaptureFrameTensors(
            frame_id="frame",
            image=CaptureTensor(
                path="frame.ppm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=3,
                values=(1.0, 0.0, 0.0, 0.5, 0.0, 0.0),
            ),
            depth=CaptureTensor(
                path="frame.pgm",
                format="Netpbm",
                backend="stdlib",
                width=2,
                height=1,
                channels=1,
                values=(2.0, 2.0),
            ),
        ),
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        (frame,),
        tensors,
        tile_size=1,
        max_targets_per_batch=1,
    )
    original_converter = torch_optimizer_module.torch_capture_training_batch_from_packed
    converted_batch_indices = []

    def counted_converter(packed, *, device=None):
        converted_batch_indices.append(packed.batch_index)
        return original_converter(packed, device=device)

    monkeypatch.setattr(torch_optimizer_module, "torch_capture_training_batch_from_packed", counted_converter)

    result = torch_optimize_capture_batches(
        scene,
        packed_batches,
        TorchOptimizationConfig(iterations=2, color_learning_rate=0.5, max_samples_per_batch=1),
        device="cpu",
    )

    assert len(result.steps) == 4
    assert converted_batch_indices == [0, 1]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_enforces_sample_cap():
    scene = AuraScene(
        name="torch_optimizer_cap_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
            ),
        ),
    )
    frame = TrainingFrame(
        id="frame",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm",
                    format="Netpbm",
                    backend="stdlib",
                    width=2,
                    height=1,
                    channels=3,
                    values=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm",
                    format="Netpbm",
                    backend="stdlib",
                    width=2,
                    height=1,
                    channels=1,
                    values=(2.0, 2.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    with pytest.raises(ValueError, match="max_samples_per_batch"):
        torch_optimize_capture_batch(
            scene,
            batch,
            TorchOptimizationConfig(iterations=1, max_samples_per_batch=1),
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
            "target_normal": None,
            "target_normal_present": None,
        },
    )()


# ---- New Deliverable Tests ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimization_config_adam_is_valid():
    config = TorchOptimizationConfig(optimizer_type='adam')
    assert config.optimizer_type == 'adam'
    assert config.position_learning_rate == pytest.approx(1.6e-4)
    assert config.scale_learning_rate == pytest.approx(5e-3)
    assert config.rotation_learning_rate == pytest.approx(1e-3)
    assert config.opacity_learning_rate == pytest.approx(5e-2)
    assert config.feature_learning_rate == pytest.approx(2.5e-3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimization_config_rejects_invalid_optimizer_type():
    with pytest.raises(ValueError, match="optimizer_type"):
        TorchOptimizationConfig(optimizer_type='rmsprop')


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_with_adam_reduces_loss():
    scene = AuraScene(
        name="adam_optimizer_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
                depth=CaptureTensor(
                    path="frame.pgm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=1, values=(2.0,),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=3,
            color_learning_rate=0.5,
            optimizer_type='adam',
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert result.steps[0].image_loss > result.steps[-1].image_loss


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_adam_reports_adam_optimizer():
    scene = AuraScene(
        name="adam_label_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=1,
            color_learning_rate=0.5,
            optimizer_type='adam',
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert result.steps[0].optimizer == "adam"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_sgd_stays_backward_compatible():
    scene = AuraScene(
        name="sgd_compat_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=1,
            color_learning_rate=0.5,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert result.steps[0].optimizer == "sgd"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_lr_schedule_position_decay():
    config = TorchOptimizationConfig(
        optimizer_type='adam',
        position_learning_rate=1e-3,
        position_lr_final=1e-6,
        lr_decay_steps=10,
        color_learning_rate=0.1,
        loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
    )
    # LR at step 0 should be initial
    lr_0 = torch_optimizer_module._compute_position_lr(config, 0)
    # LR at step 10 should be at or near final
    lr_10 = torch_optimizer_module._compute_position_lr(config, 10)
    # LR at step 5 should be in between
    lr_5 = torch_optimizer_module._compute_position_lr(config, 5)
    assert lr_0 == pytest.approx(1e-3)
    assert lr_10 == pytest.approx(1e-6)
    assert lr_5 < lr_0
    assert lr_5 > lr_10


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_grad_stats_accumulated_when_enabled():
    scene = AuraScene(
        name="grad_accum_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=5,
            color_learning_rate=0.1,
            grad_accum_window=5,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    # After 5 steps, at least the last step should have non-empty grad_stats
    assert len(result.steps) == 5
    # Some step should have grad_stats
    all_stats = [step.grad_stats for step in result.steps]
    assert any(len(s) > 0 for s in all_stats)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_opacity_reset_signals_in_report():
    scene = AuraScene(
        name="opacity_reset_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.9,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=4,
            color_learning_rate=0.1,
            opacity_reset_interval=2,
            opacity_reset_value=0.01,
            recovery_window=100,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert len(result.steps) == 4
    # Step at iteration 2 (index 2) should have opacity_reset_due=True
    reset_steps = [step for step in result.steps if step.opacity_reset_due]
    assert len(reset_steps) >= 1
    assert reset_steps[0].opacity_reset_due is True


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_importance_scores_function():
    opacities = {"carrier_a": 0.8, "carrier_b": 0.5}
    transmittances = {"carrier_a": 0.9, "carrier_b": 0.3}
    scores = torch_optimizer_module.compute_importance_scores(opacities, transmittances)
    assert scores["carrier_a"] == pytest.approx(0.72)
    assert scores["carrier_b"] == pytest.approx(0.15)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_importance_scores_function_missing_transmittance():
    opacities = {"carrier_x": 0.6}
    transmittances = {}
    scores = torch_optimizer_module.compute_importance_scores(opacities, transmittances)
    # Defaults transmittance to 1.0 if missing
    assert scores["carrier_x"] == pytest.approx(0.6)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_budget_ceiling_over_budget_signal():
    scene = AuraScene(
        name="budget_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    # max_carriers=0 means unlimited => over_budget=False
    result_unlimited = torch_optimize_capture_batch(
        scene, batch,
        TorchOptimizationConfig(
            iterations=1, color_learning_rate=0.1, max_carriers=0,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert result_unlimited.steps[0].over_budget is False
    assert result_unlimited.steps[0].carrier_count == 1

    # max_carriers=1 with 1 element => not over budget
    result_exact = torch_optimize_capture_batch(
        scene, batch,
        TorchOptimizationConfig(
            iterations=1, color_learning_rate=0.1, max_carriers=1,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert result_exact.steps[0].over_budget is False
    assert result_exact.steps[0].carrier_count == 1

    # max_carriers=0 means unlimited, still False
    result_over = torch_optimize_capture_batch(
        scene, batch,
        TorchOptimizationConfig(
            iterations=1, color_learning_rate=0.1, max_carriers=0,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert result_over.steps[0].over_budget is False


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_coarse_to_fine_schedule_returns_correct_scale():
    schedule = ((0, 0.5), (5, 1.0))

    scale_at_0 = torch_optimizer_module._coarse_to_fine_scale(schedule, 0)
    assert scale_at_0 == pytest.approx(0.5)

    scale_at_3 = torch_optimizer_module._coarse_to_fine_scale(schedule, 3)
    assert scale_at_3 == pytest.approx(0.5)

    scale_at_5 = torch_optimizer_module._coarse_to_fine_scale(schedule, 5)
    assert scale_at_5 == pytest.approx(1.0)

    scale_empty = torch_optimizer_module._coarse_to_fine_scale((), 10)
    assert scale_empty == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_coarse_to_fine_in_step_report():
    scene = AuraScene(
        name="c2f_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene, batch,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.1,
            coarse_to_fine_schedule=((0, 0.5), (1, 1.0)),
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    assert result.steps[0].resolution_scale == pytest.approx(0.5)
    assert result.steps[1].resolution_scale == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_recovery_phase_after_opacity_reset():
    scene = AuraScene(
        name="recovery_phase_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.9,
                normal=(0.0, 0.0, -1.0),
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
            CaptureFrameTensors(
                frame_id="frame",
                image=CaptureTensor(
                    path="frame.ppm", format="Netpbm", backend="stdlib",
                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)

    result = torch_optimize_capture_batch(
        scene, batch,
        TorchOptimizationConfig(
            iterations=5,
            color_learning_rate=0.1,
            opacity_reset_interval=2,
            opacity_reset_value=0.01,
            recovery_window=100,  # large window so all post-reset steps are in recovery
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    # Step 2 (index 2, iteration 2) should have opacity_reset_due=True
    # Steps after a reset should have recovery_phase=True
    reset_step = next((s for s in result.steps if s.opacity_reset_due), None)
    assert reset_step is not None
    # Steps after reset should be in recovery (iteration 3, 4 -> step indices 3, 4)
    recovery_steps = [s for s in result.steps if s.recovery_phase]
    assert len(recovery_steps) > 0
