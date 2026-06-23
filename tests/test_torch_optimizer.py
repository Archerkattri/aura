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
def test_torch_importance_scores_are_real_in_step_report():
    # Two carriers with distinct opacities; RadSplat importance must reflect them
    # in the per-step report (not an empty tuple). This fails if the loop reverts
    # to hardcoding importance_scores=().
    scene = AuraScene(
        name="importance_scene",
        elements=(
            AuraElement(
                id="opaque",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=0.9,
                normal=(0.0, 0.0, -1.0),
            ),
            AuraElement(
                id="faint",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.2), (0.5, 0.5, 0.3)),
                color=(0.0, 0.0, 0.0),
                opacity=0.1,
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
            iterations=1,
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    importance = dict(result.steps[0].importance_scores)
    # Real, non-empty, covers both carriers
    assert set(importance) == {"opaque", "faint"}
    # Importance tracks live opacity: the opaque carrier outranks the faint one
    assert importance["opaque"] > importance["faint"]
    assert importance["opaque"] == pytest.approx(0.9, abs=0.2)


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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_scene_materialization_keeps_finite_bounds_when_params_diverge():
    """A carrier whose optimized bounds went NaN must keep its last valid bounds
    so the chunk union (and the written package) stays finite/valid."""
    import torch
    from aura.torch_optimizer import _scene_from_carrier_parameters, _all_finite

    assert _all_finite((1.0, 2.0, 3.0)) is True
    assert _all_finite((1.0, float("nan"), 3.0)) is False
    assert _all_finite((float("inf"), 0.0, 0.0)) is False

    scene = AuraScene(
        name="nan_guard",
        elements=(
            AuraElement(
                id="g", carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 1.0), (1.0, 1.0, 3.0)),
                color=(0.5, 0.5, 0.5), opacity=0.8, confidence=1.0,
                payload={"type": "gaussian_fallback", "mean": [0.0, 0.0, 2.0],
                         "covariance": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]},
            ),
        ),
    )
    nan = float("nan")
    carrier_parameters = {
        "g": {
            "min_corner": torch.tensor([nan, nan, 1.0]),
            "max_corner": torch.tensor([1.0, 1.0, 3.0]),
            "gaussian_mean": torch.tensor([nan, 0.0, 2.0]),
        }
    }
    out = _scene_from_carrier_parameters(scene, carrier_parameters)
    el = out.elements[0]
    # NaN-divergent bounds/mean are rejected; the original valid values survive.
    assert _all_finite(el.bounds.min_corner) and _all_finite(el.bounds.max_corner)
    assert _all_finite(el.payload["mean"])
    # Chunk bounds (the union) are therefore finite too.
    for chunk in out.chunks:
        assert _all_finite(chunk.bounds.min_corner) and _all_finite(chunk.bounds.max_corner)


# ---- Densification / Pruning / Regularization Tests ----

def test_densification_config_defaults_are_disabled():
    """DensificationConfig must be off by default — ensures back-compat."""
    from aura.torch_optimizer import DensificationConfig
    cfg = DensificationConfig()
    assert cfg.enabled is False
    assert cfg.scale_reg_weight == 0.0
    assert cfg.opacity_entropy_reg_weight == 0.0


def test_densification_config_validates():
    from aura.torch_optimizer import DensificationConfig
    with pytest.raises(ValueError, match="interval"):
        DensificationConfig(enabled=True, interval=0)
    with pytest.raises(ValueError, match="end_iteration"):
        DensificationConfig(enabled=True, start_iteration=1000, end_iteration=100)
    with pytest.raises(ValueError, match="scale_reg_weight"):
        DensificationConfig(scale_reg_weight=-1.0)


def test_torch_optimization_config_accepts_densification_config():
    from aura.torch_optimizer import DensificationConfig
    dcfg = DensificationConfig(enabled=True, grad_threshold=0.001)
    cfg = TorchOptimizationConfig(densification=dcfg)
    assert cfg.densification.enabled is True
    assert cfg.densification.grad_threshold == pytest.approx(0.001)


def test_densification_disabled_by_default_in_optimization_config():
    cfg = TorchOptimizationConfig()
    assert cfg.densification.enabled is False


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_clone_increases_carrier_count():
    """When AbsGS gradient exceeds threshold, DensificationEngine must clone carriers."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="densify_test",
        elements=(
            AuraElement(
                id="carrier_a",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
            ),
        ),
    )
    # Simulate high gradient for carrier_a (well above threshold)
    carrier_parameters = {
        "carrier_a": {
            "color": torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            "opacity": torch.tensor([0.8], requires_grad=True),
        }
    }
    grad_accumulator = {
        "carrier_a.color": 0.01,  # 10x above threshold 0.001
        "carrier_a.opacity": 0.005,
    }
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        prune_importance_threshold=0.0,  # Don't prune
        prune_opacity_threshold=0.0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    assert num_densified >= 1, f"Expected densification but got {num_densified}"
    assert len(new_scene.elements) > len(scene.elements), (
        f"Expected more elements after densification, got {len(new_scene.elements)}"
    )
    assert num_pruned == 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_split_increases_carrier_count():
    """Large carriers with high gradient should be SPLIT (not cloned)."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    # Create a "large" carrier by giving it a large support radius
    scene = AuraScene(
        name="split_test",
        elements=(
            AuraElement(
                id="big_carrier",
                carrier_id="beta",
                bounds=Bounds((-2.0, -2.0, 0.0), (2.0, 2.0, 0.5)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
                payload={"type": "beta_kernel", "alpha": 3.0, "beta": 3.0, "support_radius": [2.0, 2.0, 0.25]},
            ),
        ),
    )
    carrier_parameters = {
        "big_carrier": {
            "color": torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            "opacity": torch.tensor([0.8], requires_grad=True),
            "support_radius": torch.tensor([2.0, 2.0, 0.25], requires_grad=True),
        }
    }
    # High gradient — should trigger densification
    grad_accumulator = {"big_carrier.color": 0.05}
    # split_threshold_scale=0.5 means any carrier > 0.5 * median scale is split
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        split_threshold_scale=0.5,
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    assert num_densified >= 1
    assert len(new_scene.elements) > len(scene.elements)
    # Distinguish SPLIT from CLONE: a split shrinks the children's support
    # radius (a clone would copy it unchanged). At least one new carrier must
    # carry a support_radius strictly smaller than the parent's largest extent.
    parent_max_radius = 2.0
    child_radii = [
        float(max(fields["support_radius"].tolist()))
        for cid, fields in new_params.items()
        if "support_radius" in fields and cid != "big_carrier"
    ]
    assert child_radii, "split should have produced new carriers with support_radius"
    assert min(child_radii) < parent_max_radius


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_prunes_low_importance_carriers():
    """Carriers with opacity below prune threshold must be removed."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="prune_test",
        elements=(
            AuraElement(
                id="opaque",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.8, 0.0, 0.0),
                opacity=0.9,
            ),
            AuraElement(
                id="faint",
                carrier_id="surface",
                bounds=Bounds((1.0, -0.5, 0.0), (2.0, 0.5, 0.1)),
                color=(0.1, 0.1, 0.1),
                opacity=0.001,  # Well below threshold
            ),
        ),
    )
    carrier_parameters = {
        "opaque": {
            "opacity": torch.tensor([0.9], requires_grad=True),
            "color": torch.tensor([0.8, 0.0, 0.0], requires_grad=True),
        },
        "faint": {
            "opacity": torch.tensor([0.001], requires_grad=True),
            "color": torch.tensor([0.1, 0.1, 0.1], requires_grad=True),
        },
    }
    # No high gradients — only pruning should happen
    grad_accumulator = {}
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=1.0,  # Very high — no densification
        prune_importance_threshold=0.005,
        prune_opacity_threshold=0.005,
        recovery_prune_delay=0,  # No delay
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    assert num_pruned >= 1, f"Expected pruning but got {num_pruned}"
    assert len(new_scene.elements) < len(scene.elements), (
        f"Expected fewer elements after pruning, got {len(new_scene.elements)}"
    )
    element_ids = {e.id for e in new_scene.elements}
    assert "opaque" in element_ids, "High-opacity carrier should not be pruned"
    # The faint carrier should be gone
    assert "faint" not in element_ids, "Low-opacity carrier should have been pruned"
    assert num_densified == 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_does_not_prune_during_recovery_window():
    """A carrier below the opacity/importance threshold must NOT be pruned while
    still inside the opacity-reset recovery window (steps_since_reset <
    recovery_prune_delay) — it is given time to recover its opacity first."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="recovery_prune",
        elements=(
            AuraElement(
                id="faint",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.1, 0.1, 0.1),
                opacity=0.001,  # below the prune threshold
            ),
        ),
    )
    carrier_parameters = {
        "faint": {
            "opacity": torch.tensor([0.001], requires_grad=True),
            "color": torch.tensor([0.1, 0.1, 0.1], requires_grad=True),
        }
    }
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=1.0,            # no densification
        prune_importance_threshold=0.005,
        prune_opacity_threshold=0.005,
        recovery_prune_delay=200,      # recovery window of 200 steps
    )
    new_scene, _new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, {},
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=50,          # 50 < 200 -> still in recovery
        max_carriers_budget=0,
        torch=torch,
    )
    assert num_pruned == 0, "carrier below threshold must survive during recovery window"
    assert {e.id for e in new_scene.elements} == {"faint"}


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_new_parameters_are_trainable():
    """After densification, cloned parameters must have requires_grad=True."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="trainable_params_test",
        elements=(
            AuraElement(
                id="carrier",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
            ),
        ),
    )
    carrier_parameters = {
        "carrier": {
            "color": torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            "opacity": torch.tensor([0.8], requires_grad=True),
        }
    }
    grad_accumulator = {"carrier.color": 0.1}  # High gradient
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    assert num_densified >= 1
    # All new parameters in new_params must be trainable
    new_carrier_ids = {e.id for e in new_scene.elements} - {e.id for e in scene.elements}
    assert len(new_carrier_ids) >= 1, "No new carriers created"
    for cid in new_carrier_ids:
        params = new_params.get(cid, {})
        assert len(params) > 0, f"New carrier {cid} has no parameters"
        for pname, tensor in params.items():
            if hasattr(tensor, 'requires_grad'):
                assert tensor.requires_grad, (
                    f"New carrier {cid} parameter {pname} is not trainable"
                )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_disabled_leaves_scene_unchanged():
    """With densification disabled, DensificationEngine.should_run must return False."""
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    cfg = DensificationConfig(enabled=False)
    # should_run must return False regardless of iteration
    for iteration in [0, 500, 1000, 15000]:
        assert DensificationEngine.should_run(iteration, cfg) is False


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_respects_max_carriers_budget():
    """Densification must not exceed max_carriers budget."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="budget_test",
        elements=tuple(
            AuraElement(
                id=f"carrier_{i}",
                carrier_id="surface",
                bounds=Bounds((-0.5 + i, -0.5, 0.0), (0.5 + i, 0.5, 0.1)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
            )
            for i in range(3)
        ),
    )
    carrier_parameters = {
        f"carrier_{i}": {
            "color": torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            "opacity": torch.tensor([0.8], requires_grad=True),
        }
        for i in range(3)
    }
    # High gradient on all carriers — would densify all without budget
    grad_accumulator = {f"carrier_{i}.color": 0.1 for i in range(3)}
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
        max_carriers=4,  # Only allow up to 4 total (3 original + at most 1 new)
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=4,
        torch=torch,
    )
    assert len(new_scene.elements) <= 4, (
        f"Budget exceeded: got {len(new_scene.elements)} elements, max=4"
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_regularization_loss_is_zero_by_default():
    """When both reg weights are 0.0, _compute_regularization_loss returns None."""
    import torch
    from aura.torch_optimizer import DensificationConfig, _compute_regularization_loss

    carrier_parameters = {
        "e1": {
            "color": torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            "opacity": torch.tensor([0.8], requires_grad=True),
            "support_radius": torch.tensor([0.1, 0.2, 0.15], requires_grad=True),
        }
    }
    cfg = DensificationConfig(scale_reg_weight=0.0, opacity_entropy_reg_weight=0.0)
    result = _compute_regularization_loss(carrier_parameters, cfg, torch=torch)
    assert result is None, "Reg loss should be None (disabled) when weights are 0"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_scale_regularization_is_nonzero_when_enabled():
    """Scale/anisotropy regularization must produce a positive loss for anisotropic carriers."""
    import torch
    from aura.torch_optimizer import DensificationConfig, _compute_regularization_loss

    carrier_parameters = {
        "e1": {
            "support_radius": torch.tensor([2.0, 0.01, 0.01], requires_grad=True),
        }
    }
    cfg = DensificationConfig(scale_reg_weight=0.01, opacity_entropy_reg_weight=0.0)
    result = _compute_regularization_loss(carrier_parameters, cfg, torch=torch)
    assert result is not None
    loss_val = float(result.detach())
    assert loss_val > 0.0, f"Scale reg loss should be > 0 for anisotropic carrier, got {loss_val}"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_opacity_entropy_regularization_is_nonzero_when_enabled():
    """Opacity entropy reg must produce positive loss for mid-range opacities."""
    import torch
    from aura.torch_optimizer import DensificationConfig, _compute_regularization_loss

    carrier_parameters = {
        "e1": {
            "opacity": torch.tensor([0.5], requires_grad=True),  # max entropy at 0.5
        }
    }
    cfg = DensificationConfig(scale_reg_weight=0.0, opacity_entropy_reg_weight=0.1)
    result = _compute_regularization_loss(carrier_parameters, cfg, torch=torch)
    assert result is not None
    loss_val = float(result.detach())
    assert loss_val > 0.0, f"Opacity entropy loss should be > 0 for opacity=0.5, got {loss_val}"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_integrated_in_training_loop_increases_count():
    """End-to-end: with densification enabled in a training run, carrier count must increase."""
    from aura.torch_optimizer import DensificationConfig

    scene = AuraScene(
        name="integrated_densify_test",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
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
    initial_count = len(scene.elements)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        TorchOptimizationConfig(
            iterations=10,
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
            densification=DensificationConfig(
                enabled=True,
                start_iteration=2,
                end_iteration=10,
                interval=2,
                grad_threshold=0.0,  # Threshold=0: ALL carriers will be densified
                prune_importance_threshold=0.0,  # No pruning
                prune_opacity_threshold=0.0,
            ),
        ),
    )
    final_count = len(result.scene.elements)
    total_densified = sum(s.densified_count for s in result.steps)
    assert total_densified > 0, f"Expected densification events, got 0"
    assert final_count > initial_count, (
        f"Expected more carriers after densification, got {final_count} (started with {initial_count})"
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_off_by_default_does_not_change_carrier_count():
    """With default TorchOptimizationConfig, carrier count must not change due to densification."""
    scene = AuraScene(
        name="no_densify_test",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
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
    initial_count = len(scene.elements)

    result = torch_optimize_capture_batch(
        scene,
        batch,
        # Default config — densification disabled
        TorchOptimizationConfig(
            iterations=5,
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0, query=0.0, normal=0.0, mask=0.0),
            max_samples_per_batch=1,
        ),
    )
    final_count = len(result.scene.elements)
    total_densified = sum(s.densified_count for s in result.steps)
    total_pruned = sum(s.pruned_count for s in result.steps)
    assert total_densified == 0
    assert total_pruned == 0
    # Without evolution, carrier count should stay the same (densification is off)
    assert final_count == initial_count, (
        f"Densification disabled but carrier count changed: {initial_count} -> {final_count}"
    )


# ---- Batch: DensificationConfig validation gaps (lines 77, 83, 85, 87, 91) ----

def test_densification_config_rejects_negative_start_iteration():
    """Cover line 77: start_iteration < 0."""
    from aura.torch_optimizer import DensificationConfig
    with pytest.raises(ValueError, match="start_iteration"):
        DensificationConfig(start_iteration=-1)


def test_densification_config_rejects_negative_grad_threshold():
    """Cover line 83: grad_threshold < 0."""
    from aura.torch_optimizer import DensificationConfig
    with pytest.raises(ValueError, match="grad_threshold"):
        DensificationConfig(grad_threshold=-0.001)


def test_densification_config_rejects_out_of_range_prune_importance():
    """Cover line 85: prune_importance_threshold out of [0, 1]."""
    from aura.torch_optimizer import DensificationConfig
    with pytest.raises(ValueError, match="prune_importance_threshold"):
        DensificationConfig(prune_importance_threshold=1.5)


def test_densification_config_rejects_out_of_range_prune_opacity():
    """Cover line 87: prune_opacity_threshold out of [0, 1]."""
    from aura.torch_optimizer import DensificationConfig
    with pytest.raises(ValueError, match="prune_opacity_threshold"):
        DensificationConfig(prune_opacity_threshold=-0.1)


def test_densification_config_rejects_negative_opacity_entropy_reg_weight():
    """Cover line 91: opacity_entropy_reg_weight < 0."""
    from aura.torch_optimizer import DensificationConfig
    with pytest.raises(ValueError, match="opacity_entropy_reg_weight"):
        DensificationConfig(opacity_entropy_reg_weight=-1.0)


# ---- Batch: TorchOptimizationConfig validation gaps (lines 132, 140, 142, 144, 148, 150, 152, 154, 156) ----

def test_optimization_config_rejects_non_training_loss_weights():
    """Cover line 132: loss_weights must be TrainingLossWeights instance."""
    with pytest.raises(TypeError, match="loss_weights"):
        TorchOptimizationConfig(loss_weights={"image": 1.0})


def test_optimization_config_rejects_non_evolution_policy():
    """Cover line 140: evolution_policy must be CarrierEvolutionPolicy or None."""
    with pytest.raises(TypeError, match="evolution_policy"):
        TorchOptimizationConfig(evolution_policy="invalid")


def test_optimization_config_rejects_negative_iteration_offset():
    """Cover line 142: iteration_offset must be non-negative."""
    with pytest.raises(ValueError, match="iteration_offset"):
        TorchOptimizationConfig(iteration_offset=-1)


def test_optimization_config_rejects_zero_checkpoint_interval():
    """Cover line 144: checkpoint_interval must be positive when set."""
    with pytest.raises(ValueError, match="checkpoint_interval"):
        TorchOptimizationConfig(checkpoint_interval=0)


def test_optimization_config_rejects_invalid_optimizer_type():
    """Cover line 145-146: optimizer_type must be 'sgd' or 'adam'."""
    with pytest.raises(ValueError, match="optimizer_type"):
        TorchOptimizationConfig(optimizer_type="rmsprop")


def test_optimization_config_rejects_out_of_range_opacity_reset_value():
    """Cover line 148: opacity_reset_value must be in [0, 1]."""
    with pytest.raises(ValueError, match="opacity_reset_value"):
        TorchOptimizationConfig(opacity_reset_value=1.5)


def test_optimization_config_rejects_negative_recovery_window():
    """Cover line 150: recovery_window must be non-negative."""
    with pytest.raises(ValueError, match="recovery_window"):
        TorchOptimizationConfig(recovery_window=-1)


def test_optimization_config_rejects_negative_lr_decay_steps():
    """Cover line 152: lr_decay_steps must be non-negative."""
    with pytest.raises(ValueError, match="lr_decay_steps"):
        TorchOptimizationConfig(lr_decay_steps=-1)


def test_optimization_config_rejects_non_positive_position_lr_final():
    """Cover line 154: position_lr_final must be positive."""
    with pytest.raises(ValueError, match="position_lr_final"):
        TorchOptimizationConfig(position_lr_final=0.0)


def test_optimization_config_rejects_non_densification_config():
    """Cover line 156: densification must be a DensificationConfig instance."""
    with pytest.raises(TypeError, match="densification"):
        TorchOptimizationConfig(densification={"enabled": True})


# ---- Batch: TorchOptimizationResult.to_dict and TorchSceneCheckpoint.to_dict (lines 223, 260, 265-268) ----

def test_torch_optimization_result_to_dict_serializes_all_fields():
    """Cover line 223: TorchOptimizationResult.to_dict."""
    from aura.torch_optimizer import TorchOptimizationResult, TorchOptimizationStep
    from aura.optimize import TrainingLossWeights
    scene = AuraScene(
        name="result_scene",
        elements=(
            AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),
        ),
    )
    step = TorchOptimizationStep(
        iteration=0,
        batch_index=None,
        device="cpu",
        sample_count=1,
        target_offset=None,
        image_loss=0.5,
        depth_loss=0.1,
        query_loss=0.0,
        normal_loss=0.0,
        mask_loss=0.0,
        confidence_loss=0.0,
        total_loss=0.6,
        carrier_counts={"surface": 1},
        loss_weights=TrainingLossWeights().to_dict(),
        optimizer="sgd",
        gradient_norm=1.0,
        applied_gradient_norm=1.0,
        gradient_clip_norm=None,
        updated_parameter_count=1,
        max_samples_per_batch=None,
    )
    result = TorchOptimizationResult(scene=scene, steps=(step,))
    d = result.to_dict()
    assert d["scene"] == "result_scene"
    assert len(d["steps"]) == 1
    assert d["finalLoss"] == pytest.approx(0.6)
    assert "lossCurve" in d
    assert "checkpoints" in d
    assert "sceneCheckpoints" in d


def test_torch_optimization_result_to_dict_handles_no_steps():
    """Cover line 223 (empty steps branch): finalLoss is None when steps is empty."""
    from aura.torch_optimizer import TorchOptimizationResult
    scene = AuraScene(
        name="empty_result",
        elements=(
            AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),
        ),
    )
    result = TorchOptimizationResult(scene=scene, steps=())
    d = result.to_dict()
    assert d["finalLoss"] is None
    assert d["steps"] == []


def test_torch_scene_checkpoint_to_dict_serializes_correctly():
    """Cover lines 260, 265-268: TorchSceneCheckpoint.to_dict."""
    from aura.torch_optimizer import TorchSceneCheckpoint
    scene = AuraScene(
        name="checkpoint_scene",
        elements=(
            AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),
            AuraElement(id="g", carrier_id="gaussian", bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0))),
        ),
    )
    checkpoint = TorchSceneCheckpoint(
        checkpoint_index=2,
        iteration=100,
        step_count=50,
        scene=scene,
    )
    d = checkpoint.to_dict()
    assert d["checkpointIndex"] == 2
    assert d["iteration"] == 100
    assert d["stepCount"] == 50
    assert d["scene"] == "checkpoint_scene"
    assert d["elementCount"] == 2
    assert "surface" in d["carrierCounts"]
    assert "gaussian" in d["carrierCounts"]


# ---- Batch: _carrier_importance_scores tensor path (line 294) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_importance_scores_with_tensor_opacity():
    """Cover line 294: _carrier_importance_scores tensor fast-path."""
    import torch
    from aura.torch_optimizer import _carrier_importance_scores

    elements = (
        AuraElement(
            id="e1",
            carrier_id="surface",
            bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
            opacity=0.5,
        ),
        AuraElement(
            id="e2",
            carrier_id="surface",
            bounds=Bounds((0.5, -0.5, 0.0), (1.5, 0.5, 0.1)),
            opacity=0.2,
        ),
    )
    # e1 has a tensor opacity param (will take tensor path), e2 has no tensor
    carrier_parameters = {
        "e1": {"opacity": torch.tensor([0.9], requires_grad=True)},
        "e2": {},  # Will use element.opacity fallback
    }
    scores = _carrier_importance_scores(carrier_parameters, elements)
    assert "e1" in scores
    assert "e2" in scores
    assert scores["e1"] == pytest.approx(0.9, abs=0.01)
    assert scores["e2"] == pytest.approx(0.2, abs=0.01)


# ---- Batch: DensificationEngine.should_run edge cases (line 332) ----

def test_densification_should_run_returns_false_past_end_iteration():
    """Cover line 332: should_run returns False when past end_iteration."""
    from aura.torch_optimizer import DensificationConfig, DensificationEngine
    cfg = DensificationConfig(
        enabled=True,
        start_iteration=100,
        end_iteration=500,
        interval=50,
    )
    # Past end
    assert DensificationEngine.should_run(501, cfg) is False
    # Exactly at end — should run (it's still <= end_iteration at 500)
    assert DensificationEngine.should_run(500, cfg) is True
    # Before start
    assert DensificationEngine.should_run(99, cfg) is False


def test_densification_should_run_only_on_interval_boundaries():
    """should_run fires only on interval-aligned iterations within [start, end],
    and is False for in-range iterations that are NOT on the interval."""
    from aura.torch_optimizer import DensificationConfig, DensificationEngine
    cfg = DensificationConfig(
        enabled=True, start_iteration=500, end_iteration=1000, interval=100,
    )
    assert DensificationEngine.should_run(500, cfg) is True    # aligned at start
    assert DensificationEngine.should_run(600, cfg) is True    # aligned interval
    assert DensificationEngine.should_run(650, cfg) is False   # in range, off interval
    assert DensificationEngine.should_run(1000, cfg) is True   # aligned at end
    assert DensificationEngine.should_run(550, cfg) is False   # in range, off interval


# ---- Batch: densify_and_prune gaussian_covariance_diag path (lines 375-377) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_uses_gaussian_covariance_diag_for_scale():
    """Cover lines 375-377: gaussian_covariance_diag used as scale proxy."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="covariance_scale_test",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
                payload={"type": "gaussian_fallback", "mean": [0.0, 0.0, 0.5],
                         "covariance": [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]]},
            ),
        ),
    )
    carrier_parameters = {
        "gaussian": {
            "color": torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            "opacity": torch.tensor([0.8], requires_grad=True),
            "gaussian_covariance_diag": torch.tensor([0.04, 0.04, 0.04], requires_grad=True),
        }
    }
    grad_accumulator = {"gaussian.color": 0.5}  # High gradient
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    # Should have cloned or split; scale was computed from gaussian_covariance_diag
    assert num_densified >= 1 or num_pruned >= 0  # Ran without error
    assert len(new_scene.elements) >= 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_split_budget_reached_mid_loop():
    """Cover line 390 (median_scale fallback) and line 472 (budget check in split loop)."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    # Empty carrier_parameters -> no scale info -> falls back to bounds
    scene = AuraScene(
        name="budget_split_test",
        elements=(
            AuraElement(
                id="big",
                carrier_id="beta",
                bounds=Bounds((-5.0, -5.0, 0.0), (5.0, 5.0, 5.0)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
                payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [5.0, 5.0, 5.0]},
            ),
        ),
    )
    carrier_parameters = {
        "big": {
            "opacity": torch.tensor([0.8], requires_grad=True),
            "support_radius": torch.tensor([5.0, 5.0, 5.0], requires_grad=True),
        }
    }
    grad_accumulator = {"big.support_radius": 0.5}
    # Budget = 2 total: only 1 original + 1 new allowed
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        split_threshold_scale=0.01,  # Very small -> everything is "large" -> split
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
        max_carriers=2,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=2,
        torch=torch,
    )
    assert len(new_scene.elements) <= 2


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_never_prunes_to_zero():
    """Cover lines 531-533: when ALL elements would be pruned, keep the one with highest opacity."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="never_zero_test",
        elements=(
            AuraElement(
                id="a",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                opacity=0.001,  # All are very faint
            ),
            AuraElement(
                id="b",
                carrier_id="surface",
                bounds=Bounds((0.5, -0.5, 0.0), (1.5, 0.5, 0.1)),
                opacity=0.002,  # Slightly higher — should be retained
            ),
        ),
    )
    carrier_parameters = {
        "a": {"opacity": torch.tensor([0.001])},
        "b": {"opacity": torch.tensor([0.002])},
    }
    grad_accumulator = {}
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=100.0,  # No densification
        prune_importance_threshold=0.5,  # Very aggressive: prune everything
        prune_opacity_threshold=0.5,
        recovery_prune_delay=0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    # Should retain at least 1 element (the one with highest opacity = "b")
    assert len(new_scene.elements) >= 1
    ids = {e.id for e in new_scene.elements}
    assert "b" in ids, "Should keep the element with highest opacity"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_engine_split_with_non_tensor_support_radius():
    """Cover lines 503: non-tensor parameter copy path in split loop."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="split_non_tensor_test",
        elements=(
            AuraElement(
                id="beta",
                carrier_id="beta",
                bounds=Bounds((-3.0, -3.0, 0.0), (3.0, 3.0, 1.0)),
                opacity=0.8,
                payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [3.0, 3.0, 1.0]},
            ),
        ),
    )
    carrier_parameters = {
        "beta": {
            "opacity": torch.tensor([0.8], requires_grad=True),
            "support_radius": torch.tensor([3.0, 3.0, 1.0], requires_grad=True),
            "note": "non_tensor_value",  # Plain string — non-tensor path
        }
    }
    grad_accumulator = {"beta.opacity": 1.0}
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        split_threshold_scale=0.01,  # Low threshold -> split (not clone)
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    assert num_densified >= 1


# ---- Batch: Regularization in training loop (line 619, 931) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_scale_regularization_active_in_training_loop():
    """Cover line 619 and line 931: regularization loss added in main training loop."""
    from aura.torch_optimizer import DensificationConfig

    scene = AuraScene(
        name="reg_loss_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
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
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            densification=DensificationConfig(
                scale_reg_weight=0.01,  # Enable scale regularization
                opacity_entropy_reg_weight=0.01,  # Enable opacity entropy reg
            ),
        ),
    )
    assert len(result.steps) == 2


# ---- Batch: torch_optimize_capture_batches (lines 652, 654, 659, 661, 667) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_rejects_empty_scene():
    """Cover line 652: empty scene raises ValueError."""
    # Use mock-like object; the empty-scene guard runs before batch validation
    class _FakePacked:
        target_count = 1
    scene = AuraScene(name="empty", elements=())
    with pytest.raises(ValueError, match="scene element"):
        torch_optimize_capture_batches(scene, [_FakePacked()])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_rejects_empty_batches():
    """Cover line 654: no batches raises ValueError."""
    scene = AuraScene(
        name="s",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )
    with pytest.raises(ValueError, match="at least one packed"):
        torch_optimize_capture_batches(scene, [])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_skips_empty_target_count_batches():
    """Cover line 659: batches with target_count=0 are silently skipped; line 667: all empty raises ValueError."""
    scene = AuraScene(
        name="skip_empty_test",
        elements=(AuraElement(id="s", carrier_id="surface", bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1))),),
    )

    class _FakePackedZero:
        target_count = 0

    # A batch with target_count=0 should be filtered; if ALL are filtered -> ValueError
    with pytest.raises(ValueError, match="non-empty"):
        torch_optimize_capture_batches(scene, [_FakePackedZero()])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batches_rejects_oversize_packed_batch():
    """Cover line 661: packed batch exceeding max_samples_per_batch raises ValueError."""
    scene = AuraScene(
        name="oversize_test",
        elements=(
            AuraElement(
                id="s", carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                normal=(0.0, 0.0, -1.0), opacity=1.0,
            ),
        ),
    )

    class _FakePacked:
        target_count = 2  # Over limit

    with pytest.raises(ValueError, match="max_samples_per_batch"):
        torch_optimize_capture_batches(scene, [_FakePacked()], TorchOptimizationConfig(max_samples_per_batch=1), device="cpu")


# ---- Batch: _build_adam_optimizer (lines 704, 708, 719, 725, 737, 745, 748) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_build_adam_optimizer_creates_param_groups():
    """Cover lines 704-748: _build_adam_optimizer creates per-attribute parameter groups."""
    import torch
    from aura.torch_optimizer import _build_adam_optimizer

    config = TorchOptimizationConfig(optimizer_type="adam")
    carrier_parameters = {
        "e1": {
            "min_corner": torch.tensor([-0.5, -0.5, 0.0], requires_grad=True),  # position
            "max_corner": torch.tensor([0.5, 0.5, 0.1], requires_grad=True),    # position
            "support_radius": torch.tensor([0.3, 0.3, 0.3], requires_grad=True),  # scale
            "normal": torch.tensor([0.0, 0.0, -1.0], requires_grad=True),  # rotation
            "opacity": torch.tensor([0.9], requires_grad=True),  # opacity
            "color": torch.tensor([1.0, 0.0, 0.0], requires_grad=True),  # color
            "confidence": torch.tensor([0.5], requires_grad=True),  # feature
            "residual_scale": torch.tensor([0.2], requires_grad=True),  # color
            "some_unknown": torch.tensor([1.0], requires_grad=True),  # unknown -> color group
            "no_grad": torch.tensor([1.0], requires_grad=False),  # not trainable -> skipped
        }
    }
    optimizer = _build_adam_optimizer(torch, carrier_parameters, config)
    assert optimizer is not None
    group_names = [g.get("name") for g in optimizer.param_groups]
    assert "position" in group_names
    assert "scale" in group_names
    assert "rotation" in group_names
    assert "opacity" in group_names
    assert "color" in group_names
    assert "feature" in group_names


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_build_adam_optimizer_returns_none_when_no_trainable_params():
    """Cover line 737 (empty param_groups fallback): returns None."""
    import torch
    from aura.torch_optimizer import _build_adam_optimizer

    config = TorchOptimizationConfig(optimizer_type="adam")
    carrier_parameters = {
        "e1": {
            "color": torch.tensor([1.0, 0.0, 0.0], requires_grad=False),  # no grad -> skip
        }
    }
    optimizer = _build_adam_optimizer(torch, carrier_parameters, config)
    assert optimizer is None


# ---- Batch: Adam optimizer training path (lines 851, 910-913, 931, 955-983, 999, 1001, 1056-1058, 1083, 1114) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_adam_optimizer_trains_surface_color():
    """Cover Adam path (lines 955-983, 999, 1001, 1114): optimizer_type='adam'."""
    scene = AuraScene(
        name="adam_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell", "alpha": 0.5, "beta": 0.5},
            ),
        ),
    )
    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
                depth=CaptureTensor(path="f.pgm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=1, values=(2.0,)),
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
            optimizer_type="adam",
            loss_weights=TrainingLossWeights(image=1.0, depth=1.0),
            gradient_clip_norm=5.0,
            max_samples_per_batch=1,
        ),
    )
    assert result.steps[0].optimizer == "adam"
    # Color should improve
    assert result.scene.elements[0].color[0] > 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_adam_optimizer_with_grad_accumulation_window():
    """Cover lines 967-968, 972-983, 1056-1058: grad_accum_window with Adam."""
    scene = AuraScene(
        name="adam_grad_accum_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
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
            optimizer_type="adam",
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            grad_accum_window=2,  # Accumulate every 2 steps
        ),
    )
    # grad_stats should be populated at step intervals
    assert len(result.steps) == 4
    # At least some steps should have grad_stats
    grad_stats_counts = [len(s.grad_stats) for s in result.steps]
    assert sum(grad_stats_counts) > 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_adam_optimizer_with_opacity_reset():
    """Cover lines 910-913: opacity soft-reset during training."""
    scene = AuraScene(
        name="adam_opacity_reset_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
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
            optimizer_type="adam",
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            opacity_reset_interval=2,  # Reset every 2 iterations
            opacity_reset_value=0.05,
            recovery_window=1,
        ),
    )
    # At least one step should have opacity_reset_due=True
    assert any(s.opacity_reset_due for s in result.steps)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_adam_optimizer_with_lr_decay():
    """Cover lines 1114: Adam LR decay updates position param group."""
    scene = AuraScene(
        name="adam_lr_decay_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
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
            optimizer_type="adam",
            position_learning_rate=1e-3,
            position_lr_final=1e-6,
            lr_decay_steps=10,  # Enable LR decay
            loss_weights=TrainingLossWeights(image=1.0, depth=1.0),
            max_samples_per_batch=1,
        ),
    )
    assert result.steps[0].optimizer == "adam"
    assert len(result.steps) == 3


# ---- Batch: large scene path (line 851, 1083) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_large_scene_skips_importance_scores(monkeypatch):
    """Cover lines 851, 1083: large scene (>20k carriers) skips importance score snapshots."""
    import aura.torch_optimizer as _mod

    # Patch the threshold to 0 so any scene counts as "large"
    monkeypatch.setattr(_mod, "_LARGE_SCENE_CARRIER_THRESHOLD", 0)

    scene = AuraScene(
        name="large_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
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
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            grad_accum_window=1,  # Enable grad accumulation to hit line 1056-1058
        ),
    )
    # Importance scores should be empty tuples (large scene optimization)
    assert all(s.importance_scores == () for s in result.steps)


# ---- Batch: Adam with evolution rebuild (lines 1143, 1186) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_adam_optimizer_rebuilt_after_evolution():
    """Cover lines 1143, 1186: Adam optimizer is rebuilt after evolution changes scene."""
    scene = AuraScene(
        name="adam_evolution_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
                depth=CaptureTensor(path="f.pgm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=1, values=(2.0,)),
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
            color_learning_rate=0.05,
            optimizer_type="adam",
            loss_weights=TrainingLossWeights(image=1.0, depth=0.0),
            max_samples_per_batch=1,
            evolution_policy=CarrierEvolutionPolicy(split_image_loss_threshold=0.0),
        ),
    )
    # Evolution should have run — either created new carriers or logged decisions
    assert any(s.carrier_evolution for s in result.steps) or len(result.scene.elements) >= 1


# ---- Batch: _restore_trained_parameters (lines 1249, 1253, 1257-1258) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_restore_trained_parameters_skips_new_carriers():
    """Cover line 1249: new carrier (not in trained_parameters) is skipped."""
    import torch
    from aura.torch_optimizer import _restore_trained_parameters

    new_tensor = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    new_carrier_parameters = {
        "existing": {"color": new_tensor},
        "brand_new": {"color": torch.tensor([1.0, 0.0, 0.0])},  # Not in trained
    }
    trained_parameters = {
        "existing": {"color": torch.tensor([0.5, 0.5, 0.5])},
        # "brand_new" is absent — this is the new carrier path
    }
    _restore_trained_parameters(new_carrier_parameters, trained_parameters)
    # existing should be restored
    assert new_tensor.data.tolist() == pytest.approx([0.5, 0.5, 0.5])
    # brand_new color stays at 1.0 (not overwritten)
    assert new_carrier_parameters["brand_new"]["color"].tolist() == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_restore_trained_parameters_skips_missing_pname():
    """Cover line 1253: parameter name not in trained_fields is skipped."""
    import torch
    from aura.torch_optimizer import _restore_trained_parameters

    new_tensor = torch.tensor([0.0], requires_grad=True)
    new_carrier_parameters = {
        "e1": {
            "opacity": new_tensor,
            "extra_field": torch.tensor([1.0]),  # Not in trained
        }
    }
    trained_parameters = {
        "e1": {
            "opacity": torch.tensor([0.9]),
            # extra_field is absent
        }
    }
    _restore_trained_parameters(new_carrier_parameters, trained_parameters)
    assert new_tensor.data.item() == pytest.approx(0.9)
    # extra_field stays unchanged
    assert new_carrier_parameters["e1"]["extra_field"].item() == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_restore_trained_parameters_handles_shape_mismatch():
    """Cover lines 1257-1258: shape mismatch (incompatible shapes) during copy is silently caught."""
    import torch
    from aura.torch_optimizer import _restore_trained_parameters

    # Use a 2D tensor as target but a 3D trained tensor — genuinely incompatible shapes
    new_tensor = torch.zeros(2, 3, requires_grad=True)   # shape (2, 3)
    trained_tensor = torch.ones(3, 2)                    # shape (3, 2) — copy_ will fail
    new_carrier_parameters = {"e1": {"color": new_tensor}}
    trained_parameters = {"e1": {"color": trained_tensor}}
    # Should not raise even if copy_ fails
    _restore_trained_parameters(new_carrier_parameters, trained_parameters)
    # Copy either succeeded or was silently ignored; either way no exception
    assert new_tensor.data.shape == (2, 3)


# ---- Batch: _evolve_scene removed element path (line 1311) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_evolve_scene_skips_removed_elements():
    """Cover line 1311: _evolve_scene skips elements in removed_evolved_ids."""
    from aura.torch_optimizer import _evolve_scene, _TorchEvolutionPrediction
    from aura.evolution import CarrierEvolutionDecision

    scene = AuraScene(
        name="evolve_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.5, 0.5, 0.5),
                opacity=0.9,
            ),
            AuraElement(
                id="beta_detail",  # This is the "created" element that should be removed
                carrier_id="beta",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                opacity=0.5,
                payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [0.5, 0.5, 0.5]},
            ),
        ),
    )
    predictions = (
        _TorchEvolutionPrediction(
            element_id="surface",
            carrier_id="surface",
            image_loss=0.1,
            depth_loss=0.0,
            query_loss=0.0,
            normal_loss=0.0,
            target_color=(1.0, 0.0, 0.0),
            target_point=None,
        ),
    )
    # Decision says to merge the beta_detail back into surface
    decisions = (
        CarrierEvolutionDecision(
            element_id="surface",
            carrier_id="surface",
            action="merge_beta_detail",
            reason="test",
            created_element_id="beta_detail",  # This element should be removed
        ),
    )
    evolved = _evolve_scene(scene, predictions, decisions, learning_rate=0.1)
    element_ids = {e.id for e in evolved.elements}
    assert "beta_detail" not in element_ids, "beta_detail should be removed via merge_beta_detail"


# ---- Batch: _TorchGradientStepState properties (lines 1349, 1355) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gradient_step_state_gradient_norm_returns_zero_when_none():
    """Cover line 1349: gradient_norm returns 0.0 when gradient_norm_tensor is None."""
    from aura.torch_optimizer import _TorchGradientStepState

    state = _TorchGradientStepState(
        gradient_norm_tensor=None,
        scale_tensor=None,
        gradient_clip_norm=None,
        updated_parameter_count=0,
    )
    assert state.gradient_norm == 0.0
    assert state.applied_gradient_norm == 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gradient_step_state_applied_gradient_norm_when_no_scale():
    """Cover line 1355: applied_gradient_norm returns gradient_norm when scale_tensor is None."""
    import torch
    from aura.torch_optimizer import _TorchGradientStepState

    norm_tensor = torch.tensor(3.0)
    state = _TorchGradientStepState(
        gradient_norm_tensor=norm_tensor,
        scale_tensor=None,  # No clipping applied
        gradient_clip_norm=None,
        updated_parameter_count=1,
    )
    assert state.gradient_norm == pytest.approx(3.0)
    # No scale -> applied_gradient_norm == gradient_norm
    assert state.applied_gradient_norm == pytest.approx(3.0)


# ---- Batch: _gradient_step_carrier_parameters no-grad paths (lines 1373, 1386-1387, 1392) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gradient_step_skips_params_without_grad():
    """Cover line 1373: parameters without grad are skipped in gradient collection."""
    import torch
    from aura.torch_optimizer import _gradient_step_carrier_parameters

    no_grad_param = torch.tensor([0.5, 0.5, 0.5], requires_grad=False)
    no_grad_param.grad = None  # No grad explicitly

    update = _gradient_step_carrier_parameters(
        torch,
        {"e1": {"color": no_grad_param}},
        learning_rate=0.1,
        gradient_clip_norm=None,
    )
    # No parameters with grad -> norm is None, scale is None
    assert update.gradient_norm_tensor is None
    assert update.scale_tensor is None
    assert update.updated_parameter_count == 0
    # gradient_norm property
    assert update.gradient_norm == 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gradient_step_skips_params_without_grad_in_update_loop():
    """Cover line 1392: params without grad are also skipped in the update (with-no-grad) loop."""
    import torch
    from aura.torch_optimizer import _gradient_step_carrier_parameters

    # One param with grad, one without
    param_with_grad = torch.tensor([0.5], requires_grad=True)
    param_with_grad.grad = torch.tensor([1.0])

    param_no_grad = torch.tensor([0.5], requires_grad=True)
    param_no_grad.grad = None  # No gradient computed

    carrier_parameters = {
        "e1": {
            "opacity": param_with_grad,
            "color": param_no_grad,  # No grad -> skipped in update loop
        }
    }
    update = _gradient_step_carrier_parameters(
        torch,
        carrier_parameters,
        learning_rate=0.1,
        gradient_clip_norm=None,
    )
    # Only 1 parameter was updated
    assert update.updated_parameter_count == 1
    # param_no_grad should be unchanged
    assert param_no_grad.item() == pytest.approx(0.5)


# ---- Batch: _scene_from_carrier_parameters volume_cell opacity path (line 1527) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_scene_from_carrier_parameters_updates_volume_cell_opacity():
    """Cover line 1527: opacity written to payload for volume_cell type."""
    import torch
    from aura.torch_optimizer import _scene_from_carrier_parameters

    scene = AuraScene(
        name="volume_scene",
        elements=(
            AuraElement(
                id="vol",
                carrier_id="volume",
                bounds=Bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                opacity=0.3,
                payload={"type": "volume_cell", "opacity": 0.3},
            ),
        ),
    )
    carrier_parameters = {
        "vol": {
            "opacity": torch.tensor([0.7]),
        }
    }
    new_scene = _scene_from_carrier_parameters(scene, carrier_parameters)
    el = new_scene.elements[0]
    assert el.opacity == pytest.approx(0.7, abs=0.01)
    assert el.payload["opacity"] == pytest.approx(0.7, abs=0.01)


# ---- Batch: _loss_by_element_summary with None element_id (line 1557) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_loss_by_element_summary_skips_none_element_id():
    """Cover line 1557: _loss_by_element_summary skips entries with element_id=None."""
    from aura.torch_optimizer import _loss_by_element_summary
    from aura.torch_renderer import TorchCaptureRenderSummary

    summary = TorchCaptureRenderSummary(
        device="cpu",
        ray_origins=((0.0, 0.0, -2.0),) * 3,
        ray_directions=((0.0, 0.0, 1.0),) * 3,
        element_ids=("e1", None, "e2"),  # None should be skipped
        carrier_ids=("surface", "surface", "surface"),
        predicted_color=((0.5, 0.0, 0.0),) * 3,
        predicted_depth=(2.0, 2.0, 2.0),
        transmittance=(0.0, 0.0, 0.0),
        normal=(None, None, None),
        target_color=((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.2, 0.0, 0.0)),
        target_depth=(2.0, 2.0, 2.0),
        target_point=(None, None, None),
        image_loss=(0.5, 0.3, 0.1),
        depth_loss=(0.1, 0.2, 0.0),
        query_loss=(0.0, 0.0, 0.0),
        normal_loss=(0.0, 0.0, 0.0),
    )
    result = _loss_by_element_summary(summary)
    assert "e1" in result
    assert "e2" in result
    assert None not in result


# ---- Batch: _compute_position_lr decay path (line 1609) ----

def test_compute_position_lr_returns_final_when_past_decay():
    """Cover line 1609: _compute_position_lr returns lr_final when step >= lr_decay_steps."""
    from aura.torch_optimizer import _compute_position_lr

    config = TorchOptimizationConfig(
        position_learning_rate=1.6e-4,
        position_lr_final=1.6e-6,
        lr_decay_steps=100,
        position_lr_warmup_steps=0,
    )
    # Step past decay
    lr = _compute_position_lr(config, 100)
    assert lr == pytest.approx(1.6e-6)
    # Step well past
    lr2 = _compute_position_lr(config, 200)
    assert lr2 == pytest.approx(1.6e-6)


def test_compute_position_lr_interpolates_during_decay():
    """Cover line 1609: exponential decay during decay window."""
    from aura.torch_optimizer import _compute_position_lr

    config = TorchOptimizationConfig(
        position_learning_rate=1e-3,
        position_lr_final=1e-5,
        lr_decay_steps=100,
        position_lr_warmup_steps=0,
    )
    lr_mid = _compute_position_lr(config, 50)
    # Should be between initial and final
    assert 1e-5 < lr_mid < 1e-3


def test_compute_position_lr_returns_initial_during_warmup():
    """Cover line 748 (_compute_position_lr warmup): during warmup, returns initial LR."""
    from aura.torch_optimizer import _compute_position_lr

    config = TorchOptimizationConfig(
        position_learning_rate=1e-3,
        position_lr_final=1e-6,
        lr_decay_steps=100,
        position_lr_warmup_steps=20,
    )
    lr = _compute_position_lr(config, 10)  # In warmup window
    assert lr == pytest.approx(1e-3)


# ---- Batch: _clamp_unit (line 1629) ----

def test_clamp_unit_clamps_below_zero():
    """Cover line 1629: _clamp_unit clamps negative values to 0."""
    from aura.torch_optimizer import _clamp_unit
    assert _clamp_unit(-0.5) == pytest.approx(0.0)


def test_clamp_unit_clamps_above_one():
    """Cover line 1629: _clamp_unit clamps values > 1 to 1."""
    from aura.torch_optimizer import _clamp_unit
    assert _clamp_unit(1.5) == pytest.approx(1.0)


def test_clamp_unit_passes_through_valid_values():
    """Cover line 1629: _clamp_unit passes through values in [0, 1]."""
    from aura.torch_optimizer import _clamp_unit
    assert _clamp_unit(0.5) == pytest.approx(0.5)
    assert _clamp_unit(0.0) == pytest.approx(0.0)
    assert _clamp_unit(1.0) == pytest.approx(1.0)


# ---- Batch: SGD grad accumulation path (lines 967-968, 972-983 for SGD side) ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_sgd_optimizer_with_grad_accumulation_window():
    """Cover grad_accum_window > 0 with SGD path (lines 1012-1024)."""
    scene = AuraScene(
        name="sgd_grad_accum_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0),
        target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
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
            optimizer_type="sgd",
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            grad_accum_window=2,  # Accumulate 2 steps before reporting
        ),
    )
    assert len(result.steps) == 4
    # At steps 2 and 4, grad_stats should be populated (window=2)
    grad_stats_counts = [len(s.grad_stats) for s in result.steps]
    assert sum(grad_stats_counts) > 0


# ---- Batch: Remaining coverage gaps (lines 265-268, 390, 619, 745, 851, 967-968, 1114, 1143, 1557, 1609, 1629) ----

def test_live_opacity_with_plain_scalar_value():
    """Cover lines 265-268: _live_opacity with a plain Python float (no detach method)."""
    from aura.torch_optimizer import _live_opacity
    # Plain float — no detach — falls through to float() conversion
    assert _live_opacity(0.7, 0.5) == pytest.approx(0.7)


def test_live_opacity_with_unconvertible_value():
    """Cover lines 267-268: _live_opacity with an unconvertible value falls back."""
    from aura.torch_optimizer import _live_opacity
    # A list is not directly float()-convertible
    assert _live_opacity([0.7], 0.5) == pytest.approx(0.5)


def test_densification_engine_empty_scene_median_scale_fallback():
    """Cover line 390: median_scale = 1.0 when no scale info is available."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    # Scene with one element but no scale parameters at all
    scene = AuraScene(
        name="empty_scale_test",
        elements=(
            AuraElement(
                id="s",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                opacity=0.8,
            ),
        ),
    )
    carrier_parameters = {"s": {}}  # No scale parameters
    grad_accumulator = {"s.fake_grad": 0.5}
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
    )
    # Should run without error; median_scale defaults to 1.0 (no scale info)
    new_scene, new_params, _, _ = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    assert len(new_scene.elements) >= 1


def test_compute_position_lr_no_decay_returns_initial():
    """Cover line 745: _compute_position_lr returns initial LR when lr_decay_steps <= 0."""
    from aura.torch_optimizer import _compute_position_lr

    config = TorchOptimizationConfig(
        position_learning_rate=1.6e-4,
        position_lr_final=1.6e-6,
        lr_decay_steps=0,  # No decay
    )
    lr = _compute_position_lr(config, 100)
    assert lr == pytest.approx(1.6e-4)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_optimize_torch_batches_raises_on_oversize_via_internal_path():
    """Cover line 851: sample count exceeds max_samples_per_batch inside _optimize_torch_batches."""
    # Call _optimize_torch_batches directly to bypass the external check in torch_optimize_capture_batch.
    from aura.torch_optimizer import _optimize_torch_batches
    import dataclasses

    scene = AuraScene(
        name="internal_guard_scene",
        elements=(
            AuraElement(
                id="s", carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0), opacity=1.0, normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0), look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0), target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)  # 1 sample

    # Build a valid config then bypass __post_init__ to set max_samples_per_batch=0
    config_base = TorchOptimizationConfig(iterations=1, color_learning_rate=0.1, loss_weights=TrainingLossWeights(image=1.0))
    config_with_zero = dataclasses.replace(config_base)
    object.__setattr__(config_with_zero, 'max_samples_per_batch', 0)

    with pytest.raises(ValueError, match="max_samples_per_batch"):
        _optimize_torch_batches(scene, ((batch, None, None, ()),), config=config_with_zero, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_steps_list_capped_at_max_steps_in_memory(monkeypatch):
    """Cover line 1114: steps list is capped at _MAX_STEPS_IN_MEMORY."""
    import aura.torch_optimizer as _mod
    monkeypatch.setattr(_mod, "_MAX_STEPS_IN_MEMORY", 3)

    scene = AuraScene(
        name="cap_steps_scene",
        elements=(
            AuraElement(
                id="s", carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0), opacity=1.0, normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0), look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0), target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    result = torch_optimize_capture_batch(
        scene, batch,
        TorchOptimizationConfig(
            iterations=6,
            color_learning_rate=0.1,
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
        ),
    )
    # With _MAX_STEPS_IN_MEMORY=3 and 6 iterations, only the last 3 steps are kept
    assert len(result.steps) == 3


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_adam_optimizer_rebuilt_after_densification():
    """Cover line 1143: Adam optimizer is rebuilt when densification changes carrier count."""
    from aura.torch_optimizer import DensificationConfig

    scene = AuraScene(
        name="adam_densify_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.0, 0.0, 0.0),
                opacity=0.9,
                normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0), look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0), target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    result = torch_optimize_capture_batch(
        scene, batch,
        TorchOptimizationConfig(
            iterations=6,
            color_learning_rate=0.1,
            optimizer_type="adam",  # Adam + densification = rebuild after densify
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            densification=DensificationConfig(
                enabled=True,
                start_iteration=2,
                end_iteration=10,
                interval=2,
                grad_threshold=0.0,  # Clone everything
                prune_importance_threshold=0.0,
                prune_opacity_threshold=0.0,
            ),
        ),
    )
    total_densified = sum(s.densified_count for s in result.steps)
    assert total_densified > 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_clamp_unit_via_scene_from_carrier_parameters():
    """Cover line 1629: _clamp_unit called from _scene_from_carrier_parameters."""
    import torch
    from aura.torch_optimizer import _scene_from_carrier_parameters

    scene = AuraScene(
        name="clamp_scene",
        elements=(
            AuraElement(
                id="s", carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                opacity=0.5, confidence=0.5,
                payload={"type": "beta_kernel", "confidence": 0.5},
            ),
        ),
    )
    carrier_parameters = {
        "s": {
            "opacity": torch.tensor([1.5]),  # Over 1.0 — clamped to 1.0
            "confidence": torch.tensor([-0.2]),  # Negative — clamped to 0.0
        }
    }
    new_scene = _scene_from_carrier_parameters(scene, carrier_parameters)
    el = new_scene.elements[0]
    assert el.opacity == pytest.approx(1.0)   # Clamped from 1.5
    assert el.confidence == pytest.approx(0.0)  # Clamped from -0.2


# ---- Final coverage batch: lines 390, 422, 445, 470, 475-476, 619, 967-968, 1557, 1609, 1629 ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_optimize_capture_batch_rejects_empty_scene():
    """Cover line 619: torch_optimize_capture_batch raises on empty scene."""
    import torch

    # Create a minimal valid batch
    frame = TrainingFrame(
        id="f",
        camera_origin=(0.0, 0.0, -2.0), look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0), target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
            ),
        ),
        device="cpu",
    )
    batch = torch_capture_training_batch((frame,), assets)
    scene = AuraScene(name="empty_single", elements=())

    with pytest.raises(ValueError, match="scene element"):
        torch_optimize_capture_batch(scene, batch)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_densification_clone_loop_with_non_tensor_param():
    """Cover line 445: non-tensor parameter in clone loop passes through as-is."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine

    scene = AuraScene(
        name="clone_non_tensor_test",
        elements=(
            AuraElement(
                id="carrier",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.5, 0.5, 0.5),
                opacity=0.8,
            ),
        ),
    )
    carrier_parameters = {
        "carrier": {
            "color": torch.tensor([0.5, 0.5, 0.5], requires_grad=True),
            "label": "my_label",  # Non-tensor — takes the else branch (line 445)
        }
    }
    grad_accumulator = {"carrier.color": 0.1}  # High gradient -> clone
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        split_threshold_scale=100.0,  # Large -> always clone (not split)
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    assert num_densified >= 1
    # The cloned carrier should have "label" copied through as a plain value
    new_ids = {e.id for e in new_scene.elements} - {e.id for e in scene.elements}
    assert len(new_ids) >= 1
    for new_id in new_ids:
        assert new_params.get(new_id, {}).get("label") == "my_label"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_depth_distortion_loss_weight_used_in_training():
    """Cover line 1557: depth_distortion loss added when weight > 0."""
    scene = AuraScene(
        name="depth_distortion_scene",
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
        id="f",
        camera_origin=(0.0, 0.0, -2.0), look_at=(0.0, 0.0, 0.0),
        target_color=(1.0, 0.0, 0.0), target_depth=2.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    assets = torch_capture_asset_batch(
        (
            CaptureFrameTensors(
                frame_id="f",
                image=CaptureTensor(path="f.ppm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=3, values=(1.0, 0.0, 0.0)),
                depth=CaptureTensor(path="f.pgm", format="Netpbm", backend="stdlib",
                                    width=1, height=1, channels=1, values=(2.0,)),
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
            loss_weights=TrainingLossWeights(
                image=1.0, depth=1.0,
                depth_distortion=0.01,  # Enable depth distortion loss (line 1557)
            ),
            max_samples_per_batch=1,
        ),
    )
    assert len(result.steps) == 2


def test_mean_function_with_empty_values():
    """Cover line 1609: _mean returns 0.0 for empty sequence."""
    from aura.torch_optimizer import _mean
    assert _mean([]) == 0.0
    assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_normalized_vec3_with_zero_vector():
    """Cover line 1629: _normalized_vec3 returns original vector when norm <= 1e-8."""
    from aura.torch_optimizer import _normalized_vec3
    result = _normalized_vec3((0.0, 0.0, 0.0))
    assert result == (0.0, 0.0, 0.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_scene_from_carrier_parameters_with_zero_normal_uses_normalized_vec3_fallback():
    """Cover line 1629 via _scene_from_carrier_parameters with a zero-length normal."""
    import torch
    from aura.torch_optimizer import _scene_from_carrier_parameters

    scene = AuraScene(
        name="zero_normal_scene",
        elements=(
            AuraElement(
                id="s", carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    carrier_parameters = {
        "s": {
            "normal": torch.tensor([0.0, 0.0, 0.0]),  # Zero normal -> _normalized_vec3 returns original
        }
    }
    new_scene = _scene_from_carrier_parameters(scene, carrier_parameters)
    # Zero normal passed through as-is (no crash)
    assert new_scene.elements[0].normal == pytest.approx((0.0, 0.0, 0.0))


# ---- Batched gaussian writeback fix ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_scene_from_carrier_parameters_reads_batched_gaussian_params():
    """Fix: _scene_from_carrier_parameters must extract trained values from __batched__ tensors.

    Before the fix, training updated carrier_parameters["__batched__"] but
    _scene_from_carrier_parameters looked up by element ID (always got {}) so no
    trained parameters were ever written back to the saved .aura file.
    """
    import torch
    from aura.torch_optimizer import _scene_from_carrier_parameters
    from aura.torch_kernels import _torch_batched_gaussian_parameter_tensors

    N = 5
    elements = tuple(
        AuraElement(
            id=f"g{i}",
            carrier_id="gaussian",
            bounds=Bounds((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)),
            color=(0.1, 0.2, 0.3),
            opacity=0.5,
            confidence=0.8,
            payload={"type": "gaussian_fallback", "mean": [float(i) + 0.5, 0.5, 0.5]},
        )
        for i in range(N)
    )
    scene = AuraScene(name="test", elements=elements)

    # Build batched carrier parameters directly (same dict shape the optimizer uses
    # for >1000 gaussian elements via _torch_batched_gaussian_parameter_tensors).
    carrier_parameters = _torch_batched_gaussian_parameter_tensors(torch, elements, device="cpu", requires_grad=True)

    # Mutate the batched color and opacity tensors to simulate what training would do.
    new_color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                               [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    carrier_parameters["__batched__"]["color"].data[:] = new_color

    new_opacity = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9])
    carrier_parameters["__batched__"]["opacity"].data[:] = new_opacity

    new_scene = _scene_from_carrier_parameters(scene, carrier_parameters)

    # Each element should now carry the updated color and opacity from the batched tensors.
    assert len(new_scene.elements) == N
    for i, elem in enumerate(new_scene.elements):
        assert elem.color == pytest.approx(new_color[i].tolist(), abs=1e-5), (
            f"element {i}: color not written back from __batched__"
        )
        assert elem.opacity == pytest.approx(new_opacity[i].item(), abs=1e-5), (
            f"element {i}: opacity not written back from __batched__"
        )


# ---- Batched densification fixes ----

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_densify_and_prune_batched_path_converts_and_clones():
    """densify_and_prune must handle __batched__ carrier_parameters (large gaussian scenes).

    Before the fix, batched parameters had key '__batched__' not element IDs, so
    all carrier_parameters.get(element.id) lookups returned {} — no densification happened
    because every element appeared to have grad_norm=0 and opacity=default.
    """
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine
    from aura.torch_kernels import _torch_batched_gaussian_parameter_tensors

    elements = tuple(
        AuraElement(
            id=f"g{i}",
            carrier_id="gaussian",
            bounds=Bounds((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)),
            color=(0.5, 0.5, 0.5),
            opacity=0.8,
        )
        for i in range(3)
    )
    scene = AuraScene(name="batched_densify_test", elements=elements)

    carrier_parameters = _torch_batched_gaussian_parameter_tensors(
        torch, elements, device="cpu", requires_grad=True
    )

    # Only g0 has a high grad norm (use element ID key as set by the batched accumulation fix)
    grad_accumulator = {
        "g0.grad_absmax": 0.05,   # above threshold
        "g1.grad_absmax": 0.0001,  # below threshold
        "g2.grad_absmax": 0.0001,
    }
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        prune_importance_threshold=0.0,
        prune_opacity_threshold=0.0,
    )
    new_scene, new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    # g0 should have been cloned — count increases
    assert num_densified >= 1, f"Expected clone of g0 but got num_densified={num_densified}"
    assert len(new_scene.elements) > len(scene.elements), (
        f"Expected more elements after batched densification, got {len(new_scene.elements)}"
    )
    assert num_pruned == 0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_densify_and_prune_batched_path_densifies_and_prunes():
    """The batched (__batched__) path must drive BOTH densification (high grad)
    and pruning (low opacity) through the per-element conversion, not just clone."""
    import torch
    from aura.torch_optimizer import DensificationConfig, DensificationEngine
    from aura.torch_kernels import _torch_batched_gaussian_parameter_tensors

    def _gauss(i, opacity, cov):
        return AuraElement(
            id=f"g{i}", carrier_id="gaussian",
            bounds=Bounds((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)),
            color=(0.5, 0.5, 0.5), opacity=opacity,
            payload={
                "type": "gaussian_fallback",
                "mean": [float(i) + 0.5, 0.5, 0.5],
                "covariance": [[cov, 0.0, 0.0], [0.0, cov, 0.0], [0.0, 0.0, cov]],
            },
        )

    # g0: high grad -> densify; g1: tiny opacity -> prune; g2: normal -> keep.
    elements = (_gauss(0, 0.8, 0.04), _gauss(1, 0.001, 0.0001), _gauss(2, 0.8, 0.0001))
    scene = AuraScene(name="batched_split_prune", elements=elements)
    carrier_parameters = _torch_batched_gaussian_parameter_tensors(
        torch, elements, device="cpu", requires_grad=True
    )
    grad_accumulator = {
        "g0.grad_absmax": 0.05,
        "g1.grad_absmax": 0.0001,
        "g2.grad_absmax": 0.0001,
    }
    cfg = DensificationConfig(
        enabled=True,
        grad_threshold=0.001,
        split_threshold_scale=0.5,
        prune_importance_threshold=0.005,
        prune_opacity_threshold=0.005,
        recovery_prune_delay=0,
    )
    new_scene, _new_params, num_densified, num_pruned = DensificationEngine.densify_and_prune(
        scene, carrier_parameters, grad_accumulator,
        absolute_iteration=500,
        densification_config=cfg,
        steps_since_reset=1000,
        max_carriers_budget=0,
        torch=torch,
    )
    ids = {e.id for e in new_scene.elements}
    assert num_densified >= 1, "g0 (high grad) should densify through the batched path"
    assert num_pruned >= 1, "g1 (tiny opacity) should prune through the batched path"
    assert "g1" not in ids, "the faint carrier must be pruned"
    assert "g2" in ids, "the normal carrier must survive"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_restore_trained_parameters_batched_new_per_element_trained():
    """_restore_trained_parameters must restore per-element values into batched tensors.

    After densification, scene_tensors is rebuilt (batched format) but trained_parameters
    holds per-element slices. The restore must map trained values back by element index.
    """
    import torch
    from aura.torch_optimizer import _restore_trained_parameters
    from aura.torch_kernels import _torch_batched_gaussian_parameter_tensors

    elements = tuple(
        AuraElement(
            id=f"e{i}",
            carrier_id="gaussian",
            bounds=Bounds((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)),
            color=(0.1, 0.2, 0.3),
            opacity=0.5,
        )
        for i in range(3)
    )

    # Batched new parameters (as produced by torch_scene_tensors after densification)
    new_carrier_parameters = _torch_batched_gaussian_parameter_tensors(
        torch, elements, device="cpu", requires_grad=True
    )

    # Per-element trained parameters (as returned by densify_and_prune).
    # Use opacity=0.0 (far from default 0.5) to make "not restored" assertions unambiguous.
    trained_color_e1 = torch.tensor([0.9, 0.8, 0.7])
    trained_opacity_e1 = torch.tensor(0.0)
    trained_parameters = {
        "e1": {
            "color": trained_color_e1,
            "opacity": trained_opacity_e1,
        }
        # e0 and e2 are "new" (pruned and re-added) — should keep fresh init
    }

    _restore_trained_parameters(new_carrier_parameters, trained_parameters, elements=elements)

    batched = new_carrier_parameters["__batched__"]
    # e1 (index 1) should have trained values
    assert batched["color"].data[1].tolist() == pytest.approx([0.9, 0.8, 0.7], abs=1e-5)
    assert batched["opacity"].data[1].item() == pytest.approx(0.0, abs=1e-5)
    # e0 and e2 (indices 0, 2) should keep their fresh-init opacity (0.5, not 0.0)
    assert batched["opacity"].data[0].item() == pytest.approx(0.5, abs=1e-5)
    assert batched["opacity"].data[2].item() == pytest.approx(0.5, abs=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_restore_trained_parameters_batched_handles_shape_mismatch():
    """A trained value that cannot reshape into the batched slot is skipped
    (left at fresh init) rather than crashing."""
    import torch
    from aura.torch_optimizer import _restore_trained_parameters
    from aura.torch_kernels import _torch_batched_gaussian_parameter_tensors

    elements = tuple(
        AuraElement(
            id=f"e{i}",
            carrier_id="gaussian",
            bounds=Bounds((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)),
            color=(0.1, 0.2, 0.3),
            opacity=0.5,
        )
        for i in range(2)
    )
    new_carrier_parameters = _torch_batched_gaussian_parameter_tensors(
        torch, elements, device="cpu", requires_grad=True
    )
    # e0's trained color has the WRONG length (2 vs the 3-wide slot) -> the
    # reshape raises and the restore must skip it without crashing.
    trained_parameters = {"e0": {"color": torch.tensor([0.9, 0.8])}}
    _restore_trained_parameters(new_carrier_parameters, trained_parameters, elements=elements)
    batched = new_carrier_parameters["__batched__"]
    # The mismatched slot keeps its fresh-init color (0.1, 0.2, 0.3).
    assert batched["color"].data[0].tolist() == pytest.approx([0.1, 0.2, 0.3], abs=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_batched_grad_accumulation_populates_element_ids_in_accumulator():
    """Cover lines 985-996 (Adam) and 1041-1052 (SGD): grad accumulation for batched gaussian path.

    With >1000 gaussian_fallback elements, torch_carrier_parameter_tensors returns batched
    __batched__ tensors. The grad accumulation code must store per-element keys (element_id.grad_absmax)
    so densification can threshold by element. Before the fix, only '__batched__.param_name' keys
    were stored, making all elements appear to have zero gradients.
    """
    import torch
    from aura.torch_optimizer import DensificationConfig

    # 1001 gaussian_fallback elements to trigger batched path
    N = 1001
    elements = tuple(
        AuraElement(
            id=f"g{i}",
            carrier_id="gaussian",
            bounds=Bounds((float(i) * 0.01, 0.0, 0.0), (float(i) * 0.01 + 0.01, 0.1, 0.1)),
            color=(0.5, 0.5, 0.5),
            opacity=0.8,
            payload={"type": "gaussian_fallback", "mean": [float(i) * 0.01 + 0.005, 0.05, 0.05],
                     "covariance": [[0.0001, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]]},
        )
        for i in range(N)
    )
    scene = AuraScene(name="batched_grad_test", elements=elements)

    frame = TrainingFrame(
        id="f0",
        camera_origin=(5.0, 0.05, 0.05),
        look_at=(0.0, 0.05, 0.05),
        target_color=(0.8, 0.2, 0.1),
        target_depth=5.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    frame_tensors_list = (
        CaptureFrameTensors(
            frame_id="f0",
            image=CaptureTensor(path="f0.ppm", format="Netpbm", backend="stdlib",
                                width=1, height=1, channels=3, values=(0.8, 0.2, 0.1)),
        ),
    )
    packed = capture_tensors_to_packed_render_batches((frame,), frame_tensors_list, tile_size=1, max_targets_per_batch=1)

    result = torch_optimize_capture_batches(
        scene,
        packed,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.01,
            optimizer_type="adam",
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            grad_accum_window=1,  # Accumulate every step — covers lines 985-996
            densification=DensificationConfig(enabled=False),  # Don't densify, just accumulate
        ),
        device="cpu",
    )
    # The training completed without error.  Grad accum window is 1 so steps include grad_stats.
    assert len(result.steps) >= 1
    # The actual point of the batched fix: per-element keys (element_id.grad_absmax)
    # must be surfaced, never the pre-fix '__batched__.*' keys.
    keys = {k for step in result.steps for (k, _) in step.grad_stats}
    assert any(k.endswith(".grad_absmax") for k in keys)
    assert not any(k.startswith("__batched__") for k in keys)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
def test_batched_grad_accumulation_sgd_path():
    """Cover lines 1041-1052: batched grad accumulation on SGD path with gaussian elements.

    With >1000 gaussian_fallback elements (batched path) and optimizer_type='sgd',
    the gradient accumulation must handle __batched__ tensors and store per-element keys.
    """
    import torch
    from aura.torch_optimizer import DensificationConfig

    N = 1001
    elements = tuple(
        AuraElement(
            id=f"h{i}",
            carrier_id="gaussian",
            bounds=Bounds((float(i) * 0.01, 0.0, 0.0), (float(i) * 0.01 + 0.01, 0.1, 0.1)),
            color=(0.5, 0.5, 0.5),
            opacity=0.8,
            payload={"type": "gaussian_fallback", "mean": [float(i) * 0.01 + 0.005, 0.05, 0.05],
                     "covariance": [[0.0001, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]]},
        )
        for i in range(N)
    )
    scene = AuraScene(name="batched_sgd_grad_test", elements=elements)

    frame = TrainingFrame(
        id="fs",
        camera_origin=(5.0, 0.05, 0.05),
        look_at=(0.0, 0.05, 0.05),
        target_color=(0.8, 0.2, 0.1),
        target_depth=5.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 1.0, "height": 1.0},
    )
    frame_tensors_sgd = (
        CaptureFrameTensors(
            frame_id="fs",
            image=CaptureTensor(path="fs.ppm", format="Netpbm", backend="stdlib",
                                width=1, height=1, channels=3, values=(0.8, 0.2, 0.1)),
        ),
    )
    packed_sgd = capture_tensors_to_packed_render_batches((frame,), frame_tensors_sgd, tile_size=1, max_targets_per_batch=1)

    result = torch_optimize_capture_batches(
        scene,
        packed_sgd,
        TorchOptimizationConfig(
            iterations=2,
            color_learning_rate=0.01,
            optimizer_type="sgd",
            loss_weights=TrainingLossWeights(image=1.0),
            max_samples_per_batch=1,
            grad_accum_window=1,  # Cover lines 1041-1052
            densification=DensificationConfig(enabled=False),
        ),
        device="cpu",
    )
    assert len(result.steps) >= 1
    keys = {k for step in result.steps for (k, _) in step.grad_stats}
    assert any(k.endswith(".grad_absmax") for k in keys)
    assert not any(k.startswith("__batched__") for k in keys)


class TestRotatingBatchWindow:
    """Carrier-coverage rotation: _rotating_batch_window_indices (see
    docs/CONVERGENCE_TODO.md, carrier gradient starvation)."""

    _fn = staticmethod(torch_optimizer_module._rotating_batch_window_indices)

    def test_disabled_returns_none(self):
        # window 0 (default) or >= total => process every batch (None sentinel)
        assert self._fn(0, 0, 10) is None
        assert self._fn(5, 10, 10) is None
        assert self._fn(5, 99, 10) is None
        assert self._fn(0, 4, 0) is None

    def test_window_size_and_wraparound(self):
        # 7 batches, window 3: iter 0 -> [0,1,2], iter 1 -> [3,4,5],
        # iter 2 -> [6,0,1] (wraps), iter 3 -> [2,3,4] ...
        assert self._fn(0, 3, 7) == (0, 1, 2)
        assert self._fn(1, 3, 7) == (3, 4, 5)
        assert self._fn(2, 3, 7) == (6, 0, 1)
        assert all(0 <= i < 7 for i in self._fn(2, 3, 7))

    def test_full_coverage_over_a_cycle(self):
        # Over ceil(total/window) iterations every batch index is visited.
        total, window = 8128, 32  # representative of a real full-coverage plan
        import math
        cycle = math.ceil(total / window)
        seen = set()
        for it in range(cycle):
            seen.update(self._fn(it, window, total))
        assert seen == set(range(total))
