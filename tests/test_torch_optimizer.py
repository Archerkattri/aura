import importlib.util

import pytest

from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    CaptureFrameTensors,
    CaptureTensor,
    TorchOptimizationConfig,
    TrainingFrame,
    torch_capture_asset_batch,
    torch_capture_training_batch,
    torch_optimize_capture_batch,
)


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
        TorchOptimizationConfig(iterations=2, color_learning_rate=0.5),
    )

    assert result.steps[0].sample_count == 1
    assert result.steps[0].device == "cpu"
    assert result.steps[0].carrier_counts == {"surface": 1}
    assert result.steps[0].image_loss > result.steps[1].image_loss
    assert result.steps[0].normal_loss == pytest.approx(0.0)
    assert result.scene.elements[0].color[0] > scene.elements[0].color[0]
    assert result.scene.elements[0].metadata["optimized_by"] == "aura-core-torch-autograd-reference"
    assert result.to_dict()["finalLoss"] == result.steps[-1].total_loss


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
            "target_normal": None,
            "target_normal_present": None,
        },
    )()
