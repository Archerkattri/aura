"""Comprehensive CLI command tests to improve aura.cli coverage.

All heavy operations (reconstruct, optimize, package, load) are mocked so
these tests run quickly without GPU/filesystem dependencies.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from aura.cli import (
    _add_densification_args,
    _add_reconstruction_config_args,
    _add_render_backend_args,
    _densification_config_from_args,
    _load_resume_training_state,
    _reconstruction_config_from_args,
    _render_package_image,
    _resume_iteration_offset,
    _scene_from_training_dataset,
    _stable_json_hash,
    _training_resume_state,
    _turntable_frame_renderer,
    _validate_resume_training_state,
    _write_training_checkpoints,
    main,
)
from aura.optimize import TrainingLossWeights
from aura.torch_optimizer import DensificationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_result(pkg_dir: pathlib.Path):
    """Build a minimal TorchOptimizationResult mock."""
    step = MagicMock()
    step.carrier_count = 5
    step.densified_count = 0
    step.pruned_count = 0
    step.iteration = 0

    result = MagicMock()
    result.scene = MagicMock()
    result.scene.name = "mock_scene"
    result.to_dict.return_value = {}
    result.steps = [step]
    result.scene_checkpoints = []
    return result


def _make_mock_packaged(pkg_dir: pathlib.Path):
    mock_pkg = MagicMock()
    mock_pkg.write.return_value = pkg_dir
    return mock_pkg


def _make_train_mocks(tmp_path: pathlib.Path):
    """Return a dict of patches for the heavy `train` command dependencies."""
    mock_manifest = MagicMock()
    mock_manifest.to_dict.return_value = {}
    mock_manifest.frames = []

    mock_dataset = MagicMock()
    mock_dataset.frames = []
    mock_dataset.regions = []

    mock_sampling_plan = MagicMock()
    mock_sampling_plan.max_targets_per_batch = 10
    mock_sampling_plan.to_dict.return_value = {}

    pkg_dir = tmp_path / "scene.aura"
    pkg_dir.mkdir()

    return {
        "manifest": mock_manifest,
        "dataset": mock_dataset,
        "sampling_plan": mock_sampling_plan,
        "result": _make_mock_result(pkg_dir),
        "packaged": _make_mock_packaged(pkg_dir),
        "pkg_dir": pkg_dir,
        "real_loss_weights": TrainingLossWeights(),
    }


# ---------------------------------------------------------------------------
# write-native-demo-package / build-native-demo
# ---------------------------------------------------------------------------

class TestWriteNativeDemoPackage:
    def test_write_native_demo_package(self, tmp_path):
        mock_scene = MagicMock()
        mock_packaged = MagicMock()
        mock_packaged.write.return_value = tmp_path / "out.aura"
        with patch("aura.cli.native_demo_scene", return_value=mock_scene), \
             patch("aura.cli.package_scene", return_value=mock_packaged):
            result = main(["write-native-demo-package", "--output-dir", str(tmp_path)])
        assert result == 0

    def test_build_native_demo(self, tmp_path):
        mock_scene = MagicMock()
        mock_packaged = MagicMock()
        mock_packaged.write.return_value = tmp_path / "out.aura"
        with patch("aura.cli.native_demo_scene", return_value=mock_scene), \
             patch("aura.cli.package_scene", return_value=mock_packaged):
            result = main(["build-native-demo", "--output-dir", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# reconstruct-demo
# ---------------------------------------------------------------------------

class TestReconstructDemo:
    def _make_reconstruct_result(self, pkg_dir: pathlib.Path):
        result = MagicMock()
        result.scene = MagicMock()
        result.report = MagicMock()
        result.report.to_dict.return_value = {}
        mock_packaged = MagicMock()
        mock_packaged.write.return_value = pkg_dir
        return result, mock_packaged

    def test_reconstruct_demo_no_frames(self, tmp_path):
        pkg_dir = tmp_path / "out.aura"
        pkg_dir.mkdir()
        result, mock_packaged = self._make_reconstruct_result(pkg_dir)

        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.reconstruct_demo_scene", return_value=result), \
             patch("aura.cli.package_scene", return_value=mock_packaged):
            ret = main(["reconstruct-demo", "--output-dir", str(tmp_path), "--iterations", "2"])
        assert ret == 0

    def test_reconstruct_demo_with_frames(self, tmp_path):
        pkg_dir = tmp_path / "out.aura"
        pkg_dir.mkdir()
        result, mock_packaged = self._make_reconstruct_result(pkg_dir)

        mock_dataset = MagicMock()
        mock_dataset.frames = []
        mock_dataset.regions = []

        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_training_dataset", return_value=mock_dataset), \
             patch("aura.cli.reconstruct_demo_scene", return_value=result), \
             patch("aura.cli.package_scene", return_value=mock_packaged):
            frames_path = tmp_path / "frames.json"
            frames_path.write_text("{}")
            ret = main([
                "reconstruct-demo",
                "--output-dir", str(tmp_path),
                "--iterations", "2",
                "--frames", str(frames_path),
            ])
        assert ret == 0


# ---------------------------------------------------------------------------
# write-training-frames-demo
# ---------------------------------------------------------------------------

class TestWriteTrainingFramesDemo:
    def test_write_training_frames_demo(self, tmp_path):
        output = tmp_path / "frames.json"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.write_synthetic_training_frames", return_value=output):
            result = main(["write-training-frames-demo", "--output", str(output)])
        assert result == 0


# ---------------------------------------------------------------------------
# write-capture-manifest-template
# ---------------------------------------------------------------------------

class TestWriteCaptureManifestTemplate:
    def test_write_capture_manifest_template(self, tmp_path):
        output = tmp_path / "capture-manifest.json"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.write_capture_manifest_template", return_value=output):
            result = main(["write-capture-manifest-template", "--output", str(output)])
        assert result == 0


# ---------------------------------------------------------------------------
# capture-manifest-to-training
# ---------------------------------------------------------------------------

class TestCaptureManifestToTraining:
    def test_capture_manifest_to_training(self, tmp_path):
        mock_manifest = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.to_dict.return_value = {"frames": [], "regions": []}
        mock_manifest.to_training_dataset.return_value = mock_dataset

        output = tmp_path / "out.json"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=mock_manifest):
            result = main([
                "capture-manifest-to-training",
                "some_manifest.json",
                "--output", str(output),
            ])
        assert result == 0
        assert output.exists()

    def test_capture_manifest_to_training_with_load_assets(self, tmp_path):
        mock_manifest = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.to_dict.return_value = {"frames": [], "regions": []}
        mock_manifest.to_training_dataset.return_value = mock_dataset

        output = tmp_path / "out.json"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=mock_manifest):
            result = main([
                "capture-manifest-to-training",
                "some_manifest.json",
                "--output", str(output),
                "--load-assets",
            ])
        assert result == 0
        mock_manifest.to_training_dataset.assert_called_once_with(load_assets=True)


# ---------------------------------------------------------------------------
# torch-optimize-capture-manifest
# ---------------------------------------------------------------------------

class TestTorchOptimizeCaptureManifest:
    def test_torch_optimize_capture_manifest(self, tmp_path):
        pkg_dir = tmp_path / "out.aura"
        pkg_dir.mkdir()

        mock_manifest = MagicMock()
        mock_manifest.to_dict.return_value = {}

        mock_dataset = MagicMock()
        mock_dataset.frames = []
        mock_dataset.regions = []

        mock_sampling_plan = MagicMock()
        mock_sampling_plan.max_targets_per_batch = 10
        mock_sampling_plan.to_dict.return_value = {}

        mock_result = _make_mock_result(pkg_dir)
        real_lw = TrainingLossWeights()

        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=mock_manifest), \
             patch("aura.cli.load_capture_asset_tensors", return_value=[]), \
             patch("aura.cli.capture_tensors_to_training_dataset", return_value=mock_dataset), \
             patch("aura.cli.plan_capture_tensor_sampling", return_value=mock_sampling_plan), \
             patch("aura.cli._scene_from_training_dataset", return_value=MagicMock()), \
             patch("aura.cli.capture_tensors_to_packed_render_batches", return_value=[]), \
             patch("aura.cli.torch_optimize_capture_batches", return_value=mock_result), \
             patch("aura.cli.package_scene", return_value=_make_mock_packaged(pkg_dir)), \
             patch("aura.cli.torch_renderer_status") as mock_trs, \
             patch("aura.cli._loss_weights_from_args", return_value=real_lw):
            mock_trs.return_value.to_dict.return_value = {}
            result = main([
                "torch-optimize-capture-manifest",
                "some_manifest.json",
                "--iterations", "2",
                "--output-dir", str(pkg_dir),
            ])
        assert result == 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTrainCommand:
    def _run_train(self, tmp_path, extra_args=None):
        m = _make_train_mocks(tmp_path)
        extra_args = extra_args or []
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=m["manifest"]), \
             patch("aura.cli.load_capture_asset_tensors", return_value=[]), \
             patch("aura.cli.capture_tensors_to_training_dataset", return_value=m["dataset"]), \
             patch("aura.cli.plan_capture_tensor_sampling", return_value=m["sampling_plan"]), \
             patch("aura.cli._scene_from_training_dataset", return_value=MagicMock()), \
             patch("aura.cli.capture_tensors_to_packed_render_batches", return_value=[]), \
             patch("aura.cli.torch_optimize_capture_batches", return_value=m["result"]), \
             patch("aura.cli.package_scene", return_value=m["packaged"]), \
             patch("aura.cli.torch_renderer_status") as mock_trs, \
             patch("aura.cli._loss_weights_from_args", return_value=m["real_loss_weights"]):
            mock_trs.return_value.to_dict.return_value = {}
            result = main([
                "train", "some_manifest.json",
                "--iterations", "2",
                "--output", str(m["pkg_dir"]),
            ] + extra_args)
        return result

    def test_train_basic(self, tmp_path):
        assert self._run_train(tmp_path) == 0

    def test_train_disable_evolution(self, tmp_path):
        assert self._run_train(tmp_path, ["--disable-evolution"]) == 0

    def test_train_with_densification(self, tmp_path):
        assert self._run_train(tmp_path, ["--densify"]) == 0

    def test_train_with_checkpointing(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        assert self._run_train(tmp_path, [
            "--checkpoint-dir", str(ckpt_dir),
            "--checkpoint-interval", "1",
        ]) == 0

    def test_train_with_resume(self, tmp_path):
        """Test resume path: load previous training state and validate."""
        m = _make_train_mocks(tmp_path)

        # Create a "previous" package dir with a training_report.json that matches
        resume_dir = tmp_path / "prev.aura"
        resume_dir.mkdir()
        # We'll mock load_package and _load_resume_training_state
        compatible_state = {
            "format": "AURA_TRAINING_STATE",
            "resumeCompatibilityFingerprint": "dummy",
            "manifestFingerprint": "a",
            "samplingPlanFingerprint": "b",
            "optimizerConfigFingerprint": "c",
        }

        mock_loaded_pkg = MagicMock()
        mock_loaded_pkg.scene = MagicMock()

        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=m["manifest"]), \
             patch("aura.cli.load_capture_asset_tensors", return_value=[]), \
             patch("aura.cli.capture_tensors_to_training_dataset", return_value=m["dataset"]), \
             patch("aura.cli.plan_capture_tensor_sampling", return_value=m["sampling_plan"]), \
             patch("aura.cli.capture_tensors_to_packed_render_batches", return_value=[]), \
             patch("aura.cli.torch_optimize_capture_batches", return_value=m["result"]), \
             patch("aura.cli.package_scene", return_value=m["packaged"]), \
             patch("aura.cli.torch_renderer_status") as mock_trs, \
             patch("aura.cli._loss_weights_from_args", return_value=m["real_loss_weights"]), \
             patch("aura.cli._training_resume_state", return_value=compatible_state), \
             patch("aura.cli._load_resume_training_state", return_value=compatible_state), \
             patch("aura.cli._validate_resume_training_state"), \
             patch("aura.cli._resume_iteration_offset", return_value=5), \
             patch("aura.cli.load_package", return_value=mock_loaded_pkg):
            mock_trs.return_value.to_dict.return_value = {}
            result = main([
                "train", "some_manifest.json",
                "--iterations", "2",
                "--output", str(m["pkg_dir"]),
                "--resume-from", str(resume_dir),
            ])
        assert result == 0


# ---------------------------------------------------------------------------
# benchmark-plan
# ---------------------------------------------------------------------------

class TestBenchmarkPlan:
    def test_benchmark_plan(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.default_benchmark_suite") as mock_suite:
            mock_suite.return_value.to_dict.return_value = {"plan": "data"}
            result = main(["benchmark-plan"])
        assert result == 0


# ---------------------------------------------------------------------------
# compare-renders
# ---------------------------------------------------------------------------

class TestCompareRenders:
    def test_compare_renders_pass(self, tmp_path):
        ppm1 = tmp_path / "expected.ppm"
        ppm2 = tmp_path / "actual.ppm"
        ppm1.write_bytes(b"")
        ppm2.write_bytes(b"")

        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.read_ppm", side_effect=[MagicMock(), MagicMock()]), \
             patch("aura.cli.compare_images", return_value={"passed": True, "mse": 0.0, "psnr": 99.0}):
            result = main(["compare-renders", str(ppm1), str(ppm2)])
        assert result == 0

    def test_compare_renders_fail(self, tmp_path):
        ppm1 = tmp_path / "expected.ppm"
        ppm2 = tmp_path / "actual.ppm"
        ppm1.write_bytes(b"")
        ppm2.write_bytes(b"")

        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.read_ppm", side_effect=[MagicMock(), MagicMock()]), \
             patch("aura.cli.compare_images", return_value={"passed": False, "mse": 99.0, "psnr": 5.0}):
            result = main(["compare-renders", str(ppm1), str(ppm2)])
        assert result == 1


# ---------------------------------------------------------------------------
# render-package / render
# ---------------------------------------------------------------------------

class TestRenderPackage:
    def _make_package_mock(self):
        mock_pkg = MagicMock()
        mock_pkg.scene = MagicMock()
        mock_img = MagicMock()
        mock_img.write_ppm.return_value = pathlib.Path("outputs/preview.ppm")
        return mock_pkg, mock_img

    def test_render_package_cpu(self, tmp_path):
        mock_pkg, mock_img = self._make_package_mock()
        output = tmp_path / "preview.ppm"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli._render_package_image", return_value=mock_img):
            result = main(["render-package", str(tmp_path), "--output", str(output)])
        assert result == 0

    def test_render_command(self, tmp_path):
        mock_pkg, mock_img = self._make_package_mock()
        output = tmp_path / "render.ppm"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli._render_package_image", return_value=mock_img):
            result = main(["render", str(tmp_path), "--output", str(output)])
        assert result == 0

    def test_render_exr_format(self, tmp_path):
        mock_pkg, mock_img = self._make_package_mock()
        output = tmp_path / "render.exr"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli._render_package_image", return_value=mock_img), \
             patch("aura.cli.write_radiance_image", return_value=output) as mock_wri:
            result = main(["render", str(tmp_path), "--output", str(output), "--format", "exr"])
        assert result == 0
        mock_wri.assert_called_once()


# ---------------------------------------------------------------------------
# render-video
# ---------------------------------------------------------------------------

class TestRenderVideo:
    def test_render_video(self, tmp_path):
        mock_pkg = MagicMock()
        mock_pkg.scene = MagicMock()
        mock_frames = [MagicMock()]
        mock_video_result = MagicMock()
        mock_video_result.to_dict.return_value = {"format": "mp4"}

        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli.render_turntable_frames", return_value=mock_frames), \
             patch("aura.cli.write_video", return_value=mock_video_result):
            result = main(["render-video", str(tmp_path), "--frames", "4", "--output", str(tmp_path / "out.mp4")])
        assert result == 0


# ---------------------------------------------------------------------------
# torch-renderer-status
# ---------------------------------------------------------------------------

class TestTorchRendererStatus:
    def test_torch_renderer_status(self):
        mock_status = MagicMock()
        mock_status.to_dict.return_value = {"available": True}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.torch_renderer_status", return_value=mock_status):
            result = main(["torch-renderer-status"])
        assert result == 0


# ---------------------------------------------------------------------------
# torch-kernel-report
# ---------------------------------------------------------------------------

class TestTorchKernelReport:
    def test_torch_kernel_report(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.torch_carrier_kernel_report", return_value={"status": "ok"}):
            result = main(["torch-kernel-report"])
        assert result == 0


# ---------------------------------------------------------------------------
# cuda-kernel-build-report
# ---------------------------------------------------------------------------

class TestCudaKernelBuildReport:
    def test_cuda_kernel_build_report(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.cuda_kernel_extension_report", return_value={"built": False}):
            result = main(["cuda-kernel-build-report"])
        assert result == 0

    def test_cuda_kernel_build_report_with_build(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.cuda_kernel_extension_report", return_value={"built": True}) as mock_report:
            result = main(["cuda-kernel-build-report", "--build", "--verbose"])
        assert result == 0
        mock_report.assert_called_once_with(build=True, verbose=True)


# ---------------------------------------------------------------------------
# cuda-renderer-report
# ---------------------------------------------------------------------------

class TestCudaRendererReport:
    def test_cuda_renderer_report(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.cuda_renderer_report", return_value={"available": False}):
            result = main(["cuda-renderer-report"])
        assert result == 0


# ---------------------------------------------------------------------------
# benchmark-core
# ---------------------------------------------------------------------------

class TestBenchmarkCore:
    def test_benchmark_core(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.run_core_reconstruction_benchmark", return_value={"score": 1.0}):
            result = main(["benchmark-core", "--iterations", "2"])
        assert result == 0


# ---------------------------------------------------------------------------
# benchmark-reference
# ---------------------------------------------------------------------------

class TestBenchmarkReference:
    def test_benchmark_reference(self, tmp_path):
        mock_pkg = MagicMock()
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli.run_reference_benchmark", return_value={"score": 1.0}):
            result = main(["benchmark-reference", str(tmp_path)])
        assert result == 0

    def test_benchmark_reference_with_ablations(self, tmp_path):
        mock_pkg = MagicMock()
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli.run_ablation_benchmarks", return_value={"score": 1.0}) as mock_ablation:
            result = main(["benchmark-reference", str(tmp_path), "--include-ablations"])
        assert result == 0
        mock_ablation.assert_called_once()


# ---------------------------------------------------------------------------
# benchmark-cuda-runtime
# ---------------------------------------------------------------------------

class TestBenchmarkCudaRuntime:
    def test_benchmark_cuda_runtime_native(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.run_cuda_runtime_benchmark", return_value={"throughput": 0}):
            result = main(["benchmark-cuda-runtime"])
        assert result == 0

    def test_benchmark_cuda_runtime_package(self, tmp_path):
        mock_pkg = MagicMock()
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli.run_cuda_runtime_benchmark", return_value={"throughput": 0}):
            result = main(["benchmark-cuda-runtime", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# benchmark-capture
# ---------------------------------------------------------------------------

class TestBenchmarkCapture:
    def test_benchmark_capture(self, tmp_path):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.run_capture_reconstruction_benchmark", return_value={"passed": True}):
            result = main([
                "benchmark-capture",
                "some_manifest.json",
                "--output-dir", str(tmp_path),
            ])
        assert result == 0


# ---------------------------------------------------------------------------
# benchmark-visual
# ---------------------------------------------------------------------------

class TestBenchmarkVisual:
    def test_benchmark_visual_pass(self, tmp_path):
        ref = tmp_path / "ref.ppm"
        ref.write_bytes(b"")
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.read_ppm", return_value=MagicMock()), \
             patch("aura.cli.run_visual_quality_benchmark", return_value={"passed": True, "psnr": 30.0}):
            result = main(["benchmark-visual", str(tmp_path), str(ref)])
        assert result == 0

    def test_benchmark_visual_fail(self, tmp_path):
        ref = tmp_path / "ref.ppm"
        ref.write_bytes(b"")
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.read_ppm", return_value=MagicMock()), \
             patch("aura.cli.run_visual_quality_benchmark", return_value={"passed": False, "psnr": 5.0}):
            result = main(["benchmark-visual", str(tmp_path), str(ref)])
        assert result == 1


# ---------------------------------------------------------------------------
# benchmark-real-scene
# ---------------------------------------------------------------------------

class TestBenchmarkRealScene:
    def test_benchmark_real_scene_pass(self, tmp_path):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.run_real_scene_benchmark", return_value={"passed": True}):
            result = main(["benchmark-real-scene", str(tmp_path)])
        assert result == 0

    def test_benchmark_real_scene_fail(self, tmp_path):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.run_real_scene_benchmark", return_value={"passed": False}):
            result = main(["benchmark-real-scene", str(tmp_path)])
        assert result == 1


# ---------------------------------------------------------------------------
# production-gate-report
# ---------------------------------------------------------------------------

class TestProductionGateReport:
    def test_production_gate_report(self, tmp_path):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.run_production_gate_report", return_value={"passed": True}):
            result = main(["production-gate-report", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# memory-stability-probe
# ---------------------------------------------------------------------------

class TestMemoryStabilityProbe:
    def test_memory_stability_probe_pass(self, tmp_path):
        mock_report = MagicMock()
        mock_report.stable = True
        mock_report.to_dict.return_value = {"stable": True}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.run_memory_stability_probe", return_value=mock_report):
            result = main(["memory-stability-probe", str(tmp_path), "--iterations", "4"])
        assert result == 0

    def test_memory_stability_probe_fail(self, tmp_path):
        mock_report = MagicMock()
        mock_report.stable = False
        mock_report.to_dict.return_value = {"stable": False}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.run_memory_stability_probe", return_value=mock_report):
            result = main(["memory-stability-probe", str(tmp_path)])
        assert result == 1


# ---------------------------------------------------------------------------
# benchmark-ray-query
# ---------------------------------------------------------------------------

class TestBenchmarkRayQuery:
    def test_benchmark_ray_query(self, tmp_path):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.native_demo_ray_query_expectations", return_value=[]), \
             patch("aura.cli.run_ray_query_correctness_benchmark", return_value={"passed": True}):
            result = main(["benchmark-ray-query", str(tmp_path), "--native-demo-expectations"])
        assert result == 0

    def test_benchmark_ray_query_requires_expectations(self, tmp_path):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()):
            with pytest.raises(ValueError, match="--native-demo-expectations"):
                main(["benchmark-ray-query", str(tmp_path)])


# ---------------------------------------------------------------------------
# ingest-adapters
# ---------------------------------------------------------------------------

class TestIngestAdapters:
    def test_ingest_adapters(self):
        mock_adapter = MagicMock()
        mock_adapter.to_dict.return_value = {"name": "test_adapter"}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.supported_ingest_adapters", return_value=[mock_adapter]):
            result = main(["ingest-adapters"])
        assert result == 0


# ---------------------------------------------------------------------------
# inspect-rays
# ---------------------------------------------------------------------------

class TestInspectRays:
    def test_inspect_rays_default(self, tmp_path):
        mock_inspection = MagicMock()
        mock_inspection.to_dict.return_value = {}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.inspect_scene_rays", return_value=[mock_inspection]):
            result = main(["inspect-rays", str(tmp_path)])
        assert result == 0

    def test_inspect_rays_native_probes(self, tmp_path):
        mock_inspection = MagicMock()
        mock_inspection.to_dict.return_value = {}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=MagicMock()), \
             patch("aura.cli.native_demo_interaction_probes", return_value=[mock_inspection]):
            result = main(["inspect-rays", str(tmp_path), "--native-demo-probes"])
        assert result == 0


# ---------------------------------------------------------------------------
# migration-plan
# ---------------------------------------------------------------------------

class TestMigrationPlan:
    def test_migration_plan(self, tmp_path):
        mock_pkg = MagicMock()
        mock_pkg.asset.version = "1.0.0"
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"status": "ok"}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli.migration_report", return_value=mock_report):
            result = main(["migration-plan", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# Helper function: _stable_json_hash
# ---------------------------------------------------------------------------

class TestStableJsonHash:
    def test_stable_json_hash_deterministic(self):
        h1 = _stable_json_hash({"a": 1, "b": 2})
        h2 = _stable_json_hash({"b": 2, "a": 1})
        assert h1 == h2, "Hash must be key-order-independent"

    def test_stable_json_hash_is_hex_string(self):
        h = _stable_json_hash({"x": "y"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_stable_json_hash_different_payloads(self):
        assert _stable_json_hash({"a": 1}) != _stable_json_hash({"a": 2})


# ---------------------------------------------------------------------------
# Helper function: _load_resume_training_state
# ---------------------------------------------------------------------------

class TestLoadResumeTrainingState:
    def test_load_from_training_report(self, tmp_path):
        state = {"format": "AURA_TRAINING_STATE", "resumeCompatibilityFingerprint": "abc"}
        (tmp_path / "training_report.json").write_text(json.dumps({"trainingState": state}))
        loaded = _load_resume_training_state(tmp_path)
        assert loaded["format"] == "AURA_TRAINING_STATE"

    def test_load_from_checkpoint_json(self, tmp_path):
        state = {"format": "AURA_TRAINING_STATE", "resumeCompatibilityFingerprint": "abc"}
        (tmp_path / "training_checkpoint.json").write_text(json.dumps({"trainingState": state}))
        loaded = _load_resume_training_state(tmp_path)
        assert loaded["format"] == "AURA_TRAINING_STATE"

    def test_raises_when_no_state(self, tmp_path):
        with pytest.raises(ValueError, match="lacks trainingState metadata"):
            _load_resume_training_state(tmp_path)

    def test_raises_on_invalid_json(self, tmp_path):
        (tmp_path / "training_checkpoint.json").write_text("not json!")
        with pytest.raises(ValueError, match="not valid JSON"):
            _load_resume_training_state(tmp_path)

    def test_raises_when_state_missing_in_json(self, tmp_path):
        (tmp_path / "training_report.json").write_text(json.dumps({"noState": True}))
        with pytest.raises(ValueError, match="lacks trainingState metadata"):
            _load_resume_training_state(tmp_path)


# ---------------------------------------------------------------------------
# Helper function: _validate_resume_training_state
# ---------------------------------------------------------------------------

class TestValidateResumeTrainingState:
    def _make_state(self, fp="same"):
        return {
            "format": "AURA_TRAINING_STATE",
            "resumeCompatibilityFingerprint": fp,
            "manifestFingerprint": "mfp",
            "samplingPlanFingerprint": "sfp",
            "optimizerConfigFingerprint": "ofp",
        }

    def test_matching_fingerprints_ok(self, tmp_path):
        s = self._make_state("same")
        _validate_resume_training_state(s, s, tmp_path)  # Should not raise

    def test_mismatched_fingerprint_raises(self, tmp_path):
        expected = self._make_state("fingerprint_a")
        previous = self._make_state("fingerprint_b")
        with pytest.raises(ValueError, match="incompatible"):
            _validate_resume_training_state(expected, previous, tmp_path)

    def test_wrong_format_raises(self, tmp_path):
        expected = self._make_state("same")
        previous = {"format": "UNKNOWN_FORMAT", "resumeCompatibilityFingerprint": "same"}
        with pytest.raises(ValueError, match="unsupported trainingState format"):
            _validate_resume_training_state(expected, previous, tmp_path)

    def test_mismatch_names_specific_field(self, tmp_path):
        expected = {
            "format": "AURA_TRAINING_STATE",
            "resumeCompatibilityFingerprint": "fp_e",
            "manifestFingerprint": "mfp_e",
            "samplingPlanFingerprint": "sfp",
            "optimizerConfigFingerprint": "ofp",
        }
        previous = {
            "format": "AURA_TRAINING_STATE",
            "resumeCompatibilityFingerprint": "fp_p",
            "manifestFingerprint": "mfp_p",
            "samplingPlanFingerprint": "sfp",
            "optimizerConfigFingerprint": "ofp",
        }
        with pytest.raises(ValueError, match="manifestFingerprint"):
            _validate_resume_training_state(expected, previous, tmp_path)


# ---------------------------------------------------------------------------
# Helper function: _resume_iteration_offset
# ---------------------------------------------------------------------------

class TestResumeIterationOffset:
    def test_no_files_returns_zero(self, tmp_path):
        assert _resume_iteration_offset(tmp_path) == 0

    def test_training_report_with_steps(self, tmp_path):
        report = {"steps": [{"iteration": 0}, {"iteration": 1}, {"iteration": 2}]}
        (tmp_path / "training_report.json").write_text(json.dumps(report))
        assert _resume_iteration_offset(tmp_path) == 3

    def test_training_report_empty_steps(self, tmp_path):
        report = {"steps": []}
        (tmp_path / "training_report.json").write_text(json.dumps(report))
        assert _resume_iteration_offset(tmp_path) == 0

    def test_training_report_no_steps_key(self, tmp_path):
        (tmp_path / "training_report.json").write_text(json.dumps({"other": "data"}))
        assert _resume_iteration_offset(tmp_path) == 0

    def test_checkpoint_with_iteration(self, tmp_path):
        (tmp_path / "training_checkpoint.json").write_text(json.dumps({"iteration": 5}))
        assert _resume_iteration_offset(tmp_path) == 6

    def test_invalid_report_json_returns_zero(self, tmp_path):
        (tmp_path / "training_report.json").write_text("invalid json!")
        assert _resume_iteration_offset(tmp_path) == 0

    def test_invalid_checkpoint_json_falls_back(self, tmp_path):
        (tmp_path / "training_checkpoint.json").write_text("invalid json!")
        # checkpoint with invalid JSON -> report dict is empty -> returns 0
        assert _resume_iteration_offset(tmp_path) == 0


# ---------------------------------------------------------------------------
# Helper function: _write_training_checkpoints
# ---------------------------------------------------------------------------

class TestWriteTrainingCheckpoints:
    def test_no_checkpoints_returns_empty(self, tmp_path):
        result = MagicMock()
        result.scene_checkpoints = []
        records = _write_training_checkpoints(result, checkpoint_dir=None, fallback_dir=tmp_path, training_state={})
        assert records == []

    def test_writes_checkpoints_to_fallback_dir(self, tmp_path):
        checkpoint = MagicMock()
        checkpoint.iteration = 3
        checkpoint.scene = MagicMock()
        checkpoint.to_dict.return_value = {"iteration": 3}

        result = MagicMock()
        result.scene_checkpoints = [checkpoint]

        ckpt_output = tmp_path / "iter_000003.aura"
        ckpt_output.mkdir()

        mock_pkg = MagicMock()
        mock_pkg.write.return_value = ckpt_output

        with patch("aura.cli.package_scene", return_value=mock_pkg):
            records = _write_training_checkpoints(
                result,
                checkpoint_dir=None,
                fallback_dir=tmp_path,
                training_state={"key": "value"},
            )
        assert len(records) == 1
        assert records[0]["iteration"] == 3
        assert (ckpt_output / "training_checkpoint.json").exists()

    def test_writes_checkpoints_to_explicit_dir(self, tmp_path):
        ckpt_dir = tmp_path / "explicit_ckpts"
        ckpt_dir.mkdir()

        checkpoint = MagicMock()
        checkpoint.iteration = 7
        checkpoint.scene = MagicMock()
        checkpoint.to_dict.return_value = {"iteration": 7}

        result = MagicMock()
        result.scene_checkpoints = [checkpoint]

        ckpt_output = ckpt_dir / "iter_000007.aura"
        ckpt_output.mkdir()

        mock_pkg = MagicMock()
        mock_pkg.write.return_value = ckpt_output

        with patch("aura.cli.package_scene", return_value=mock_pkg):
            records = _write_training_checkpoints(
                result,
                checkpoint_dir=ckpt_dir,
                fallback_dir=tmp_path / "fallback",
                training_state={},
            )
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Helper function: _scene_from_training_dataset
# ---------------------------------------------------------------------------

class TestSceneFromTrainingDataset:
    def test_raises_when_no_evidence(self):
        dataset = MagicMock()
        dataset.frames = []
        dataset.regions = []
        with pytest.raises(ValueError, match="at least one training region"):
            _scene_from_training_dataset(dataset, name="test")

    def test_raises_when_region_has_unknown_frame(self):
        frame = MagicMock()
        frame.id = "frame_0"

        region = MagicMock()
        region.id = "region_0"
        region.frame_id = "frame_UNKNOWN"

        dataset = MagicMock()
        dataset.frames = [frame]
        dataset.regions = [region]

        with pytest.raises(ValueError, match="unknown frame"):
            _scene_from_training_dataset(dataset, name="test")

    def test_decomposes_evidence_when_valid(self):
        frame = MagicMock()
        frame.id = "frame_0"

        evidence_sample = MagicMock()
        region = MagicMock()
        region.id = "region_0"
        region.frame_id = "frame_0"
        region.to_evidence_sample.return_value = evidence_sample

        dataset = MagicMock()
        dataset.frames = [frame]
        dataset.regions = [region]

        mock_scene = MagicMock()
        with patch("aura.cli.decompose_evidence", return_value=mock_scene) as mock_de:
            result = _scene_from_training_dataset(dataset, name="test")
        mock_de.assert_called_once_with((evidence_sample,), name="test")
        assert result is mock_scene


# ---------------------------------------------------------------------------
# Helper function: _render_package_image
# ---------------------------------------------------------------------------

class TestRenderPackageImage:
    def _make_args(self, backend="cpu", **kwargs):
        defaults = {
            "width": 16, "height": 16, "device": None,
            "require_cuda": False, "threads_per_block": 128, "max_hits": 8,
        }
        defaults.update(kwargs)
        defaults["backend"] = backend
        return argparse.Namespace(**defaults)

    def test_cpu_backend(self):
        scene = MagicMock()
        with patch("aura.cli.render_orthographic", return_value=MagicMock()) as mock_render:
            _render_package_image(scene, self._make_args("cpu"))
        mock_render.assert_called_once()

    def test_cpu_backend_with_require_cuda_raises(self):
        scene = MagicMock()
        with pytest.raises(SystemExit, match="--require-cuda"):
            _render_package_image(scene, self._make_args("cpu", require_cuda=True))

    def test_torch_backend(self):
        scene = MagicMock()
        with patch("aura.cli.render_orthographic_torch", return_value=MagicMock()) as mock_render:
            _render_package_image(scene, self._make_args("torch"))
        mock_render.assert_called_once()

    def test_cuda_backend(self):
        scene = MagicMock()
        with patch("aura.cli.render_orthographic_cuda", return_value=MagicMock()) as mock_render:
            _render_package_image(scene, self._make_args("cuda"))
        call_kwargs = mock_render.call_args.kwargs
        assert call_kwargs["fallback_backend"] == "none"
        assert call_kwargs["require_cuda"] is True

    def test_auto_backend(self):
        scene = MagicMock()
        with patch("aura.cli.render_orthographic_cuda", return_value=MagicMock()) as mock_render:
            _render_package_image(scene, self._make_args("auto"))
        call_kwargs = mock_render.call_args.kwargs
        assert call_kwargs["fallback_backend"] == "auto"


# ---------------------------------------------------------------------------
# Helper function: _turntable_frame_renderer
# ---------------------------------------------------------------------------

class TestTurntableFrameRenderer:
    def _make_args(self, backend="cpu", **kwargs):
        defaults = {
            "device": None, "require_cuda": False,
            "threads_per_block": 128, "max_hits": 8,
        }
        defaults.update(kwargs)
        defaults["backend"] = backend
        return argparse.Namespace(**defaults)

    def test_cpu_backend_returns_render_orthographic(self):
        from aura.render import render_orthographic
        fn = _turntable_frame_renderer(self._make_args("cpu"))
        assert fn is render_orthographic

    def test_torch_backend_renders(self):
        args = self._make_args("torch")
        fn = _turntable_frame_renderer(args)
        scene = MagicMock()
        with patch("aura.cli.render_orthographic_torch", return_value=MagicMock()) as mock_render:
            fn(scene, width=8, height=8)
        mock_render.assert_called_once()

    def test_cuda_backend_uses_none_fallback(self):
        args = self._make_args("cuda")
        fn = _turntable_frame_renderer(args)
        scene = MagicMock()
        with patch("aura.cli.render_orthographic_cuda", return_value=MagicMock()) as mock_render:
            fn(scene, width=8, height=8)
        assert mock_render.call_args.kwargs["fallback_backend"] == "none"
        assert mock_render.call_args.kwargs["require_cuda"] is True

    def test_auto_backend_uses_auto_fallback(self):
        args = self._make_args("auto")
        fn = _turntable_frame_renderer(args)
        scene = MagicMock()
        with patch("aura.cli.render_orthographic_cuda", return_value=MagicMock()) as mock_render:
            fn(scene, width=8, height=8)
        assert mock_render.call_args.kwargs["fallback_backend"] == "auto"


# ---------------------------------------------------------------------------
# Helper function: _reconstruction_config_from_args
# ---------------------------------------------------------------------------

class TestReconstructionConfigFromArgs:
    def test_reconstruction_config_from_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--iterations", type=int, default=4)
        _add_reconstruction_config_args(parser)
        args = parser.parse_args([])
        config = _reconstruction_config_from_args(args)
        assert config.iterations == 4
        assert config.color_learning_rate == 0.35
        assert config.render_backend == "cpu"

    def test_reconstruction_config_custom_values(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--iterations", type=int, default=4)
        _add_reconstruction_config_args(parser)
        args = parser.parse_args([
            "--iterations", "10",
            "--color-learning-rate", "0.5",
            "--render-backend", "torch",
            "--disable-adaptive-evolution",
        ])
        config = _reconstruction_config_from_args(args)
        assert config.iterations == 10
        assert config.color_learning_rate == 0.5
        assert config.render_backend == "torch"
        assert config.enable_adaptive_evolution is False


# ---------------------------------------------------------------------------
# Helper function: _densification_config_from_args
# ---------------------------------------------------------------------------

class TestDensificationConfigFromArgs:
    def test_default_densification_config(self):
        parser = argparse.ArgumentParser()
        _add_densification_args(parser)
        args = parser.parse_args([])
        config = _densification_config_from_args(args)
        assert isinstance(config, DensificationConfig)
        assert config.enabled is False

    def test_enabled_densification(self):
        parser = argparse.ArgumentParser()
        _add_densification_args(parser)
        args = parser.parse_args(["--densify", "--grad-threshold", "0.001"])
        config = _densification_config_from_args(args)
        assert config.enabled is True
        assert config.grad_threshold == 0.001

    def test_densification_pruning_thresholds(self):
        parser = argparse.ArgumentParser()
        _add_densification_args(parser)
        args = parser.parse_args([
            "--densify",
            "--prune-threshold", "0.01",
            "--prune-opacity-threshold", "0.02",
        ])
        config = _densification_config_from_args(args)
        assert config.prune_importance_threshold == 0.01
        assert config.prune_opacity_threshold == 0.02


# ---------------------------------------------------------------------------
# Helper function: _add_render_backend_args
# ---------------------------------------------------------------------------

class TestAddRenderBackendArgs:
    def test_default_render_backend_args(self):
        parser = argparse.ArgumentParser()
        _add_render_backend_args(parser)
        args = parser.parse_args([])
        assert args.backend == "cpu"
        assert args.device is None
        assert args.require_cuda is False
        assert args.threads_per_block == 128
        assert args.max_hits == 8


# ---------------------------------------------------------------------------
# Helper function: _training_resume_state
# ---------------------------------------------------------------------------

class TestTrainingResumeState:
    def test_training_resume_state_structure(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("manifest", type=pathlib.Path)
        parser.add_argument("--disable-evolution", action="store_true")
        parser.add_argument("--color-learning-rate", type=float, default=0.25)
        parser.add_argument("--split-image-loss-threshold", type=float, default=0.03)
        parser.add_argument("--depth-anchor-loss-threshold", type=float, default=0.10)
        parser.add_argument("--merge-image-loss-threshold", type=float, default=0.025)
        parser.add_argument("--merge-depth-loss-threshold", type=float, default=0.04)
        parser.add_argument("--demote-after-iteration", type=int, default=3)
        parser.add_argument("--demote-image-loss-threshold", type=float, default=0.045)
        parser.add_argument("--demote-depth-loss-threshold", type=float, default=0.02)
        parser.add_argument("--image-loss-weight", type=float, default=1.0)
        parser.add_argument("--depth-loss-weight", type=float, default=None)
        parser.add_argument("--query-loss-weight", type=float, default=1.0)
        parser.add_argument("--normal-loss-weight", type=float, default=1.0)
        parser.add_argument("--mask-loss-weight", type=float, default=1.0)
        parser.add_argument("--confidence-loss-weight", type=float, default=0.0)

        args = parser.parse_args(["some_manifest.json"])
        args.depth_loss_weight = 0.0  # must be resolved before calling

        mock_manifest = MagicMock()
        mock_manifest.to_dict.return_value = {}

        mock_sampling_plan = MagicMock()
        mock_sampling_plan.max_targets_per_batch = 10
        mock_sampling_plan.to_dict.return_value = {}

        state = _training_resume_state(args, mock_manifest, mock_sampling_plan)
        assert state["format"] == "AURA_TRAINING_STATE"
        assert "resumeCompatibilityFingerprint" in state
        assert "manifestFingerprint" in state
        assert "samplingPlanFingerprint" in state
        assert "optimizerConfigFingerprint" in state

    def test_training_resume_state_evolution_disabled(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("manifest", type=pathlib.Path)
        parser.add_argument("--disable-evolution", action="store_true")
        parser.add_argument("--color-learning-rate", type=float, default=0.25)
        parser.add_argument("--split-image-loss-threshold", type=float, default=0.03)
        parser.add_argument("--depth-anchor-loss-threshold", type=float, default=0.10)
        parser.add_argument("--merge-image-loss-threshold", type=float, default=0.025)
        parser.add_argument("--merge-depth-loss-threshold", type=float, default=0.04)
        parser.add_argument("--demote-after-iteration", type=int, default=3)
        parser.add_argument("--demote-image-loss-threshold", type=float, default=0.045)
        parser.add_argument("--demote-depth-loss-threshold", type=float, default=0.02)
        parser.add_argument("--image-loss-weight", type=float, default=1.0)
        parser.add_argument("--depth-loss-weight", type=float, default=None)
        parser.add_argument("--query-loss-weight", type=float, default=1.0)
        parser.add_argument("--normal-loss-weight", type=float, default=1.0)
        parser.add_argument("--mask-loss-weight", type=float, default=1.0)
        parser.add_argument("--confidence-loss-weight", type=float, default=0.0)

        args = parser.parse_args(["some_manifest.json", "--disable-evolution"])
        args.depth_loss_weight = 0.0

        mock_manifest = MagicMock()
        mock_manifest.to_dict.return_value = {}
        mock_sampling_plan = MagicMock()
        mock_sampling_plan.max_targets_per_batch = 10
        mock_sampling_plan.to_dict.return_value = {}

        state = _training_resume_state(args, mock_manifest, mock_sampling_plan)
        assert state["optimizerConfig"]["adaptiveEvolutionEnabled"] is False
        assert state["optimizerConfig"]["evolutionPolicy"] is None


# ---------------------------------------------------------------------------
# inspect-capture-assets
# ---------------------------------------------------------------------------

class TestInspectCaptureAssets:
    def test_inspect_capture_assets(self):
        mock_manifest = MagicMock()
        mock_asset = MagicMock()
        mock_asset.to_dict.return_value = {"asset": "data"}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=mock_manifest), \
             patch("aura.cli.load_capture_assets", return_value=[mock_asset]):
            result = main(["inspect-capture-assets", "some_manifest.json"])
        assert result == 0


# ---------------------------------------------------------------------------
# inspect-capture-tensors
# ---------------------------------------------------------------------------

class TestInspectCaptureTensors:
    def test_inspect_capture_tensors(self):
        mock_manifest = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.to_dict.return_value = {"tensor": "data"}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=mock_manifest), \
             patch("aura.cli.load_capture_asset_tensors", return_value=[mock_tensor]):
            result = main(["inspect-capture-tensors", "some_manifest.json"])
        assert result == 0


# ---------------------------------------------------------------------------
# plan-capture-sampling
# ---------------------------------------------------------------------------

class TestPlanCaptureSampling:
    def test_plan_capture_sampling(self):
        mock_manifest = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.frames = []
        mock_plan = MagicMock()
        mock_plan.to_dict.return_value = {"plan": "data"}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_capture_manifest", return_value=mock_manifest), \
             patch("aura.cli.load_capture_asset_tensors", return_value=[]), \
             patch("aura.cli.capture_tensors_to_training_dataset", return_value=mock_dataset), \
             patch("aura.cli.plan_capture_tensor_sampling", return_value=mock_plan):
            result = main(["plan-capture-sampling", "some_manifest.json"])
        assert result == 0


# ---------------------------------------------------------------------------
# colmap-to-capture-manifest
# ---------------------------------------------------------------------------

class TestColmapToCaptureManifest:
    def test_colmap_to_capture_manifest(self, tmp_path):
        output = tmp_path / "manifest.json"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.write_colmap_capture_manifest", return_value=output):
            result = main(["colmap-to-capture-manifest", str(tmp_path), "--output", str(output)])
        assert result == 0


# ---------------------------------------------------------------------------
# write-demo-package
# ---------------------------------------------------------------------------

class TestWriteDemoPackage:
    def test_write_demo_package(self, tmp_path):
        mock_packaged = MagicMock()
        mock_packaged.write.return_value = tmp_path / "demo.aura"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.package_scene", return_value=mock_packaged):
            result = main(["write-demo-package", "--output-dir", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# write-splat-demo-package
# ---------------------------------------------------------------------------

class TestWriteSplatDemoPackage:
    def test_write_splat_demo_package(self, tmp_path):
        input_file = tmp_path / "tiny_3dgs.json"
        input_file.write_text("{}")
        mock_scene = MagicMock()
        mock_packaged = MagicMock()
        mock_packaged.write.return_value = tmp_path / "splat.aura"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_3dgs_scene", return_value=mock_scene), \
             patch("aura.cli.package_scene", return_value=mock_packaged):
            result = main(["write-splat-demo-package", "--input", str(input_file), "--output-dir", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# import-3dgs
# ---------------------------------------------------------------------------

class TestImport3dgs:
    def test_import_3dgs(self, tmp_path):
        input_file = tmp_path / "scene.ply"
        input_file.write_bytes(b"PLY")
        mock_package = MagicMock()
        mock_package.write.return_value = tmp_path / "imported.aura"
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.package_3dgs_export", return_value=mock_package):
            result = main(["import-3dgs", str(input_file), "--output-dir", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# query-demo
# ---------------------------------------------------------------------------

class TestQueryDemo:
    def test_query_demo(self):
        mock_scene = MagicMock()
        mock_scene.ray_query.return_value = MagicMock()
        with patch("aura.cli.native_demo_scene", return_value=mock_scene):
            result = main(["query-demo", "--x", "0.1", "--y", "0.2"])
        assert result == 0
        mock_scene.ray_query.assert_called_once()


# ---------------------------------------------------------------------------
# validate-package
# ---------------------------------------------------------------------------

class TestValidatePackage:
    def test_validate_package(self, tmp_path):
        mock_pkg = MagicMock()
        mock_pkg.summary.return_value = {
            "name": "test",
            "version": "1.0",
            "elementCount": 5,
            "chunkCount": 2,
        }
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg):
            result = main(["validate-package", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# inspect-package
# ---------------------------------------------------------------------------

class TestInspectPackage:
    def test_inspect_package(self, tmp_path):
        mock_pkg = MagicMock()
        mock_pkg.summary.return_value = {"name": "test", "version": "1.0"}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg):
            result = main(["inspect-package", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# export-report
# ---------------------------------------------------------------------------

class TestExportReport:
    def test_export_report(self, tmp_path):
        mock_pkg = MagicMock()
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"status": "ok"}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.load_package", return_value=mock_pkg), \
             patch("aura.cli.runtime_export_report", return_value=mock_report):
            result = main(["export-report", str(tmp_path)])
        assert result == 0


# ---------------------------------------------------------------------------
# readiness-report
# ---------------------------------------------------------------------------

class TestReadinessReport:
    def test_readiness_report(self):
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"ready": True}
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
             patch("aura.cli.production_readiness_report", return_value=mock_report):
            result = main(["readiness-report"])
        assert result == 0


# ---------------------------------------------------------------------------
# __main__ guard (line 1439)
# ---------------------------------------------------------------------------

class TestMainEntryPoint:
    def test_main_raises_on_unknown_command(self):
        with patch("aura.cli.native_demo_scene", return_value=MagicMock()):
            # We need to trick the parser into accepting an unknown command
            # by patching argparse to return a fake known command that falls
            # through all branches. The easiest is to test the ValueError branch.
            with patch("aura.cli.native_demo_scene", return_value=MagicMock()), \
                 patch("argparse.ArgumentParser.parse_args") as mock_parse:
                # Simulate a namespace with an unrecognized command
                mock_parse.return_value = argparse.Namespace(command="__unknown_command__")
                with pytest.raises(ValueError, match="__unknown_command__"):
                    main(["build-native-demo"])  # argv doesn't matter since parse_args is mocked

    def test_main_as_module_calls_main(self):
        """Verify __main__ guard: raise SystemExit(main()) is correct flow."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "aura.cli", "--help"],
            capture_output=True,
            text=True,
        )
        # --help exits with 0 and prints usage
        assert result.returncode == 0
        assert "aura" in result.stdout
