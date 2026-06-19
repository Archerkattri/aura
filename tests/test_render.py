import importlib.util
import subprocess
import sys

import pytest
import aura.render as render_module
import aura.torch_renderer as torch_renderer_module

from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    RenderImage,
    compare_images,
    image_lpips_proxy,
    image_mse,
    image_psnr,
    image_ssim,
    package_scene,
    read_ppm,
    render_orthographic,
    render_orthographic_cuda,
    render_orthographic_torch,
)
from aura.cli import demo_scene


def test_render_orthographic_produces_nonblank_preview():
    image = render_orthographic(demo_scene(), width=8, height=8)

    assert image.width == 8
    assert image.height == 8
    assert len(image.pixels) == 64
    assert any(pixel != (0.0, 0.0, 0.0) for pixel in image.pixels)
    assert image.pixel(4, 4)[0] > 0.0


def test_render_orthographic_respects_scene_bounds():
    scene = AuraScene(
        name="single",
        elements=(
            AuraElement(
                id="red",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
            ),
        ),
    )

    image = render_orthographic(scene, width=3, height=3)

    assert all(pixel == (1.0, 0.0, 0.0) for pixel in image.pixels)


def test_render_orthographic_uses_ordered_native_compositing():
    scene = AuraScene(
        name="composited",
        elements=(
            AuraElement(
                id="front",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.5,
            ),
            AuraElement(
                id="back",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.2), (0.5, 0.5, 0.3)),
                color=(0.0, 0.0, 1.0),
                opacity=0.5,
            ),
        ),
    )

    image = render_orthographic(scene, width=1, height=1)

    assert image.pixel(0, 0) == pytest.approx((0.5, 0.0, 0.25))


def test_render_orthographic_cuda_cpu_fallback_matches_cpu_preview():
    scene = demo_scene()

    image = render_orthographic_cuda(scene, width=4, height=4, fallback_backend="cpu")
    expected = render_orthographic(scene, width=4, height=4)

    assert image.width == 4
    assert image.height == 4
    for pixel, expected_pixel in zip(image.pixels, expected.pixels):
        assert pixel == pytest.approx(expected_pixel)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_render_orthographic_torch_matches_cpu_preview():
    scene = AuraScene(
        name="torch_preview",
        elements=(
            AuraElement(
                id="red",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    image = render_orthographic_torch(scene, width=3, height=3, device="cpu")
    expected = render_orthographic(scene, width=3, height=3)

    assert image.width == 3
    assert image.height == 3
    for pixel, expected_pixel in zip(image.pixels, expected.pixels):
        assert pixel == pytest.approx(expected_pixel)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_render_orthographic_torch_builds_rays_as_tensors(monkeypatch):
    scene = AuraScene(
        name="torch_tensor_grid_preview",
        elements=(
            AuraElement(
                id="red",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    def fail_python_ray_grid(*_args, **_kwargs):
        raise AssertionError("torch orthographic render should not build Python ray tuples")

    monkeypatch.setattr(render_module, "orthographic_camera_rays", fail_python_ray_grid)

    image = render_orthographic_torch(scene, width=2, height=2, device="cpu")

    assert image.width == 2
    assert image.height == 2
    assert all(pixel == pytest.approx((1.0, 0.0, 0.0)) for pixel in image.pixels)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_render_orthographic_torch_skips_ordered_trace_serialization(monkeypatch):
    scene = AuraScene(
        name="torch_trace_free_preview",
        elements=(
            AuraElement(
                id="red",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    def fail_ordered_trace_serialization(*_args, **_kwargs):
        raise AssertionError("torch preview render should not serialize ordered hit traces")

    monkeypatch.setattr(torch_renderer_module, "_torch_ordered_hit_traces", fail_ordered_trace_serialization)

    image = render_orthographic_torch(scene, width=2, height=2, device="cpu")

    assert image.width == 2
    assert image.height == 2
    assert all(pixel == pytest.approx((1.0, 0.0, 0.0)) for pixel in image.pixels)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_render_orthographic_torch_uses_color_tensor_path(monkeypatch):
    scene = AuraScene(
        name="torch_color_tensor_preview",
        elements=(
            AuraElement(
                id="red",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    def fail_full_batch_render(*_args, **_kwargs):
        raise AssertionError("torch preview render should not build a full render batch")

    monkeypatch.setattr(torch_renderer_module, "torch_render_rays", fail_full_batch_render)
    monkeypatch.setattr(torch_renderer_module, "_torch_render_tensor_targets", fail_full_batch_render)

    image = render_orthographic_torch(scene, width=2, height=2, device="cpu")

    assert image.width == 2
    assert image.height == 2
    assert all(pixel == pytest.approx((1.0, 0.0, 0.0)) for pixel in image.pixels)


def test_render_image_writes_ascii_ppm(tmp_path):
    image = RenderImage(width=2, height=1, pixels=((1.0, 0.0, 0.0), (0.0, 0.5, 1.0)))

    path = image.write_ppm(tmp_path / "preview.ppm")

    assert path.read_text(encoding="ascii").splitlines() == [
        "P3",
        "2 1",
        "255",
        "255 0 0",
        "0 128 255",
    ]
    loaded = read_ppm(path)
    assert loaded.width == 2
    assert loaded.height == 1
    assert loaded.pixel(0, 0) == (1.0, 0.0, 0.0)
    assert loaded.pixel(1, 0) == (0.0, pytest.approx(128 / 255), 1.0)


def test_image_metrics_report_identity_and_difference():
    left = RenderImage(width=1, height=1, pixels=((1.0, 0.0, 0.0),))
    right = RenderImage(width=1, height=1, pixels=((0.0, 0.0, 0.0),))

    assert image_mse(left, left) == 0.0
    assert image_psnr(left, left) == float("inf")
    assert image_ssim(left, left) == 1.0
    assert image_lpips_proxy(left, left) == 0.0
    assert image_mse(left, right) == pytest.approx(1.0 / 3.0)
    assert image_psnr(left, right) > 0.0
    assert image_ssim(left, right) < 1.0
    assert image_lpips_proxy(left, right) == pytest.approx(1.0 / 3.0)
    identical = compare_images(left, left, min_psnr=99.0)
    different = compare_images(left, right, min_psnr=99.0)
    assert identical["passed"] is True
    assert identical["ssim"] == 1.0
    assert identical["lpipsProxy"] == 0.0
    assert different["passed"] is False


def test_render_package_cli_writes_preview(tmp_path):
    package_dir = tmp_path / "demo.aura"
    preview_path = tmp_path / "preview.ppm"
    package_scene(demo_scene()).write(package_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "render-package",
            str(package_dir),
            "--output",
            str(preview_path),
            "--width",
            "4",
            "--height",
            "4",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert str(preview_path) in result.stdout
    assert preview_path.read_text(encoding="ascii").startswith("P3\n4 4\n255\n")


def test_render_cli_alias_writes_preview(tmp_path):
    package_dir = tmp_path / "demo.aura"
    preview_path = tmp_path / "render.ppm"
    package_scene(demo_scene()).write(package_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "render",
            str(package_dir),
            "--output",
            str(preview_path),
            "--width",
            "4",
            "--height",
            "4",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert str(preview_path) in result.stdout
    assert preview_path.read_text(encoding="ascii").startswith("P3\n4 4\n255\n")


def test_render_cli_can_use_cuda_boundary_cpu_fallback(tmp_path):
    package_dir = tmp_path / "demo.aura"
    preview_path = tmp_path / "render_cuda_boundary.ppm"
    package_scene(demo_scene()).write(package_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "render",
            str(package_dir),
            "--output",
            str(preview_path),
            "--width",
            "4",
            "--height",
            "4",
            "--backend",
            "auto",
            "--device",
            "cpu",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert str(preview_path) in result.stdout
    assert preview_path.read_text(encoding="ascii").startswith("P3\n4 4\n255\n")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_render_cli_can_use_direct_torch_backend(tmp_path):
    package_dir = tmp_path / "demo.aura"
    preview_path = tmp_path / "render_torch.ppm"
    package_scene(demo_scene()).write(package_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "render",
            str(package_dir),
            "--output",
            str(preview_path),
            "--width",
            "4",
            "--height",
            "4",
            "--backend",
            "torch",
            "--device",
            "cpu",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert str(preview_path) in result.stdout
    assert preview_path.read_text(encoding="ascii").startswith("P3\n4 4\n255\n")


def test_compare_renders_cli_reports_json_metrics(tmp_path):
    expected = tmp_path / "expected.ppm"
    actual = tmp_path / "actual.ppm"
    RenderImage(width=1, height=1, pixels=((1.0, 0.0, 0.0),)).write_ppm(expected)
    RenderImage(width=1, height=1, pixels=((1.0, 0.0, 0.0),)).write_ppm(actual)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "compare-renders", str(expected), str(actual), "--min-psnr", "60"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert '"passed": true' in result.stdout
    assert '"psnr": null' in result.stdout
    assert '"psnrInfinite": true' in result.stdout


def test_compare_renders_cli_fails_when_threshold_is_missed(tmp_path):
    expected = tmp_path / "expected.ppm"
    actual = tmp_path / "actual.ppm"
    RenderImage(width=1, height=1, pixels=((1.0, 0.0, 0.0),)).write_ppm(expected)
    RenderImage(width=1, height=1, pixels=((0.0, 0.0, 0.0),)).write_ppm(actual)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "compare-renders", str(expected), str(actual), "--min-psnr", "60"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 1
    assert '"passed": false' in result.stdout


# --- Additional coverage tests ---

def test_render_image_rejects_zero_dimensions():
    """Cover RenderImage.__post_init__ dimension checks (lines 29, 31)."""
    with pytest.raises(ValueError, match="dimensions must be positive"):
        RenderImage(width=0, height=1, pixels=())
    with pytest.raises(ValueError, match="dimensions must be positive"):
        RenderImage(width=1, height=0, pixels=())


def test_render_image_rejects_wrong_pixel_count():
    """Cover RenderImage.__post_init__ pixel count check (line 31)."""
    with pytest.raises(ValueError, match="pixel count"):
        RenderImage(width=2, height=2, pixels=((1.0, 0.0, 0.0),))


def test_render_image_pixel_out_of_bounds():
    """Cover RenderImage.pixel out-of-bounds check (line 36)."""
    image = RenderImage(width=2, height=2, pixels=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.5)))
    with pytest.raises(IndexError, match="out of bounds"):
        image.pixel(-1, 0)
    with pytest.raises(IndexError, match="out of bounds"):
        image.pixel(0, 5)
    with pytest.raises(IndexError, match="out of bounds"):
        image.pixel(5, 0)


def test_read_ppm_rejects_invalid_format(tmp_path):
    """Cover read_ppm format checks (lines 54, 59, 63)."""
    # Not P3 format
    bad_format = tmp_path / "bad.ppm"
    bad_format.write_text("P6\n1 1\n255\n", encoding="ascii")
    with pytest.raises(ValueError, match="P3"):
        from aura.render import read_ppm as _read_ppm
        _read_ppm(bad_format)

    # Too few tokens
    too_short = tmp_path / "short.ppm"
    too_short.write_text("P3", encoding="ascii")
    with pytest.raises(ValueError, match="P3"):
        from aura.render import read_ppm as _read_ppm
        _read_ppm(too_short)

    # max_value <= 0
    bad_max = tmp_path / "badmax.ppm"
    bad_max.write_text("P3\n1 1\n0\n0 0 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="max value"):
        from aura.render import read_ppm as _read_ppm
        _read_ppm(bad_max)

    # Wrong number of values
    wrong_count = tmp_path / "wrongcount.ppm"
    wrong_count.write_text("P3\n2 2\n255\n0 0 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="expected"):
        from aura.render import read_ppm as _read_ppm
        _read_ppm(wrong_count)


def test_render_orthographic_rejects_zero_dimensions():
    """Cover render_orthographic dimension check (line 104)."""
    from aura.cli import demo_scene
    with pytest.raises(ValueError, match="dimensions must be positive"):
        render_orthographic(demo_scene(), width=0, height=4)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        render_orthographic(demo_scene(), width=4, height=0)


def test_render_orthographic_torch_rejects_non_cuda_when_required(monkeypatch):
    """Cover render_orthographic_torch require_cuda guard (line 176)."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not available")
    scene = AuraScene(
        name="test",
        elements=(
            AuraElement(id="s", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))),
        ),
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        render_orthographic_torch(scene, width=2, height=2, device="cpu", require_cuda=True)


def test_orthographic_camera_ray_tensors_rejects_zero_dimensions():
    """Cover _orthographic_camera_ray_tensors dimension check (line 210)."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not available")
    import torch
    from aura.render import _orthographic_camera_ray_tensors
    scene = AuraScene(
        name="t",
        elements=(AuraElement(id="e", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))),),
    )
    with pytest.raises(ValueError, match="dimensions must be positive"):
        _orthographic_camera_ray_tensors(torch, scene, width=0, height=4, device="cpu")


def test_orthographic_camera_rays_rejects_zero_dimensions():
    """Cover orthographic_camera_rays dimension check (line 239)."""
    from aura.render import orthographic_camera_rays
    scene = AuraScene(
        name="t",
        elements=(AuraElement(id="e", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))),),
    )
    with pytest.raises(ValueError, match="dimensions must be positive"):
        orthographic_camera_rays(scene, width=0, height=4)


def test_render_uses_element_bounds_when_no_chunks():
    """Cover _scene_bounds element fallback (lines 267-275)."""
    scene = AuraScene(
        name="no_chunks",
        elements=(
            AuraElement(id="e", carrier_id="surface", bounds=Bounds((0.5, 0.5, 0.0), (1.5, 1.5, 0.1))),
        ),
        chunks=(),
    )
    # Rendering should succeed and use element bounds (not chunk bounds)
    image = render_orthographic(scene, width=4, height=4)
    assert image.width == 4
    assert image.height == 4


def test_scene_bounds_raises_for_empty_scene():
    """Cover _scene_bounds empty scene error (line 377)."""
    from aura.render import _scene_bounds
    empty_scene = AuraScene(name="empty", elements=(), chunks=())
    with pytest.raises(ValueError, match="empty scene"):
        _scene_bounds(empty_scene)


def test_turntable_camera_path_single_frame():
    """Cover turntable_camera_path single-frame branch (lines 267-275 in render.py)."""
    from aura.render import turntable_camera_path
    scene = AuraScene(
        name="t",
        elements=(AuraElement(id="e", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),),
    )
    path = turntable_camera_path(scene, frames=1)
    assert len(path) == 1


def test_turntable_camera_path_rejects_zero_frames():
    """Cover turntable_camera_path frame count check (line 267)."""
    from aura.render import turntable_camera_path
    scene = AuraScene(
        name="t",
        elements=(AuraElement(id="e", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),),
    )
    with pytest.raises(ValueError, match="frame count must be positive"):
        turntable_camera_path(scene, frames=0)


def test_render_turntable_frames_produces_correct_count():
    """Cover render_turntable_frames (lines 294-297)."""
    from aura.render import render_turntable_frames
    scene = AuraScene(
        name="t",
        elements=(AuraElement(id="e", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)), color=(1.0, 0.0, 0.0), opacity=1.0),),
    )
    frames = render_turntable_frames(scene, frames=3, width=2, height=2)
    assert len(frames) == 3
    assert all(isinstance(f, RenderImage) for f in frames)


def test_image_mse_raises_for_dimension_mismatch():
    """Cover image_mse dimension check (line 306)."""
    left = RenderImage(width=2, height=1, pixels=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    right = RenderImage(width=1, height=1, pixels=((1.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="matching dimensions"):
        image_mse(left, right)


def test_image_ssim_returns_one_for_uniform_identical_images():
    """Cover image_ssim denominator=0 branch (line 346)."""
    # When both images are uniform (all same pixel), denominator approaches the SSIM
    # special case path. Use identical uniform images to trigger denominator == 0.
    uniform = RenderImage(width=2, height=2, pixels=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    result = image_ssim(uniform, uniform)
    assert result == 1.0


def test_require_matching_dimensions_raises():
    """Cover _require_matching_dimensions error (line 365)."""
    left = RenderImage(width=2, height=1, pixels=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    right = RenderImage(width=1, height=1, pixels=((1.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="matching dimensions"):
        image_ssim(left, right)
    with pytest.raises(ValueError, match="matching dimensions"):
        image_lpips_proxy(left, right)
