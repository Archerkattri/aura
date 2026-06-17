import subprocess
import sys

import pytest

from aura import (
    AuraElement,
    AuraScene,
    Bounds,
    RenderImage,
    compare_images,
    image_mse,
    image_psnr,
    package_scene,
    read_ppm,
    render_orthographic,
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
    assert image_mse(left, right) == pytest.approx(1.0 / 3.0)
    assert image_psnr(left, right) > 0.0
    assert compare_images(left, left, min_psnr=99.0)["passed"] is True
    assert compare_images(left, right, min_psnr=99.0)["passed"] is False


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
