import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aura import (
    load_package,
    package_scene,
    run_real_scene_benchmark,
)
from aura.cli import demo_scene
from aura.imaging import read_reference_image
from aura.render import render_orthographic

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _cli_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env


def _package(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    return load_package(package_dir), package_dir


def test_fixture_mode_degrades_without_reference_dir(tmp_path):
    package, package_dir = _package(tmp_path)
    report = run_real_scene_benchmark(package, package_dir=package_dir, fixture_view_count=3)
    assert report["format"] == "AURA_REAL_SCENE_BENCHMARK"
    assert report["referenceSource"] == "deterministic_fixture"
    assert report["referenceDir"] is None
    assert report["aggregate"]["viewCount"] == 3
    assert report["passed"] is True
    assert len(report["views"]) == 3
    json.dumps(report)


def test_external_reference_dir_scores_each_view(tmp_path):
    package, package_dir = _package(tmp_path)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    for index in range(3):
        image = render_orthographic(demo_scene(), width=8, height=8)
        image.write_ppm(ref_dir / f"view_{index:03d}.ppm")

    report = run_real_scene_benchmark(
        package,
        reference_dir=ref_dir,
        baseline_label="colmap",
        min_psnr=15.0,
    )
    assert report["referenceSource"] == "external_reference_dir"
    assert report["baseline"] == "colmap"
    assert report["aggregate"]["viewCount"] == 3
    assert report["aggregate"]["meanPsnr"] is not None
    assert report["aggregate"]["meanSsim"] > 0.0
    assert report["passed"] is True
    assert all(view["passed"] for view in report["views"])


def test_min_psnr_gate_fails_against_divergent_reference(tmp_path):
    package, package_dir = _package(tmp_path)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    # A solid-color reference that does not match the rendered scene.
    from aura.render import RenderImage

    constant = RenderImage(width=6, height=6, pixels=((0.0, 1.0, 0.0),) * 36)
    constant.write_ppm(ref_dir / "view_000.ppm")

    report = run_real_scene_benchmark(
        package,
        reference_dir=ref_dir,
        min_psnr=60.0,
    )
    assert report["aggregate"]["viewCount"] == 1
    assert report["passed"] is False


def test_max_views_limits_loaded_references(tmp_path):
    package, _ = _package(tmp_path)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    for index in range(5):
        render_orthographic(demo_scene(), width=8, height=8).write_ppm(
            ref_dir / f"view_{index:03d}.ppm"
        )
    report = run_real_scene_benchmark(package, reference_dir=ref_dir, max_views=2)
    assert report["aggregate"]["viewCount"] == 2


def test_empty_reference_dir_raises(tmp_path):
    package, _ = _package(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        run_real_scene_benchmark(package, reference_dir=empty)


def test_pfm_reference_roundtrips_through_harness(tmp_path):
    package, _ = _package(tmp_path)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    from aura.imaging import write_pfm_image

    image = render_orthographic(demo_scene(), width=8, height=8)
    write_pfm_image(image, ref_dir / "view_000.pfm")
    loaded = read_reference_image(ref_dir / "view_000.pfm")
    assert loaded.width == 8
    report = run_real_scene_benchmark(package, reference_dir=ref_dir, min_psnr=15.0)
    assert report["passed"] is True


def test_real_scene_benchmark_cli_fixture_mode(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-real-scene",
            str(package_dir),
            "--fixture-view-count",
            "2",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_cli_env(),
    )
    payload = json.loads(result.stdout)
    assert payload["referenceSource"] == "deterministic_fixture"
    assert payload["aggregate"]["viewCount"] == 2


def test_real_scene_benchmark_cli_external_dir(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    render_orthographic(demo_scene(), width=8, height=8).write_ppm(ref_dir / "view_000.ppm")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-real-scene",
            str(package_dir),
            "--reference-dir",
            str(ref_dir),
            "--baseline-label",
            "3dgs",
            "--min-psnr",
            "15",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_cli_env(),
    )
    payload = json.loads(result.stdout)
    assert payload["referenceSource"] == "external_reference_dir"
    assert payload["baseline"] == "3dgs"
    assert payload["aggregate"]["viewCount"] == 1
