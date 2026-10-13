import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aura import load_package, package_scene, run_real_scene_benchmark
from aura.cli import demo_scene
from aura.imaging import read_reference_image, write_pfm_image
from aura.render import RenderImage, render_orthographic

pytestmark = pytest.mark.integration

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env


def _package(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    return load_package(package_dir), package_dir


def test_fixture_mode_runs_without_external_reference_data(tmp_path):
    package, package_dir = _package(tmp_path)

    report = run_real_scene_benchmark(package, package_dir=package_dir, fixture_view_count=3)

    assert report["format"] == "AURA_REAL_SCENE_BENCHMARK"
    assert report["referenceSource"] == "deterministic_fixture"
    assert report["referenceDir"] is None
    assert report["aggregate"]["viewCount"] == 3
    assert report["passed"] is True
    assert len(report["views"]) == 3
    json.dumps(report)


def test_external_reference_directory_scores_each_view(tmp_path):
    package, _ = _package(tmp_path)
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    for index in range(3):
        render_orthographic(demo_scene(), width=8, height=8).write_ppm(reference_dir / f"view_{index:03d}.ppm")

    report = run_real_scene_benchmark(
        package,
        reference_dir=reference_dir,
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
    package, _ = _package(tmp_path)
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    RenderImage(width=6, height=6, pixels=((0.0, 1.0, 0.0),) * 36).write_ppm(
        reference_dir / "view_000.ppm"
    )

    report = run_real_scene_benchmark(package, reference_dir=reference_dir, min_psnr=60.0)

    assert report["aggregate"]["viewCount"] == 1
    assert report["passed"] is False


def test_max_views_limits_loaded_references(tmp_path):
    package, _ = _package(tmp_path)
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    for index in range(5):
        render_orthographic(demo_scene(), width=8, height=8).write_ppm(
            reference_dir / f"view_{index:03d}.ppm"
        )

    report = run_real_scene_benchmark(package, reference_dir=reference_dir, max_views=2)

    assert report["aggregate"]["viewCount"] == 2


def test_empty_reference_directory_is_rejected(tmp_path):
    package, _ = _package(tmp_path)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError):
        run_real_scene_benchmark(package, reference_dir=empty_dir)


def test_pfm_reference_round_trips_through_benchmark_harness(tmp_path):
    package, _ = _package(tmp_path)
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    image = render_orthographic(demo_scene(), width=8, height=8)
    write_pfm_image(image, reference_dir / "view_000.pfm")

    loaded = read_reference_image(reference_dir / "view_000.pfm")
    report = run_real_scene_benchmark(package, reference_dir=reference_dir, min_psnr=15.0)

    assert loaded.width == 8
    assert report["passed"] is True


def test_benchmark_cli_runs_fixture_mode(tmp_path):
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
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    payload = json.loads(result.stdout)

    assert payload["referenceSource"] == "deterministic_fixture"
    assert payload["aggregate"]["viewCount"] == 2


def test_benchmark_cli_scores_external_reference_directory(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    render_orthographic(demo_scene(), width=8, height=8).write_ppm(reference_dir / "view_000.ppm")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "benchmark-real-scene",
            str(package_dir),
            "--reference-dir",
            str(reference_dir),
            "--baseline-label",
            "3dgs",
            "--min-psnr",
            "15",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    payload = json.loads(result.stdout)

    assert payload["referenceSource"] == "external_reference_dir"
    assert payload["baseline"] == "3dgs"
    assert payload["aggregate"]["viewCount"] == 1
