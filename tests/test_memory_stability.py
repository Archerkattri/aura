import json
import os
import subprocess
import sys
from pathlib import Path

from aura import (
    package_scene,
    run_memory_stability_probe,
)
from aura.cli import demo_scene

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _cli_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env


def test_render_query_loop_is_memory_stable():
    report = run_memory_stability_probe(
        demo_scene(),
        iterations=48,
        width=12,
        height=12,
        sample_interval=8,
    )
    assert report.iterations == 48
    assert report.stable, report.to_dict()
    # A clean loop should not grow allocation meaningfully per iteration.
    assert report.growth_bytes_per_iteration <= report.threshold_bytes_per_iteration


def test_probe_detects_a_real_leak():
    leaked: list[bytes] = []

    def leaky(iteration: int) -> None:
        # Grow unbounded with distinct Python objects tracemalloc tracks well.
        leaked.append(bytes([iteration % 256]) * 2048 + str(iteration).encode())

    report = run_memory_stability_probe(
        demo_scene(),
        iterations=64,
        workload=leaky,
        sample_interval=8,
        growth_threshold_bytes_per_iteration=512.0,
    )
    assert not report.stable, report.to_dict()
    assert report.growth_bytes_per_iteration > report.threshold_bytes_per_iteration


def test_memory_report_serializes():
    report = run_memory_stability_probe(demo_scene(), iterations=16, width=8, height=8)
    payload = report.to_dict()
    assert payload["format"] == "AURA_MEMORY_STABILITY"
    assert payload["iterations"] == 16
    assert "growthBytesPerIteration" in payload
    json.dumps(payload)  # must be JSON serializable


def test_memory_stability_cli(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "memory-stability-probe",
            str(package_dir),
            "--iterations",
            "32",
            "--width",
            "8",
            "--height",
            "8",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_cli_env(),
    )
    payload = json.loads(result.stdout)
    assert payload["stable"] is True
    assert payload["iterations"] == 32
