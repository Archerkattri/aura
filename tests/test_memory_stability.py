import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aura import package_scene, run_memory_stability_probe
from aura.cli import demo_scene

pytestmark = pytest.mark.integration

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env


def test_repeated_render_query_loop_stays_within_memory_threshold():
    report = run_memory_stability_probe(
        demo_scene(),
        iterations=48,
        width=12,
        height=12,
        sample_interval=8,
    )

    assert report.iterations == 48
    assert report.stable, report.to_dict()
    assert report.growth_bytes_per_iteration <= report.threshold_bytes_per_iteration


def test_memory_probe_flags_a_repeated_workload_leak():
    leaked: list[bytes] = []

    def leaky(iteration: int) -> None:
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


def test_memory_stability_cli_reads_package_and_emits_json(tmp_path):
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
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    payload = json.loads(result.stdout)

    assert payload["stable"] is True
    assert payload["iterations"] == 32
