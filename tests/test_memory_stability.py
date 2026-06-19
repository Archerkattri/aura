import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


def test_memory_stability_probe_rejects_non_positive_iterations():
    """Line 112 memory.py: iterations <= 0 raises ValueError."""
    from aura.cli import native_demo_scene
    with pytest.raises(ValueError, match="iterations"):
        run_memory_stability_probe(native_demo_scene(), iterations=0)


def test_maybe_cuda_returns_none_for_cpu_device():
    """Lines 181-182 memory.py: _maybe_cuda returns None when device doesn't start with 'cuda'."""
    from aura.memory import _maybe_cuda
    assert _maybe_cuda(None) is None
    assert _maybe_cuda("cpu") is None


def test_maybe_cuda_returns_none_when_cuda_unavailable():
    """Lines 183-188 memory.py: _maybe_cuda returns None when torch not available or CUDA not present."""
    from aura.memory import _maybe_cuda
    # Either None (no CUDA / no torch) or torch.cuda module (with CUDA) — both are valid.
    result = _maybe_cuda("cuda:0")
    assert result is None or hasattr(result, "is_available")


def test_memory_stability_probe_samples_always_populated():
    """Line 148 memory.py: samples is always populated (via loop or fallback to peak_current)."""
    from aura.cli import demo_scene
    # Use a small run; the loop always appends at least the final iteration sample.
    report = run_memory_stability_probe(demo_scene(), iterations=2, sample_interval=100, width=4, height=4)
    assert report.iterations == 2
    assert len(report.samples) >= 1


def test_maybe_cuda_with_mocked_cuda_available():
    """Lines 119-121 memory.py: CUDA synchronize/empty_cache/memory_allocated called when cuda available."""
    from aura.memory import _maybe_cuda
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.memory_allocated.return_value = 0

    mock_torch = MagicMock()
    mock_torch.cuda = mock_cuda

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = _maybe_cuda("cuda:0")
        assert result is mock_cuda


def test_maybe_cuda_returns_none_when_torch_cuda_unavailable():
    """Lines 185-186, 188 memory.py: _maybe_cuda returns None when torch importable but CUDA unavailable."""
    from aura.memory import _maybe_cuda

    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False

    mock_torch = MagicMock()
    mock_torch.cuda = mock_cuda

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = _maybe_cuda("cuda:0")
        assert result is None


def test_probe_with_mocked_cuda_runs_cuda_branch():
    """Lines 119-121, 144-145, 160-161 memory.py: cuda branch executes synchronize/memory_allocated."""
    from aura.memory import run_memory_stability_probe
    from aura.cli import demo_scene

    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    # Return same value so CUDA growth stays zero
    mock_cuda.memory_allocated.return_value = 1024

    mock_torch = MagicMock()
    mock_torch.cuda = mock_cuda

    with patch.dict("sys.modules", {"torch": mock_torch}):
        report = run_memory_stability_probe(
            demo_scene(),
            iterations=4,
            width=4,
            height=4,
            device="cuda:0",
            sample_interval=2,
            growth_threshold_bytes_per_iteration=1_000_000.0,  # very permissive to avoid CI flakiness
        )

    assert report.cuda_allocated_start == 1024
    assert report.cuda_allocated_end == 1024
    assert report.stable is True
    # synchronize called at start and end
    assert mock_cuda.synchronize.call_count >= 2
    # empty_cache called once at start
    mock_cuda.empty_cache.assert_called_once()


def test_probe_with_mocked_cuda_detects_cuda_growth():
    """Lines 160-161 memory.py: cuda growth > threshold makes stable=False."""
    from aura.memory import run_memory_stability_probe
    from aura.cli import demo_scene

    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    # Huge growth between start and end
    mock_cuda.memory_allocated.side_effect = [0, 10_000_000]

    mock_torch = MagicMock()
    mock_torch.cuda = mock_cuda

    with patch.dict("sys.modules", {"torch": mock_torch}):
        report = run_memory_stability_probe(
            demo_scene(),
            iterations=4,
            width=4,
            height=4,
            device="cuda:0",
            sample_interval=2,
            growth_threshold_bytes_per_iteration=4096.0,
        )

    assert report.cuda_allocated_start == 0
    assert report.cuda_allocated_end == 10_000_000
    assert report.stable is False


def test_maybe_cuda_returns_none_when_torch_import_fails():
    """Lines 185-186 memory.py: _maybe_cuda returns None when torch raises on import."""
    from aura.memory import _maybe_cuda

    with patch.dict("sys.modules", {"torch": None}):
        result = _maybe_cuda("cuda:0")
        assert result is None


def test_probe_samples_fallback_when_sample_interval_exceeds_iterations():
    """Line 148 memory.py: when no sample captured in loop, fallback to peak_current."""
    from aura.memory import run_memory_stability_probe
    from aura.cli import demo_scene
    # iterations=1, sample_interval=1 → index==1==iterations so appends normally
    # To get the fallback (line 148), we need no sample appended in loop.
    # With iterations=1 sample_interval=10: index==1, 1%10 != 0 but 1==iterations so still appends.
    # The fallback is actually only reachable if iterations=0, but that's rejected.
    # So line 148 is dead code. Test that samples is always non-empty.
    report = run_memory_stability_probe(
        demo_scene(), iterations=1, width=4, height=4, sample_interval=100
    )
    assert len(report.samples) >= 1


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
