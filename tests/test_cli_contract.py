"""Tests for the asset-contract CLI commands (ray-query, confidence, export-splat)."""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.carrier_io import save_carriers, load_carriers  # noqa: E402
from aura.cli import _ray_query_command, _confidence_command, _export_splat_command  # noqa: E402


def _make_carriers(d):
    save_carriers(d, means=np.array([[0.0, 0.0, 2.0]]), scales=np.array([[0.3, 0.3, 0.05]]),
                  quats=np.array([[1.0, 0, 0, 0]]), opacity=np.array([0.8]),
                  colors=np.array([[0.2, 0.8, 0.4]]), sh_degree=0)


def _frame():
    return {"intrinsics": {"width": 64, "height": 64, "fx": 64, "fy": 64, "cx": 32, "cy": 32},
            "camera_origin": [0.0, 0.0, -3.0], "look_at": [0.0, 0.0, 0.0], "up": [0.0, -1.0, 0.0]}


def test_ray_query_command_returns_payload(tmp_path):
    _make_carriers(tmp_path)
    out = _ray_query_command(Namespace(source=tmp_path, origin=[0, 0, 0], direction=[0, 0, 1],
                                       min_confidence=0.0, device="cpu"))
    payload = json.loads(out)
    assert payload["provenance"] == "carrier_query"
    assert abs(payload["depth"] - 2.0) < 1e-3
    assert payload["color"][1] > payload["color"][0]


def test_confidence_command_writes_field(tmp_path):
    _make_carriers(tmp_path)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"root": ".", "frames": [_frame()] * 4}))
    _confidence_command(Namespace(source=tmp_path, manifest=manifest, scale=1.0,
                                  saturate=12.0, device="cpu"))
    c = load_carriers(tmp_path, device="cpu")
    assert "confidence" in c and 0.0 <= float(c["confidence"][0]) <= 1.0


def test_export_splat_command_writes_glb(tmp_path):
    _make_carriers(tmp_path)
    out = _export_splat_command(Namespace(source=tmp_path, output=tmp_path / "x.glb"), load_carriers)
    assert out.exists() and out.stat().st_size > 0
