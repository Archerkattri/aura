"""Tests for per-carrier multi-view confidence (CPU)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.confidence import multiview_confidence, attach_confidence  # noqa: E402
from aura import gltf_splat as G  # noqa: E402


def _frame(origin, look_at):
    return {
        "intrinsics": {"width": 100, "height": 100, "fx": 100, "fy": 100, "cx": 50, "cy": 50},
        "camera_origin": origin,
        "look_at": look_at,
        "up": [0.0, -1.0, 0.0],
    }


def _manifest(frames):
    return {"root": ".", "frames": frames}


def test_confidence_in_unit_range_and_monotone_in_views():
    # one carrier at origin, viewed by cameras looking at it from +Z increasingly
    c = {"means": torch.tensor([[0.0, 0.0, 0.0]])}
    one_view = _manifest([_frame([0, 0, -3], [0, 0, 0])])
    many_views = _manifest([_frame([0, 0, -3], [0, 0, 0])] * 10)
    a = multiview_confidence(c, one_view, scale=1.0, device="cpu")
    b = multiview_confidence(c, many_views, scale=1.0, device="cpu")
    assert 0.0 <= float(a) <= 1.0 and 0.0 <= float(b) <= 1.0
    assert float(b) > float(a)  # more observing views -> higher confidence


def test_unobserved_carrier_has_low_confidence():
    # carrier far behind every camera (out of frustum) -> ~0
    c = {"means": torch.tensor([[1000.0, 1000.0, 1000.0]])}
    m = _manifest([_frame([0, 0, -3], [0, 0, 0])] * 5)
    conf = multiview_confidence(c, m, scale=1.0, device="cpu")
    assert float(conf) < 0.1


def test_attach_and_export_confidence_attribute(tmp_path):
    c = {
        "means": torch.zeros(4, 3), "scales": torch.ones(4, 3),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(4, 1),
        "opacity": torch.full((4,), 0.5), "colors": torch.full((4, 3), 0.5),
        "sh_degree": 0,
    }
    m = _manifest([_frame([0, 0, -3], [0, 0, 0])] * 6)
    c = attach_confidence(c, m, scale=1.0, device="cpu")
    assert "confidence" in c and c["confidence"].shape == (4,)
    g, _ = G.build_splat_gltf(c)
    attrs = g["meshes"][0]["primitives"][0]["attributes"]
    assert "_AURA_CONFIDENCE" in attrs
    assert g["accessors"][attrs["_AURA_CONFIDENCE"]]["type"] == "SCALAR"
