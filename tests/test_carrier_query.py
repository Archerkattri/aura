"""Tests for the unified ray query over trained carriers (CPU)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.carrier_query import carrier_ray_query  # noqa: E402


def _one_carrier_at(p, color=(0.2, 0.8, 0.4), conf=0.9):
    return {
        "means": torch.tensor([p], dtype=torch.float32),
        "scales": torch.tensor([[0.3, 0.3, 0.05]], dtype=torch.float32),
        "quats": torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float32),
        "opacity": torch.tensor([0.8], dtype=torch.float32),
        "colors": torch.tensor([color], dtype=torch.float32),
        "confidence": torch.tensor([conf], dtype=torch.float32),
        "sh_degree": 0,
    }


def test_hit_returns_full_payload():
    c = _one_carrier_at([0.0, 0.0, 2.0])
    r = carrier_ray_query(c, [0, 0, 0], [0, 0, 1], device="cpu")
    assert r.provenance == "carrier_query"
    assert r.depth is not None and abs(r.depth - 2.0) < 1e-3
    assert r.normal is not None and abs(sum(x * x for x in r.normal) - 1.0) < 1e-4
    assert 0.0 <= r.confidence <= 1.0 and r.confidence == pytest.approx(0.9, abs=1e-5)


def test_miss_when_ray_points_away():
    c = _one_carrier_at([0.0, 0.0, 2.0])
    r = carrier_ray_query(c, [0, 0, 0], [0, 0, -1], device="cpu")
    assert r.provenance == "miss" and r.confidence == 0.0


def test_nearest_carrier_wins():
    c = {
        "means": torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        "scales": torch.tensor([[0.3, 0.3, 0.05]] * 2, dtype=torch.float32),
        "quats": torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=torch.float32),
        "opacity": torch.tensor([0.8, 0.8], dtype=torch.float32),
        "colors": torch.tensor([[1.0, 0, 0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "confidence": torch.tensor([0.5, 0.9], dtype=torch.float32),
        "sh_degree": 0,
    }
    r = carrier_ray_query(c, [0, 0, 0], [0, 0, 1], device="cpu")
    assert abs(r.depth - 2.0) < 1e-3            # the nearer carrier (z=2)
    assert r.color[1] > r.color[0]              # green, not red


def test_normal_oriented_toward_ray():
    c = _one_carrier_at([0.0, 0.0, 2.0])
    r = carrier_ray_query(c, [0, 0, 0], [0, 0, 1], device="cpu")
    # normal should face back toward the camera (negative z component)
    assert r.normal[2] < 0
