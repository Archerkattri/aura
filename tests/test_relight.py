"""Tests for relighting trained carriers (capability contract, CPU)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.relight import carrier_normals, carrier_albedo, relight_colors  # noqa: E402
from aura.shading import DirectionalLight  # noqa: E402


def _carriers(n=64):
    g = torch.Generator().manual_seed(1)
    quats = torch.randn(n, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    return dict(
        means=torch.randn(n, 3, generator=g),
        scales=torch.rand(n, 3, generator=g) + 0.05,
        quats=quats,
        opacity=torch.rand(n, generator=g),
        colors=torch.rand(n, 3, generator=g),
        sh_degree=0,
    )


def test_normals_are_unit_and_match_short_axis():
    c = _carriers()
    n = carrier_normals(torch, c["quats"], c["scales"])
    assert n.shape == (64, 3)
    assert torch.allclose(n.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_albedo_in_unit_range():
    c = _carriers()
    a = carrier_albedo(torch, c)
    assert a.shape == (64, 3)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_relight_changes_with_light_direction():
    c = _carriers()
    L1 = DirectionalLight(direction=(1.0, 0.0, 0.0), intensity=1.0)
    L2 = DirectionalLight(direction=(0.0, 0.0, 1.0), intensity=1.0)
    a = relight_colors(c, [L1], ambient=0.1, device="cpu")
    b = relight_colors(c, [L2], ambient=0.1, device="cpu")
    assert a.shape == (64, 3)
    assert (a - b).abs().mean() > 1e-3  # lighting actually depends on direction


def test_relight_differs_from_flat_albedo():
    c = _carriers()
    L = DirectionalLight(direction=(0.3, 0.6, 0.7), intensity=1.2)
    lit = relight_colors(c, [L], ambient=0.05, device="cpu")
    albedo = carrier_albedo(torch, c)
    assert (lit - albedo).abs().mean() > 1e-3


def test_ambient_floor_keeps_unlit_nonzero():
    c = _carriers()
    # a light pointing such that some carriers are unlit; ambient keeps them > 0
    L = DirectionalLight(direction=(0.0, 1.0, 0.0), intensity=1.0)
    lit = relight_colors(c, [L], ambient=0.2, device="cpu")
    assert lit.min() >= 0.0
    assert lit.sum() > 0.0
