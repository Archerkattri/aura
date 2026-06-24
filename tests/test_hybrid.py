"""Tests for the PRISM-extends-gsplat hybrid renderer."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("hybrid renderer needs CUDA (gsplat)", allow_module_level=True)
pytest.importorskip("gsplat")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.hybrid import extension_mask, render_hybrid, FOOTPRINT_CODES  # noqa: E402


def test_default_extension_mask_keeps_beta_in_quality_backend():
    ft = torch.tensor([
        FOOTPRINT_CODES["gaussian"],
        FOOTPRINT_CODES["beta"],
        FOOTPRINT_CODES["gabor"],
        FOOTPRINT_CODES["neural"],
    ], device="cuda")

    mask = extension_mask(ft)

    assert mask.tolist() == [False, False, True, True]


def _scene(n=40, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    means = torch.randn(n, 3, generator=g, device="cuda") * 0.3
    means[:, 2] += 3.0
    quats = torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n, 1)
    scales = (torch.rand(n, 3, generator=g, device="cuda") * 0.05 + 0.02)
    opac = torch.rand(n, generator=g, device="cuda") * 0.5 + 0.3
    colors = torch.rand(n, 3, generator=g, device="cuda")
    vm = torch.eye(4, device="cuda")
    K = torch.tensor([[64.0, 0, 32], [0, 64.0, 32], [0, 0, 1]], device="cuda")
    return means, quats, scales, opac, colors, vm, K


def test_all_gaussian_matches_gsplat():
    from gsplat import rasterization
    means, quats, scales, opac, colors, vm, K = _scene()
    ft = torch.zeros(means.shape[0], dtype=torch.long, device="cuda")
    hyb = render_hybrid(means, quats, scales, opac, colors, ft, vm, K, 64, 64, sh_degree=None)
    ref, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                              colors=colors, viewmats=vm.unsqueeze(0), Ks=K.unsqueeze(0),
                              width=64, height=64, render_mode="RGB+ED")
    ref = ref[0, ..., :3]
    assert torch.allclose(hyb, ref.clamp(0, 1), atol=1e-4), "all-Gaussian hybrid must equal gsplat"


def test_gabor_extensions_differ_and_compose():
    means, quats, scales, opac, colors, vm, K = _scene()
    ft_all_g = torch.zeros(means.shape[0], dtype=torch.long, device="cuda")
    base = render_hybrid(means, quats, scales, opac, colors, ft_all_g, vm, K, 64, 64, sh_degree=None)
    ft_mixed = ft_all_g.clone()
    ft_mixed[: means.shape[0] // 2] = FOOTPRINT_CODES["gabor"]   # Gabor becomes PRISM extension
    mixed = render_hybrid(means, quats, scales, opac, colors, ft_mixed, vm, K, 64, 64, sh_degree=None)
    assert mixed.shape == base.shape
    assert (mixed - base).abs().mean() > 1e-4, "routing half to PRISM must change the image"
    assert mixed.min() >= 0 and mixed.max() <= 1


def test_beta_is_not_prism_extension_by_default():
    means, quats, scales, opac, colors, vm, K = _scene()
    ft_all_g = torch.zeros(means.shape[0], dtype=torch.long, device="cuda")
    base = render_hybrid(means, quats, scales, opac, colors, ft_all_g, vm, K, 64, 64, sh_degree=None)
    ft_beta = torch.full((means.shape[0],), FOOTPRINT_CODES["beta"], dtype=torch.long, device="cuda")

    beta_primary = render_hybrid(means, quats, scales, opac, colors, ft_beta, vm, K, 64, 64, sh_degree=None)

    assert torch.allclose(beta_primary, base, atol=1e-4), "Beta is a quality-backend carrier, not a PRISM extension"


def test_pure_prism_layer_renders():
    means, quats, scales, opac, colors, vm, K = _scene()
    ft = torch.full((means.shape[0],), FOOTPRINT_CODES["gabor"], dtype=torch.long, device="cuda")
    freq = torch.rand(means.shape[0], 2, device="cuda") * 0.3 + 0.1
    phase = torch.rand(means.shape[0], device="cuda")
    img = render_hybrid(means, quats, scales, opac, colors, ft, vm, K, 64, 64, freq=freq, phase=phase)
    assert img.shape == (64, 64, 3) and img.sum() > 0
