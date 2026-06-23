"""Tests for the AURA-native differentiable rasterizer (prism.py).

The rasterizer is the post-3DGS substrate: a differentiable GPU alpha
compositor with a pluggable per-carrier footprint, so non-Gaussian carriers
(Beta/Gabor) train with gradients. These tests need torch and run on CPU when no
GPU is present (small scenes); the gsplat-parity test additionally needs gsplat.
"""

import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_GSPLAT = importlib.util.find_spec("gsplat") is not None
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_gsplat = pytest.mark.skipif(not _HAS_GSPLAT, reason="gsplat not installed")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@requires_torch
def test_gradients_flow_through_rasterizer():
    import torch
    from aura.prism import render_gaussians

    dev = _device()
    torch.manual_seed(0)
    n, w, h = 20, 32, 32
    means = (torch.randn(n, 3, device=dev) * 0.4 + torch.tensor([0, 0, 3.0], device=dev)).requires_grad_(True)
    quats = torch.zeros(n, 4, device=dev); quats[:, 0] = 1
    scales = torch.full((n, 3), 0.1, device=dev)
    opac = torch.full((n,), 0.7, device=dev)
    colors = torch.rand(n, 3, device=dev)
    K = torch.tensor([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1.0]], device=dev)
    viewmat = torch.eye(4, device=dev)
    img = render_gaussians(means, quats, scales, opac, colors, viewmat, K, w, h, device=dev)
    assert img.shape == (h, w, 3)
    img.sum().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()


@requires_torch
@requires_gsplat
def test_matches_gsplat_on_gaussians():
    import math
    import torch
    from gsplat import rasterization
    from aura.prism import render_gaussians

    dev = _device()
    torch.manual_seed(0)
    n, w, h = 200, 80, 80
    means = torch.randn(n, 3, device=dev) * 0.6 + torch.tensor([0, 0, 3.0], device=dev)
    quats = torch.randn(n, 4, device=dev); quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(n, 3, device=dev) * 0.08 + 0.02
    opac = torch.rand(n, device=dev) * 0.5 + 0.4
    colors = torch.rand(n, 3, device=dev)
    K = torch.tensor([[120.0, 0, w / 2], [0, 120.0, h / 2], [0, 0, 1.0]], device=dev)
    viewmat = torch.eye(4, device=dev)
    g, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                            colors=colors, viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
                            width=w, height=h, packed=False)
    native = render_gaussians(means, quats, scales, opac, colors, viewmat, K, w, h, device=dev)
    mse = float(((g[0] - native) ** 2).mean())
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    # Different cutoff/tiling than gsplat, so not pixel-identical, but the
    # projection + Gaussian falloff math must agree closely.
    assert psnr > 25.0, f"native rasterizer diverges from gsplat: {psnr:.1f} dB"


@requires_torch
def test_beta_typed_carrier_trains():
    """The genuine post-3DGS check: a NON-Gaussian (Beta, bounded-polynomial)
    carrier optimised end-to-end with gradients through the native rasterizer."""
    import torch
    from aura.prism import (
        project_gaussians, quats_scales_to_cov3d, composite,
        beta_footprint, gaussian_footprint,
    )

    dev = _device()
    torch.manual_seed(1)
    w = h = 48
    K = torch.tensor([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1.0]], device=dev)
    viewmat = torch.eye(4, device=dev)

    def render(means, quats, scales, opac, colors, footprint):
        cov = quats_scales_to_cov3d(quats, scales, torch)
        proj = project_gaussians(means, cov, viewmat, K, w, h, torch)
        return composite(proj, colors, opac, w, h, torch, footprint=footprint)

    gt_means = torch.tensor([[-0.4, -0.2, 3.0], [0.4, 0.1, 3.0]], device=dev)
    gt_q = torch.zeros(2, 4, device=dev); gt_q[:, 0] = 1
    gt_s = torch.full((2, 3), 0.2, device=dev)
    gt_o = torch.full((2,), 0.9, device=dev)
    gt_c = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.8, 0.3]], device=dev)
    target = render(gt_means, gt_q, gt_s, gt_o, gt_c, gaussian_footprint).detach()

    m = 6
    means = (torch.randn(m, 3, device=dev) * 0.3 + torch.tensor([0, 0, 3.0], device=dev)).requires_grad_(True)
    logscale = torch.log(torch.full((m, 3), 0.2, device=dev)).requires_grad_(True)
    quats = torch.zeros(m, 4, device=dev); quats[:, 0] = 1; quats = quats.requires_grad_(True)
    logit_o = torch.logit(torch.full((m,), 0.5, device=dev)).requires_grad_(True)
    colors = torch.rand(m, 3, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([means, logscale, quats, logit_o, colors], lr=0.05)

    def beta(dx, dy, conic, t):
        return beta_footprint(dx, dy, conic, t, beta=2.0)

    first = last = None
    for it in range(200):
        cov = quats_scales_to_cov3d(quats, torch.exp(logscale), torch)
        proj = project_gaussians(means, cov, viewmat, K, w, h, torch)
        img = composite(proj, colors.clamp(0, 1), torch.sigmoid(logit_o), w, h, torch, footprint=beta)
        loss = torch.abs(img - target).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first * 0.5, f"beta carrier did not converge: {first:.4f} -> {last:.4f}"
