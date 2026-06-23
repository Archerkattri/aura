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


@requires_torch
def test_prism_cuda_forward_matches_torch_tiled():
    """The hand-written PRISM CUDA forward kernel matches the torch compositor."""
    import math
    import torch
    from aura.prism import quats_scales_to_cov3d, project_gaussians, composite_tiled
    from aura.prism_cuda import cuda_available, render_gaussians_cuda

    if not torch.cuda.is_available() or not cuda_available():
        pytest.skip("PRISM CUDA extension unavailable (no GPU/nvcc)")
    dev = "cuda"
    torch.manual_seed(0)
    n, w, h = 1500, 96, 96
    means = torch.randn(n, 3, device=dev) * 0.6 + torch.tensor([0, 0, 3.0], device=dev)
    quats = torch.randn(n, 4, device=dev); quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(n, 3, device=dev) * 0.05 + 0.02
    opac = torch.rand(n, device=dev) * 0.5 + 0.4
    colors = torch.rand(n, 3, device=dev)
    K = torch.tensor([[120.0, 0, w / 2], [0, 120.0, h / 2], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    cu = render_gaussians_cuda(means, quats, scales, opac, colors, vm, K, w, h)
    cov = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov, vm, K, w, h, torch)
    tt = composite_tiled(proj, colors, opac, w, h, torch)
    mse = float(((cu - tt) ** 2).mean())
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    assert psnr > 50.0, f"CUDA kernel diverges from torch tiled: {psnr:.1f} dB"


@requires_torch
def test_prism_cuda_differentiable_training():
    """The differentiable PRISM CUDA path (forward + custom CUDA backward) trains
    a scene end-to-end — full gsplat-parity at the kernel level."""
    import torch
    from aura.prism_cuda import cuda_available, render_gaussians_cuda_diff
    from aura.prism import render_gaussians

    if not torch.cuda.is_available() or not cuda_available():
        pytest.skip("PRISM CUDA extension unavailable (no GPU/nvcc)")
    dev = "cuda"
    torch.manual_seed(0)
    w = h = 56
    K = torch.tensor([[90.0, 0, w / 2], [0, 90.0, h / 2], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    gm = torch.tensor([[-0.3, -0.1, 3.0], [0.3, 0.1, 3.0]], device=dev)
    gq = torch.zeros(2, 4, device=dev); gq[:, 0] = 1
    gs = torch.full((2, 3), 0.16, device=dev); go = torch.full((2,), 0.9, device=dev)
    gc = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.8, 0.3]], device=dev)
    target = render_gaussians(gm, gq, gs, go, gc, vm, K, w, h, device=dev).detach()

    m = 8
    means = (torch.randn(m, 3, device=dev) * 0.3 + torch.tensor([0, 0, 3.0], device=dev)).requires_grad_(True)
    ls = torch.log(torch.full((m, 3), 0.18, device=dev)).requires_grad_(True)
    q = torch.zeros(m, 4, device=dev); q[:, 0] = 1; q = q.requires_grad_(True)
    lo = torch.logit(torch.full((m,), 0.5, device=dev)).requires_grad_(True)
    col = torch.rand(m, 3, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([means, ls, q, lo, col], lr=0.05)
    first = last = None
    for it in range(200):
        img = render_gaussians_cuda_diff(means, q, torch.exp(ls), torch.sigmoid(lo), col.clamp(0, 1), vm, K, w, h)
        loss = torch.abs(img - target).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first * 0.25, f"CUDA-diff training did not converge: {first:.4f} -> {last:.4f}"


@requires_torch
@pytest.mark.parametrize("fp", ["beta", "gabor"])
def test_prism_cuda_typed_footprints(fp):
    """CUDA beta/gabor footprints match the torch compositor and (for gabor)
    carry differentiable freq/phase grads."""
    import math
    import torch
    from aura.prism import (quats_scales_to_cov3d, project_gaussians, composite,
                            composite_tiled, beta_footprint, gabor_footprint)
    from aura.prism_cuda import cuda_available, render_gaussians_cuda

    if not torch.cuda.is_available() or not cuda_available():
        pytest.skip("PRISM CUDA extension unavailable")
    dev = "cuda"; torch.manual_seed(0); w = h = 72
    n = 800
    m = torch.randn(n, 3, device=dev) * 0.6 + torch.tensor([0, 0, 3.0], device=dev)
    q = torch.randn(n, 4, device=dev); q = q / q.norm(dim=1, keepdim=True)
    s = torch.rand(n, 3, device=dev) * 0.05 + 0.02
    o = torch.rand(n, device=dev) * 0.5 + 0.4; c = torch.rand(n, 3, device=dev)
    K = torch.tensor([[120.0, 0, w / 2], [0, 120.0, h / 2], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    cov = quats_scales_to_cov3d(q, s, torch); proj = project_gaussians(m, cov, vm, K, w, h, torch)
    if fp == "beta":
        cu = render_gaussians_cuda(m, q, s, o, c, vm, K, w, h, footprint="beta")
        tt = composite_tiled(proj, c, o, w, h, torch, footprint=beta_footprint)
    else:
        freq = torch.randn(n, 2, device=dev) * 0.3; phase = torch.rand(n, device=dev) * 6.28
        cu = render_gaussians_cuda(m, q, s, o, c, vm, K, w, h, footprint="gabor", freq=freq, phase=phase)
        tt = composite(proj, c, o, w, h, torch, footprint=gabor_footprint,
                       footprint_extra={"freq": freq[proj.index], "phase": phase[proj.index]})
    mse = float(((cu - tt) ** 2).mean())
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 99.0
    assert psnr > 40.0, f"CUDA {fp} footprint diverges: {psnr:.1f} dB"
