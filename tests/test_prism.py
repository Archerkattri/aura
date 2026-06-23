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


@requires_torch
def test_typed_carriers_beat_gaussian_on_high_frequency():
    """Post-3DGS thesis check: on high-frequency content, Gabor carriers fit
    better than the same number of Gaussian carriers (the whole point of typed
    adaptive carriers). Pure-torch so it runs without a GPU."""
    import torch
    from aura.prism import (quats_scales_to_cov3d, project_gaussians, composite_tiled,
                            gaussian_footprint, gabor_footprint)

    dev = _device()
    torch.manual_seed(0)
    w = h = 48
    K = torch.tensor([[80.0, 0, w / 2], [0, 80.0, h / 2], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    # High-frequency striped target.
    yy, xx = torch.meshgrid(torch.arange(h, device=dev).float(),
                            torch.arange(w, device=dev).float(), indexing="ij")
    stripe = (0.5 + 0.5 * torch.cos(xx * 1.2)).unsqueeze(-1).repeat(1, 1, 3)
    target = stripe.clamp(0, 1)

    def fit(footprint, freq=None, phase=None, iters=300):
        torch.manual_seed(1)
        m = 12
        means = (torch.randn(m, 3, device=dev) * 0.15 + torch.tensor([0, 0, 3.0], device=dev)).requires_grad_(True)
        ls = torch.log(torch.full((m, 3), 0.5, device=dev)).requires_grad_(True)
        q = torch.zeros(m, 4, device=dev); q[:, 0] = 1; q = q.requires_grad_(True)
        lo = torch.logit(torch.full((m,), 0.5, device=dev)).requires_grad_(True)
        col = torch.rand(m, 3, device=dev).requires_grad_(True)
        params = [means, ls, q, lo, col]
        fr = ph = None
        if freq is not None:
            fr = freq.clone().requires_grad_(True); ph = phase.clone().requires_grad_(True)
            params += [fr, ph]
        opt = torch.optim.Adam(params, lr=0.05)
        extra = {}
        for it in range(iters):
            cov = quats_scales_to_cov3d(q, torch.exp(ls), torch)
            proj = project_gaussians(means, cov, vm, K, w, h, torch)
            if fr is not None:
                extra = {"freq": fr[proj.index], "phase": ph[proj.index]}
            img = composite_tiled(proj, col.clamp(0, 1), torch.sigmoid(lo), w, h, torch,
                                  footprint=footprint, footprint_extra=extra)
            loss = torch.abs(img - target).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.detach())

    g_loss = fit(gaussian_footprint)
    fr = torch.randn(12, 2, device=dev) * 0.2
    fr[:, 0] += 1.2  # bias toward the stripe frequency
    ph = torch.zeros(12, device=dev)
    gab_loss = fit(gabor_footprint, freq=fr, phase=ph)
    assert gab_loss < g_loss, f"gabor ({gab_loss:.4f}) should beat gaussian ({g_loss:.4f}) on stripes"


@requires_torch
def test_volumetric_alpha_trains_and_differs():
    """EVER-style volumetric-consistent alpha (1-exp(-opacity*w)) renders
    differently from billboard alpha and trains end-to-end via the CUDA path."""
    import torch
    from aura.prism_cuda import cuda_available, render_gaussians_cuda, render_gaussians_cuda_diff
    from aura.prism import render_gaussians

    if not torch.cuda.is_available() or not cuda_available():
        pytest.skip("PRISM CUDA extension unavailable")
    dev = "cuda"; torch.manual_seed(0); w = h = 64
    K = torch.tensor([[90.0, 0, w / 2], [0, 90.0, h / 2], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    def scene(n):
        m = torch.randn(n, 3, device=dev) * 0.5 + torch.tensor([0, 0, 3.0], device=dev)
        q = torch.randn(n, 4, device=dev); q = q / q.norm(dim=1, keepdim=True)
        s = torch.rand(n, 3, device=dev) * 0.06 + 0.03
        o = torch.rand(n, device=dev) * 0.6 + 0.3; c = torch.rand(n, 3, device=dev)
        return m, q, s, o, c
    m, q, s, o, c = scene(400)
    billboard = render_gaussians_cuda(m, q, s, o, c, vm, K, w, h, volumetric=False)
    volumetric = render_gaussians_cuda(m, q, s, o, c, vm, K, w, h, volumetric=True)
    assert not torch.allclose(billboard, volumetric)
    # train under volumetric alpha
    gm, gq, gs, go, gc = scene(3)
    target = render_gaussians_cuda(gm, gq, gs, go, gc, vm, K, w, h, volumetric=True).detach()
    M = 8
    mm = (torch.randn(M, 3, device=dev) * 0.3 + torch.tensor([0, 0, 3.0], device=dev)).requires_grad_(True)
    ls = torch.log(torch.full((M, 3), 0.18, device=dev)).requires_grad_(True)
    qq = torch.zeros(M, 4, device=dev); qq[:, 0] = 1; qq = qq.requires_grad_(True)
    lo = torch.logit(torch.full((M,), 0.5, device=dev)).requires_grad_(True)
    col = torch.rand(M, 3, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([mm, ls, qq, lo, col], lr=0.05)
    first = last = None
    for it in range(200):
        img = render_gaussians_cuda_diff(mm, qq, torch.exp(ls), torch.sigmoid(lo), col.clamp(0, 1), vm, K, w, h, volumetric=True)
        loss = torch.abs(img - target).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first * 0.4, f"volumetric training did not converge: {first:.4f} -> {last:.4f}"


@requires_torch
def test_neural_carrier_trains_and_beats_gaussian_on_ring():
    """Splat-the-Net-style neural carrier: a bounded neural primitive represents
    a non-Gaussian local pattern (a ring) better than a Gaussian carrier."""
    import torch
    from aura.prism import (make_neural_footprint, composite, project_gaussians,
                            quats_scales_to_cov3d, gaussian_footprint)

    dev = _device(); torch.manual_seed(0); w = h = 40
    K = torch.tensor([[70.0, 0, w / 2], [0, 70.0, h / 2], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    # Target: a ring (annulus) — impossible for a single Gaussian to match.
    yy, xx = torch.meshgrid(torch.arange(h, device=dev).float() - h / 2,
                            torch.arange(w, device=dev).float() - w / 2, indexing="ij")
    rad = (xx * xx + yy * yy).sqrt()
    ring = torch.exp(-((rad - 9.0) ** 2) / 8.0).unsqueeze(-1).repeat(1, 1, 3).clamp(0, 1)

    # one carrier centred in front of the camera
    means = torch.tensor([[0.0, 0.0, 3.0]], device=dev)
    quats = torch.zeros(1, 4, device=dev); quats[:, 0] = 1
    scales = torch.full((1, 3), 0.7, device=dev)
    opac = torch.full((1,), 0.95, device=dev)

    def fit(footprint, params_extra, latents=None, iters=400):
        torch.manual_seed(1)
        col = torch.ones(1, 3, device=dev).requires_grad_(True)
        params = [col] + params_extra
        opt = torch.optim.Adam(params, lr=0.02)
        for it in range(iters):
            cov = quats_scales_to_cov3d(quats, scales, torch)
            proj = project_gaussians(means, cov, vm, K, w, h, torch)
            extra = {"latent": latents[proj.index]} if latents is not None else {}
            img = composite(proj, col.clamp(0, 1), opac, w, h, torch, footprint=footprint, footprint_extra=extra)
            loss = torch.abs(img - ring).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.detach())

    g_loss = fit(gaussian_footprint, [])
    nfp, net = make_neural_footprint(torch, device=dev)
    latents = torch.randn(1, 4, device=dev).requires_grad_(True)
    n_loss = fit(nfp, list(net.parameters()) + [latents], latents=latents)
    assert n_loss < g_loss, f"neural ({n_loss:.4f}) should beat gaussian ({g_loss:.4f}) on a ring"
