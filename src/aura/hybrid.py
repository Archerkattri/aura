"""Hybrid renderer — PRISM as a true extension of the gsplat engine.

The Gaussian rasterizer (gsplat) is fast and high-quality and now even ray-traces;
there is no point re-implementing it. So AURA routes work by carrier type:

  * **Gaussian** carriers  → gsplat  (the engine: speed + quality)
  * **non-Gaussian** carriers (Beta / Gabor / neural) → PRISM  (the typed footprints
    gsplat cannot express)

and depth-composites the two layers into one image. A scene that is all Gaussians
renders *exactly* as gsplat (zero overhead); a scene that mixes types gets gsplat
quality on the Gaussian bulk plus PRISM's typed carriers where they help — the
extension, not a replacement.
"""
from __future__ import annotations

FOOTPRINT_CODES = {"gaussian": 0, "beta": 1, "gabor": 2, "neural": 3}


def _prism_layer(means, quats, scales, opacities, colors, ftypes, freq, phase,
                 viewmat, K, width, height, torch):
    """Front-to-back composite of the (non-Gaussian) PRISM carriers, returning
    (rgb [H,W,3], alpha [H,W], depth [H,W]) so the layer can be merged with gsplat."""
    from .prism import (project_gaussians, quats_scales_to_cov3d,
                        gaussian_footprint, beta_footprint, gabor_footprint)
    dev = means.device
    cov = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov, viewmat, K, width, height, torch)
    rgb = torch.zeros((height, width, 3), device=dev)
    T = torch.ones((height, width), device=dev)
    depth = torch.zeros((height, width), device=dev)
    M = proj.index.shape[0]
    if M == 0:
        return rgb, 1.0 - T, depth
    ys = torch.arange(height, device=dev, dtype=torch.float32)
    xs = torch.arange(width, device=dev, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    sub_c = colors[proj.index]; sub_o = opacities[proj.index]
    sub_ft = ftypes[proj.index]; sub_fr = freq[proj.index]; sub_ph = phase[proj.index]
    order = torch.argsort(proj.depths)
    for m in order.tolist():
        dx = gx - proj.means2d[m, 0]; dy = gy - proj.means2d[m, 1]
        conic = proj.conics[m]; code = int(sub_ft[m])
        if code == FOOTPRINT_CODES["gabor"]:
            w = gabor_footprint(dx, dy, conic, torch, freq=sub_fr[m], phase=sub_ph[m])
        elif code == FOOTPRINT_CODES["beta"]:
            w = beta_footprint(dx, dy, conic, torch, beta=2.0)
        else:
            w = gaussian_footprint(dx, dy, conic, torch)
        alpha = (sub_o[m] * w).clamp(0.0, 0.999)
        contrib = T * alpha
        rgb = rgb + contrib.unsqueeze(-1) * sub_c[m].view(1, 1, 3)
        depth = depth + contrib * proj.depths[m]
        T = T * (1.0 - alpha)
    return rgb, 1.0 - T, depth


def render_hybrid(means, quats, scales, opacities, colors, ftypes, viewmat, K,
                  width, height, *, freq=None, phase=None, sh_degree=None, device="cuda"):
    """Render a mixed-carrier scene: gsplat for Gaussian carriers, PRISM for the
    rest, depth-composited. ``ftypes`` is a per-carrier int (0=gaussian, 1=beta,
    2=gabor, 3=neural). Returns rgb [H,W,3]. All-Gaussian → identical to gsplat."""
    import torch
    from gsplat import rasterization

    ftypes = ftypes.to(device)
    is_g = ftypes == FOOTPRINT_CODES["gaussian"]
    vm = viewmat.unsqueeze(0) if viewmat.dim() == 2 else viewmat
    Ks = K.unsqueeze(0) if K.dim() == 2 else K
    n = means.shape[0]
    if freq is None:
        freq = torch.zeros(n, 2, device=device)
    if phase is None:
        phase = torch.zeros(n, device=device)

    # --- Gaussian layer via gsplat (RGB + expected depth + alpha) ---
    g = torch.nonzero(is_g, as_tuple=False).squeeze(-1)
    rgb_g = torch.zeros((height, width, 3), device=device)
    a_g = torch.zeros((height, width), device=device)
    d_g = torch.full((height, width), 1e9, device=device)
    if g.numel() > 0:
        out, alphas, _ = rasterization(
            means=means[g], quats=quats[g], scales=scales[g], opacities=opacities[g],
            colors=colors[g], viewmats=vm, Ks=Ks, width=width, height=height,
            sh_degree=sh_degree, render_mode="RGB+ED")
        rgb_g = out[0, ..., :3]; d_g = out[0, ..., 3].clamp(min=1e-6)
        a_g = alphas[0, ..., 0]

    # --- non-Gaussian layer via PRISM ---
    p = torch.nonzero(~is_g, as_tuple=False).squeeze(-1)
    if p.numel() == 0:
        return rgb_g                       # pure-Gaussian scene == gsplat exactly
    p_colors = colors[p]
    if p_colors.dim() == 3:                # SH → use DC term as flat colour for PRISM
        p_colors = (0.5 + 0.28209479177387814 * p_colors[:, 0, :]).clamp(0, 1)
    rgb_p, a_p, d_p = _prism_layer(means[p], quats[p], scales[p], opacities[p],
                                   p_colors, ftypes[p], freq[p], phase[p],
                                   vm[0], Ks[0], width, height, torch)

    # --- depth-correct 2-layer over-composite (front layer wins per pixel) ---
    front_is_p = (d_p <= d_g).float().unsqueeze(-1)
    a_g3, a_p3 = a_g.unsqueeze(-1), a_p.unsqueeze(-1)
    # p in front:  p over g
    pg = rgb_p * a_p3 + rgb_g * a_g3 * (1 - a_p3)
    # g in front:  g over p
    gp = rgb_g * a_g3 + rgb_p * a_p3 * (1 - a_g3)
    return (front_is_p * pg + (1 - front_is_p) * gp).clamp(0, 1)
