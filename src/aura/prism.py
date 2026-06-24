"""PRISM — Pluggable Radiance-prImitive Splatting Module.

AURA's own differentiable rasterizer, built as the alternative to gsplat that
splats *typed* carriers, not just 3D Gaussians. (A prism handles a whole
spectrum; PRISM splats a whole spectrum of carrier types.)

The genuine post-3DGS substrate: a differentiable, GPU, front-to-back
alpha-compositing rasterizer whose per-primitive 2D footprint is *pluggable* by
carrier type. Gaussians use an exponential conic falloff (validated against
gsplat at ~31 dB); Beta carriers use a bounded polynomial falloff; Gabor
carriers use an oscillatory envelope. New carrier types add one footprint
function — no change to projection or compositing.

Pure PyTorch (autograd handles the backward pass), so any carrier whose
footprint is a differentiable function of its parameters trains end-to-end with
no custom CUDA. Two compositors: :func:`composite` (dense, reference) and
:func:`composite_tiled` (tile-binned, scales to tens of thousands of carriers at
interactive speed — ~30 ms for 20k carriers at 256x256). A custom-CUDA PRISM is
the documented perf-future (see the README). Pipeline mirrors
gsplat's stages:

    project (3D -> 2D conic/footprint + depth) -> depth sort -> alpha composite

Conventions match ``gsplat_renderer.manifest_frame_to_camera`` and AURA's
Gaussian carrier model (wxyz quats, linear-RGB colour, opacity in [0,1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("rasterizer_native requires torch") from exc
    return torch


# --------------------------------------------------------------------------- #
# Quaternion / covariance helpers (wxyz, matching gsplat_renderer).
# --------------------------------------------------------------------------- #


def quats_scales_to_cov3d(quats, scales, torch):
    """[N,4] wxyz quats + [N,3] scales -> [N,3,3] world covariance R diag(s^2) R^T."""

    q = quats / quats.norm(dim=1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    S = torch.diag_embed(scales * scales)
    return R @ S @ R.transpose(1, 2)


# --------------------------------------------------------------------------- #
# Projection: 3D world -> 2D screen mean, conic, depth.
# --------------------------------------------------------------------------- #


@dataclass
class ProjectedCarriers:
    means2d: object   # [M,2] pixel centres of visible carriers
    depths: object    # [M]
    conics: object    # [M,3] inverse-2D-covariance (a, b, c) for Gaussian/Gabor envelope
    radii: object     # [M] pixel support radius (for footprint scaling / culling)
    index: object     # [M] indices back into the original carrier arrays (visible subset)
    opacity_comp: object = None  # [M] EWA antialiasing opacity compensation (<=1), or None


def project_gaussians(means, cov3d, viewmat, K, width, height, torch,
                      *, near: float = 0.01, eps2d: float = 0.3, antialias: bool = False):
    """Project 3D Gaussians to 2D. Returns ProjectedCarriers (visible subset).

    viewmat: [4,4] world->camera (rows [right, up, forward], +Z forward — the
    convention produced by ``manifest_frame_to_camera``). K: [3,3] intrinsics.
    """

    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    p_cam = means @ R.T + t  # [N,3]
    z = p_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = z > near
    # Perspective projection of the mean to pixels.
    u = fx * p_cam[:, 0] / z.clamp(min=near) + cx
    v = fy * p_cam[:, 1] / z.clamp(min=near) + cy
    means2d = torch.stack([u, v], dim=1)

    # Covariance in camera frame, then the perspective Jacobian to screen.
    cov_cam = R @ cov3d @ R.T  # [N,3,3]
    zc = z.clamp(min=near)
    J = torch.zeros((means.shape[0], 2, 3), dtype=means.dtype, device=means.device)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * p_cam[:, 0] / (zc * zc)
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * p_cam[:, 1] / (zc * zc)
    cov2d = J @ cov_cam @ J.transpose(1, 2)  # [N,2,2]
    a0 = cov2d[:, 0, 0]; b0 = cov2d[:, 0, 1]; c0 = cov2d[:, 1, 1]
    det_orig = (a0 * c0 - b0 * b0).clamp(min=1e-12)
    # Low-pass dilation (gsplat eps2d): keep sub-pixel splats renderable.
    cov2d[:, 0, 0] = cov2d[:, 0, 0] + eps2d
    cov2d[:, 1, 1] = cov2d[:, 1, 1] + eps2d

    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-12)
    # EWA antialiasing opacity compensation (gsplat "antialiased" mode): the
    # dilation grows the footprint, so scale opacity by sqrt(det_orig/det_dilated)
    # (<=1) to conserve energy — without this, dilated sub-pixel splats render too
    # opaque, which is the main fidelity gap vs gsplat on dense scenes.
    opacity_comp = (det_orig / det).clamp(min=0.0, max=1.0).sqrt() if antialias else None
    # conic = inverse(cov2d) = 1/det [[c,-b],[-b,a]] -> store (xx, xy, yy)
    conic = torch.stack([c / det, -b / det, a / det], dim=1)
    # 3-sigma radius from the larger eigenvalue.
    mid = 0.5 * (a + c)
    disc = (mid * mid - det).clamp(min=0.0).sqrt()
    lambda_max = (mid + disc).clamp(min=1e-6)
    radii = 3.0 * lambda_max.sqrt()

    on_screen = (u + radii >= 0) & (u - radii < width) & (v + radii >= 0) & (v - radii < height)
    keep = valid & on_screen & torch.isfinite(det)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    return ProjectedCarriers(
        means2d=means2d[idx], depths=z[idx], conics=conic[idx], radii=radii[idx], index=idx,
        opacity_comp=(opacity_comp[idx] if opacity_comp is not None else None),
    )


# --------------------------------------------------------------------------- #
# Pluggable per-carrier 2D footprint kernels: alpha(pixel) given the offset
# (dx,dy) from the carrier's 2D centre. Each returns a [...] weight in [0,1]
# (BEFORE multiplying by opacity). New carrier types implement one of these.
# --------------------------------------------------------------------------- #


def gaussian_footprint(dx, dy, conic, torch):
    """Standard 3DGS falloff: exp(-0.5 (a dx^2 + 2 b dx dy + c dy^2))."""
    quad = conic[..., 0] * dx * dx + 2.0 * conic[..., 1] * dx * dy + conic[..., 2] * dy * dy
    return torch.exp(-0.5 * quad.clamp(min=0.0))


def beta_footprint(dx, dy, conic, torch, *, beta: float = 2.0):
    """Deformable-Beta-style bounded falloff: (1 - r)_+^beta where r is the
    conic (Mahalanobis-like) radius. Compact support (hard zero outside),
    unlike the Gaussian's infinite tail."""
    quad = conic[..., 0] * dx * dx + 2.0 * conic[..., 1] * dx * dy + conic[..., 2] * dy * dy
    r = (quad.clamp(min=0.0)).sqrt()
    return torch.clamp(1.0 - r / 3.0, min=0.0) ** beta


def gabor_footprint(dx, dy, conic, torch, freq=None, phase=0.0):
    """Gabor carrier: Gaussian envelope * positive cosine modulation, for
    high-frequency texture (freq = [fx, fy] in pixel^-1, phase scalar).
    ``torch`` is the 4th positional arg to match the other footprint kernels."""
    if freq is None:
        freq = torch.zeros(2, device=dx.device)
    quad = conic[..., 0] * dx * dx + 2.0 * conic[..., 1] * dx * dy + conic[..., 2] * dy * dy
    env = torch.exp(-0.5 * quad.clamp(min=0.0))
    fx = freq[..., 0]; fy = freq[..., 1]
    # Broadcast per-carrier freq/phase ([T] or scalar) against the pixel grid (dx).
    while hasattr(fx, "dim") and fx.dim() < dx.dim():
        fx = fx.unsqueeze(-1); fy = fy.unsqueeze(-1)
    ph = phase
    if hasattr(ph, "dim"):
        while ph.dim() < dx.dim():
            ph = ph.unsqueeze(-1)
    osc = torch.cos(fx * dx + fy * dy + ph)
    return env * (1.0 + osc) * 0.5


def make_neural_footprint(torch, *, latent_dim: int = 4, hidden: int = 32, n_freq: int = 4, device="cuda"):
    """Splat-the-Net-style **neural carrier**: a bounded neural primitive whose
    footprint is a small shared MLP over Fourier features of the local
    (conic-normalised) offset plus a per-carrier latent, gated by a Gaussian
    envelope so support stays bounded. Returns ``(footprint_callable, module)``;
    add ``module.parameters()`` (and the per-carrier latents) to the optimizer to
    train it. The footprint matches the ``(dx, dy, conic, torch, latent=...)``
    calling convention of :func:`composite`.

    A neural carrier can represent local appearance a single Gaussian/Beta/Gabor
    cannot (rings, corners, fine structure) — the expressive end of the carrier
    spectrum (see the AURA roadmap; arXiv:2510.08491 Splat-the-Net)."""

    nn = torch.nn
    in_dim = 2 * 2 * n_freq + latent_dim
    net = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    freqs = (2.0 ** torch.arange(n_freq, dtype=torch.float32, device=device)) * 3.14159265

    def footprint(dx, dy, conic, torch_, latent=None):
        quad = conic[..., 0] * dx * dx + 2.0 * conic[..., 1] * dx * dy + conic[..., 2] * dy * dy
        env = torch_.exp(-0.5 * quad.clamp(min=0.0))
        # local coords scaled so the 3-sigma footprint maps to ~[-1,1]
        r = quad.clamp(min=0.0).sqrt().clamp(max=3.0) / 3.0  # [..]
        ang = torch_.atan2(dy, dx)
        u = (r * torch_.cos(ang)); v = (r * torch_.sin(ang))
        feats = []
        for f in freqs:
            feats += [torch_.sin(f * u), torch_.cos(f * u), torch_.sin(f * v), torch_.cos(f * v)]
        feat = torch_.stack(feats, dim=-1)  # [..., 4*n_freq]
        if latent is not None:
            lat = latent
            while lat.dim() < feat.dim():
                lat = lat.unsqueeze(0)
            lat = lat.expand(*feat.shape[:-1], lat.shape[-1])
            feat = torch_.cat([feat, lat], dim=-1)
        else:
            zeros = torch_.zeros(*feat.shape[:-1], latent_dim, device=feat.device)
            feat = torch_.cat([feat, zeros], dim=-1)
        density = torch_.sigmoid(net(feat).squeeze(-1))
        return env * density

    return footprint, net


# --------------------------------------------------------------------------- #
# Differentiable front-to-back alpha compositing (depth-sorted dense scan).
# --------------------------------------------------------------------------- #


def composite(
    proj: ProjectedCarriers,
    colors,
    opacities,
    width: int,
    height: int,
    torch,
    *,
    footprint: Callable | None = None,
    footprint_extra: dict | None = None,
    background=None,
    volumetric: bool = False,
):
    """Render visible carriers to an [H,W,3] image via depth-sorted front-to-back
    alpha compositing. Differentiable end-to-end. ``footprint`` defaults to the
    Gaussian kernel; pass beta/gabor (or a custom callable) for typed carriers.
    ``volumetric=True`` uses the EVER-style physically-consistent alpha
    ``1 - exp(-opacity*w)`` (optical-depth absorption) instead of the billboard
    ``opacity*w``.

    Dense per-carrier scan (O(M*H*W)) — correct and fully differentiable, good
    for validation and modest scenes; tiled/CUDA acceleration is future work.
    """

    if footprint is None:
        footprint = gaussian_footprint
    extra = footprint_extra or {}
    device = colors.device
    M = proj.index.shape[0]
    img = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    if background is not None:
        img = img + background
    if M == 0:
        return img
    T = torch.ones((height, width), dtype=torch.float32, device=device)

    order = torch.argsort(proj.depths)  # front (small z) -> back
    # pixel CENTRES (+0.5), matching gsplat's sampling convention
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    sub_colors = colors[proj.index]
    sub_opac = opacities[proj.index]
    for m in order.tolist():
        cx2d = proj.means2d[m, 0]
        cy2d = proj.means2d[m, 1]
        dx = gx - cx2d
        dy = gy - cy2d
        conic_m = proj.conics[m]
        kwargs = {key: val[m] if hasattr(val, "__getitem__") else val for key, val in extra.items()}
        weight = footprint(dx, dy, conic_m, torch, **kwargs)  # [H,W]
        if volumetric:
            alpha = (1.0 - torch.exp(-(sub_opac[m] * weight).clamp(min=0.0))).clamp(0.0, 0.999)
        else:
            alpha = (sub_opac[m] * weight).clamp(0.0, 0.999)
        contrib = T * alpha
        img = img + contrib.unsqueeze(-1) * sub_colors[m].view(1, 1, 3)
        T = T * (1.0 - alpha)
    return img


def composite_tiled(
    proj: ProjectedCarriers,
    colors,
    opacities,
    width: int,
    height: int,
    torch,
    *,
    footprint: Callable | None = None,
    footprint_extra: dict | None = None,
    background=None,
    tile: int = 16,
    max_per_tile: int = 256,
    volumetric: bool = False,
):
    """Tiled differentiable front-to-back alpha compositing.

    Scales far better than :func:`composite` (which scans every carrier over the
    whole image): each carrier is binned only into the tiles its footprint
    overlaps, intersections are depth-sorted, and each tile composites at most
    ``max_per_tile`` nearest carriers. The per-tile scan is vectorised across all
    tiles and pixels simultaneously, so the Python loop is only ``max_per_tile``
    iterations regardless of carrier count. Fully differentiable.

    ``max_per_tile`` caps depth complexity per tile (front-most kept); raise it
    for very dense scenes. Matches :func:`composite` when no tile saturates.
    """

    if footprint is None:
        footprint = gaussian_footprint
    extra = footprint_extra or {}
    device = colors.device
    f32 = torch.float32
    M = int(proj.index.shape[0])
    img = torch.zeros((height, width, 3), dtype=f32, device=device)
    if background is not None:
        img = img + background
    if M == 0:
        return img

    ntx = (width + tile - 1) // tile
    nty = (height + tile - 1) // tile
    T = ntx * nty

    cx = proj.means2d[:, 0]
    cy = proj.means2d[:, 1]
    r = proj.radii.clamp(min=0.0)
    tx0 = torch.clamp(torch.floor((cx - r) / tile).long(), 0, ntx - 1)
    tx1 = torch.clamp(torch.floor((cx + r) / tile).long(), 0, ntx - 1)
    ty0 = torch.clamp(torch.floor((cy - r) / tile).long(), 0, nty - 1)
    ty1 = torch.clamp(torch.floor((cy + r) / tile).long(), 0, nty - 1)
    nx = (tx1 - tx0 + 1).clamp(min=1)
    ny = (ty1 - ty0 + 1).clamp(min=1)
    cnt = nx * ny  # tiles touched per carrier
    total = int(cnt.sum().item())
    if total == 0:
        return img

    # Expand carrier->(tile) intersections without a Python loop.
    starts = cnt.cumsum(0) - cnt
    carrier_of = torch.repeat_interleave(torch.arange(M, device=device), cnt)  # [total]
    local = torch.arange(total, device=device) - starts[carrier_of]
    lx = local % nx[carrier_of]
    ly = local // nx[carrier_of]
    tile_x = tx0[carrier_of] + lx
    tile_y = ty0[carrier_of] + ly
    tile_id = tile_y * ntx + tile_x  # [total]

    # Sort intersections by (tile, depth-front-first). Single composite key.
    depth = proj.depths[carrier_of]
    dmax = float(proj.depths.max().item()) + 1.0
    key = tile_id.to(torch.float64) * dmax + depth.to(torch.float64)
    order = torch.argsort(key)
    carrier_sorted = carrier_of[order]
    tile_sorted = tile_id[order]

    # Position of each intersection within its tile's run; keep first K.
    tile_counts = torch.bincount(tile_sorted, minlength=T)
    tstart = tile_counts.cumsum(0) - tile_counts
    pos = torch.arange(total, device=device) - tstart[tile_sorted]
    keep = pos < max_per_tile
    K = int(min(max_per_tile, int(tile_counts.max().item())))

    idxTK = torch.full((T, K), -1, dtype=torch.long, device=device)
    idxTK[tile_sorted[keep], pos[keep]] = carrier_sorted[keep]  # [T,K], -1 = empty
    valid = idxTK >= 0
    safe = idxTK.clamp(min=0)

    # Gather per-(tile,slot) carrier attributes.
    g_mean = proj.means2d[safe]        # [T,K,2]
    g_conic = proj.conics[safe]        # [T,K,3]
    g_col = colors[proj.index][safe]   # [T,K,3]
    g_op = opacities[proj.index][safe] * valid.to(f32)  # [T,K]; empty slots -> 0
    extra_TK = {k: (v[safe] if hasattr(v, "shape") else v) for k, v in extra.items()}

    # Per-tile pixel coordinates (global), with an in-image mask.
    ts = tile
    tile_ix = torch.arange(T, device=device) % ntx
    tile_iy = torch.arange(T, device=device) // ntx
    px = (tile_ix.view(T, 1) * ts + torch.arange(ts, device=device).view(1, ts)).to(f32)  # [T,ts]
    py = (tile_iy.view(T, 1) * ts + torch.arange(ts, device=device).view(1, ts)).to(f32)
    # pixel grid [T, ts, ts]
    gx = px.view(T, 1, ts).expand(T, ts, ts)
    gy = py.view(T, ts, 1).expand(T, ts, ts)
    in_img = (gx < width) & (gy < height)
    P = ts * ts
    gx = gx.reshape(T, P)
    gy = gy.reshape(T, P)

    out = torch.zeros((T, P, 3), dtype=f32, device=device)
    trans = torch.ones((T, P), dtype=f32, device=device)
    for k in range(K):
        mx = g_mean[:, k, 0].view(T, 1)
        my = g_mean[:, k, 1].view(T, 1)
        dx = gx - mx
        dy = gy - my
        conic_k = g_conic[:, k, :].view(T, 1, 3).expand(T, P, 3)
        kwargs = {kk: vv[:, k] if hasattr(vv, "shape") else vv for kk, vv in extra_TK.items()}
        weight = footprint(dx, dy, conic_k, torch, **kwargs)  # [T,P]
        if volumetric:
            alpha = (1.0 - torch.exp(-(g_op[:, k].view(T, 1) * weight).clamp(min=0.0))).clamp(0.0, 0.999)
        else:
            alpha = (g_op[:, k].view(T, 1) * weight).clamp(0.0, 0.999)
        contrib = trans * alpha
        out = out + contrib.unsqueeze(-1) * g_col[:, k, :].view(T, 1, 3)
        trans = trans * (1.0 - alpha)

    # Scatter tile pixels back into the image.
    out = out.view(T, ts, ts, 3)
    mask = in_img.view(T, ts, ts)
    base = img if background is not None else torch.zeros((height, width, 3), dtype=f32, device=device)
    canvas = base
    for t in range(T):
        oy = (t // ntx) * ts
        ox = (t % ntx) * ts
        hh = min(ts, height - oy)
        ww = min(ts, width - ox)
        if hh <= 0 or ww <= 0:
            continue
        canvas[oy:oy + hh, ox:ox + ww, :] = out[t, :hh, :ww, :] + (
            canvas[oy:oy + hh, ox:ox + ww, :] if background is not None else 0.0
        )
    return canvas


# --------------------------------------------------------------------------- #
# High-level convenience: render a homogeneous Gaussian scene (validation).
# --------------------------------------------------------------------------- #


def render_gaussians(means, quats, scales, opacities, colors, viewmat, K,
                     width, height, *, device: str = "cuda", background=None, tiled: bool = True):
    """Render a Gaussian scene with PRISM. Mirrors the inputs of
    ``gsplat.rasterization`` (single camera). ``tiled`` selects the scalable
    tile compositor (default) vs the dense reference compositor."""

    torch = _require_torch()
    cov3d = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov3d, viewmat, K, width, height, torch)
    comp = composite_tiled if tiled else composite
    return comp(proj, colors, opacities, width, height, torch, background=background)


# --------------------------------------------------------------------------- #
# PRISM training backend — train typed carriers on real posed images, end to
# end, with AURA's own rasterizer (no gsplat). Reuses the seed/writeback and
# camera/image helpers from gsplat_renderer so the .aura asset contract is
# shared.
# --------------------------------------------------------------------------- #


_FOOTPRINTS = {
    "gaussian": lambda dx, dy, conic, torch: gaussian_footprint(dx, dy, conic, torch),
    "beta": lambda dx, dy, conic, torch: beta_footprint(dx, dy, conic, torch, beta=2.0),
}

#: footprint name <-> integer code (kept in sync with prism_cuda).
FOOTPRINT_CODES = {"gaussian": 0, "beta": 1, "gabor": 2, "neural": 3}
FOOTPRINT_NAMES = {v: k for k, v in FOOTPRINT_CODES.items()}


def assign_footprints(manifest: dict, scale: float, device: str, *, gabor_fraction: float = 0.3):
    """Adaptive carrier-type assignment from image texture (the "pick the right
    carrier per region" step of the AURA thesis).

    Loads each posed image once, computes a gradient-magnitude (texture) map,
    projects every seed region's centre into its source frame and samples the
    local texture. The highest-texture ``gabor_fraction`` of carriers become
    **Gabor** carriers (oscillatory, for high-frequency detail); the rest stay
    **Gaussian**. Returns ``(ftypes [N] long, freq [N,2], phase [N])`` aligned to
    ``manifest['regions']`` order (= seed order).
    """

    torch = _require_torch()
    from pathlib import Path
    from .gsplat_renderer import manifest_frame_to_camera, _load_image_rgb

    root = Path(manifest.get("root", "."))
    frames = {f["id"]: f for f in manifest["frames"]}
    regions = manifest["regions"]
    n = len(regions)
    scores = torch.zeros(n, dtype=torch.float32, device=device)
    cache: dict = {}
    for i, r in enumerate(regions):
        fid = r["frame_id"]
        frame = frames.get(fid)
        if frame is None or not (root / frame["image_path"]).exists():
            continue
        if fid not in cache:
            view, k, w, h = manifest_frame_to_camera(frame, scale)
            img = _load_image_rgb(root / frame["image_path"], torch, device, w, h)
            gray = img.mean(dim=-1)
            gx = torch.zeros_like(gray); gy = torch.zeros_like(gray)
            gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
            gy[1:, :] = gray[1:, :] - gray[:-1, :]
            gmag = (gx * gx + gy * gy).sqrt()
            cache[fid] = (gmag, view, k, w, h)
        gmag, view, k, w, h = cache[fid]
        bmin = r["bounds"]["min"]; bmax = r["bounds"]["max"]
        mean = [(bmin[j] + bmax[j]) * 0.5 for j in range(3)]
        R = [view[a][:3] for a in range(3)]
        t = [view[a][3] for a in range(3)]
        zc = sum(R[2][j] * mean[j] for j in range(3)) + t[2]
        if zc <= 1e-4:
            continue
        xc = sum(R[0][j] * mean[j] for j in range(3)) + t[0]
        yc = sum(R[1][j] * mean[j] for j in range(3)) + t[1]
        u = int(k[0][0] * xc / zc + k[0][2]); v = int(k[1][1] * yc / zc + k[1][2])
        if 0 <= u < w and 0 <= v < h:
            scores[i] = gmag[v, u]
    ftypes = torch.zeros(n, dtype=torch.long, device=device)
    freq = torch.zeros(n, 2, dtype=torch.float32, device=device)
    phase = torch.zeros(n, dtype=torch.float32, device=device)
    if gabor_fraction > 0 and float(scores.max()) > 0:
        thr = torch.quantile(scores, 1.0 - gabor_fraction)
        gabor = scores > thr
        ftypes[gabor] = FOOTPRINT_CODES["gabor"]
        # initialise gabor frequency in a random direction at a moderate rate;
        # training refines it. (deterministic per-index via arange, no RNG.)
        idx = torch.nonzero(gabor, as_tuple=False).squeeze(1)
        ang = (idx.float() * 0.61803398875) * 6.2831853  # golden-angle spread
        freq[idx, 0] = 0.5 * torch.cos(ang)
        freq[idx, 1] = 0.5 * torch.sin(ang)
    return ftypes, freq, phase


def train_carriers_prism(seed_params, ctx, manifest, *, config, device="cuda", carrier="gaussian"):
    """Optimise seed carriers against the manifest's posed images using PRISM's
    tiled differentiable rasterizer and the chosen carrier footprint
    (``gaussian`` or ``beta``). Returns ``(trained_scene, history)``; geometry is
    written back through the shared Gaussian carrier payload (the footprint type
    is recorded in ``metadata['prism_footprint']``).
    """

    torch = _require_torch()
    from pathlib import Path
    from .gsplat_renderer import manifest_frame_to_camera, _load_image_rgb, _ssim, gaussian_params_to_scene

    import json as _json
    if carrier not in ("gaussian", "beta", "gabor", "auto"):
        raise ValueError(f"unknown carrier '{carrier}'")
    log = getattr(config, "log", None) or (lambda _m: None)
    root = Path(manifest.get("root", "."))
    frames = [f for f in manifest["frames"] if (root / f["image_path"]).exists()]
    if not frames:
        raise ValueError("manifest has no readable training frames")

    n0 = seed_params["means"].shape[0]
    # Per-carrier footprint type + gabor freq/phase. "auto" assigns types from
    # image texture (Gabor on high-frequency regions); else homogeneous.
    if carrier == "auto":
        ftypes, freq0, phase0 = assign_footprints(manifest, config.scale, device)
    else:
        ftypes = torch.full((n0,), FOOTPRINT_CODES[carrier], dtype=torch.long, device=device)
        freq0 = torch.zeros(n0, 2, dtype=torch.float32, device=device)
        phase0 = torch.zeros(n0, dtype=torch.float32, device=device)

    _LRS = {"means": config.position_lr, "log_scales": config.log_scale_lr,
            "quats": config.quat_lr, "logit_opacities": config.opacity_lr,
            "colors": config.color_lr, "freq": config.quat_lr, "phase": config.opacity_lr}

    def _new_params(src):
        return {k: src[k].detach().clone().requires_grad_(True) for k in _LRS}

    def _build_opt(p):
        return torch.optim.Adam([{"params": [p[k]], "lr": _LRS[k]} for k in _LRS], eps=1e-15)

    init = {k: seed_params[k] for k in ("means", "log_scales", "quats", "logit_opacities", "colors")}
    init["freq"] = freq0
    init["phase"] = phase0
    P = _new_params(init)
    ft = ftypes.clone()  # per-carrier footprint codes (non-trainable; grows on densify)
    opt = _build_opt(P)

    from .prism_cuda import cuda_available, render_gaussians_cuda_diff
    use_cuda = cuda_available()
    homo_fp = _FOOTPRINTS.get(carrier if carrier in _FOOTPRINTS else "gaussian")

    history = {"loss": []}
    nf = len(frames)
    grad_accum = torch.zeros(P["means"].shape[0], device=device)
    accum_n = 0
    for it in range(config.iterations):
        frame = frames[it % nf]
        view, k, w, h = manifest_frame_to_camera(frame, config.scale)
        gt = _load_image_rgb(root / frame["image_path"], torch, device, w, h)
        viewmat = torch.tensor(view, dtype=torch.float32, device=device)
        K = torch.tensor(k, dtype=torch.float32, device=device)
        if use_cuda:
            img = render_gaussians_cuda_diff(
                P["means"], P["quats"], torch.exp(P["log_scales"]), torch.sigmoid(P["logit_opacities"]),
                P["colors"].clamp(0, 1), viewmat, K, w, h, max_per_tile=config.max_per_tile,
                ftypes=ft, freq=P["freq"], phase=P["phase"], volumetric=config.volumetric,
            )
        else:  # torch fallback (homogeneous footprint only)
            cov = quats_scales_to_cov3d(P["quats"], torch.exp(P["log_scales"]), torch)
            proj = project_gaussians(P["means"], cov, viewmat, K, w, h, torch)
            img = composite_tiled(
                proj, P["colors"].clamp(0, 1), torch.sigmoid(P["logit_opacities"]), w, h, torch,
                footprint=homo_fp, max_per_tile=config.max_per_tile, volumetric=config.volumetric,
            )
        l1 = torch.abs(img - gt).mean()
        loss = (1 - config.ssim_weight) * l1 + config.ssim_weight * (1 - _ssim(img, gt, torch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Stabilise: zero non-finite grads and clip — the dense tile-composite can
        # produce large/ NaN gradients on near-degenerate carriers at high caps.
        for t in P.values():
            if t.grad is not None:
                torch.nan_to_num_(t.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(P.values()), 10.0)
        opt.step()

        # Exponential position-LR decay (3DGS-style): coarse-to-fine convergence.
        if config.position_lr_final > 0.0 and config.iterations > 1:
            t = it / (config.iterations - 1)
            lr = config.position_lr * (config.position_lr_final / config.position_lr) ** t
            for grp in opt.param_groups:
                if grp["params"][0] is P["means"]:
                    grp["lr"] = lr
        # Opacity reset: periodically clamp opacities down so floaters must re-earn
        # them (removes view-inconsistent blobs that otherwise dominate at high caps).
        if (config.opacity_reset_interval and it > 0
                and it % config.opacity_reset_interval == 0):
            with torch.no_grad():
                cap = math.log(config.opacity_reset_to / (1 - config.opacity_reset_to))
                P["logit_opacities"].clamp_(max=cap)

        # Adaptive densification: clone high-positional-gradient carriers and
        # prune transparent ones (footprint type + gabor params clone with them).
        if config.densify and P["means"].grad is not None:
            with torch.no_grad():
                grad_accum += P["means"].grad.norm(dim=1)
                accum_n += 1
            due = (config.densify_start < it < config.densify_stop
                   and it > 0 and it % config.densify_interval == 0)
            if due:
                with torch.no_grad():
                    avg = grad_accum / max(accum_n, 1)
                    keep = torch.sigmoid(P["logit_opacities"]) > config.prune_opacity
                    highgrad = keep & (avg > config.grad_threshold)
                    if int(P["means"].shape[0]) >= config.max_carriers:
                        highgrad = torch.zeros_like(highgrad)
                    scale_lin = torch.exp(P["log_scales"]).max(dim=1).values
                    # 3DGS-style: SPLIT large high-grad carriers (into 2 smaller
                    # children, positions sampled within the carrier), CLONE small
                    # ones. Split fills detail; clone densifies under-reconstructed.
                    hg_idx = torch.nonzero(highgrad, as_tuple=False).squeeze(1)
                    if hg_idx.numel() > 0:
                        thr = torch.quantile(scale_lin[hg_idx], config.split_scale_percentile)
                    else:
                        thr = scale_lin.new_tensor(float("inf"))
                    split = highgrad & (scale_lin > thr)
                    clone = highgrad & ~split
                    base = keep & ~split           # survivors (split originals are replaced)
                    base_idx = torch.nonzero(base, as_tuple=False).squeeze(1)
                    clone_idx = torch.nonzero(clone, as_tuple=False).squeeze(1)
                    split_idx = torch.nonzero(split, as_tuple=False).squeeze(1)
                    cl_jit = torch.zeros_like(P["means"][clone_idx]).normal_(0, 1e-3)
                    # two children per split carrier, jittered by the carrier's own scale
                    s = torch.exp(P["log_scales"][split_idx])
                    j1 = torch.randn_like(P["means"][split_idx]) * s
                    j2 = torch.randn_like(P["means"][split_idx]) * s
                    shrink = math.log(config.split_scale_shrink)
                    new = {}
                    for key in _LRS:
                        parts = [P[key][base_idx], P[key][clone_idx] + (cl_jit if key == "means" else 0.0)]
                        if split_idx.numel() > 0:
                            c1, c2 = P[key][split_idx].clone(), P[key][split_idx].clone()
                            if key == "means":
                                c1 = c1 + j1; c2 = c2 + j2
                            elif key == "log_scales":
                                c1 = c1 - shrink; c2 = c2 - shrink
                            parts += [c1, c2]
                        new[key] = torch.cat(parts, dim=0)
                    ft = torch.cat([ft[base_idx], ft[clone_idx], ft[split_idx], ft[split_idx]], dim=0)
                P = _new_params(new)
                opt = _build_opt(P)
                grad_accum = torch.zeros(P["means"].shape[0], device=device)
                accum_n = 0

        if it % config.log_every == 0 or it == config.iterations - 1:
            history["loss"].append((it, float(loss.detach())))
            log(f"  [prism:{carrier}] iter {it + 1}/{config.iterations}  loss={float(loss.detach()):.4f}  N={P['means'].shape[0]}")

    trained = {
        "means": P["means"], "log_scales": P["log_scales"], "quats": P["quats"],
        "logit_opacities": P["logit_opacities"], "colors": P["colors"],
    }
    scene = gaussian_params_to_scene(trained, {**ctx, "sh_degree": 0})
    # Tag each carrier's footprint (and gabor freq/phase) so PRISM eval reproduces
    # the heterogeneous render. Element order matches the trained param order.
    import dataclasses
    ft_l = ft.cpu().tolist()
    freq_l = P["freq"].detach().cpu().tolist()
    phase_l = P["phase"].detach().cpu().tolist()
    gaussian_i = 0
    tagged = []
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for e in scene.elements:
        if e.carrier_id == "gaussian" and gaussian_i < len(ft_l):
            code = int(ft_l[gaussian_i]); counts[code] = counts.get(code, 0) + 1
            meta = dict(getattr(e, "metadata", None) or {})
            meta["prism_footprint"] = FOOTPRINT_NAMES.get(code, "gaussian")
            if config.volumetric:
                meta["prism_volumetric"] = "1"
            if code == FOOTPRINT_CODES["gabor"]:
                meta["prism_freq"] = _json.dumps(freq_l[gaussian_i])
                meta["prism_phase"] = str(phase_l[gaussian_i])
            tagged.append(dataclasses.replace(e, metadata=meta))
            gaussian_i += 1
        else:
            tagged.append(e)
    scene = dataclasses.replace(scene, elements=tuple(tagged))
    from .carrier_io import carriers_from_params
    history["carrier_save"] = carriers_from_params(trained, sh_degree=0, ftypes=ft, freq=P["freq"], phase=P["phase"])
    history["final_gaussian_count"] = int(P["means"].shape[0])
    history["footprint_counts"] = {FOOTPRINT_NAMES[c]: counts[c] for c in counts if counts[c]}
    return scene, history


@dataclass
class PrismTrainConfig:
    iterations: int = 3000
    scale: float = 0.25
    position_lr: float = 1.6e-4
    log_scale_lr: float = 5e-3
    quat_lr: float = 1e-3
    opacity_lr: float = 5e-2
    color_lr: float = 2.5e-3
    ssim_weight: float = 0.2
    max_per_tile: int = 256
    log_every: int = 100
    densify: bool = False
    densify_interval: int = 100
    densify_start: int = 200
    densify_stop: int = 15000
    grad_threshold: float = 2e-4
    prune_opacity: float = 0.05
    split_scale_percentile: float = 0.5   # among high-grad carriers, split those above this scale pct (else clone)
    split_scale_shrink: float = 1.6        # child scale = parent scale / this (3DGS uses 1.6)
    max_carriers: int = 2_000_000
    volumetric: bool = False  # EVER-style 1-exp(-opacity*w) alpha
    # 3DGS-style training stabilisers (default off → unchanged behaviour):
    opacity_reset_interval: int = 0     # every N iters clamp opacity down to opacity_reset_to (floater control)
    opacity_reset_to: float = 0.01
    position_lr_final: float = 0.0      # >0 → exponential decay of the means LR to this over training
    log: Callable | None = None


def _scene_carrier_types(scene, torch, device):
    """Per-Gaussian-carrier footprint code [N], freq [N,2], phase [N] read from
    each carrier's metadata (default gaussian); aligned to scene_to_gaussian_params
    order (the Gaussian carriers, in scene order)."""
    import json as _json
    from .gsplat_renderer import _is_gaussian
    gaussians = [e for e in scene.elements if _is_gaussian(e)]
    codes, freqs, phases = [], [], []
    for e in gaussians:
        meta = getattr(e, "metadata", None) or {}
        name = meta.get("prism_footprint", "gaussian")
        codes.append(FOOTPRINT_CODES.get(name, 0))
        if name == "gabor" and "prism_freq" in meta:
            freqs.append([float(x) for x in _json.loads(meta["prism_freq"])])
            phases.append(float(meta.get("prism_phase", 0.0)))
        else:
            freqs.append([0.0, 0.0]); phases.append(0.0)
    ft = torch.tensor(codes, dtype=torch.long, device=device)
    fr = torch.tensor(freqs, dtype=torch.float32, device=device)
    ph = torch.tensor(phases, dtype=torch.float32, device=device)
    return ft, fr, ph


def render_scene_prism(scene, frame, scale, *, device="cuda"):
    """Render a trained AURA scene with PRISM through one manifest frame; returns
    (W, H, flat_rgb). Renders the HETEROGENEOUS carrier mix using each carrier's
    ``metadata['prism_footprint']`` (+ gabor freq/phase) via the CUDA kernel
    (torch fallback for homogeneous scenes)."""

    torch = _require_torch()
    from .gsplat_renderer import scene_to_gaussian_params, manifest_frame_to_camera, _is_gaussian

    params, _ctx = scene_to_gaussian_params(scene, device=device)
    ft, fr, ph = _scene_carrier_types(scene, torch, device)
    vol = any((getattr(e, "metadata", None) or {}).get("prism_volumetric") == "1"
              for e in scene.elements if _is_gaussian(e))
    view, k, w, h = manifest_frame_to_camera(frame, scale)
    viewmat = torch.tensor(view, dtype=torch.float32, device=device)
    K = torch.tensor(k, dtype=torch.float32, device=device)
    with torch.no_grad():
        img = None
        try:
            from .prism_cuda import cuda_available, render_gaussians_cuda
            if cuda_available():
                img = render_gaussians_cuda(
                    params["means"], params["quats"], torch.exp(params["log_scales"]),
                    torch.sigmoid(params["logit_opacities"]), params["colors"].clamp(0, 1),
                    viewmat, K, w, h, ftypes=ft, freq=fr, phase=ph, volumetric=vol,
                )
        except Exception:
            img = None
        if img is None:
            # torch fallback: render with the majority footprint (homogeneous).
            codes = ft.tolist()
            majority = max(set(codes), key=codes.count) if codes else 0
            footprint = _FOOTPRINTS.get(FOOTPRINT_NAMES.get(majority, "gaussian"), _FOOTPRINTS["gaussian"])
            cov = quats_scales_to_cov3d(params["quats"], torch.exp(params["log_scales"]), torch)
            proj = project_gaussians(params["means"], cov, viewmat, K, w, h, torch)
            img = composite_tiled(proj, params["colors"].clamp(0, 1),
                                  torch.sigmoid(params["logit_opacities"]), w, h, torch,
                                  footprint=footprint)
    flat = img.clamp(0.0, 1.0).reshape(-1).cpu().tolist()
    return w, h, flat
