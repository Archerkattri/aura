"""AURA-native differentiable rasterizer — an alternative to gsplat that
supports *typed* carriers, not just 3D Gaussians.

This is the genuine post-3DGS substrate: a differentiable, GPU, front-to-back
alpha-compositing rasterizer whose per-primitive 2D footprint is *pluggable* by
carrier type. Gaussians use an exponential conic falloff (validated against
gsplat); Beta carriers use a bounded polynomial falloff; Gabor carriers use an
oscillatory envelope. New carrier types add one footprint function — no change
to projection or compositing.

It is implemented in pure PyTorch (autograd handles the backward pass), so any
carrier whose footprint is a differentiable function of its parameters can be
trained end-to-end with no custom CUDA. The first milestone targets correctness
and typed-carrier training on modest scenes; a tiled / CUDA fast path is future
work (see docs/STATUS_AND_ROADMAP.md). The pipeline mirrors gsplat's stages:

    project (3D -> 2D conic/footprint + depth) -> depth sort -> alpha composite

Conventions match ``gsplat_renderer.manifest_frame_to_camera`` and AURA's
Gaussian carrier model (wxyz quats, linear-RGB colour, opacity in [0,1]).
"""

from __future__ import annotations

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


def project_gaussians(means, cov3d, viewmat, K, width, height, torch,
                      *, near: float = 0.01, eps2d: float = 0.3):
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
    # Low-pass dilation (gsplat eps2d): keep sub-pixel splats renderable.
    cov2d[:, 0, 0] = cov2d[:, 0, 0] + eps2d
    cov2d[:, 1, 1] = cov2d[:, 1, 1] + eps2d

    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-12)
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
        means2d=means2d[idx], depths=z[idx], conics=conic[idx], radii=radii[idx], index=idx
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


def gabor_footprint(dx, dy, conic, freq, phase, torch):
    """Gabor carrier: Gaussian envelope * positive cosine modulation, for
    high-frequency texture (freq = [fx, fy] in pixel^-1, phase scalar)."""
    quad = conic[..., 0] * dx * dx + 2.0 * conic[..., 1] * dx * dy + conic[..., 2] * dy * dy
    env = torch.exp(-0.5 * quad.clamp(min=0.0))
    osc = torch.cos(freq[..., 0] * dx + freq[..., 1] * dy + phase)
    return env * (1.0 + osc) * 0.5


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
):
    """Render visible carriers to an [H,W,3] image via depth-sorted front-to-back
    alpha compositing. Differentiable end-to-end. ``footprint`` defaults to the
    Gaussian kernel; pass beta/gabor (or a custom callable) for typed carriers.

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
        alpha = (sub_opac[m] * weight).clamp(0.0, 0.999)
        contrib = T * alpha
        img = img + contrib.unsqueeze(-1) * sub_colors[m].view(1, 1, 3)
        T = T * (1.0 - alpha)
    return img


# --------------------------------------------------------------------------- #
# High-level convenience: render a homogeneous Gaussian scene (validation).
# --------------------------------------------------------------------------- #


def render_gaussians(means, quats, scales, opacities, colors, viewmat, K,
                     width, height, *, device: str = "cuda", background=None):
    """Render a Gaussian scene with the native rasterizer. Mirrors the inputs
    of ``gsplat.rasterization`` (single camera) for validation."""

    torch = _require_torch()
    cov3d = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov3d, viewmat, K, width, height, torch)
    return composite(proj, colors, opacities, width, height, torch, background=background)
