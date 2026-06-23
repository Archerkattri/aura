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
the documented perf-future (see docs/STATUS_AND_ROADMAP.md). Pipeline mirrors
gsplat's stages:

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
    py = (tile_iy.view(T, 1) * ts + torch.arange(ts, device=device).view(1, ts)).to(f32)  # [T,ts]
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

    if carrier not in _FOOTPRINTS:
        raise ValueError(f"unknown carrier '{carrier}'; choose from {sorted(_FOOTPRINTS)}")
    footprint = _FOOTPRINTS[carrier]
    log = getattr(config, "log", None) or (lambda _m: None)
    root = Path(manifest.get("root", "."))
    frames = [f for f in manifest["frames"] if (root / f["image_path"]).exists()]
    if not frames:
        raise ValueError("manifest has no readable training frames")

    means = seed_params["means"].detach().clone().requires_grad_(True)
    log_scales = seed_params["log_scales"].detach().clone().requires_grad_(True)
    quats = seed_params["quats"].detach().clone().requires_grad_(True)
    logit_op = seed_params["logit_opacities"].detach().clone().requires_grad_(True)
    colors = seed_params["colors"].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": [means], "lr": config.position_lr},
        {"params": [log_scales], "lr": config.log_scale_lr},
        {"params": [quats], "lr": config.quat_lr},
        {"params": [logit_op], "lr": config.opacity_lr},
        {"params": [colors], "lr": config.color_lr},
    ], eps=1e-15)

    history = {"loss": []}
    nf = len(frames)
    for it in range(config.iterations):
        frame = frames[it % nf]
        view, k, w, h = manifest_frame_to_camera(frame, config.scale)
        gt = _load_image_rgb(root / frame["image_path"], torch, device, w, h)
        viewmat = torch.tensor(view, dtype=torch.float32, device=device)
        K = torch.tensor(k, dtype=torch.float32, device=device)
        cov = quats_scales_to_cov3d(quats, torch.exp(log_scales), torch)
        proj = project_gaussians(means, cov, viewmat, K, w, h, torch)
        img = composite_tiled(
            proj, colors.clamp(0, 1), torch.sigmoid(logit_op), w, h, torch,
            footprint=footprint, max_per_tile=config.max_per_tile,
        )
        l1 = torch.abs(img - gt).mean()
        loss = (1 - config.ssim_weight) * l1 + config.ssim_weight * (1 - _ssim(img, gt, torch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % config.log_every == 0 or it == config.iterations - 1:
            history["loss"].append((it, float(loss.detach())))
            log(f"  [prism:{carrier}] iter {it + 1}/{config.iterations}  loss={float(loss.detach()):.4f}  N={means.shape[0]}")

    trained = {
        "means": means, "log_scales": log_scales, "quats": quats,
        "logit_opacities": logit_op, "colors": colors,
    }
    scene = gaussian_params_to_scene(trained, {**ctx, "sh_degree": 0})
    # Tag the footprint type so PRISM eval renders the right kernel.
    import dataclasses
    tagged = []
    for e in scene.elements:
        if e.carrier_id == "gaussian":
            meta = dict(getattr(e, "metadata", None) or {})
            meta["prism_footprint"] = carrier
            tagged.append(dataclasses.replace(e, metadata=meta))
        else:
            tagged.append(e)
    scene = dataclasses.replace(scene, elements=tuple(tagged))
    history["final_gaussian_count"] = int(means.shape[0])
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
    log: Callable | None = None


def render_scene_prism(scene, frame, scale, *, device="cuda"):
    """Render a trained AURA scene's Gaussian carriers with PRISM through one
    manifest frame; returns (W, H, flat_rgb) like the eval harness expects.
    Uses each carrier's ``metadata['prism_footprint']`` (default gaussian)."""

    torch = _require_torch()
    from .gsplat_renderer import scene_to_gaussian_params, manifest_frame_to_camera

    params, _ctx = scene_to_gaussian_params(scene, device=device)
    # Footprint: use the scene's recorded prism_footprint if uniform, else gaussian.
    fps = {((getattr(e, "metadata", None) or {}).get("prism_footprint", "gaussian"))
           for e in scene.elements if e.carrier_id == "gaussian"}
    footprint = _FOOTPRINTS.get(next(iter(fps)) if len(fps) == 1 else "gaussian", _FOOTPRINTS["gaussian"])
    cov = quats_scales_to_cov3d(params["quats"], torch.exp(params["log_scales"]), torch)
    view, k, w, h = manifest_frame_to_camera(frame, scale)
    viewmat = torch.tensor(view, dtype=torch.float32, device=device)
    K = torch.tensor(k, dtype=torch.float32, device=device)
    with torch.no_grad():
        proj = project_gaussians(params["means"], cov, viewmat, K, w, h, torch)
        img = composite_tiled(proj, params["colors"].clamp(0, 1),
                              torch.sigmoid(params["logit_opacities"]), w, h, torch,
                              footprint=footprint)
    flat = img.clamp(0.0, 1.0).reshape(-1).cpu().tolist()
    return w, h, flat
