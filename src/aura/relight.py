"""Relighting trained carriers — the 'relightable asset' capability over real
gsplat/DBS-trained carriers (not just demo scenes).

3DGS bakes lighting into view-dependent colour and cannot be relit. AURA treats a
carrier as a surface element: its **normal** is the short axis of the Gaussian
(the direction of least spatial extent ≈ surface normal), and its **albedo** is the
diffuse colour (SH DC term). Given those, `shading.py`'s Lambertian / Cook-Torrance
BRDFs produce a *relit* per-carrier colour under arbitrary lights, which gsplat then
rasterizes. This is an explicit, editable lighting model on top of the same
carriers — a core post-3DGS differentiator.

Scope: normals from covariance are unsigned and noisy for
near-isotropic carriers, and albedo from baked colour still contains residual
shading. This is a usable relighting layer, not a full inverse-rendering material
decomposition.
"""
from __future__ import annotations

_C0 = 0.28209479177387814  # SH band-0 constant


def _quat_to_rotmat(torch, quats):
    """wxyz unit quats [N,4] -> rotation matrices [N,3,3]."""
    q = quats / torch.clamp(torch.linalg.norm(quats, dim=-1, keepdim=True), min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def carrier_normals(torch, quats, scales):
    """Per-carrier normal = the rotation axis with the SMALLEST scale (the short
    axis of the Gaussian ≈ local surface normal). Returns unit normals [N,3]."""
    R = _quat_to_rotmat(torch, quats)          # [N,3,3], columns are principal axes
    short = torch.argmin(scales, dim=-1)        # [N] index of smallest extent
    n = R.gather(2, short.reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)  # [N,3]
    return n / torch.clamp(torch.linalg.norm(n, dim=-1, keepdim=True), min=1e-8)


def carrier_albedo(torch, carriers):
    """Diffuse albedo [N,3] in [0,1] from carriers (SH DC or flat colour)."""
    if "sh" in carriers and carriers["sh"] is not None:
        dc = carriers["sh"][:, 0, :]
        return torch.clamp(0.5 + _C0 * dc, 0.0, 1.0)
    return torch.clamp(carriers["colors"], 0.0, 1.0)


def relight_colors(carriers, lights, *, ambient=0.1, view_dir=None, device="cuda"):
    """Compute relit per-carrier RGB [N,3] under directional `lights`.

    Lambertian + an ambient term so unlit faces are not pure black. If `view_dir`
    (a 3-vector toward the camera) is given, a Cook-Torrance specular lobe is added.
    """
    import torch
    from .shading import lambertian_shade, DirectionalLight

    quats = carriers["quats"].to(device)
    scales = carriers["scales"].to(device)
    n = carrier_normals(torch, quats, scales)
    albedo = carrier_albedo(torch, {k: (v.to(device) if hasattr(v, "to") else v)
                                     for k, v in carriers.items() if k in ("sh", "colors")})
    if not isinstance(lights, (list, tuple)):
        lights = [lights]
    lit = lambertian_shade(torch, albedo, n, list(lights))
    # normals from covariance are unsigned — also light the back face so a carrier
    # facing away from every light is not spuriously black, then take the brighter.
    lit_flip = lambertian_shade(torch, albedo, -n, list(lights))
    lit = torch.maximum(lit, lit_flip)
    out = ambient * albedo + lit
    return torch.clamp(out, 0.0, 1.0)


def render_relit(carriers, frame, scale, lights, *, ambient=0.1, device="cuda"):
    """Relight carriers and rasterize through one manifest frame (gsplat).
    Returns (W, H, flat_rgb)."""
    import torch
    from gsplat import rasterization
    from .gsplat_renderer import manifest_frame_to_camera

    colors = relight_colors(carriers, lights, ambient=ambient, device=device)
    view, k, w, h = manifest_frame_to_camera(frame, scale)
    vm = torch.tensor(view, dtype=torch.float32, device=device).unsqueeze(0)
    K = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out, _, _ = rasterization(
            means=carriers["means"].to(device), quats=carriers["quats"].to(device),
            scales=carriers["scales"].to(device), opacities=carriers["opacity"].to(device),
            colors=colors, viewmats=vm, Ks=K, width=w, height=h,
        )
    return w, h, out[0].clamp(0, 1).reshape(-1).cpu().tolist()
