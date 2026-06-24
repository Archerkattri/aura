"""Unified ray query over trained carriers — the capability contract on real data.

`scene.AuraScene.ray_query` answers a ray on native/demo scenes. This module
answers the same query directly over a `carriers.npz` tensor set (gsplat/DBS
trained, millions of carriers), returning the full contract payload:

    rayQuery(r) -> RayQueryResult{ color, depth, normal, confidence,
                                   semantic_id, transmittance, ... }

First-hit model: a carrier is hit if the ray passes within k·(largest axis scale)
of its centre and in front of the origin; among hits the nearest (smallest t) wins,
its colour is the diffuse albedo, its normal the Gaussian short axis, and its
confidence/semantic come from the per-carrier fields when present. This is the
geometric query layer (front-most surface), not the full volumetric integral the
rasterizer computes.
"""
from __future__ import annotations


def carrier_ray_query(carriers, origin, direction, *, k=3.0, min_opacity=0.0,
                      min_confidence=0.0, device="cpu"):
    """Answer one ray over carrier tensors. Returns an ``aura.ray.RayQueryResult``.

    origin / direction are length-3 sequences (world space). ``min_opacity`` and
    ``min_confidence`` reject carriers below those thresholds before hit-testing —
    use ``min_confidence`` (with the multi-view confidence field) to skip
    speculative floaters. Honest caveat: a geometric first-surface query over a raw
    (unpruned) 3DGS/MCMC cloud is sensitive to near-camera floaters; the full
    volumetric integral lives in the rasterizer. Opacity/confidence filtering and
    densification pruning mitigate it."""
    import torch
    from .ray import RayQueryResult
    from .relight import carrier_albedo, carrier_normals

    means = carriers["means"].to(device).float()
    scales = carriers["scales"].to(device).float()
    keep = torch.ones(means.shape[0], dtype=torch.bool, device=device)
    if min_opacity > 0.0:
        keep &= carriers["opacity"].to(device).float() >= min_opacity
    if min_confidence > 0.0 and "confidence" in carriers:
        keep &= carriers["confidence"].to(device).float() >= min_confidence
    o = torch.tensor(origin, dtype=torch.float32, device=device)
    d = torch.tensor(direction, dtype=torch.float32, device=device)
    d = d / torch.clamp(torch.linalg.norm(d), min=1e-8)

    rel = means - o                                   # [N,3]
    t = rel @ d                                        # [N] projection (signed depth)
    closest = o + t.unsqueeze(-1) * d                  # [N,3] nearest point on ray
    perp = torch.linalg.norm(means - closest, dim=-1)  # [N] perpendicular distance
    radius = k * scales.max(dim=-1).values             # [N] carrier extent
    hit = (t > 1e-4) & (perp < radius) & keep
    if not bool(hit.any()):
        return RayQueryResult(color=(0.0, 0.0, 0.0), transmittance=1.0,
                              confidence=0.0, provenance="miss")

    # Front-to-back opacity accumulation: the "surface" is the carrier where the
    # accumulated alpha crosses 0.5. This ignores wispy near-camera floaters that a
    # naive nearest-centre hit would wrongly pick, matching what the rasterizer sees.
    idx_hits = torch.nonzero(hit, as_tuple=False).squeeze(-1)
    order = torch.argsort(t[idx_hits])
    ordered = idx_hits[order]
    opac = carriers["opacity"].to(device).float()[ordered].clamp(0, 1)
    trans = torch.cumprod(1.0 - opac + 1e-6, dim=0)    # transmittance before each
    accum = 1.0 - trans                                 # accumulated alpha
    cross = torch.nonzero(accum >= 0.5, as_tuple=False)
    sel = int(cross[0]) if cross.numel() else int(torch.argmax(accum))
    i = int(ordered[sel])

    albedo = carrier_albedo(torch, {kk: carriers[kk] for kk in carriers if kk in ("sh", "colors")})
    color = tuple(float(x) for x in albedo[i].clamp(0, 1).tolist())
    n = carrier_normals(torch, carriers["quats"].to(device).float()[i:i + 1],
                        scales[i:i + 1])[0]
    # orient the normal toward the ray origin (covariance normals are unsigned)
    if float(torch.dot(n, -d)) < 0:
        n = -n
    normal = tuple(float(x) for x in n.tolist())
    opacity = float(carriers["opacity"].to(device).float()[i].clamp(0, 1))
    conf = float(carriers["confidence"][i]) if "confidence" in carriers else opacity
    semantic = None
    if "semantic_id" in carriers and carriers["semantic_id"] is not None:
        semantic = str(carriers["semantic_id"][i])

    return RayQueryResult(
        color=color, transmittance=max(0.0, 1.0 - opacity),
        confidence=max(0.0, min(1.0, conf)), depth=float(t[i]),
        normal=normal, semantic_id=semantic, provenance="carrier_query",
    )
