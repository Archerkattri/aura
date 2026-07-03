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

Acceleration (audit M2): the hit test is an isotropic sphere of radius
``k*max(scale)`` per carrier, so a BVH over the cubes ``mean ± k*max(scale)`` finds
a candidate superset of the true hits in sub-linear node visits, batched across
rays. The candidate set is then resolved with the *exact* brute-force selection
(`_resolve_ray`), so ``method="bvh"`` (default) returns results identical to the old
``method="brute"`` O(N) path — see `aura.bvh` and `docs/P5_BVH_RAY_QUERY.md`.
"""
from __future__ import annotations


def _miss():
    from .ray import RayQueryResult
    return RayQueryResult(color=(0.0, 0.0, 0.0), transmittance=1.0,
                          confidence=0.0, provenance="miss")


def _build_ctx(carriers, device="cpu"):
    """Pre-convert the carrier tensors to float/device ONCE so a batch (or repeated
    single) query does not re-cast the full N-carrier arrays on every ray. Returns an
    opaque context consumed by :func:`_resolve_ray`."""
    import torch

    def cast(key):
        v = carriers.get(key) if hasattr(carriers, "get") else (
            carriers[key] if key in carriers else None)
        return None if v is None else v.to(device).float()

    ctx = {
        "means": cast("means"),
        "scales": cast("scales"),
        "quats": cast("quats"),
        "opacity": cast("opacity"),
        "confidence": cast("confidence"),
        "device": device,
    }
    if "sh" in carriers and carriers["sh"] is not None:
        ctx["color_key"], ctx["color"] = "sh", cast("sh")
    else:
        ctx["color_key"], ctx["color"] = "colors", cast("colors")
    ctx["semantic_id"] = carriers["semantic_id"] if (
        "semantic_id" in carriers and carriers["semantic_id"] is not None) else None
    return ctx


def _resolve_ray(ctx, cand_idx, origin, direction, *, k=3.0, min_opacity=0.0,
                 min_confidence=0.0, device="cpu"):
    """Exact first-surface resolution over a candidate carrier subset.

    ``ctx`` is a context from :func:`_build_ctx` (or a raw carriers mapping, wrapped
    on the fly). ``cand_idx`` is a 1-D LongTensor of carrier indices (ascending) to
    consider, or ``None`` for all carriers (the brute-force path). The hit test,
    front-to-back opacity accumulation and payload extraction are the original
    brute-force arithmetic — the BVH path merely restricts the input set to a
    superset of the true hits, so both paths return the same result.
    """
    import torch
    from .relight import carrier_albedo, carrier_normals

    if "means" not in ctx or "color_key" not in ctx:          # a raw carriers mapping
        ctx = _build_ctx(ctx, device)
    device = ctx["device"]
    means_all = ctx["means"]
    scales_all = ctx["scales"]
    opacity_all = ctx["opacity"]
    N = means_all.shape[0]
    if cand_idx is None:
        idx = torch.arange(N, device=device)
    else:
        idx = cand_idx if torch.is_tensor(cand_idx) else torch.as_tensor(cand_idx)
        idx = idx.to(device=device, dtype=torch.long)
    if idx.numel() == 0:
        return _miss()

    means = means_all.index_select(0, idx)
    scales = scales_all.index_select(0, idx)
    keep = torch.ones(idx.shape[0], dtype=torch.bool, device=device)
    if min_opacity > 0.0:
        keep &= opacity_all.index_select(0, idx) >= min_opacity
    if min_confidence > 0.0 and ctx["confidence"] is not None:
        keep &= ctx["confidence"].index_select(0, idx) >= min_confidence
    o = torch.as_tensor(origin, dtype=torch.float32, device=device)
    d = torch.as_tensor(direction, dtype=torch.float32, device=device)
    d = d / torch.clamp(torch.linalg.norm(d), min=1e-8)

    rel = means - o                                   # [C,3]
    t = rel @ d                                        # [C] projection (signed depth)
    closest = o + t.unsqueeze(-1) * d                  # [C,3] nearest point on ray
    perp = torch.linalg.norm(means - closest, dim=-1)  # [C] perpendicular distance
    radius = k * scales.max(dim=-1).values             # [C] carrier extent
    hit = (t > 1e-4) & (perp < radius) & keep
    if not bool(hit.any()):
        return _miss()

    # Front-to-back opacity accumulation: the "surface" is the carrier where the
    # accumulated alpha crosses 0.5. This ignores wispy near-camera floaters that a
    # nearest-centre hit would otherwise pick, matching what the rasterizer sees.
    local_hits = torch.nonzero(hit, as_tuple=False).squeeze(-1)
    order = torch.argsort(t[local_hits], stable=True)  # ties -> ascending carrier idx
    ordered = idx.index_select(0, local_hits.index_select(0, order))
    opac = opacity_all.index_select(0, ordered).clamp(0, 1)
    trans = torch.cumprod(1.0 - opac + 1e-6, dim=0)    # transmittance before each
    accum = 1.0 - trans                                 # accumulated alpha
    cross = torch.nonzero(accum >= 0.5, as_tuple=False)
    sel = int(cross[0]) if cross.numel() else int(torch.argmax(accum))
    i = int(ordered[sel])

    albedo = carrier_albedo(torch, {ctx["color_key"]: ctx["color"][i:i + 1]})
    color = tuple(float(x) for x in albedo[0].clamp(0, 1).tolist())
    n = carrier_normals(torch, ctx["quats"][i:i + 1], scales_all[i:i + 1])[0]
    # orient the normal toward the ray origin (covariance normals are unsigned)
    if float(torch.dot(n, -d)) < 0:
        n = -n
    normal = tuple(float(x) for x in n.tolist())
    opacity = float(opacity_all[i].clamp(0, 1))
    conf = float(ctx["confidence"][i]) if ctx["confidence"] is not None else opacity
    semantic = None
    if ctx["semantic_id"] is not None:
        semantic = str(ctx["semantic_id"][i])

    from .ray import RayQueryResult
    # depth is the projection of the selected carrier centre (matches brute force)
    depth = float((means_all[i] - o) @ d)
    return RayQueryResult(
        color=color, transmittance=max(0.0, 1.0 - opacity),
        confidence=max(0.0, min(1.0, conf)), depth=depth,
        normal=normal, semantic_id=semantic, provenance="carrier_query",
    )


def _resolve_batch(ctx, cand_ray, cand_prim, origins, directions, M, *, k=3.0,
                   min_opacity=0.0, min_confidence=0.0, device="cpu"):
    """Vectorised first-surface resolution for a whole batch of rays.

    ``cand_ray``/``cand_prim`` are the flat candidate ``(ray, carrier)`` pairs from
    the BVH traversal (a superset of the true hits). Every step mirrors the per-ray
    :func:`_resolve_ray` exactly — same hit test, same ordering (t ascending, ties by
    ascending carrier index), same ``cumprod`` accumulation and 0.5 crossing — but
    amortised across all rays, so results match ``method='brute'`` while paying the
    Python/torch launch overhead once instead of once per ray. Returns a list of
    ``M`` ``RayQueryResult``.
    """
    import numpy as np
    import torch
    from .ray import RayQueryResult
    from .relight import carrier_albedo, carrier_normals

    if "means" not in ctx or "color_key" not in ctx:
        ctx = _build_ctx(ctx, device)
    device = ctx["device"]
    means = ctx["means"]
    scales = ctx["scales"]
    opacity = ctx["opacity"]
    confidence = ctx["confidence"]

    o_all = torch.as_tensor(np.ascontiguousarray(origins), dtype=torch.float32, device=device)
    d_raw = torch.as_tensor(np.ascontiguousarray(directions), dtype=torch.float32, device=device)
    d_all = d_raw / torch.clamp(torch.linalg.norm(d_raw, dim=1, keepdim=True), min=1e-8)

    miss = _miss()
    results = [miss] * M
    if cand_prim.size == 0:
        return results

    ray = torch.from_numpy(np.ascontiguousarray(cand_ray)).long()
    prim = torch.from_numpy(np.ascontiguousarray(cand_prim)).long()

    o = o_all.index_select(0, ray)                    # [R,3]
    d = d_all.index_select(0, ray)                    # [R,3]
    cm = means.index_select(0, prim)                  # [R,3]
    rel = cm - o
    t = (rel * d).sum(dim=1)                          # [R]
    closest = o + t.unsqueeze(1) * d
    perp = torch.linalg.norm(cm - closest, dim=1)
    radius = k * scales.index_select(0, prim).max(dim=1).values
    hit = (t > 1e-4) & (perp < radius)
    if min_opacity > 0.0:
        hit &= opacity.index_select(0, prim) >= min_opacity
    if min_confidence > 0.0 and confidence is not None:
        hit &= confidence.index_select(0, prim) >= min_confidence
    if not bool(hit.any()):
        return results

    hp = torch.nonzero(hit, as_tuple=False).squeeze(1)
    h_ray = ray.index_select(0, hp).cpu().numpy()
    h_prim = prim.index_select(0, hp).cpu().numpy()
    h_t = t.index_select(0, hp).detach().cpu().numpy()

    # order hits per ray by (t ascending, carrier index ascending) — matches the
    # stable argsort over ascending-index candidates used by _resolve_ray
    perm = np.lexsort((h_prim, h_t, h_ray))
    h_ray = h_ray[perm]
    h_prim = h_prim[perm]

    counts = np.bincount(h_ray, minlength=M)          # hits per ray
    hit_rays = np.nonzero(counts)[0]
    Kmax = int(counts.max())
    seg_off = np.zeros(M, np.int64)
    seg_off[1:] = np.cumsum(counts)[:-1]
    rank = np.arange(h_ray.shape[0]) - seg_off[h_ray]  # within-ray position

    prim_t = torch.from_numpy(h_prim).long()
    opac_flat = opacity.index_select(0, prim_t).clamp(0, 1)
    P = torch.zeros((M, Kmax), dtype=torch.float32, device=device)
    r_idx = torch.from_numpy(np.ascontiguousarray(h_ray)).long()
    c_idx = torch.from_numpy(np.ascontiguousarray(rank)).long()
    P[r_idx, c_idx] = opac_flat
    PRIM = torch.full((M, Kmax), -1, dtype=torch.long, device=device)
    PRIM[r_idx, c_idx] = prim_t

    trans = torch.cumprod(1.0 - P + 1e-6, dim=1)      # padded tail (opac 0) is inert
    accum = 1.0 - trans
    valid = torch.arange(Kmax, device=device).unsqueeze(0) < torch.from_numpy(
        np.ascontiguousarray(counts)).to(device).unsqueeze(1)
    ge = (accum >= 0.5) & valid
    has_cross = ge.any(dim=1)
    weight = torch.arange(Kmax, 0, -1, device=device)  # earlier columns rank higher
    first_cross = torch.argmax(ge.to(torch.int64) * weight, dim=1)
    accum_masked = torch.where(valid, accum, torch.full_like(accum, -1.0))
    arg_accum = torch.argmax(accum_masked, dim=1)
    sel_col = torch.where(has_cross, first_cross, arg_accum)
    sel_prim = PRIM[torch.arange(M, device=device), sel_col]  # [M], -1 for miss rays

    hr = torch.from_numpy(np.ascontiguousarray(hit_rays)).long()
    i = sel_prim.index_select(0, hr)                  # selected carrier per hit ray
    o_hr = o_all.index_select(0, hr)
    d_hr = d_all.index_select(0, hr)
    depth = ((means.index_select(0, i) - o_hr) * d_hr).sum(dim=1)
    opacity_i = opacity.index_select(0, i).clamp(0, 1)
    trans_i = torch.clamp(1.0 - opacity_i, min=0.0)
    if confidence is not None:
        conf_i = confidence.index_select(0, i).clamp(0, 1)
    else:
        conf_i = opacity_i
    albedo = carrier_albedo(torch, {ctx["color_key"]: ctx["color"].index_select(0, i)}).clamp(0, 1)
    normals = carrier_normals(torch, ctx["quats"].index_select(0, i), scales.index_select(0, i))
    flip = (normals * (-d_hr)).sum(dim=1) < 0
    normals = torch.where(flip.unsqueeze(1), -normals, normals)

    depth_l = depth.detach().cpu().tolist()
    trans_l = trans_i.detach().cpu().tolist()
    conf_l = conf_i.detach().cpu().tolist()
    albedo_l = albedo.detach().cpu().tolist()
    normal_l = normals.detach().cpu().tolist()
    prim_l = i.detach().cpu().tolist()
    sem = ctx["semantic_id"]
    for j, m in enumerate(hit_rays.tolist()):
        semantic = str(sem[prim_l[j]]) if sem is not None else None
        results[m] = RayQueryResult(
            color=tuple(albedo_l[j]), transmittance=max(0.0, float(trans_l[j])),
            confidence=max(0.0, min(1.0, float(conf_l[j]))), depth=max(0.0, float(depth_l[j])),
            normal=tuple(normal_l[j]), semantic_id=semantic, provenance="carrier_query",
        )
    return results


def carrier_ray_query(carriers, origin, direction, *, k=3.0, min_opacity=0.0,
                      min_confidence=0.0, device="cpu", method="bvh"):
    """Answer one ray over carrier tensors. Returns an ``aura.ray.RayQueryResult``.

    origin / direction are length-3 sequences (world space). ``min_opacity`` and
    ``min_confidence`` reject carriers below those thresholds before hit-testing —
    use ``min_confidence`` (with the multi-view confidence field) to skip
    speculative floaters. Note: a geometric first-surface query over a raw
    (unpruned) 3DGS/MCMC cloud is sensitive to near-camera floaters; the full
    volumetric integral lives in the rasterizer. Opacity/confidence filtering and
    densification pruning mitigate it.

    ``method="bvh"`` (default) resolves the ray through a per-carrier BVH
    (`aura.bvh`); ``method="brute"`` runs the O(N) distance-to-all-carriers
    reference. Both return the identical result — ``method`` exists for parity
    testing and for the rare single-shot call where building a tree is not worth it.
    For many rays use :func:`carrier_ray_query_batch`; for streaming, build a handle
    once with :func:`aura.bvh.build_carrier_bvh` and call ``bvh.query``.
    """
    if method == "brute":
        ctx = _build_ctx(carriers, device)
        return _resolve_ray(ctx, None, origin, direction, k=k,
                            min_opacity=min_opacity, min_confidence=min_confidence,
                            device=device)
    if method != "bvh":
        raise ValueError(f"method must be 'bvh' or 'brute', got {method!r}")
    from .bvh import get_or_build_bvh
    bvh = get_or_build_bvh(carriers, k=k, device=device)
    return bvh.query(origin, direction, min_opacity=min_opacity,
                     min_confidence=min_confidence)


def carrier_ray_query_batch(carriers, origins, directions, *, k=3.0, min_opacity=0.0,
                            min_confidence=0.0, device="cpu", method="bvh", bvh=None,
                            return_stats=False):
    """Answer M rays over carrier tensors.

    ``origins`` / ``directions`` are [M,3] array-likes. Returns a list of M
    ``RayQueryResult``. ``method="bvh"`` (default) builds one BVH for the whole batch
    (or reuses ``bvh`` if a prebuilt :class:`aura.bvh.CarrierBVH` handle is passed —
    the streaming-friendly path). ``method="brute"`` answers each ray with the O(N)
    reference. With ``return_stats=True`` and ``method="bvh"`` returns
    ``(results, stats)`` including measured node visits per ray.
    """
    import numpy as np

    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)

    if method == "brute":
        ctx = _build_ctx(carriers, device)
        results = [
            _resolve_ray(ctx, None, origins[m], directions[m], k=k,
                        min_opacity=min_opacity, min_confidence=min_confidence,
                        device=device)
            for m in range(origins.shape[0])
        ]
        return (results, None) if return_stats else results
    if method != "bvh":
        raise ValueError(f"method must be 'bvh' or 'brute', got {method!r}")

    from .bvh import build_carrier_bvh
    if bvh is None:
        bvh = build_carrier_bvh(carriers, k=k, device=device)
    return bvh.query_batch(origins, directions, min_opacity=min_opacity,
                           min_confidence=min_confidence, return_stats=return_stats)
