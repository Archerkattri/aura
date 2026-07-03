"""Bounding-volume hierarchy over trained carriers for accelerated ray queries.

`carrier_query.carrier_ray_query` originally computed the perpendicular distance
from a ray to *every* carrier and argsorted the hits — O(N) per ray, one ray at a
time. For a 129k-carrier truck asset that is 129k distance evaluations per ray.

This module builds a deterministic, pure-numpy **median-split BVH** over the same
carrier hit-volumes the brute-force query uses, and traverses it **batched across
rays** with a stack-based numpy loop. The BVH only narrows the candidate set; the
final hit test and front-to-back surface selection are the *exact* brute-force
arithmetic (`carrier_query._resolve_ray`) applied to the candidates, so results are
identical to the O(N) path (that parity is the acceptance criterion).

Design choice — median-split over LBVH/Morton
----------------------------------------------
The brute-force hit test accepts a carrier when the ray passes within
``k * max(scale)`` of its centre — an **isotropic sphere**. A carrier's leaf AABB
is therefore the axis-aligned cube ``mean ± k*max(scale)`` that tightly bounds that
sphere (the sphere already contains the anisotropic covariance ellipsoid
``k*scale`` rotated by the quaternion, so the cube conservatively bounds both). Any
ray that intersects the sphere intersects this cube, and every ancestor AABB
contains the cube, so a leaf holding a true hit is never pruned: the candidate set
is a guaranteed superset of the brute-force hit set → exact parity.

A **median-split** (object-median, split-by-count) tree is preferred over a
Morton-code LBVH here because:

* It is robust to **duplicate carrier positions** (an explicit edge case): identical
  Morton codes collapse Karras' internal-node split and produce degenerate,
  order-dependent hierarchies, whereas splitting a segment at ``count//2`` always
  makes strictly smaller children and terminates regardless of coordinate ties.
* It is robust to **degenerate flat carriers** (a zero scale axis) — the cube uses
  ``max(scale)`` so the AABB is never a zero-volume slab unless every scale is zero.
* Construction is short, deterministic, and easy to audit in pure numpy; the tree is
  balanced (depth = ceil(log2(N/leaf_size))), and the flat node arrays traverse with
  the same batched stack loop a GPU LBVH would use.

Honest scope: this is an algorithmic fix (O(N) → sub-linear node visits + a batched
API) validated for correctness on a real trained asset. It is a CPU-numpy structure;
it is NOT and does not claim to be a wall-clock match for the CUDA LBVH ray tracers
in 3DGRT / 3DGUT (now landing in gsplat main) — that remains the deferred bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class CarrierBVH:
    """A built BVH handle. Reuse across many ray batches for streaming queries.

    Holds the tree geometry (numpy) plus a reference to the carriers it was built
    from, so ``query_batch`` / ``query`` are self-contained. The tree is built over
    ALL carriers; per-query ``min_opacity`` / ``min_confidence`` filtering is applied
    at hit-resolution time (never baked into the tree), so one handle serves every
    threshold — the streaming-friendly path.
    """

    node_min: np.ndarray          # [Nn,3] float32 — AABB lower corner per node
    node_max: np.ndarray          # [Nn,3] float32 — AABB upper corner per node
    node_left: np.ndarray         # [Nn] int64 — left child node id, -1 if leaf
    node_right: np.ndarray        # [Nn] int64 — right child node id, -1 if leaf
    leaf_start: np.ndarray        # [Nn] int64 — start into prim_order, -1 if internal
    leaf_count: np.ndarray        # [Nn] int64 — #prims in leaf, 0 if internal
    prim_order: np.ndarray        # [N] int64 — carrier indices in leaf order
    k: float
    leaf_size: int
    max_depth: int
    n_carriers: int
    carriers: Mapping[str, Any]
    device: str = "cpu"
    _ctx: Any = field(default=None, repr=False, compare=False)

    def _context(self):
        """Carrier tensors pre-cast to float/device once and cached on the handle,
        so repeated batches over this tree never re-cast the full carrier arrays."""
        if self._ctx is None:
            from .carrier_query import _build_ctx
            self._ctx = _build_ctx(self.carriers, self.device)
        return self._ctx

    # ---- queries ---------------------------------------------------------------
    def query_batch(self, origins, directions, *, min_opacity=0.0,
                    min_confidence=0.0, return_stats=False):
        """Answer M rays. ``origins``/``directions`` are [M,3]. Returns a list of
        ``aura.ray.RayQueryResult`` (one per ray). With ``return_stats=True`` returns
        ``(results, stats)`` where stats includes measured node visits per ray.

        Candidate gathering (BVH traversal) and hit resolution are both batched: the
        traversal never normalises the direction (the slab test is scale-invariant),
        so the resolver applies the exact same single normalisation the brute path
        does — keeping results identical to ``method='brute'``."""
        from .carrier_query import _resolve_batch  # local import avoids import cycle

        origins = np.ascontiguousarray(_to_numpy(origins), dtype=np.float64).reshape(-1, 3)
        directions = np.ascontiguousarray(_to_numpy(directions), dtype=np.float64).reshape(-1, 3)
        M = origins.shape[0]

        cand_ray, cand_prim, node_visits = _traverse(self, origins, directions)
        ctx = self._context()
        results = _resolve_batch(ctx, cand_ray, cand_prim, origins, directions, M,
                                 k=self.k, min_opacity=min_opacity,
                                 min_confidence=min_confidence, device=self.device)
        if return_stats:
            stats = {
                "rays": int(M),
                "node_visits": node_visits.astype(np.int64),
                "candidates": np.bincount(cand_ray, minlength=M).astype(np.int64),
                "nodes": int(self.node_min.shape[0]),
                "max_depth": int(self.max_depth),
                "n_carriers": int(self.n_carriers),
            }
            return results, stats
        return results

    def query(self, origin, direction, *, min_opacity=0.0, min_confidence=0.0):
        """Answer one ray. Returns a single ``RayQueryResult`` (same as brute-force)."""
        res = self.query_batch(np.asarray(origin, dtype=np.float64)[None, :],
                               np.asarray(direction, dtype=np.float64)[None, :],
                               min_opacity=min_opacity, min_confidence=min_confidence)
        return res[0]


def build_carrier_bvh(carriers, *, k=3.0, leaf_size=64, device="cpu") -> CarrierBVH:
    """Build a median-split BVH over carrier hit-volumes (``mean ± k*max(scale)``).

    Pure numpy, deterministic. Build once and reuse the handle across ray batches
    for streaming queries (this is the persistent, streaming-friendly path)."""
    means = np.ascontiguousarray(_to_numpy(carriers["means"]), dtype=np.float64)
    scales = np.ascontiguousarray(_to_numpy(carriers["scales"]), dtype=np.float64)
    N = int(means.shape[0])

    if N == 0:
        z = np.zeros((1, 3), np.float32)
        return CarrierBVH(z.copy(), z.copy(), np.array([-1], np.int64),
                          np.array([-1], np.int64), np.array([0], np.int64),
                          np.array([0], np.int64), np.zeros(0, np.int64),
                          float(k), int(leaf_size), 0, 0, carriers, device)

    r = k * np.max(scales, axis=1)                 # [N] isotropic hit radius
    prim_min = means - r[:, None]                  # [N,3] carrier AABB corners
    prim_max = means + r[:, None]
    centroids = means                              # AABB centre == carrier mean

    max_nodes = 2 * N + 1
    node_min = np.zeros((max_nodes, 3), np.float64)
    node_max = np.zeros((max_nodes, 3), np.float64)
    node_left = np.full(max_nodes, -1, np.int64)
    node_right = np.full(max_nodes, -1, np.int64)
    leaf_start = np.full(max_nodes, -1, np.int64)
    leaf_count = np.zeros(max_nodes, np.int64)
    order = np.arange(N, dtype=np.int64)

    next_node = 1
    max_depth = 0
    # explicit stack of (node_id, start, end, depth); deterministic LIFO order
    stack = [(0, 0, N, 0)]
    while stack:
        nid, s, e, depth = stack.pop()
        if depth > max_depth:
            max_depth = depth
        seg = order[s:e]
        node_min[nid] = prim_min[seg].min(axis=0)
        node_max[nid] = prim_max[seg].max(axis=0)
        cnt = e - s
        if cnt <= leaf_size:
            leaf_start[nid] = s
            leaf_count[nid] = cnt
            continue
        c = centroids[seg]
        cmin = c.min(axis=0)
        cmax = c.max(axis=0)
        axis = int(np.argmax(cmax - cmin))
        if cmax[axis] - cmin[axis] > 0.0:
            # object-median: stable sort by axis coord, ties broken by carrier index
            perm = np.lexsort((seg, c[:, axis]))
            order[s:e] = seg[perm]
        # else: all centroids identical (duplicate positions) — split by count only
        mid = cnt // 2
        left = next_node
        right = next_node + 1
        next_node += 2
        node_left[nid] = left
        node_right[nid] = right
        # push right first so left is processed first (deterministic pre-order)
        stack.append((right, s + mid, e, depth + 1))
        stack.append((left, s, s + mid, depth + 1))

    sl = slice(0, next_node)
    return CarrierBVH(
        node_min[sl].astype(np.float32).copy(),
        node_max[sl].astype(np.float32).copy(),
        node_left[sl].copy(),
        node_right[sl].copy(),
        leaf_start[sl].copy(),
        leaf_count[sl].copy(),
        order.copy(),
        float(k), int(leaf_size), int(max_depth), N, carriers, device,
    )


def _traverse(bvh: CarrierBVH, origins: np.ndarray, directions: np.ndarray):
    """Batched, stack-based BVH traversal (numpy).

    Returns ``(cand_ray, cand_prim, node_visits)``: flat, aligned arrays of candidate
    ``(ray, carrier)`` pairs (a guaranteed superset of the true hits), and the [M]
    AABB-tests-per-ray count (evidence of sub-linear traversal). Ordering within the
    resolver is re-established there, so pairs are returned in traversal order."""
    M = origins.shape[0]
    node_min = bvh.node_min.astype(np.float64)
    node_max = bvh.node_max.astype(np.float64)
    node_left = bvh.node_left
    node_right = bvh.node_right
    leaf_start = bvh.leaf_start
    leaf_count = bvh.leaf_count
    prim_order = bvh.prim_order

    if M == 0 or bvh.n_carriers == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(M, np.int64)

    # sign-preserving epsilon avoids 0*inf NaNs on axis-aligned rays
    d = directions.copy()
    tiny = 1e-12
    zero = np.abs(d) < tiny
    d[zero] = np.where(d[zero] >= 0, tiny, -tiny)
    inv = 1.0 / d                                  # [M,3]

    max_stack = 2 * bvh.max_depth + 8
    stack = np.zeros((M, max_stack), np.int64)
    sp = np.ones(M, np.int64)                      # push root (node 0)
    node_visits = np.zeros(M, np.int64)

    # collect candidate (ray, prim) pairs incrementally
    cand_rays_parts: list[np.ndarray] = []
    cand_prims_parts: list[np.ndarray] = []

    ray_ids = np.arange(M)
    active = sp > 0
    while active.any():
        ar = ray_ids[active]                       # rays with a non-empty stack
        top = stack[ar, sp[ar] - 1]                # pop top node id
        sp[ar] -= 1
        node_visits[ar] += 1

        o = origins[ar]
        di = inv[ar]
        t1 = (node_min[top] - o) * di
        t2 = (node_max[top] - o) * di
        tmin = np.maximum.reduce(np.minimum(t1, t2), axis=1)
        tmax = np.minimum.reduce(np.maximum(t1, t2), axis=1)
        hitbox = tmax >= np.maximum(tmin, 0.0)     # forward semi-infinite ray

        is_leaf = leaf_start[top] >= 0
        leaf_hit = hitbox & is_leaf
        int_hit = hitbox & ~is_leaf

        if leaf_hit.any():
            lr = ar[leaf_hit]
            lnode = top[leaf_hit]
            ls = leaf_start[lnode]
            lc = leaf_count[lnode]
            total = int(lc.sum())
            if total:
                # expand each hit leaf into its prim indices (concatenated aranges)
                rep_ray = np.repeat(lr, lc)
                # offsets 0..lc-1 within each leaf, then + leaf start
                offs = np.arange(total) - np.repeat(np.cumsum(lc) - lc, lc)
                positions = np.repeat(ls, lc) + offs
                cand_rays_parts.append(rep_ray)
                cand_prims_parts.append(prim_order[positions])

        if int_hit.any():
            ir = ar[int_hit]
            inode = top[int_hit]
            left = node_left[inode]
            right = node_right[inode]
            # each ray in ir is unique → scatter is safe
            stack[ir, sp[ir]] = left
            sp[ir] += 1
            stack[ir, sp[ir]] = right
            sp[ir] += 1

        active = sp > 0

    if cand_rays_parts:
        cand_rays = np.concatenate(cand_rays_parts).astype(np.int64)
        cand_prims = np.concatenate(cand_prims_parts).astype(np.int64)
    else:
        cand_rays = np.zeros(0, np.int64)
        cand_prims = np.zeros(0, np.int64)
    return cand_rays, cand_prims, node_visits


# --- small build cache for the single-ray / auto path -------------------------
# Repeated single-ray queries over the SAME carriers (e.g. the publication
# secondary-ray gate casts 24 rays over one asset) reuse the tree instead of
# rebuilding it. Keyed by object identity of the means tensor (a strong reference is
# held so the id cannot be recycled onto different data). Explicit build_carrier_bvh
# never consults this cache.
_BVH_CACHE: list[tuple] = []
_BVH_CACHE_MAX = 2


def get_or_build_bvh(carriers, *, k=3.0, leaf_size=64, device="cpu") -> CarrierBVH:
    means = carriers["means"]
    key = (id(means), int(_to_numpy(means).shape[0]), round(float(k), 9), int(leaf_size), device)
    for entry_key, ref, bvh in _BVH_CACHE:
        if entry_key == key and ref is means:
            return bvh
    bvh = build_carrier_bvh(carriers, k=k, leaf_size=leaf_size, device=device)
    _BVH_CACHE.append((key, means, bvh))
    if len(_BVH_CACHE) > _BVH_CACHE_MAX:
        _BVH_CACHE.pop(0)
    return bvh
