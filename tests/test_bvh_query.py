"""Parity + edge-case tests for the carrier BVH ray query (audit M2).

The acceptance criterion is parity: the BVH path (``method='bvh'``, default) must
return the *same* nearest-hit result as the brute-force O(N) path (``method='brute'``)
for the same rays — same hit carrier (⇒ same depth, colour, normal, confidence,
transmittance, provenance) to float tolerance. These tests assert that on synthetic
scenes (including the tricky edge cases: miss rays, rays whose origin is inside a
carrier AABB, degenerate flat carriers, duplicate carrier positions, min_opacity /
min_confidence filtering) and on the real 129,531-carrier truck asset.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.carrier_query import (  # noqa: E402
    carrier_ray_query,
    carrier_ray_query_batch,
)
from aura.bvh import build_carrier_bvh  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRUCK = REPO / "outputs" / "truck-sidecar.aura" / "carriers.npz"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _assert_same(a, b, *, tol=1e-4):
    """Two RayQueryResults must agree on every field to float tolerance."""
    assert a.provenance == b.provenance
    if a.provenance == "miss":
        return
    assert (a.depth is None) == (b.depth is None)
    if a.depth is not None:
        assert abs(a.depth - b.depth) <= tol, (a.depth, b.depth)
    assert abs(a.transmittance - b.transmittance) <= tol
    assert abs(a.confidence - b.confidence) <= tol
    assert max(abs(x - y) for x, y in zip(a.color, b.color)) <= tol
    if a.normal is not None and b.normal is not None:
        assert max(abs(x - y) for x, y in zip(a.normal, b.normal)) <= 10 * tol
    assert a.semantic_id == b.semantic_id


def _scene(means, scales=None, opacity=None, colors=None, quats=None,
           confidence=None):
    n = len(means)
    means = np.asarray(means, dtype=np.float32)
    if scales is None:
        scales = np.tile([0.3, 0.3, 0.05], (n, 1)).astype(np.float32)
    if opacity is None:
        opacity = np.full(n, 0.8, dtype=np.float32)
    if colors is None:
        colors = np.tile([0.2, 0.7, 0.4], (n, 1)).astype(np.float32)
    if quats is None:
        quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    if confidence is None:
        confidence = np.linspace(0.4, 0.95, n).astype(np.float32)
    return {
        "means": torch.tensor(means),
        "scales": torch.tensor(np.asarray(scales, dtype=np.float32)),
        "quats": torch.tensor(np.asarray(quats, dtype=np.float32)),
        "opacity": torch.tensor(np.asarray(opacity, dtype=np.float32)),
        "colors": torch.tensor(np.asarray(colors, dtype=np.float32)),
        "confidence": torch.tensor(np.asarray(confidence, dtype=np.float32)),
        "sh_degree": 0,
    }


def _random_rays(centroid, radius, m, seed, means=None):
    rng = np.random.default_rng(seed)
    origins = centroid + rng.normal(size=(m, 3)) * radius
    if means is not None:
        targets = means[rng.integers(0, means.shape[0], size=m)]
    else:
        targets = centroid + rng.normal(size=(m, 3)) * radius * 0.2
    dirs = targets - origins
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9, None)
    return origins, dirs


def _parity_over_rays(carriers, origins, dirs, *, min_opacity=0.0, min_confidence=0.0):
    brute = carrier_ray_query_batch(carriers, origins, dirs, method="brute",
                                    min_opacity=min_opacity, min_confidence=min_confidence)
    bvh = carrier_ray_query_batch(carriers, origins, dirs, method="bvh",
                                  min_opacity=min_opacity, min_confidence=min_confidence)
    assert len(brute) == len(bvh) == len(origins)
    for rb, rv in zip(brute, bvh):
        _assert_same(rv, rb)
    return brute, bvh


# --------------------------------------------------------------------------- #
# single-ray API parity
# --------------------------------------------------------------------------- #
def test_single_ray_default_is_bvh_and_matches_brute():
    c = _scene([[0, 0, 5.0], [0, 0, 2.0]],
               colors=[[1.0, 0, 0], [0.0, 1.0, 0.0]],
               confidence=[0.5, 0.9])
    default = carrier_ray_query(c, [0, 0, 0], [0, 0, 1])
    brute = carrier_ray_query(c, [0, 0, 0], [0, 0, 1], method="brute")
    _assert_same(default, brute)
    assert default.provenance == "carrier_query"
    assert abs(default.depth - 2.0) < 1e-3     # nearer carrier at z=2 wins


def test_single_ray_miss_parity():
    c = _scene([[0, 0, 2.0]])
    for method in ("brute", "bvh"):
        r = carrier_ray_query(c, [0, 0, 0], [0, 0, -1], method=method)
        assert r.provenance == "miss" and r.confidence == 0.0


def test_invalid_method_raises():
    c = _scene([[0, 0, 2.0]])
    with pytest.raises(ValueError):
        carrier_ray_query(c, [0, 0, 0], [0, 0, 1], method="octree")


# --------------------------------------------------------------------------- #
# batched parity on a general synthetic cloud
# --------------------------------------------------------------------------- #
def test_batch_parity_random_cloud():
    rng = np.random.default_rng(7)
    means = rng.normal(size=(2000, 3)).astype(np.float32) * 2.0
    scales = (rng.uniform(0.02, 0.4, size=(2000, 3))).astype(np.float32)
    c = _scene(means, scales=scales,
               opacity=rng.uniform(0.05, 0.95, size=2000).astype(np.float32))
    origins, dirs = _random_rays(np.zeros(3), 4.0, 400, seed=1, means=means)
    _parity_over_rays(c, origins, dirs)


def test_batch_parity_with_axis_aligned_rays():
    # axis-aligned directions exercise the zero-component slab branch
    means = np.array([[0, 0, 3.0], [0.5, 0, 3.0], [0, 0, 6.0], [-1.0, 0, 2.0]], np.float32)
    c = _scene(means)
    origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], np.float64)
    dirs = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float64)
    _parity_over_rays(c, origins, dirs)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_edge_ray_origin_inside_aabb():
    # origin sits inside the first carrier's hit sphere; tmin is negative
    c = _scene([[0, 0, 0.0], [0, 0, 4.0]])
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[0.0, 0.0, 1.0]])
    _parity_over_rays(c, origins, dirs)


def test_edge_degenerate_flat_carriers():
    # a fully flat carrier (one zero scale axis) — AABB uses max(scale) so it is
    # still a valid box; some carriers are entirely zero-scale (point AABB)
    means = np.array([[0, 0, 2.0], [0.2, 0, 2.5], [0, 0, 5.0]], np.float32)
    scales = np.array([[0.3, 0.3, 0.0], [0.0, 0.0, 0.0], [0.4, 0.1, 0.2]], np.float32)
    c = _scene(means, scales=scales)
    origins, dirs = _random_rays(np.array([0, 0, 3.0]), 3.0, 200, seed=3, means=means)
    _parity_over_rays(c, origins, dirs)


def test_edge_duplicate_positions():
    # many carriers at the exact same position — the case that breaks Morton LBVH
    pos = [0.0, 0.0, 3.0]
    means = np.array([pos] * 40 + [[1.0, 0.0, 2.0]], np.float32)
    opacity = np.concatenate([np.full(40, 0.05, np.float32), np.array([0.9], np.float32)])
    c = _scene(means, opacity=opacity)
    origins, dirs = _random_rays(np.array(pos), 3.0, 200, seed=4, means=means)
    _parity_over_rays(c, origins, dirs)


def test_edge_all_duplicate_positions_build_terminates():
    # a pathological cloud of identical points must still build a finite tree
    means = np.array([[1.0, 2.0, 3.0]] * 500, np.float32)
    bvh = build_carrier_bvh(_scene(means))
    assert bvh.n_carriers == 500
    assert bvh.max_depth < 40  # split-by-count keeps depth ~ log2(N)
    r = carrier_ray_query(_scene(means), [1.0, 2.0, 0.0], [0, 0, 1])
    assert r.provenance in ("carrier_query", "miss")


def test_edge_min_opacity_filtering_parity():
    means = np.array([[0, 0, 2.0], [0, 0, 4.0], [0, 0, 6.0]], np.float32)
    opacity = np.array([0.05, 0.9, 0.9], np.float32)  # first below threshold
    c = _scene(means, opacity=opacity, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origins = np.array([[0, 0, 0.0]])
    dirs = np.array([[0, 0, 1.0]])
    # without filtering the low-opacity floater is a candidate; with min_opacity it is
    # skipped — both paths must agree in each regime
    _parity_over_rays(c, origins, dirs, min_opacity=0.0)
    b, _ = _parity_over_rays(c, origins, dirs, min_opacity=0.1)
    assert b[0].provenance == "carrier_query"


def test_edge_min_confidence_filtering_parity():
    means = np.array([[0, 0, 2.0], [0, 0, 4.0]], np.float32)
    conf = np.array([0.1, 0.95], np.float32)
    c = _scene(means, confidence=conf)
    origins = np.array([[0, 0, 0.0]])
    dirs = np.array([[0, 0, 1.0]])
    _parity_over_rays(c, origins, dirs, min_confidence=0.5)


def test_empty_scene_returns_miss():
    c = _scene(np.zeros((0, 3), np.float32))
    for method in ("brute", "bvh"):
        r = carrier_ray_query(c, [0, 0, 0], [0, 0, 1], method=method)
        assert r.provenance == "miss"


# --------------------------------------------------------------------------- #
# persistent handle + stats
# --------------------------------------------------------------------------- #
def test_prebuilt_handle_matches_and_reports_sublinear_stats():
    rng = np.random.default_rng(11)
    means = rng.normal(size=(3000, 3)).astype(np.float32) * 2.0
    c = _scene(means, scales=rng.uniform(0.02, 0.3, size=(3000, 3)).astype(np.float32))
    origins, dirs = _random_rays(np.zeros(3), 4.0, 256, seed=2, means=means)

    bvh = build_carrier_bvh(c)
    reused = bvh.query_batch(origins, dirs)
    fresh = carrier_ray_query_batch(c, origins, dirs)  # builds its own tree
    brute = carrier_ray_query_batch(c, origins, dirs, method="brute")
    for a, b, br in zip(reused, fresh, brute):
        _assert_same(a, br)
        _assert_same(b, br)

    _, stats = bvh.query_batch(origins, dirs, return_stats=True)
    assert stats["node_visits"].mean() < stats["n_carriers"]        # sub-linear
    assert stats["candidates"].mean() < stats["n_carriers"]


# --------------------------------------------------------------------------- #
# real trained asset — the acceptance-critical parity check
# --------------------------------------------------------------------------- #
@pytest.mark.local_data
def test_truck_asset_parity_random_rays():
    if not TRUCK.exists():
        pytest.skip("truck asset not present")
    torch.set_num_threads(2)
    data = np.load(TRUCK)
    carriers = {k: torch.tensor(data[k], dtype=torch.float32)
                for k in ("means", "scales", "quats", "opacity", "colors", "confidence")
                if k in data.files}
    means = data["means"]
    assert means.shape[0] == 129531
    centroid = means.mean(axis=0)
    radius = float(np.linalg.norm(means.std(axis=0)) * 3.0 + 2.0)
    origins, dirs = _random_rays(centroid, radius, 300, seed=0, means=means)

    brute, bvh = _parity_over_rays(carriers, origins, dirs, min_opacity=0.1)
    hits = sum(1 for r in brute if r.provenance != "miss")
    assert hits > 100  # the rays are aimed into the cloud; most should hit

    _, stats = carrier_ray_query_batch(carriers, origins, dirs, min_opacity=0.1,
                                       return_stats=True)
    # sub-linear traversal: mean node visits and candidates are a small fraction of N
    assert stats["node_visits"].mean() < 0.2 * stats["n_carriers"]
    assert stats["candidates"].mean() < 0.2 * stats["n_carriers"]
