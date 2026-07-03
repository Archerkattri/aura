#!/usr/bin/env python3
"""BVH carrier-ray-query benchmark (audit M2): brute O(N) vs BVH sub-linear traversal.

Measures rays/sec for the brute-force per-ray query and the BVH-accelerated query,
for single-ray and batched (M rays) workloads, at carrier counts N ∈ {1e3, 1e4, 1e5}
where 1e5 is the REAL trained 129,531-carrier truck asset. Also reports measured node
visits and candidates per ray as evidence of sub-linear traversal, and verifies the
two paths agree (parity) on the exact rays timed.

Honest scope: CPU-only Python/numpy. This quantifies an *algorithmic* fix
(O(N) → sub-linear node visits + a batched API) and its CPU wall-clock; it is NOT a
wall-clock comparison against the CUDA LBVH ray tracers in 3DGRT / 3DGUT (gsplat
main), which remains the deferred bar. See docs/P5_BVH_RAY_QUERY.md.

Usage:
    PYTHONPATH=src .gpu_venv/bin/python experiments/bvh_query_benchmark.py \
        --out outputs/bvh_query_benchmark.json
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _synthetic(n, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    means = (rng.normal(size=(n, 3)) * 2.0).astype(np.float32)
    scales = rng.uniform(0.02, 0.3, size=(n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    opacity = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
    colors = rng.uniform(0, 1, size=(n, 3)).astype(np.float32)
    confidence = rng.uniform(0.2, 0.99, size=n).astype(np.float32)
    return {
        "means": torch.tensor(means), "scales": torch.tensor(scales),
        "quats": torch.tensor(quats), "opacity": torch.tensor(opacity),
        "colors": torch.tensor(colors), "confidence": torch.tensor(confidence),
        "sh_degree": 0,
    }, means


def _load_truck():
    import torch
    path = ROOT / "outputs" / "truck-sidecar.aura" / "carriers.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    carriers = {k: torch.tensor(data[k], dtype=torch.float32)
                for k in ("means", "scales", "quats", "opacity", "colors", "confidence")
                if k in data.files}
    return carriers, data["means"]


def _rays(means, m, seed):
    rng = np.random.default_rng(seed)
    centroid = means.mean(axis=0)
    radius = float(np.linalg.norm(means.std(axis=0)) * 3.0 + 2.0)
    origins = centroid + rng.normal(size=(m, 3)) * radius
    targets = means[rng.integers(0, means.shape[0], size=m)]
    dirs = targets - origins
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9, None)
    return origins, dirs


def _time(fn, *, repeats=3):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _parity(a, b, tol=1e-3):
    mism = 0
    for x, y in zip(a, b):
        if x.provenance != y.provenance:
            mism += 1
            continue
        if x.provenance == "miss":
            continue
        if abs((x.depth or 0.0) - (y.depth or 0.0)) > tol:
            mism += 1
            continue
        if abs(x.transmittance - y.transmittance) > tol or abs(x.confidence - y.confidence) > tol:
            mism += 1
            continue
        if max(abs(p - q) for p, q in zip(x.color, y.color)) > tol:
            mism += 1
    return mism


def _batch_measure(carriers, bvh, bo, bd, min_opacity):
    from aura.carrier_query import carrier_ray_query_batch
    tbb, rbb = _time(lambda: carrier_ray_query_batch(carriers, bo, bd, method="brute",
                                                     min_opacity=min_opacity))
    tvb, rvb = _time(lambda: bvh.query_batch(bo, bd, min_opacity=min_opacity))
    return {
        "min_opacity": min_opacity,
        "brute_rays_per_s": round(len(bo) / tbb, 1),
        "bvh_reuse_rays_per_s": round(len(bo) / tvb, 1),
        "speedup_reuse_vs_brute": round(tbb / tvb, 2),
        "parity_mismatches": _parity(rbb, rvb),
    }


def bench_case(name, carriers, means, *, single=50, batch=1024, seed=0):
    from aura.carrier_query import carrier_ray_query, carrier_ray_query_batch
    from aura.bvh import build_carrier_bvh

    N = int(means.shape[0])
    row = {"scene": name, "carriers": N}

    # ---- single-ray: brute vs bvh. BVH one-shot pays a full tree build (via the
    # auto-cache), so it is expected to LOSE on CPU for one-off single rays; the
    # streaming column (prebuilt handle, per-ray) isolates traversal+resolve. -----
    so, sd = _rays(means, single, seed=seed + 1)
    bvh = build_carrier_bvh(carriers)
    build_t, _ = _time(lambda: build_carrier_bvh(carriers))
    tb, rb = _time(lambda: [carrier_ray_query(carriers, so[i], sd[i], method="brute")
                            for i in range(single)])
    tv, rv = _time(lambda: [carrier_ray_query(carriers, so[i], sd[i], method="bvh")
                            for i in range(single)])
    ts, rs = _time(lambda: [bvh.query(so[i], sd[i]) for i in range(single)])
    row["single"] = {
        "rays": single,
        "brute_rays_per_s": round(single / tb, 1),
        "bvh_autocache_rays_per_s": round(single / tv, 1),
        "bvh_streaming_prebuilt_rays_per_s": round(single / ts, 1),
        "build_seconds": round(build_t, 4),
        "parity_mismatches": _parity(rb, rv) + _parity(rb, rs),
    }

    # ---- batched: brute (per-ray loop) vs bvh (prebuilt handle) ------------------
    bo, bd = _rays(means, batch, seed=seed + 2)
    tvb_build, _ = _time(lambda: carrier_ray_query_batch(carriers, bo, bd, method="bvh"))
    row["batch"] = {
        "rays": batch,
        "bvh_incl_build_rays_per_s": round(batch / tvb_build, 1),
        "geometry_no_filter": _batch_measure(carriers, bvh, bo, bd, 0.0),
        "filtered_min_opacity_0p1": _batch_measure(carriers, bvh, bo, bd, 0.1),
    }
    _, stats = bvh.query_batch(bo, bd, return_stats=True)
    row["traversal"] = {
        "tree_nodes": int(stats["nodes"]),
        "max_depth": int(stats["max_depth"]),
        "node_visits_per_ray_mean": round(float(stats["node_visits"].mean()), 1),
        "node_visits_per_ray_max": int(stats["node_visits"].max()),
        "candidates_per_ray_mean": round(float(stats["candidates"].mean()), 1),
        "node_visits_frac_of_N": round(float(stats["node_visits"].mean()) / N, 4),
        "candidates_frac_of_N": round(float(stats["candidates"].mean()) / N, 4),
    }
    print(json.dumps(row, indent=2), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bvh_query_benchmark.json")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    rows = []
    for n in (1000, 10000):
        carriers, means = _synthetic(n, seed=n)
        rows.append(bench_case(f"synthetic_{n}", carriers, means, batch=args.batch, seed=n))

    truck, tmeans = _load_truck()
    if truck is not None:
        rows.append(bench_case("truck_129531", truck, tmeans, batch=args.batch, seed=99))
    else:
        print("truck asset absent — skipping N=1e5 real-asset row", flush=True)

    payload = {
        "format": "AURA_BVH_RAY_QUERY_BENCHMARK",
        "honest_scope": (
            "CPU-only numpy/torch. Quantifies an algorithmic fix (O(N) -> sub-linear "
            "node visits + batched API) and its CPU wall-clock. NOT a wall-clock "
            "comparison against the CUDA LBVH tracers in 3DGRT/3DGUT (gsplat main), "
            "which remains the deferred bar. Secondary rays are readiness probes."
        ),
        "machine": {
            "cpu_only": True,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": args.threads,
        },
        "parity_note": (
            "parity_mismatches counts rays where the BVH result differs from the "
            "brute-force result over the exact rays timed; 0 == exact parity."
        ),
        "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
