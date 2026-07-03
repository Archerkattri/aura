# P5 — BVH-accelerated batched carrier ray query

P5 removes the brute-force cost from AURA's on-asset ray query (audit item **M2**).
`carrier_query.carrier_ray_query` used to compute the perpendicular distance from a
ray to **every** carrier and argsort the hits — `O(N)` per ray, one ray at a time. On
the 129,531-carrier truck asset that is 129k distance evaluations for a single ray.

P5 adds a deterministic, pure-numpy **BVH** (`src/aura/bvh.py`) over the carrier
hit-volumes and a **batched** query API, replacing the `O(N)` scan with a sub-linear,
stack-based traversal that is vectorised across rays. The public single-ray function
keeps its exact signature and return type; a `method="bvh"|"brute"` switch (default
`bvh`) preserves the old path for parity testing.

Artifacts: `src/aura/bvh.py`, `src/aura/carrier_query.py`
(`carrier_ray_query`, `carrier_ray_query_batch`, `build_carrier_bvh`),
`tests/test_bvh_query.py`, `experiments/bvh_query_benchmark.py`
→ `outputs/bvh_query_benchmark.json`.

---

## Honest scope (read first)

This is an **algorithmic** fix — `O(N)` → sub-linear node visits, plus a batched
API — **validated for correctness on a real trained asset**. It is a CPU-numpy
structure.

It is **NOT** a wall-clock comparison against the CUDA LBVH ray tracers in **3DGRT /
3DGUT** (now landing in gsplat `main`: nonlinear cameras + secondary rays, native
CUDA MCMC). We cannot and do not claim to match their GPU wall-clock from CPU Python —
that remains the **deferred bar**. AURA's secondary rays (shadow/reflection in the
publication gate) remain **readiness probes**, not a physically-based tracer.

What P5 *does* claim, with artifacts: the BVH returns **bit-for-parity** identical
results to the brute-force path, examines a small **sub-linear** fraction of the
scene per ray, and gives a real CPU throughput gain on batches.

---

## Design

### The hit-volume, and why the AABB is a cube of `k·max(scale)`

The brute-force query accepts a carrier when the ray passes within `k·max(scale)` of
its centre **and** in front of the origin, then, among hits, walks front-to-back and
takes the carrier where accumulated alpha first crosses `0.5`. The acceptance region
per carrier is therefore an **isotropic sphere** of radius `r = k·max(scale)` centred
at the mean (the query's existing convention — it uses `scales.max(dim=-1)`, not the
anisotropic axes).

Each carrier's leaf AABB is the axis-aligned cube `mean ± r`. That cube tightly bounds
the hit-sphere, and the sphere already contains the quaternion-rotated covariance
ellipsoid `k·scale`, so the cube conservatively bounds **both**. Consequently:

> Any ray that intersects a carrier's hit-sphere intersects its AABB, and every
> ancestor node AABB (a union of descendant AABBs) also contains it — so a leaf
> holding a true hit is **never pruned**. The BVH candidate set is a guaranteed
> **superset** of the brute-force hit set.

The BVH only *narrows* the set; the final hit test, front-to-back accumulation and
payload extraction are the **exact brute-force arithmetic**, applied to the
candidates. Superset-of-hits + identical resolution ⇒ identical result. **Parity is
the acceptance criterion, and it is structural, not incidental.**

### Median-split, not Morton/LBVH — justification

Construction is an **object-median** (split-by-count) binary BVH, built top-down with
an explicit stack, in pure numpy. Chosen over a Morton-code LBVH because:

* **Duplicate carrier positions** (an explicit edge case) collapse Morton codes to
  ties, which degenerate Karras' internal-node split into order-dependent, unbalanced
  hierarchies. Splitting a segment at `count//2` makes strictly smaller children and
  terminates regardless of coordinate ties — verified by
  `test_edge_all_duplicate_positions_build_terminates` (500 identical points → depth
  `< 40`).
* **Degenerate flat carriers** (a zero scale axis) still get a valid box because the
  cube uses `max(scale)`; the AABB is only a zero-volume point when *every* axis is
  zero, which the exact hit test then rejects consistently.
* Construction is short, deterministic and auditable; the tree is balanced
  (`depth ≈ ceil(log2(N/leaf_size))`), and the flat node arrays traverse with the same
  batched stack loop a GPU LBVH would use — the structural bar is met without the
  Morton fragility.

`leaf_size` defaults to **64** — swept on the truck asset (below), it minimises the
Python-loop traversal cost (fewer node pops) while the vectorised resolver keeps the
extra candidates cheap.

### Batched traversal

`query_batch(origins[M,3], dirs[M,3])` keeps a per-ray stack (`stack[M, depth]`,
`sp[M]`) and loops `while any(sp>0)`: each iteration pops one node per active ray,
runs the slab ray–AABB test vectorised across the active rays, pushes both children
for internal hits, and collects leaf primitives for leaf hits. The slab test uses a
sign-preserving epsilon on the direction so axis-aligned rays never hit a `0·inf`
NaN (covered by `test_batch_parity_with_axis_aligned_rays`); the test is
scale-invariant, so the traversal never normalises the direction and the resolver
applies the single normalisation the brute path does.

Candidate resolution is then **fully vectorised across the batch** (`_resolve_batch`):
one gathered hit test over all candidate pairs, a per-ray `(t, carrier-index)` order,
a padded-matrix `cumprod` accumulation that reproduces the per-ray `cumprod` exactly,
and the `0.5`-crossing (first crossing, else `argmax(accum)`) — so the whole batch
pays the Python/torch launch overhead **once** instead of once per ray. This is the
difference between the ~1.2× (per-ray resolve) and the multi-× (batched resolve)
throughput.

### API and reuse

* `carrier_ray_query(carriers, origin, direction, *, k, min_opacity, min_confidence,
  device, method="bvh")` — unchanged public signature/return; `method` selects
  bvh/brute. A small auto-cache reuses the tree across repeated single-ray calls on
  the same carriers (e.g. the publication secondary-ray gate's 24 rays).
* `carrier_ray_query_batch(carriers, origins, directions, *, …, bvh=None,
  return_stats=True)` — M rays; pass a prebuilt handle to skip the build.
* `build_carrier_bvh(carriers, *, k=3.0, leaf_size=64)` → `CarrierBVH`. **This is the
  streaming-friendly path**: build once, call `bvh.query_batch(...)` for each frame's
  rays. The tree is built over *all* carriers and `min_opacity`/`min_confidence` are
  applied at resolve time, so one handle serves every threshold.

---

## Parity guarantee (the acceptance criterion)

`tests/test_bvh_query.py` asserts BVH == brute (same hit carrier ⇒ same depth, colour,
normal, confidence, transmittance, provenance, to float tolerance) on:

* **Synthetic scenes** — general random clouds, single-ray API, and every required
  edge case: **miss rays**, **rays whose origin is inside a carrier AABB**,
  **degenerate flat carriers** (zero-scale axes, all-zero-scale points), **duplicate
  positions**, and **`min_opacity` / `min_confidence` filtering** (each regime agrees).
* **The real 129,531-carrier truck asset** (`outputs/truck-sidecar.aura/carriers.npz`),
  300 random rays, `@pytest.mark.local_data` — **0 mismatches**.

The benchmark independently re-checks parity on the exact rays it times: **0
mismatches** at every N and filter level.

---

## Benchmark (CPU-only Python; `experiments/bvh_query_benchmark.py`)

Machine: `Linux x86_64`, torch 2.11.0+cu128, **CPU only** (GPUs excluded — this is a
CPU-numpy structure), 4 torch threads. Batched = 1024 rays, best-of-3.
`N=1e5` is the **real trained truck asset**. Full JSON:
`outputs/bvh_query_benchmark.json`.

### Batched throughput — brute vs BVH (prebuilt handle)

| scene | N | brute rays/s | BVH rays/s | speedup | parity |
|---|--:|--:|--:|--:|--:|
| synthetic | 1,000 | 2,797 | 26,112 | **9.3×** | 0 |
| synthetic | 10,000 | 1,354 | 5,686 | **4.2×** | 0 |
| **truck (real)** | 129,531 | 363 | 1,151 | **3.2×** | 0 |

(rows above use the realistic `min_opacity=0.1` setting; the pure-geometry no-filter
truck case is 1.6× — every low-opacity floater becomes a hit, inflating the
accumulation matrix. Both are in the JSON.)

### Sub-linear traversal — node visits & candidates per ray (truck, N=129,531)

| metric | mean per ray | fraction of N |
|---|--:|--:|
| AABB node visits | 499 | **0.39%** |
| candidate carriers examined | 9,343 | **7.2%** |
| tree nodes / max depth | 4,095 / 11 | — |

The brute path touches **100%** of carriers per ray; the BVH touches **0.4%** of tree
nodes and examines **7.2%** of carriers — direct evidence of sub-linear traversal. The
per-ray-visit fraction shrinks with N (2.3% → 1.8% → 0.4% at 1e3 → 1e4 → 1e5).

### Single-ray, one-off

For a single one-off ray the BVH **loses** on CPU (truck: 79 vs 378 rays/s) because it
pays a full `~0.38 s` tree build with nothing to amortise it over. This is expected and
honest: the BVH is for **batches and streaming**, not one-shot single rays — which is
exactly why `carrier_ray_query` still ships the `brute` path and why streaming callers
should hold a `build_carrier_bvh` handle.

---

## Tests / CI

* `tests/test_bvh_query.py` — 13 CI tests + 1 `local_data` truck-parity test.
* Unchanged and green: `tests/test_carrier_query.py`, `tests/test_carrier_ray_query.py`,
  `tests/test_cli_contract.py`, and the untouched `tests/test_publication_gates_real.py`
  (its secondary-ray/relight gates now flow through the default BVH path via
  `publication._probe_secondary_rays`, unmodified).
* CI selection `pytest -m "not gpu and not local_data"` stays green.
