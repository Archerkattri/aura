# AURA Architecture

AURA (Adaptive Unified Radiance Asset) is an end-to-end 3D scene reconstruction
engine that sits one step beyond 3D Gaussian Splatting in the progression:

```text
COLMAP / MVS  →  NeRF  →  3D Gaussian Splatting  →  AURA
```

The core idea is that a single primitive type — Gaussian or otherwise — is
insufficient for every region of a scene. AURA instead trains a **mixed
adaptive carrier** representation where each spatial region is assigned the
most appropriate primitive type based on measured evidence.

## Prior Art and Motivation

**COLMAP** provides robust camera calibration, sparse SfM geometry, and dense
MVS reconstruction, but does not produce a photorealistic radiance
representation suitable for novel-view synthesis.

**NeRF** introduced continuous volumetric scene functions optimized from posed
images via differentiable volume rendering. High quality but slow to train and
render.

**3D Gaussian Splatting** replaced the neural field with millions of explicit
anisotropic Gaussians, enabling real-time rendering. Its known weaknesses
motivate AURA:

- Screen-space rasterization causes view-consistency artifacts and popping.
- Gaussians are weak geometry carriers; mesh extraction requires additional
  constraints.
- Real ray queries, secondary rays, reflections, and robotics-style sensing are
  awkward in a splat rasterizer.
- Semantics, editability, material behavior, confidence, and LOD are
  downstream add-ons rather than native training targets.

Work such as 2DGS, SuGaR, GOF, Mip-Splatting, StopThePop, 3DGRT, and EVER
patches individual weaknesses. AURA takes a different approach: assign each
region the right primitive type from a typed carrier family.

## Carrier Families

AURA trains a scene as a mixture of **native carrier types**. Each carrier
captures a different kind of physical evidence:

| Carrier | Best for |
|---|---|
| Surface radiance cell | Stable opaque geometry, normals, collision, edit handles |
| Volumetric density cell | Translucent, fuzzy, or uncertain regions |
| Bounded beta kernel | Compact local support, fewer primitive hits per ray |
| Gabor / frequency carrier | High-frequency texture and alias control |
| Neural residual primitive | View-dependent effects that simpler carriers cannot explain |
| Semantic / object carrier | Object grouping, language anchors, per-object confidence and LOD |
| Gaussian fallback carrier | Regions where structured evidence does not justify a stronger primitive |

A Gaussian fallback carrier is explicitly labeled as such. When the training
evidence is strong enough for a surface, volume, beta, gabor, neural, or
semantic carrier, those native types dominate even if image residual remains.
This makes the scene representation auditable.

## Reconstruction Pipeline

```text
posed images / depth / masks / normals
         ↓
  capture manifest (JSON)
         ↓
  packed capture tensors   ←  PNG / PPM / COLMAP dense maps / imageio EXR
         ↓
  tiled PyTorch optimization
    • device-resident asset batching
    • mask-aware pixel sampling
    • camera ray construction
    • carrier response evaluation (color, opacity, depth, normal, confidence,
      semantics, material, residual)
    • front-to-back compositing
    • image / depth / normal / mask / confidence / query loss
    • gradient clipping + checkpoint snapshots
         ↓
  adaptive carrier evolution
    • split: high-residual volume → beta detail children
    • promote: high-residual semantic → neural residual children
    • merge: converged volume + beta children → single volume carrier
    • demote: converged semantic + neural residual children
    • hysteresis prevents immediate re-creation of removed children
         ↓
  optimized .aura package
    • queryable runtime ray-query semantics
    • confidence maps, edit metadata, semantic graph
    • JSON schemas, chunk metadata, LOD layers
```

## Adaptive Evolution Contract

Carrier evolution is a **deterministic policy contract**, not an inline side
effect. Each training iteration records one decision per predicted element:

- action (split / promote / merge / demote / retain)
- reason (threshold, measured loss values)
- element IDs created, removed, or retained

Per-iteration reports include action counts so benchmark and CLI output can be
verified without parsing free-form strings. The reference policy is
intentionally conservative and designed to be auditable as the optimizer moves
from fixture losses toward real capture residuals.

## CUDA Renderer

A compiled CUDA renderer (`aura cuda-kernel-build-report --build`) is dispatched
through a pybind11 `render_rays` binding over packed scene and ray tensors.
All seven native carrier types have measured GPU parity against the PyTorch
renderer. A production GPU BVH traversal kernel (`render_rays_bvh`) implements
flattened median-split element BVH traversal, replacing the brute-force element
scan as the dispatched production path.

## Package Format

An `.aura` scene package is a directory containing:

- `manifest.json` — top-level metadata and schema version
- `elements.json` — carrier registry with typed payloads
- `chunks.json` — adaptive decomposition / LOD chunk layout
- `exchange.json` — ingest evidence and decomposition audit trail
- `semantic_graph.json` — object grouping and language anchors
- `capture_manifest.json` — source capture asset references (training input)
- `training_dataset.json` — tiled training batch records

JSON schemas for all files are shipped in `src/aura/schemas/` and are applied
for runtime validation when a package is loaded.

## 3DGS as Evidence Ingest Only

3DGS exports (`.ply`, `point_cloud/iteration_*/point_cloud.ply`) can be ingested
as `EvidenceSample` records via `aura import-3dgs`. They become Gaussian
fallback carriers only when native carrier assignment does not justify a stronger
representation. All 3DGS-specific logic lives under `aura.ingest`; splats are
evidence inputs, not the native representation center.

## Research References

- COLMAP: https://demuc.de/colmap/
- NeRF: https://arxiv.org/abs/2003.08934
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079
- 2DGS: https://arxiv.org/abs/2403.17888
- SuGaR: https://arxiv.org/abs/2311.12775
- Gaussian Opacity Fields: https://arxiv.org/abs/2404.10772
- StopThePop: https://arxiv.org/abs/2402.00525
- Mip-Splatting: https://niujinshuchong.github.io/mip-splatting/
- 3D Gaussian Ray Tracing (3DGRT): https://arxiv.org/abs/2407.07090
- EVER: https://arxiv.org/abs/2410.01804
