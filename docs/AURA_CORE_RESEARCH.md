# AURA-Core Research Direction

AURA should be the next reconstruction step after 3D Gaussian Splatting, in the
same sense that NeRF moved beyond COLMAP/MVS and 3DGS moved beyond NeRF.

That means AURA is not primarily a file format, package wrapper, or 3DGS
converter. AURA-Core must be an end-to-end reconstruction engine:

```text
images/video -> poses/depth/priors -> native adaptive carriers -> differentiable
render/optimization -> runtime ray-query/render asset
```

## What The Previous Steps Solved

COLMAP is an end-to-end image-based 3D reconstruction pipeline for
Structure-from-Motion and Multi-View Stereo. Its strength is robust camera
calibration, sparse geometry, and dense MVS reconstruction, but it does not
directly produce a photorealistic neural/radiance representation for novel-view
rendering.

NeRF changed the center of gravity by optimizing a continuous volumetric scene
function from posed images. Its core representation is queried along camera rays
and optimized through differentiable volume rendering. The cost is slow training
and rendering.

3D Gaussian Splatting changed the center again by replacing the expensive neural
field with millions of explicit, optimizable anisotropic Gaussian primitives.
Starting from sparse calibration points, it interleaves optimization and density
control, then renders in real time with visibility-aware splatting.

## What 3DGS Still Leaves Open

Recent papers show the pressure points clearly:

- Rasterized splats approximate volume rendering with screen-space ordering,
  which causes popping and view-consistency artifacts.
- Gaussian primitives are explicit and fast, but they are weak geometry
  carriers: surfaces, normals, collision, and mesh extraction need extra
  constraints or post-processes.
- Real ray queries, secondary rays, distorted cameras, shadows, reflections, and
  robotics-style sensing are awkward in a splat rasterizer.
- Large scenes often need millions of primitives, creating storage and memory
  pressure.
- Semantics, editability, material behavior, confidence, and LOD are usually
  downstream add-ons rather than native training targets.

The field is already patching those weaknesses with 2DGS, SuGaR, GOF, Mip-
Splatting, StopThePop, 3DGRT, and EVER. AURA should absorb the lesson: the next
step is not "better Gaussian splats." It is a hybrid adaptive radiance
reconstruction system where each region gets the right primitive.

## AURA-Core Thesis

AURA-Core should train a mixed explicit scene representation directly from
images, video, and geometry priors. A Gaussian is only one possible carrier.

Native carrier families:

- surface radiance cells for stable opaque geometry, normals, collision, and
  edit handles;
- volumetric density cells for translucent, fuzzy, or uncertain regions;
- bounded beta kernels for compact local support and fewer primitive hits;
- gabor/frequency carriers for high-frequency texture and alias control;
- neural residual primitives for view-dependent effects that simpler carriers
  cannot explain;
- semantic/object carriers for grouping, language/editing, confidence, and
  object-level operations;
- Gaussian fallback carriers only where the evidence does not justify a more
  structured primitive.

The key differentiator is adaptive carrier evolution during training:

```text
initialize evidence cells
render and compare against posed images
estimate residuals, uncertainty, normals, semantics, and ray-query needs
promote/split/merge carriers
optimize carrier parameters
emit a runtime AURA scene with ray-query semantics
```

## MVP Bar

The repo should move toward these milestones, in this order:

1. `aura write-training-frames-demo` and `aura reconstruct-demo --frames`:
   a posed color/depth/semantic training dataset contract with native evidence
   region specs that is schema-validated and builds a native AURA scene without
   3DGS.
2. CPU reference loop: cast posed training rays, compute per-frame image/depth
   losses, update carrier color/depth parameters, and emit a training report
   with carrier evolution decisions.
3. Adaptive evolution: split high-residual volume regions into beta detail
   carriers, promote semantic residuals into neural residual carriers, and
   merge/demote those children when measured residuals converge.
4. Ray-query correctness: first hit, depth, normal, transmittance, material,
   confidence, provenance, shadow/reflection/collision readiness, and scored
   expected-probe benchmarks.
5. Reconstruction benchmarks: compare adaptive carrier evolution against static
   carriers with measured image/depth loss, action counts, and final package
   carrier counts.
6. GPU kernels: implement the same contract for real throughput only after the
   reference reconstruction path is correct.
7. Baselines: compare end-to-end against COLMAP outputs, NeRF-style volume
   rendering, and 3DGS on standard scenes.

## Research Sources

- COLMAP: https://demuc.de/colmap/
- NeRF: https://arxiv.org/abs/2003.08934
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079
- 2DGS: https://arxiv.org/abs/2403.17888
- SuGaR: https://arxiv.org/abs/2311.12775
- Gaussian Opacity Fields: https://arxiv.org/abs/2404.10772
- StopThePop: https://arxiv.org/abs/2402.00525
- Mip-Splatting: https://niujinshuchong.github.io/mip-splatting/
- 3D Gaussian Ray Tracing: https://arxiv.org/abs/2407.07090
- EVER: https://arxiv.org/abs/2410.01804
