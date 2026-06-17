# AURA GPU-Ready MVP

This package now contains the GPU-ready skeleton for AURA:

- adaptive carrier registry;
- native carrier payload models;
- payload/carrier consistency validation with typed payload constructor checks;
- evidence-to-carrier assignment;
- evidence-to-element adaptive decomposition with deterministic carrier/LOD
  chunking;
- package-level confidence maps and edit metadata;
- semantic/object graph package artifact;
- bounded local elements and carrier/LOD chunks;
- runtime export reports for carrier/LOD chunks and native ray-query contract
  fields;
- carrier-aware reference ray queries;
- carrier-specific reference query tests for surface, volume, beta, gabor,
  neural residual, semantic, and covariance-weighted Gaussian fallback
  carriers;
- CPU reference interaction probes for hit points, shadow transmittance,
  reflection directions, and collision proxy distances;
- front-to-back compositing;
- tiny JSON/ASCII/binary little-endian PLY 3DGS export fixture reading;
- quaternion-aware PLY covariance conversion from 3DGS log-scales;
- AURA-Ingest adapters that convert 3DGS, depth priors, semantic masks, and
  sparse point priors into `EvidenceSample` contracts;
- COLMAP binary/text sparse-model conversion into native capture manifests;
- COLMAP sparse point depth layers for native region initialization;
- standard COLMAP dense depth-map links and deterministic summary loading;
- standard COLMAP normal-map links and average-normal summary loading;
- loaded depth assets seed deterministic multi-region native surface priors;
- loaded mask assets seed native semantic/object priors;
- dependency-free PNG and PPM/PGM capture asset summaries for manifest-backed
  native training fixtures;
- chunk and LOD metadata from adaptive decomposition;
- native `.aura` package writer;
- native `.aura` package loader/validator;
- explicit `.aura` format/version compatibility checks;
- JSON package inspection output and JSON Schema documents;
- runtime JSON Schema validation for package files;
- deterministic orthographic package preview rendering and reference
  MSE/PSNR/SSIM/LPIPS-proxy image metrics;
- CPU differentiable reference ray samples that preserve image/depth/normal/
  query losses and ray-query contract outputs in reconstruction reports;
- residual-driven confidence updates and confidence maps on optimized native
  carriers;
- per-pixel capture asset tensors for PNG, PPM/PGM, COLMAP dense maps, and
  optional `imageio` EXR/HDR/video assets;
- packed host float buffers for capture tensor payloads, avoiding Python tuple
  payloads during manifest asset loading;
- torch/CUDA capture asset batching through `torch_capture_asset_batch`, which
  stacks manifest image/depth/mask/normal tensors and presence masks on the
  selected device;
- per-pixel capture training target generation through
  `capture_tensors_to_render_targets` and `torch_capture_training_batch`;
- torch reference rendering directly from capture training batches through
  `torch_render_capture_training_batch`, using carrier parameter tensors for
  every supported native/fallback carrier;
- live torch render objectives through `torch_render_target_objective`, exposing
  image/depth/normal/mask losses over carrier parameter tensors;
- torch reference optimization steps through `torch_optimize_capture_batch`,
  which runs repeated batched AURA forward passes, records image/depth/query/
  normal losses, and applies gradient updates to native carrier tensors;
- configurable adaptive carrier evolution thresholds for split/promote/merge/
  demote actions, emitted in reconstruction reports;
- explicit torch carrier kernel specs and autograd parameter tensors for
  surface, volume, beta, gabor, neural residual, semantic, and Gaussian fallback
  semantics;
- packaged CUDA carrier source symbols for surface, volume, beta, gabor, neural,
  semantic, and Gaussian fallback carriers, still gated as non-production until
  compiled extension tests and benchmarks exist;
- `cuda-kernel-build-report --build` for GPU machines to attempt native carrier
  CUDA extension compile/load without changing the default CPU-safe test path;
- `reconstruct-capture-manifest --load-assets` integration that feeds sampled
  per-pixel capture tensor targets into the CPU reference optimization loop;
- model-scored native feature proposals for image-detail and depth-edge regions
  before adaptive decomposition, with a replaceable learned-proposal contract;
- optional PyTorch renderer contract for ordered native carrier compositing,
  first-hit depth/normal/material/semantic metadata, transmittance, opacity,
  confidence, residual, provenance, ordered per-carrier hit traces, and
  query-loss outputs over `AuraScene` and `RenderTarget`;
- cached reference chunk BVH traversal metrics for native ray-query probes,
  including traversal mode and tested node counts;
- strict-JSON render comparison metrics for regression checks;
- reproducible benchmark plans plus CPU reference package/query/render timing,
  runtime export readiness, confidence-quality, and interaction-quality
  metrics;
- package-backed glTF/USD exchange-plan metadata;
- native-first CLI fixtures.

It is not yet a production renderer, trainer, CUDA kernel, autograd carrier
optimizer, or research benchmark result. The first
3DGS bridge is a fixture-sized parser that converts exported Gaussian
means/opacities/covariances from JSON or ASCII/binary little-endian PLY into
AURA Gaussian fallback elements. PLY `scale_*` fields are interpreted as 3DGS
log-scales and `rot_0..3` quaternions are applied to build world covariance. GPU
rendering and training work should implement against the native carrier
contract next, after the mixed-carrier decomposition path is the primary
fixture.

The current CPU renderer is a deterministic validation preview, and the optional
PyTorch path is a payload-aware ordered-compositing reference contract rather
than the final CUDA renderer. A future GPU renderer should match this
package/query contract while replacing the reference implementation for real
throughput. The live `torch_render_target_objective` path is the current
autograd bridge for turning carrier parameter tensors into real GPU
optimization losses.

The first CLI smoke path is `aura build-native-demo`, which builds a
mixed-carrier `.aura` package from evidence decomposition. 3DGS CLI commands are
kept as AURA-Ingest bootstrap paths after the native package path.

Use `aura inspect-capture-tensors <manifest>` on real capture manifests before
GPU training. It reports per-frame image/depth/mask/normal tensor shapes,
loader backend, and sample values so the CUDA path can consume manifest assets
without relying on summary-only statistics.
Use `aura reconstruct-capture-manifest <manifest> --load-assets --pixel-stride
N --max-targets-per-frame M` to exercise the CPU reference optimization loop on
sampled per-pixel capture tensor targets before moving the same target batches
to CUDA. Manifest-to-training conversion derives summaries, feature proposals,
depth priors, and mask priors from one loaded tensor batch to avoid duplicate
asset decode work in the reference path; the reusable API is
`capture_tensors_to_training_dataset(manifest, tensors)`.
Use the shared reconstruction flags `--split-image-loss-threshold`,
`--depth-anchor-loss-threshold`, `--merge-image-loss-threshold`,
`--merge-depth-loss-threshold`, `--demote-after-iteration`, and
`--disable-adaptive-evolution` on `reconstruct-demo` or
`reconstruct-capture-manifest` to tune or freeze the adaptive carrier evolution
policy recorded in the reconstruction report.
Use `aura torch-optimize-capture-manifest <manifest> --device cuda
--pixel-stride N --max-targets-per-frame M` to run the current torch reference
optimization scaffold from the same native capture tensor batches. It writes a
`.aura` package plus `torch_training_report.json`; it is still a scaffold until
the autograd carrier semantics are replaced by production CUDA kernels.
Use `aura torch-kernel-report` to list every native carrier kernel, its current
reference/autograd status, packaged CUDA source symbol, and missing CUDA
blockers. The surface carrier
has a tested torch autograd path, and the volume carrier has a differentiable
density parameter path. The beta carrier has differentiable bounded-shape
parameters, and the gabor carrier has differentiable frequency/phase/bandwidth
parameters. The neural residual carrier has a differentiable residual-scale
path, and the semantic carrier has differentiable confidence scoring;
the Gaussian fallback carrier has differentiable fallback color/opacity/
confidence parameters. Production readiness still requires compiling, testing,
and benchmarking CUDA kernels for every carrier.
Use `aura cuda-kernel-build-report --build` on a CUDA development machine to
attempt the packaged carrier extension compile/load gate.

Use `aura inspect-rays <package> --native-demo-probes` for material-aware
occlusion, shadow-transmittance, reflection-direction, and collision-distance
query inspection. Use
`aura benchmark-reference <package>` for CPU reference package/query/render
timing, runtime export readiness, confidence-quality, and interaction-quality
metrics, and add `--include-ablations` for carrier assignment ablation metrics;
these are contract checks, not quality or production-speed claims.
Use `aura benchmark-visual <package> <teacher.ppm>` to compare a deterministic
package render against a supplied teacher/reference image with MSE, PSNR, SSIM,
LPIPS-proxy, and render-throughput fields.
Use `aura export-report <package>` to report what the native `.aura` runtime
preserves and what glTF/USD fallback targets lose for engine workflows.
Use `aura migration-plan <package>` to report schema migration status.

3DGS-specific code lives under `aura.ingest`. That adapter converts splat exports
into `EvidenceSample` records first, then the adaptive decomposition path emits
Gaussian fallback payloads only when the evidence does not justify a stronger
carrier. Core AURA code should remain carrier- and package-centered rather than
becoming a 3DGS wrapper.

The first GPU milestone should extend the native `decompose_evidence` path and
then connect 3DGS ingest as one evidence source. Gaussian splats are allowed as
fallback carriers, but new core behavior should prove mixed non-Gaussian AURA
carriers can be packaged, loaded, queried, and inspected.
