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
- reusable native scene tensor caching through `torch_scene_tensors`, which
  keeps element/chunk bounds, carrier IDs, colors, opacities, confidence,
  element-to-chunk culling indices, and carrier parameter tensors on the
  selected device across reconstruction iterations;
- per-pixel capture training target generation through
  `capture_tensors_to_render_targets` and `torch_capture_training_batch`;
- bounded packed capture/render target batch descriptors through
  `capture_tensors_to_packed_render_batches`, with flat integer/float buffers
  for frame indices, pixels, rays, colors, depths, masks, and normals;
- direct torch tensor ingestion for one packed descriptor through
  `torch_capture_training_batch_from_packed`;
- torch reference rendering directly from capture training batches through
  `torch_render_capture_training_batch`, using carrier parameter tensors for
  every supported native/fallback carrier;
- live torch render objectives through `torch_render_target_objective`, exposing
  image/depth/normal/mask losses over carrier parameter tensors;
- torch reference optimization steps through `torch_optimize_capture_batch`,
  which runs repeated batched AURA forward passes, records image/depth/query/
  normal/mask losses plus loss weights and optimizer gradient state, enforces
  optional sampled-batch caps, and applies gradient updates to native carrier
  tensors;
- packed multi-batch torch optimization through
  `torch_optimize_capture_batches`, which streams deterministic
  `CapturePackedRenderBatch` source windows through one resident carrier tensor
  state and records batch indices, target offsets, and tile source windows per
  gradient step;
- configurable adaptive carrier evolution thresholds for split/promote/merge/
  demote actions, emitted in reconstruction reports;
- explicit torch carrier kernel specs and autograd parameter tensors for
  surface, volume, beta, gabor, neural residual, semantic, and Gaussian fallback
  semantics;
- packaged CUDA carrier source symbols for surface, volume, beta, gabor, neural,
  semantic, and Gaussian fallback carriers, still gated as non-production until
  compiled extension tests and benchmarks exist;
- packaged CUDA renderer source symbol `aura_render_rays_kernel` for batched
  ray/AABB first-hit query outputs over native element bounds, reported
  separately from carrier kernels and still gated as non-production until
  compiled dispatch, parity tests, chunk/BVH traversal, and speed benchmarks
  exist;
- packaged CUDA renderer host launcher symbol `aura_render_rays_launcher`,
  which computes a CUDA grid and launches `aura_render_rays_kernel` from the
  compiled source ABI; this is still not Python-callable renderer dispatch until
  the loaded extension symbol is verified, tensor bindings exist, and parity
  tests run on CUDA hardware;
- deterministic host-side renderer ABI buffers through
  `cuda_renderer_scene_buffers(...)` and `cuda_renderer_kernel_inputs(...)`,
  packing native element bounds, carrier IDs, material/semantic dictionaries,
  rays, and output buffer shapes for the future `aura_render_rays_kernel`
  launch;
- a CPU oracle for the packaged renderer kernel through
  `simulate_cuda_renderer_kernel(...)`, validating flat output buffers and
  first-hit parity before compiled CUDA dispatch exists;
- CPU-safe CUDA renderer callable scaffold through the legacy
  `cuda_kernels.cuda_render_rays` report plus the concrete
  `aura.cuda_renderer` launch boundary, with validated launch config
  (`rayCount`, block count, threads per block, max ordered hits) and explicit
  CPU/torch fallback batches for color, opacity, transmittance, depth, normals,
  confidence, residual, material/semantic IDs, provenance, and ordered hit
  traces when the compiled CUDA extension is unavailable;
- CUDA renderer symbol verification through
  `cuda_renderer_symbol_probe(...)`, which distinguishes an unavailable
  extension, a loaded extension with missing renderer symbols, and a loaded
  extension with both `aura_render_rays_kernel` and
  `aura_render_rays_launcher`; production dispatch still remains blocked until
  Python tensor binding, parity tests, and speed benchmarks exist;
- `cuda-kernel-build-report --build` for GPU machines to attempt native carrier
  CUDA extension compile/load without changing the default CPU-safe test path;
- `reconstruct-capture-manifest --load-assets` integration that feeds sampled
  per-pixel capture tensor targets into the CPU reference optimization loop;
- model-scored native feature proposals for image-detail and depth-edge regions
  before adaptive decomposition, with a replaceable learned-proposal contract;
- learned logistic capture proposal weights through
  `train_capture_proposal_model`, so labeled image/depth/mask/normal feature
  examples can drive native `TrainingRegion` proposal generation before a
  larger neural proposal backend replaces the reference contract;
- optional PyTorch renderer contract for ordered native carrier compositing,
  first-hit depth/normal/material/semantic metadata, transmittance, opacity,
  confidence, residual, provenance, ordered per-carrier hit traces, and
  query-loss outputs over `AuraScene` and `RenderTarget`;
- PyTorch/native Gaussian fallback parity for ray-clamped closest-point
  covariance sampling inside the carrier AABB, so fallback carriers use the
  same reference opacity/confidence semantics as CPU ray queries;
- `TorchRenderBatch.orderedHits` serialization for checking torch/CUDA ray
  outputs against the CPU `RayTraversal.orderedHits` contract;
- cached reference chunk BVH traversal metrics for native ray-query probes,
  including traversal mode and tested node counts;
- cached scene acceleration metadata for element coverage, orphan fallback
  counts, BVH node/leaf/depth shape, leaf chunk distribution, candidate
  ordering, and per-query BVH node/leaf/chunk-bound tests;
- runtime export acceleration metadata for element-linear, chunk-linear, and
  cached-BVH traversal modes, with serialized traversal metadata ready for
  engine integration reports and production GPU traversal still marked false;
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
than the final CUDA renderer. Its fallback Gaussian sampling matches the native
ray-query reference by evaluating covariance weight at the closest ray point
clamped to the carrier bounds, not only at the AABB entry. A future GPU renderer
should match this package/query contract while replacing the reference
implementation for real throughput. The live `torch_render_target_objective`
path is the current autograd bridge for turning carrier parameter tensors into
real GPU optimization losses.

The first CLI smoke path is `aura build-native-demo`, which builds a
mixed-carrier `.aura` package from evidence decomposition. 3DGS CLI commands are
kept as AURA-Ingest bootstrap paths after the native package path.

Use `aura inspect-capture-tensors <manifest>` on real capture manifests before
GPU training. It reports per-frame image/depth/mask/normal tensor shapes,
loader backend, and sample values so the CUDA path can consume manifest assets
without relying on summary-only statistics.
Use `aura plan-capture-sampling <manifest> --tile-size N --pixel-stride S
--max-targets-per-frame M` to emit the deterministic tile schedule and sampled
pixel counts that future tiled, memory-mapped, or GPU-native loaders should
match before materializing render targets. CPU and torch capture target builders
use the same mask-aware sampling semantics, skipping pixels whose mask value is
zero or negative. The plan is intentionally GPU-consumable without launching
CUDA: it records row-major tile/pixel order, per-tile target offsets,
candidate/sampled/masked pixel counts, first/last sampled pixels, and bounded
batch metadata (`maxTargetsPerBatch`, batch tile indices, target offsets, and
target counts) so production kernels can stream sampled tiles without
materializing an unbounded all-pixel list. Explicit batch caps smaller than a
sampled tile split that tile into deterministic target-offset ranges. Use
`capture_tensors_to_packed_render_batches(...)` when the next stage needs
actual bounded buffers rather than only the sampling plan; it returns packed
array-backed descriptors and keeps `capture_tensors_to_render_targets(...)`
available for legacy CPU `RenderTarget` callers.
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
Use `train_capture_proposal_model(...)` when labeled capture features are
available. It emits an `AURA_CAPTURE_PROPOSAL_MODEL` payload with learned
image-detail and depth-edge logistic weights that can be passed back into
`propose_training_regions_from_tensors(...)`; this is still a lightweight
proposal contract, not a trained neural proposer.
Both commands also accept `--render-backend cpu|torch|auto`, `--device`, and
`--require-cuda`. `--render-backend torch --device cuda --require-cuda` forces
the reconstruction iterations through the native torch AURA ray-query contract
and fails instead of silently falling back when CUDA is unavailable. `auto`
selects torch when the optional backend is installed, otherwise records the CPU
reference path in the report.
Use `aura torch-optimize-capture-manifest <manifest> --device cuda
--pixel-stride N --max-targets-per-frame M --max-targets-per-batch B` to run
the current torch reference optimization scaffold from the same native capture
tensor batches. It writes a `.aura` package plus `torch_training_report.json`.
The command now optimizes over packed tiled batches instead of one monolithic
capture batch, so the report records `packedBatchCount`, `packedTargetCount`,
batch indices, target offsets, and source windows along with the loss weights,
image/depth/query/normal/mask loss components, optimizer name, gradient norm,
clipped/applied gradient norm, updated parameter count, and sample cap for each
step. It is still a scaffold until the autograd carrier semantics are replaced
by production CUDA kernels.
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
Use `aura cuda-renderer-report` on any machine to inspect the legacy future
`cuda_render_rays` launch contract. It does not compile or load CUDA; it reports
`productionReady: false` and explains that the renderer is unavailable until the
extension is compiled, loadable, parity-tested, and benchmarked. The API
contract also reports the packaged `aura_render_rays_kernel` source ABI for
batched first-hit ray queries; source availability is not compiled dispatch. For
callable MVP integration tests, import `aura.cuda_renderer.cuda_render_rays`; it
validates the launch shape and either raises when CUDA is required or returns an
explicit CPU/torch fallback batch matching the AURA ray-query contract. This
fallback is not CUDA acceleration. Use
`aura.cuda_renderer.cuda_renderer_kernel_inputs(...)` to produce deterministic
host-side buffers matching the packaged `aura_render_rays_kernel` ABI; these
buffers are parity-test inputs, not a compiled CUDA launch. Use
`aura.cuda_renderer.cuda_renderer_dispatch_contract(...)` to inspect the planned
kernel/launcher symbols, launch shape, flat arguments, output buffer shapes, and
missing dispatch work. Use `simulate_cuda_renderer_kernel(...)` as the CPU
oracle for those buffers while compiled CUDA dispatch is still unavailable.
Use `cuda_renderer_symbol_probe(...)` after a CUDA extension build/load attempt
to verify whether the compiled module object exposes the renderer kernel and
launcher symbols. A positive symbol probe is only one dispatch prerequisite; it
does not make `cuda_render_rays` production-ready until the Python tensor
binding launches the kernel and passes parity/speed gates.

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
