# AURA GPU-Ready MVP

This package now contains the GPU-ready skeleton for AURA:

- adaptive carrier registry;
- native carrier payload models;
- payload/carrier consistency validation;
- evidence-to-carrier assignment;
- evidence-to-element adaptive decomposition;
- package-level confidence maps and edit metadata;
- semantic/object graph package artifact;
- bounded local elements;
- carrier-aware reference ray queries;
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
- chunk and LOD metadata;
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
- torch/CUDA capture asset batching through `torch_capture_asset_batch`, which
  stacks manifest image/depth/mask/normal tensors and presence masks on the
  selected device;
- per-pixel capture training target generation through
  `capture_tensors_to_render_targets` and `torch_capture_training_batch`;
- torch reference rendering directly from capture training batches through
  `torch_render_capture_training_batch`, using carrier parameter tensors for
  every supported native/fallback carrier;
- torch reference optimization steps through `torch_optimize_capture_batch`,
  which runs repeated batched AURA forward passes and records image/depth/query/
  normal losses while applying bounded native carrier color updates;
- explicit torch carrier kernel specs and autograd parameter tensors for
  surface, volume, beta, gabor, neural residual, semantic, and Gaussian fallback
  semantics;
- `reconstruct-capture-manifest --load-assets` integration that feeds sampled
  per-pixel capture tensor targets into the CPU reference optimization loop;
- model-scored native feature proposals for image-detail and depth-edge regions
  before adaptive decomposition, with a replaceable learned-proposal contract;
- optional PyTorch renderer contract for batched native first-hit/depth/color,
  transmittance, opacity, confidence, normal, material, semantic, residual,
  provenance, and query-loss outputs over `AuraScene` and `RenderTarget`;
- reference chunk BVH traversal metrics for native ray-query probes, including
  traversal mode and tested node counts;
- strict-JSON render comparison metrics for regression checks;
- reproducible benchmark plans plus CPU reference package/query/render timing,
  confidence-quality, and interaction-quality metrics;
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

The current renderer is a deterministic validation preview, and the optional
PyTorch path is a payload-aware reference contract rather than the final CUDA
renderer. A future GPU renderer should match this package/query contract while
replacing the reference implementation for real throughput.

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
to CUDA.
Use `aura torch-optimize-capture-manifest <manifest> --device cuda
--pixel-stride N --max-targets-per-frame M` to run the current torch reference
optimization scaffold from the same native capture tensor batches. It writes a
`.aura` package plus `torch_training_report.json`; it is still a scaffold until
the autograd carrier semantics are replaced by production CUDA kernels.
Use `aura torch-kernel-report` to list every native carrier kernel, its current
reference/autograd status, and the missing CUDA blockers. The surface carrier
has a tested torch autograd path, and the volume carrier has a differentiable
density parameter path. The beta carrier has differentiable bounded-shape
parameters, and the gabor carrier has differentiable frequency/phase/bandwidth
parameters. The neural residual carrier has a differentiable residual-scale
path, and the semantic carrier has differentiable confidence scoring;
the Gaussian fallback carrier has differentiable fallback color/opacity/
confidence parameters. Production readiness still requires CUDA kernels for
every carrier.

Use `aura inspect-rays <package> --native-demo-probes` for material-aware
occlusion, shadow-transmittance, reflection-direction, and collision-distance
query inspection. Use
`aura benchmark-reference <package>` for CPU reference package/query/render
timing, confidence-quality, and interaction-quality metrics, and add
`--include-ablations` for carrier assignment ablation metrics; these are
contract checks, not quality or production-speed claims.
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
