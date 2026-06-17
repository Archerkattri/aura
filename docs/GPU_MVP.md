# AURA GPU-Ready MVP

This package now contains the GPU-ready skeleton for AURA:

- adaptive carrier registry;
- native carrier payload models;
- payload/carrier consistency validation;
- evidence-to-carrier assignment;
- evidence-to-element adaptive decomposition;
- package-level confidence maps and edit metadata;
- bounded local elements;
- carrier-aware reference ray queries;
- front-to-back compositing;
- tiny JSON/ASCII/binary little-endian PLY 3DGS export fixture reading;
- quaternion-aware PLY covariance conversion from 3DGS log-scales;
- direct 3DGS export/directory import adapter;
- chunk and LOD metadata;
- native `.aura` package writer;
- native `.aura` package loader/validator;
- explicit `.aura` format/version compatibility checks;
- JSON package inspection output and JSON Schema documents;
- runtime JSON Schema validation for package files;
- deterministic orthographic package preview rendering and image metrics;
- strict-JSON render comparison metrics for regression checks;
- glTF/USD exchange-plan metadata;
- CLI fixtures.

It is not yet a renderer, trainer, CUDA kernel, or benchmark result. The first
3DGS bridge is a fixture-sized parser that converts exported Gaussian
means/opacities/covariances from JSON or ASCII/binary little-endian PLY into
AURA Gaussian fallback elements. PLY `scale_*` fields are interpreted as 3DGS
log-scales and `rot_0..3` quaternions are applied to build world covariance. GPU
rendering and training work should implement against the native carrier
contract next, after the mixed-carrier decomposition path is the primary
fixture.

The current renderer is a deterministic validation preview, not the final CUDA
renderer. A future GPU renderer should match this package/query contract while
replacing the reference implementation for real throughput.

3DGS-specific code lives under `aura.ingest`. That adapter converts splat exports
into AURA elements with Gaussian fallback payloads. Core AURA code should remain
carrier- and package-centered rather than becoming a 3DGS wrapper.

The first GPU milestone should extend the native `decompose_evidence` path and
then connect 3DGS ingest as one evidence source. Gaussian splats are allowed as
fallback carriers, but new core behavior should prove mixed non-Gaussian AURA
carriers can be packaged, loaded, queried, and inspected.
