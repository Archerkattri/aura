# ChatGPT Handoff - AURA

Use this prompt when opening the private repo on another machine:

```text
You are helping me develop AURA, Adaptive Unified Radiance Asset.

Read README.md, AGENTS.md, docs/GPU_MVP.md, docs/DATASETS.md, src/aura/cli.py,
and the tests before changing code.

Goal: turn this GPU-ready contract scaffold into the first AURA MVP:
adaptive local carriers -> ray-query API -> .aura package -> 3DGS ingest as
evidence -> glTF/USD fallback metadata.

Rules:
- Do not commit datasets, third-party repos, checkpoints, outputs, .env,
  LOCAL_SECRETS.md, or tokens.
- Treat this as the GPU development path; expose CUDA device 0 by default.
- AURA is not a single splat variant. Keep the carrier registry, native carrier
  payloads, and asset contract central.
- Keep 3DGS-specific code under `aura.ingest`; splats are evidence inputs, not
  the native representation center.
- Add tests for every file parser, package schema, and query contract.

First useful tasks:
1. run tests;
2. expand mixed-carrier decomposition fixtures and query tests;
3. extend the tiny JSON/ASCII/binary PLY 3DGS export fixtures if needed;
4. harden the splat-to-AURA scaffold reader toward real baseline exports,
   preserving 3DGS log-scale and quaternion semantics;
5. add more first-hit/depth/transmittance query tests;
6. expand `.aura` package validation around migration fixtures and malformed
   cross-file references;
7. use the deterministic preview renderer and `compare-renders` as regression
   targets;
8. run `aura import-3dgs` against a real CUDA-trained 3DGS baseline scene.
```
