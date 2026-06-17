# ChatGPT Handoff - AURA

Use this prompt when opening the private repo on another machine:

```text
You are helping me develop AURA, Adaptive Unified Radiance Asset.

Read README.md, AGENTS.md, docs/GPU_MVP.md, docs/DATASETS.md, src/aura/cli.py,
and the tests before changing code.

Goal: turn this GPU-ready contract scaffold into the first AURA MVP:
3DGS bootstrap -> adaptive local carriers -> ray-query API -> .aura package ->
glTF/USD fallback metadata.

Rules:
- Do not commit datasets, third-party repos, checkpoints, outputs, .env,
  LOCAL_SECRETS.md, or tokens.
- Treat this as the GPU development path; expose CUDA device 0 by default.
- AURA is not a single splat variant. Keep the carrier registry and asset
  contract central.
- Add tests for every file parser, package schema, and query contract.

First useful tasks:
1. run tests;
2. extend the tiny JSON/ASCII/binary PLY 3DGS export fixtures if needed;
3. harden the splat-to-AURA scaffold reader toward real baseline exports,
   preserving 3DGS log-scale and quaternion semantics;
4. add more first-hit/depth/transmittance query tests;
5. expand `.aura` package validation around migration fixtures and malformed
   cross-file references;
6. use the deterministic preview renderer and `compare-renders` as regression
   targets;
7. run `aura import-3dgs` against a real CUDA-trained 3DGS baseline scene.
```
