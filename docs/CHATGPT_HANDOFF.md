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
- Keep 3DGS-specific code under `aura.ingest`; splats must become
  `EvidenceSample` inputs before decomposition, not direct native elements.
- Add tests for every file parser, package schema, and query contract.

First useful tasks:
1. run tests;
2. run `aura build-native-demo --output-dir outputs/native-demo.aura`;
3. expand mixed-carrier decomposition fixtures, semantic graph checks, and query tests;
4. run `aura inspect-rays outputs/native-demo.aura --native-demo-probes`;
5. run `aura benchmark-reference outputs/native-demo.aura --include-ablations`;
6. run `aura benchmark-plan` and fill result-producing harnesses without overclaiming metrics;
7. run `aura ingest-adapters` and keep future source adapters evidence-based;
8. extend the tiny JSON/ASCII/binary PLY 3DGS export fixtures if needed;
9. harden the splat-to-AURA scaffold reader toward real baseline exports,
   preserving 3DGS log-scale and quaternion semantics;
10. add more first-hit/depth/transmittance query tests;
11. expand `.aura` package validation around migration fixtures and malformed
   cross-file references;
12. use the deterministic preview renderer and `compare-renders` as regression
   targets;
13. run `aura import-3dgs` against a real CUDA-trained 3DGS baseline scene.
```
