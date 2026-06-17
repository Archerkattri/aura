# ChatGPT Handoff - AURA

Use this prompt when opening the private repo on another machine:

```text
You are helping me develop AURA, Adaptive Unified Radiance Asset.

Read README.md, AGENTS.md, docs/GPU_MVP.md, docs/DATASETS.md, src/aura/cli.py,
and the tests before changing code.

Goal: turn this early scaffold into AURA-Core, the next reconstruction step
after 3D Gaussian Splatting:
images/video -> poses/depth/priors -> native adaptive carriers ->
differentiable render/optimization -> ray-query/runtime AURA asset.

Rules:
- Do not commit datasets, third-party repos, checkpoints, outputs, .env,
  LOCAL_SECRETS.md, or tokens.
- Treat this as the GPU development path; expose CUDA device 0 by default.
- AURA is not a single splat variant and not a file-format wrapper. Build the
  native reconstruction/training engine around adaptive carriers.
- Keep 3DGS-specific code under `aura.ingest`; splats must become
  `EvidenceSample` inputs before decomposition, not direct native elements.
- Add tests for every file parser, package schema, and query contract.

First useful tasks:
1. run tests;
2. run `aura build-native-demo --output-dir outputs/native-demo.aura`;
3. read `docs/AURA_CORE_RESEARCH.md`;
4. implement `aura reconstruct-demo` around a posed synthetic fixture, not 3DGS;
5. add a CPU reference optimization loop with image/depth losses and a training report;
6. add adaptive carrier promote/split/merge tests;
7. expand mixed-carrier decomposition fixtures, semantic graph checks, and query tests;
8. run `aura inspect-rays outputs/native-demo.aura --native-demo-probes`;
9. run `aura benchmark-reference outputs/native-demo.aura --include-ablations`;
10. run `aura benchmark-plan` and fill result-producing harnesses without overclaiming metrics;
11. run `aura ingest-adapters` and keep future source adapters evidence-based;
12. extend the tiny JSON/ASCII/binary PLY 3DGS export fixtures if needed;
13. harden the splat-to-AURA scaffold reader toward real baseline exports,
   preserving 3DGS log-scale and quaternion semantics;
14. add more first-hit/depth/transmittance query tests;
15. expand `.aura` package validation around migration fixtures and malformed
   cross-file references;
16. use the deterministic preview renderer and `compare-renders` as regression
   targets;
17. run `aura import-3dgs` against a real CUDA-trained 3DGS baseline scene only
    after the native reconstruction path is first-class.
```
