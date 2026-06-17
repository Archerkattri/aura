# ChatGPT Handoff - AURA

Use this prompt when opening the private repo on another machine:

```text
You are helping me develop AURA, Adaptive Unified Radiance Asset.

Read README.md, AGENTS.md, docs/AURA_CORE_RESEARCH.md,
docs/PRODUCTION_HANDOFF.md, docs/GPU_MVP.md, docs/DATASETS.md,
src/aura/cli.py, and the tests before changing code.

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
2. run `aura write-capture-manifest-template --output outputs/capture-manifest.json`;
3. run `aura capture-manifest-to-training outputs/capture-manifest.json --output outputs/training-from-capture.json`;
4. for fixture captures with existing PPM/PGM assets, run `aura inspect-capture-assets data/custom-captures/<scene>/capture-manifest.json`
   and `aura capture-manifest-to-training data/custom-captures/<scene>/capture-manifest.json --output outputs/training-from-capture-assets.json --load-assets`;
5. run `aura reconstruct-capture-manifest outputs/capture-manifest.json --output-dir outputs/reconstruct-capture.aura --iterations 6`;
6. run `aura build-native-demo --output-dir outputs/native-demo.aura`;
7. read `docs/AURA_CORE_RESEARCH.md` and `docs/PRODUCTION_HANDOFF.md`;
8. use `aura write-training-frames-demo` and `aura reconstruct-demo --frames`
   around posed color/depth/semantic frames plus native evidence regions, not 3DGS;
9. replace the CPU fixture loop with a differentiable image/depth renderer;
10. add adaptive carrier promote/split/merge tests;
11. expand mixed-carrier decomposition fixtures, semantic graph checks, and query tests;
12. run `aura inspect-rays outputs/native-demo.aura --native-demo-probes`;
13. run `aura benchmark-reference outputs/native-demo.aura --include-ablations`;
14. run `aura benchmark-plan` and fill result-producing harnesses without overclaiming metrics;
15. run `aura ingest-adapters` and keep future source adapters evidence-based;
16. extend the tiny JSON/ASCII/binary PLY 3DGS export fixtures if needed;
17. harden the splat-to-AURA scaffold reader toward real baseline exports,
   preserving 3DGS log-scale and quaternion semantics;
18. add more first-hit/depth/transmittance query tests;
19. expand `.aura` package validation around migration fixtures and malformed
   cross-file references;
20. use the deterministic preview renderer and `compare-renders` as regression
   targets;
21. run `aura import-3dgs` against a real CUDA-trained 3DGS baseline scene only
    after the native reconstruction path is first-class.
```
