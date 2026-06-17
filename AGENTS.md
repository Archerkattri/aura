# Agent Instructions - AURA

You are working on AURA, Adaptive Unified Radiance Asset.

## Read First

1. `README.md`
2. `docs/AURA_CORE_RESEARCH.md`
3. `docs/PRODUCTION_HANDOFF.md`
4. `docs/GPU_MVP.md`
5. `docs/DATASETS.md`
6. `src/aura/cli.py`
7. `tests/`

## Guardrails

- Do not commit datasets, trained checkpoints, rendered outputs, tokens, `.env`,
  or `LOCAL_SECRETS.md`.
- Keep data under ignored `data/`.
- Keep generated packages under ignored `outputs/`.
- Treat this as the GPU development path and expose CUDA device 0 by default.
- Do not reduce AURA to one new splat kernel, a package format, or a 3DGS
  wrapper. AURA-Core is the native reconstruction engine over adaptive carriers.
- Keep 3DGS-specific logic under `aura.ingest`; splats are evidence inputs, not
  the native representation center.
- New ingest sources must produce `EvidenceSample` records before decomposition.

## First Tasks On A GPU Machine

1. Install with `python -m pip install -e ".[dev]"`.
2. Run `python -m pytest -q`.
3. Run `aura write-capture-manifest-template --output outputs/capture-manifest.json`.
4. Run `aura capture-manifest-to-training outputs/capture-manifest.json --output outputs/training-from-capture.json`.
5. Run `aura reconstruct-capture-manifest outputs/capture-manifest.json --output-dir outputs/reconstruct-capture.aura --iterations 6`.
6. Run `aura validate-package outputs/reconstruct-capture.aura`.
7. Run `aura build-native-demo --output-dir outputs/native-demo.aura`.
8. Run `aura inspect-rays outputs/native-demo.aura --native-demo-probes`.
9. Run `aura benchmark-reference outputs/native-demo.aura --include-ablations`.
10. Run `aura migration-plan outputs/native-demo.aura`.
11. Run `aura ingest-adapters`.
12. Replace the fixture reconstruction loop with real image/depth loading and differentiable rendering.
13. Add adaptive carrier promote/split/merge tests and real-data benchmarks.
14. Extend mixed-carrier decomposition fixtures and query tests.
15. Add more ray-query paths for first-hit/depth/transmittance.
16. Extend the tiny JSON/ASCII/binary PLY 3DGS export fixtures when parser coverage needs it.
17. Harden the 3DGS export reader for means/covariances/opacities toward real baseline exports.
18. Use `aura import-3dgs` on real baseline output directories only after the native reconstruction path is first-class.

## Research Positioning

The paper target is not better PSNR alone. The target is scene behavior: ray
queries, confidence, geometry proxies, editability, LOD, and export.
