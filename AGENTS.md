# Agent Instructions - AURA

You are working on AURA, Adaptive Unified Radiance Asset.

## Read First

1. `README.md`
2. `docs/GPU_MVP.md`
3. `docs/DATASETS.md`
4. `src/aura/cli.py`
5. `tests/`

## Guardrails

- Do not commit datasets, trained checkpoints, rendered outputs, tokens, `.env`,
  or `LOCAL_SECRETS.md`.
- Keep data under ignored `data/`.
- Keep generated packages under ignored `outputs/`.
- Treat this as the GPU development path and expose CUDA device 0 by default.
- Do not reduce AURA to one new splat kernel. AURA is the asset/ray-query
  contract over adaptive carriers and native carrier payloads.
- Keep 3DGS-specific logic under `aura.ingest`; splats are evidence inputs, not
  the native representation center.

## First Tasks On A GPU Machine

1. Install with `python -m pip install -e .`.
2. Run `python -m pytest`.
3. Run `aura write-native-demo-package --output-dir outputs/native-demo.aura`.
4. Extend mixed-carrier decomposition fixtures and query tests.
5. Add more ray-query paths for first-hit/depth/transmittance.
6. Extend the tiny JSON/ASCII/binary PLY 3DGS export fixtures when parser coverage needs it.
7. Harden the 3DGS export reader for means/covariances/opacities toward real baseline exports.
8. Use `aura import-3dgs` on real baseline output directories.

## Research Positioning

The paper target is not better PSNR alone. The target is scene behavior: ray
queries, confidence, geometry proxies, editability, LOD, and export.
