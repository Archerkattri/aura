# Agent Instructions - AURA

You are working on AURA, Adaptive Unified Radiance Asset.

## Read First

1. `README.md`
2. `docs/CPU_MVP.md`
3. `docs/DATASETS.md`
4. `src/aura/cli.py`
5. `tests/`

## Guardrails

- Do not commit datasets, trained checkpoints, rendered outputs, tokens, `.env`,
  or `LOCAL_SECRETS.md`.
- Keep data under ignored `data/`.
- Keep generated packages under ignored `outputs/`.
- Ask before GPU work unless the owner explicitly says this is the GPU machine.
- Do not reduce AURA to one new splat kernel. AURA is the asset/ray-query
  contract over adaptive carriers.

## First Tasks On A GPU Machine

1. Install with `python -m pip install -e .`.
2. Run `python -m pytest`.
3. Add one tiny scene fixture.
4. Add a 3DGS export reader for means/covariances/opacities.
5. Build the first AURA element scaffold from splat samples.
6. Add a ray-query path for first-hit/depth/transmittance.

## Research Positioning

The paper target is not better PSNR alone. The target is scene behavior: ray
queries, confidence, geometry proxies, editability, LOD, and export.

