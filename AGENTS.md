# AGENTS.md — AURA-Core

This file is read by automated coding agents (Claude Code, etc.) working on
this repository.

## Read First

1. `README.md` — project overview, install, quickstart, repository map.
2. `docs/ARCHITECTURE.md` — carrier families, reconstruction pipeline, adaptive
   evolution contract, CUDA renderer, package format.
3. `docs/DATASETS.md` — dataset conventions, baseline methods, capture asset
   contracts.
4. `CONTRIBUTING.md` — build/test setup, commit convention, branch workflow,
   guardrails.
5. `src/aura/cli.py` — all CLI commands.
6. `tests/` — deterministic contract and integration tests.

## Scope

- **Edit freely:** `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `docs/`,
  `.gitignore`, `.env.example`, `pyproject.toml`.
- **Do not edit** any `.py` files without explicit instruction. Source code
  is owned by the main development branch.

## Core Guardrails

- Do not commit datasets, checkpoints, rendered outputs, secrets, `.env`,
  or `LOCAL_SECRETS.md`.
- AURA is the native reconstruction engine over adaptive carriers. Do not
  reduce it to a 3DGS wrapper or a file-format layer.
- Keep all 3DGS-specific logic under `aura.ingest`; splats are evidence inputs,
  not native elements.
- New ingest sources must produce `EvidenceSample` records before carrier
  assignment.
- Run `python -m pytest -q` and confirm it passes before committing.

## Commit Author

```
Archerkattri <krishiattriwork@gmail.com>
```
