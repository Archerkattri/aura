# Contributing to AURA-Core

## Setup

```bash
git clone https://github.com/Archerkattri/aura.git
cd aura
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gpu,assets]"
```

## Running Tests

```bash
python -m pytest -q
```

All tests are deterministic and do not require external datasets. Tests that
require CUDA are automatically skipped when no GPU is available.

## Commit Convention

Commit subjects should be in conventional commit style:

```
feat: add gabor carrier frequency parameter sweep
fix: correct BVH leaf node AABB computation
docs: update ARCHITECTURE carrier family table
test: add surface carrier normal parity test
refactor: extract packed batch builder into separate module
```

Commit as `Archerkattri <krishiattriwork@gmail.com>`.

## Branch Workflow

- `main` — stable, tested code only.
- Feature branches: `feat/<short-description>`.
- Fix branches: `fix/<short-description>`.

Open a pull request against `main`. All tests must pass before merge.

## Guardrails

- Do not commit datasets, trained checkpoints, rendered outputs, secrets,
  `.env`, or `LOCAL_SECRETS.md`. These are covered by `.gitignore`.
- Keep generated packages under ignored `outputs/`.
- AURA is not a 3DGS wrapper. The native reconstruction engine over adaptive
  carriers is the core contribution.
- Keep all 3DGS-specific logic under `aura.ingest`. Splats are evidence inputs
  that become `EvidenceSample` records before decomposition.
- New ingest sources must produce `EvidenceSample` records before any carrier
  assignment.
- Do not reduce AURA to a single splat variant, a file-format wrapper, or a
  3DGS converter.

## Architecture Orientation

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before making significant
changes to carrier types, the reconstruction pipeline, or the adaptive
evolution policy.

The key source files:

| File | Responsibility |
|---|---|
| `src/aura/cli.py` | All CLI commands |
| `src/aura/core.py` | Reconstruction contracts and adaptive evolution policy |
| `src/aura/torch_renderer.py` | Torch render batches and compositing |
| `src/aura/torch_optimizer.py` | Tiled capture optimization and checkpoints |
| `src/aura/torch_kernels.py` | Carrier parameter tensors and responses |
| `src/aura/cuda_renderer.py` | CUDA renderer ABI and dispatch boundary |
| `src/aura/package.py` | `.aura` package IO and validation |
| `src/aura/scene.py` | Ray-query traversal and response assembly |
| `src/aura/ingest/` | Capture, COLMAP, 3DGS, depth, semantic adapters |
