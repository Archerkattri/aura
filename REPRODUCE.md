# Reproducing AURA's headline results — CPU only, from the repo alone

This is a verified, GPU-free walkthrough. Starting from a clean clone on a
CPU-only laptop, it reproduces AURA's three headline results — **calibrated
confidence**, the **distribution-free pruning certificate**, and the **certified
LOD/streaming plan** — bit-for-bit from artifacts committed to the repository. No
GPU, no CUDA, no dataset download, and no training are required.

It works because the per-carrier reliability signals
(`outputs/reliability_<scene>.npz`) and the target result files
(`outputs/calib_*.json`, `outputs/cert_sweep.json`, `outputs/lod_certified.json`)
are committed, and the pipeline scripts recompute the results from the `.npz`
with pure NumPy at seed 0 — deterministically. The recomputation reproduces every
committed number exactly (zero `git diff`).

Retraining carriers, the trained `.aura` assets, and any FPS/timing numbers are
GPU work and are **not** needed for the results above — they are listed at the end
only for completeness.

## Prerequisites

- **OS:** Linux, macOS, or Windows via WSL. Nothing platform-specific.
- **Python:** 3.11 or 3.12 (the versions CI runs). This guide uses 3.11.
- **No GPU / no CUDA.** The reproduction is entirely CPU + NumPy.
- **Disk:** ~1.2 GB for the clone (committed figures and GIFs dominate; the
  reliability `.npz` add ~23 MB) plus ~1.1 GB for the CPU-only virtualenv (the
  CPU PyTorch wheel is the bulk).
- **Network:** once, for `git clone` and the CPU PyTorch / pip install.

## Setup

Identical to what `.github/workflows/ci.yml` installs, plus a CPU PyTorch wheel:

```bash
git clone https://github.com/Archerkattri/aura.git
cd aura

python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"   # pytest + matplotlib + imageio + pillow (same as CI)
```

## Verified command sequence

Run these from the repo root with the venv active. Wall times are from a
reference workstation (Linux, Python 3.11); a laptop will be slower, and the
one-time downloads depend on your network.

```bash
# a. CPU test suite — the calibration/certificate/LOD math, exporters, and CLI.
pytest -m "not gpu and not local_data" -q

# b. Artifact-backed publication gate report (content-checks the committed results).
python -m aura.cli publication-validation-report

# c. Recompute calibrated confidence + certificate for Truck from the committed npz.
python experiments/calibrate_confidence.py \
  --reliability outputs/reliability_truck.npz --scene truck

# d. Recompute the certified LOD/streaming plan across all four scenes.
python experiments/lod_certified_eval.py

# e. The shipped CLI that emits a certified LOD plan from a reliability signal.
python -m aura.cli lod-plan outputs/reliability_truck.npz --scene truck

# (bonus) Recompute the certificate operating study (all scenes x both labels).
python experiments/cert_sweep.py
```

Steps `c`, `d`, and `bonus` write to their committed default paths
(`outputs/calib_truck.json`, `outputs/lod_certified.json`,
`outputs/cert_sweep.json`). The reproduction is exact, so **the built-in check is
that nothing changed**:

```bash
git diff --stat outputs/calib_truck.json outputs/lod_certified.json outputs/cert_sweep.json
# (empty output = you reproduced the committed artifacts bit-for-bit)
```

### Expected output and timing

| Step | Command | Wall time | Expected result |
|---|---|---:|---|
| — | `git clone` (+ `.npz`) | ~20 s* | ~1.2 GB clone |
| — | venv + CPU torch + deps | ~35 s* | torch `2.x+cpu`, numpy 2.x, `aura` CLI on PATH |
| a | `pytest -m "not gpu and not local_data" -q` | ~110 s | **1947 passed, 9 skipped, 23 deselected, 0 failed** (exit 0)** |
| b | `publication-validation-report` | <1 s | **15/17 gates pass**, `remainingGateIds` = `secondary_ray_reflection`, `inverse_materials` (see below) |
| c | `calibrate_confidence.py … truck` | <1 s | matches `outputs/calib_truck.json` exactly |
| d | `lod_certified_eval.py` | ~1 s | "All certified bounds hold on the eval half" — 16/16 bounds, matches `outputs/lod_certified.json` |
| e | `lod-plan … truck` | <1 s | JSON plan, scene `truck`, 4 levels (keep 0.10/0.25/0.50/1.00) |
| bonus | `cert_sweep.py` | ~12 s | matches `outputs/cert_sweep.json` exactly |

\* Network- and machine-dependent. \*\* The suite runs at double-quiet
(`addopts` already has `-q`), so pytest 9.x omits the green summary line; the
exit code is the source of truth. `local_data`/`gpu` tests are deselected because
they need the trained assets or a GPU.

## What each step proves

| Step | Artifact reproduced | What it establishes | Backs |
|---|---|---|---|
| a | (code contracts) | Isotonic (PAVA) calibration, the conformal pruning certificate, the certified-LOD math, and the KHR/USD exporters hold under 1947 CPU tests. | The method sections; `src/aura/calibration.py`, `lod.py` |
| b | live gate report | 15 gates content-check the committed real-scene results (calibration ECE, pruning certificate, cross-scene transfer, full-res + render-loss, certified LOD, registry honesty, and the local/external quality tables). | README "Results and validation" |
| c | `calib_truck.json` | Raw view-count-heuristic ECE **0.5855 → 0.0014 calibrated** (~418×); selection AUC **calibrated 0.5808 vs opacity 0.3668**, within ~3% of the oracle ceiling 0.6009; the ε=0.6, α=0.1 conformal certificate is certified. | README "The killer property"; `docs/P0_CALIBRATED_CONFIDENCE.md` |
| d | `lod_certified.json` | All **16** certified LOD bounds (4 scenes × 4 keep levels) hold on the disjoint eval half under family-wise Bonferroni accounting over the R = 3 non-trivial levels (α/R ≈ 0.0333; the full-keep level is deterministic, ε = 0; 1−α = 0.9), zero violations. | README P0/certified-LOD; `docs/P4_CERTIFIED_LOD.md` |
| e | (CLI surface) | The shipped `lod-plan` command turns a reliability signal into a certified, Bonferroni-valid streaming plan — the product surface of the certificate. | README certified-LOD |
| bonus | `cert_sweep.json` | The certificate's selective regime: per-scene onset ε* **0.47–0.62** across both reliability labels. | README P1; `docs/P1_CROSS_SCENE.md` |

## Expected-unverified gates (this is correct, not a failure)

Step `b` reports `publicationReady: false` with two gates **unverified** on a
CPU-only clone:

- `secondary_ray_reflection`
- `inverse_materials`

Both probe a **trained** asset — `outputs/truck-sidecar.aura/carriers.npz`
(129,531 real carriers) — which is a large, GPU-produced artifact that is not
committed. With the asset absent, these gates return the explicit `unverified`
status **by design**: they never silently pass, and they are a distinct state
from `failed`. The other 15 gates are fully content-checked against the committed
real-scene results, so a passing `15/17` with exactly those two unverified is the
expected CPU-only outcome. To turn them green you must regenerate the trained
asset (GPU — see below).

## Optional: GPU-only regeneration (NOT needed for anything above)

These require CUDA, the datasets, and (for FPS) an idle RTX 5090 (sm_120). None of
them are needed to reproduce the calibrated-confidence, certificate, or
certified-LOD results.

- **Retrain carriers** from raw captures: `aura colmap-to-capture-manifest …`,
  `aura train-gsplat …`, and the isolated DBS-Beta fork (`.dbs_venv`).
- **Regenerate the reliability `.npz`** from trained assets:
  `experiments/per_carrier_reliability.py`, `experiments/render_loss_reliability.py`.
- **Produce `outputs/truck-sidecar.aura`** to green the two asset-probe gates above.
- **Garden native render-loss label** (17.4 MP): OOMs under concurrent GPU load —
  needs a memory-idle GPU. Command in `docs/P2_FULLRES_RENDERLOSS.md`.
- **FPS / timing rows**: `experiments/real_scene_fps_sweep.py` and the PRISM CUDA
  path need an idle GPU; these are timing measurements, not accuracy claims.

## Why it is deterministic

Every recompute script fixes NumPy's RNG at seed 0 and uses the same 50/50
carrier calibration/eval split, so the isotonic calibrator, the conformal
certificate, and the LOD plan are reproduced exactly — the recomputed JSON is
byte-identical to the committed artifact.
