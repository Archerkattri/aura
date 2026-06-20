# Reproducing the AURA Benchmark

## Environment

```bash
pip install -e ".[dev,gpu,assets]"
```

## Datasets

See `scripts/fetch_datasets.py --list` for dataset instructions.

Datasets are NOT committed to git (binary, large). Each scene is ingested from
its COLMAP sparse model into an AURA capture manifest:
```bash
python -m aura.cli colmap-to-capture-manifest <scene_dir>/sparse/0 \
    --root <scene_dir> \
    --image-dir <scene_dir>/images \
    --output outputs/<scene>-manifest.json \
    --point-seeded            # one carrier per SfM point (3DGS-style; ~129k for truck)
# omit --point-seeded (and pass --max-seed-regions N) for voxel-cluster seeding instead
```

## Training

```bash
python -m aura.cli train outputs/<scene>-manifest.json \
    --output outputs/<scene>-run.aura \
    --iterations 3000 \
    --tile-size 256 \
    --max-targets-per-batch 256 \
    --max-targets-per-frame 16 \
    --skip-validation \
    --disable-evolution \
    --position-lr 1.6e-4 \
    --position-lr-final 1.6e-6 \
    --lr-decay-steps 3000 \
    --opacity-reset-interval 600 \
    --depth-distortion-weight 0.001
```

### Current Training Run (truck, 129k carriers)

| Parameter | Value |
|---|---|
| Run | truck-3k-run6 |
| Carriers | 129,531 |
| Iterations | 3000 |
| Loss at convergence | ~0.02 |
| Checkpoint | `outputs/truck-3k-run6.aura` |

> **Note**: run6 uses the fixed batched-gaussian writeback (commit 0f3797d) and
> the writeback does work (≈23k carriers updated), but 3,000 iterations is far
> too few to converge — run6 evaluates at 6.89 dB, essentially the untrained
> floor. A competitive result needs many more iterations / denser supervision.

### Training with densification (recommended for higher PSNR)

```bash
python -m aura.cli train outputs/<scene>-manifest.json \
    --output outputs/<scene>-densify-run.aura \
    --iterations 3000 \
    --tile-size 256 \
    --max-targets-per-batch 256 \
    --max-targets-per-frame 16 \
    --skip-validation \
    --disable-evolution \
    --position-lr 1.6e-4 \
    --position-lr-final 1.6e-6 \
    --lr-decay-steps 3000 \
    --opacity-reset-interval 600 \
    --depth-distortion-weight 0.001 \
    --densify \
    --densify-start-iter 500 \
    --densify-end-iter 2500 \
    --densify-interval 100
```

## Evaluation

```bash
# Fast evaluation using compiled CUDA renderer (recommended for 129k+ carrier scenes)
python scripts/eval_psnr.py outputs/<scene>-run.aura outputs/<scene>-manifest.json \
    --frames 20 --device cuda --renderer cuda --scale 0.125

# Full-resolution evaluation (slow for large scenes; may OOM on <8 GB GPU)
python scripts/eval_psnr.py outputs/<scene>-run.aura outputs/<scene>-manifest.json \
    --frames 5 --device cuda --renderer torch --ray-batch 64
```

Reports PSNR, SSIM (11×11 Gaussian window), and LPIPS (if `pip install lpips` is available).
Use `--scale 0.125` to render at 1/8 resolution for faster evaluation; GT is downsampled to match.

### Reference numbers (T&T Truck scene)

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 3DGS (Kerbl et al. 2023) | ~25.19 dB | ~0.879 | ~0.148 |
| MP-GS (multi-primitive) | ~25–27 dB | — | — |
| AURA truck-3k-run6 (3,000 iters, 0.125× eval) | 6.89 dB | 0.044 | — |

> **The AURA number is real but badly under-converged** (3,000 iters; the
> per-iteration loss on memory-constrained random tiles stays noisy-flat, and
> ~82% of carriers never receive a gradient). It is NOT competitive with 3DGS
> and should not be read as such — it is the honest current state, measured
> on-GPU with the CUDA renderer. See README "Results" for the visual and the
> convergence caveat.

## Seeds

Runs are deterministic without an explicit seed flag:
- Carrier seeding: deterministic from the COLMAP sparse model (one carrier per
  SfM point with `--point-seeded`, or per occupied voxel cluster otherwise), so
  the same manifest reproduces the same initial scene.
- Training: deterministic given a fixed manifest and fixed hyper-parameters
  (the optimizer iterates carriers/targets in a fixed order). There is no
  `--seed` flag on `aura.cli train`.
- Evaluation: deterministic ray ordering, no random sampling.
