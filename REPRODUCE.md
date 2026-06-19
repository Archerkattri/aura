# Reproducing the AURA Benchmark

## Environment

```bash
pip install -e ".[torch,dev]"
```

## Datasets

See `scripts/fetch_datasets.py --list` for dataset instructions.

Datasets are NOT committed to git (binary, large). Each scene is ingested via:
```bash
python -m aura.cli ingest <scene_dir>/ --output outputs/<scene>-manifest.json
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
| Run | truck-3k-run5 |
| Carriers | 129,531 |
| Iterations | 3000 |
| Loss at convergence | ~0.02 |
| Checkpoint | `outputs/truck-3k-run5.aura` |

## Evaluation

```bash
python scripts/eval_psnr.py outputs/<scene>-run.aura outputs/<scene>-manifest.json \
    --frames 20 --device cuda
```

Reports PSNR, SSIM (11×11 Gaussian window), and LPIPS (if `pip install lpips` is available).

### Reference numbers (T&T Truck scene)

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 3DGS (Kerbl et al. 2023) | ~25.19 dB | ~0.857 | ~0.177 |
| MP-GS (multi-primitive) | ~25–27 dB | — | — |
| AURA truck-3k-run5 | TBD | TBD | TBD |

## Seeds

All random seeds are fixed:
- Training: controlled by `--seed` flag (default 0) in `aura.cli`
- Evaluation: deterministic ray ordering, no random sampling
