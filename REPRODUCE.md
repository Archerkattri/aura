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
    --disable-evolution
```

## Evaluation

```bash
python scripts/eval_psnr.py outputs/<scene>-run.aura outputs/<scene>-manifest.json \
    --frames 10 --device cuda
```

Reports PSNR and SSIM. Reference numbers for 3DGS (Kerbl et al. 2023): Truck ~25.19 dB PSNR, ~0.857 SSIM.

## Seeds

All random seeds are fixed:
- Training: controlled by `--seed` flag (default 0) in `aura.cli`
- Evaluation: deterministic ray ordering, no random sampling
