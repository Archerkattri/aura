#!/usr/bin/env bash
# Real on-GPU AURA-Core training of the Tanks&Temples truck scene with the
# carrier-coverage rotation fix (full-frame supervision + rotating batch
# window), then held-out PSNR/SSIM eval. This is the run that supersedes the
# starved run6 (~6.9 dB floor) documented in docs/CONVERGENCE_TODO.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source .gpu_venv/bin/activate
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST=outputs/truck-pts129k-manifest.json
OUT=outputs/truck-coverage.aura

echo "[train] start $(date)"
python -m aura.cli train "$MANIFEST" \
  --output "$OUT" \
  --iterations 425 --pixel-stride 8 --tile-size 256 \
  --max-targets-per-frame 100000 --max-targets-per-batch 256 \
  --batches-per-iteration 64 \
  --skip-validation --disable-evolution \
  --position-lr 1.6e-4 --position-lr-final 1.6e-6 --lr-decay-steps 425 \
  --opacity-reset-interval 150 --device cuda
echo "[train] done $(date)"

echo "[eval] AURA held-out PSNR/SSIM"
python scripts/eval_psnr.py "$OUT" "$MANIFEST" \
  --frames 5 --device cuda --renderer cuda --scale 0.125 \
  | tee outputs/eval_truck_coverage.txt
echo "[eval] done $(date)"
