#!/usr/bin/env bash
# Matched-budget typed-carrier ablation on Tanks&Temples Truck, all in the DBS
# (Deformable Beta Splatting) harness so data / eval-split (llffhold=8) / metric
# / MCMC densification / iteration budget are IDENTICAL across arms. The only
# difference is the carrier TYPE degrees of freedom:
#   beta  : deformable Beta kernel (learnable) + spherical-Beta colour (sb=2)   [DBS]
#   gauss : frozen kernel (beta_lr 0) + plain SH colour (sb_number 0)           [3DGS-like control]
# This isolates "does an adaptive typed carrier beat a fixed Gaussian-style one
# at matched budget" — AURA make-or-break claim #1.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T="$REPO/data/tanks/truck"
OUT="${1:-/tmp/dbs_out}"
ITERS="${2:-30000}"
cd /tmp/dbs
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0

echo "=== ARM gauss (frozen kernel + SH colour) ==="
python train.py -s "$T" --eval --iterations "$ITERS" --sb_number 0 --beta_lr 0 \
  --disable_viewer --model_path "$OUT/truck_gauss"

echo "=== ARM beta (deformable Beta + spherical-Beta colour) ==="
python train.py -s "$T" --eval --iterations "$ITERS" \
  --disable_viewer --model_path "$OUT/truck_beta"

echo "=== RESULTS ==="
for arm in gauss beta; do
  echo "--- $arm ---"
  cat "$OUT/truck_$arm/point_cloud/iteration_best/metrics.json" 2>/dev/null || echo "(no metrics)"
done
