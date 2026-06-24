#!/usr/bin/env bash
# Compactness: does the adaptive Beta carrier reach a given quality with FEWER
# carriers than a fixed Gaussian? Sweep cap_max for both arms (same harness/eval)
# to trace quality-vs-budget. If Beta's curve dominates — and especially if Beta at
# a smaller cap matches Gaussian at a larger cap — the typed carrier is more compact
# (DBS's "45% params" claim). Pairs already on disk: beta@1M=26.352, gauss@1M=26.017.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T="$REPO/data/tanks/truck"
OUT="${1:-/tmp/dbs_out}"
ITERS="${2:-30000}"
cd /tmp/dbs
export OMP_NUM_THREADS=4

beta()  { CUDA_VISIBLE_DEVICES=$1 python train.py -s "$T" --eval --iterations "$ITERS" \
            --cap_max "$2" --disable_viewer --model_path "$OUT/truck_beta_$2" > "$OUT/beta_$2.log" 2>&1; }
gauss() { CUDA_VISIBLE_DEVICES=$1 python "$REPO/experiments/dbs_uniform_beta.py" -s "$T" --eval \
            --iterations "$ITERS" --sb_number 0 --beta_lr 0 --cap_max "$2" \
            --disable_viewer --model_path "$OUT/truck_gauss_$2" > "$OUT/gauss_$2.log" 2>&1; }

# GPU0: beta@250k, gauss@250k ; GPU1: beta@500k, gauss@500k
( beta 0 250000 ; gauss 0 250000 ) &
( beta 1 500000 ; gauss 1 500000 ) &
wait

echo "=== COMPACTNESS (PSNR @ cap_max) ==="
echo "beta @1M : 26.352 | gauss @1M : 26.017  (already on disk)"
for cap in 250000 500000; do
  for arm in beta gauss; do
    m="$OUT/truck_${arm}_${cap}/point_cloud/iteration_best/metrics.json"
    echo "$arm @$cap: $(tr -d '\n ' < "$m" 2>/dev/null || echo missing)"
  done
done
