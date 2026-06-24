#!/usr/bin/env bash
# Track 1b: does adaptive per-region β (learned, full DBS = 26.352 dB) beat the
# BEST SINGLE uniform β? Sweep frozen uniform β with spherical-Beta colour held ON
# (sb_number 2) so only the kernel-shape DOF differs from full DBS. Split across
# both GPUs (GPU0 + GPU1). The learned-β arm is /tmp/dbs_out/truck_beta (already run).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T="$REPO/data/tanks/truck"
OUT="${1:-/tmp/dbs_out}"
ITERS="${2:-30000}"
cd /tmp/dbs
export OMP_NUM_THREADS=4

run() {  # gpu beta
  local gpu=$1 beta=$2
  UNIFORM_BETA=$beta CUDA_VISIBLE_DEVICES=$gpu python "$REPO/experiments/dbs_uniform_beta.py" \
    -s "$T" --eval --iterations "$ITERS" --sb_number 2 --beta_lr 0 \
    --disable_viewer --model_path "$OUT/truck_ub$beta" \
    > "$OUT/ub${beta}.log" 2>&1
}

# GPU0 gets beta=2,16 ; GPU1 gets beta=6,50 — two arms per GPU, sequential within.
( run 0 2 ; run 0 16 ) &
P0=$!
( run 1 6 ; run 1 50 ) &
P1=$!
wait $P0 $P1

echo "=== ROUTING SWEEP RESULTS (uniform frozen β, sb on) ==="
echo "learned-β (adaptive routing): $(tr -d '\n ' < "$OUT/truck_beta/point_cloud/iteration_best/metrics.json")"
for b in 2 6 16 50; do
  m="$OUT/truck_ub$b/point_cloud/iteration_best/metrics.json"
  echo "uniform β=$b: $(tr -d '\n ' < "$m" 2>/dev/null || echo missing)"
done
