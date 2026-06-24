#!/usr/bin/env bash
# Run every AURA benchmark in sequence and collect one combined report.
# Heavy DBS training arms are deterministic — skipped if their metrics already exist.
# Usage: bash experiments/run_all_benchmarks.sh [OUT] [ITERS]
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:-/tmp/dbs_out}"; ITERS="${2:-30000}"
DBS_PY="$REPO/.dbs_venv/bin/python"; GPU_PY="$REPO/.gpu_venv/bin/python"

have() { [ -f "$OUT/$1/point_cloud/iteration_best/metrics.json" ]; }

echo "### 1/4 typed-carrier ablation (Beta vs fixed Gaussian, $ITERS iters)"
have truck_beta || bash experiments/dbs_truck_ablation.sh "$OUT" "$ITERS"
echo "### 2/4 compactness sweep (250k/500k)"
have truck_beta_500000 || bash experiments/dbs_compactness_sweep.sh "$OUT" "$ITERS"
echo "### 3/4 routing sweep (uniform beta)"
have truck_ub6 || bash experiments/dbs_routing_sweep.sh "$OUT" "$ITERS"
echo "### 4/4 PRISM-native A/B (stabilisers) + max-push (clone+split)"
"$GPU_PY" experiments/prism_quality_ab.py --iterations 3000 --scale 0.25 || true
"$GPU_PY" experiments/prism_maxpush.py --iters 3000 7000 --scale 0.25 || true

echo; echo "### combined report"
"$GPU_PY" experiments/collect_results.py --out "$OUT"
