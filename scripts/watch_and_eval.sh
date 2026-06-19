#!/usr/bin/env bash
# Wait for truck-3k-run5.aura checkpoint and run PSNR/SSIM/LPIPS eval.
# Usage: bash scripts/watch_and_eval.sh [device]
set -euo pipefail

DEVICE="${1:-cuda}"
CHECKPOINT="outputs/truck-3k-run5.aura"
MANIFEST="outputs/truck-pts129k-manifest.json"
RESULT="outputs/eval_truck_3k_run5.txt"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Watching for $CHECKPOINT ..."
while [ ! -d "$ROOT/$CHECKPOINT" ] && [ ! -f "$ROOT/$CHECKPOINT" ]; do
    sleep 60
done

echo "Checkpoint found! Running eval ..."
cd "$ROOT"
python scripts/eval_psnr.py "$CHECKPOINT" "$MANIFEST" \
    --frames 20 --device "$DEVICE" \
    2>&1 | tee "$RESULT"

echo "Results saved to $RESULT"
