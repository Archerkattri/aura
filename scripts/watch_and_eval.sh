#!/usr/bin/env bash
# Wait for an AURA checkpoint and run PSNR/SSIM/LPIPS eval.
# Usage: bash scripts/watch_and_eval.sh [device] [checkpoint] [scale]
set -euo pipefail

DEVICE="${1:-cuda}"
CHECKPOINT="${2:-outputs/truck-3k-run6.aura}"
SCALE="${3:-0.25}"
MANIFEST="outputs/truck-pts129k-manifest.json"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Derive the result name from the checkpoint so the report can never drift from
# the run being evaluated (the previous hard-coded name caused run5/run6 mix-ups).
base="$(basename "$CHECKPOINT" .aura)"
RESULT="outputs/eval_${base}.txt"

echo "Watching for $CHECKPOINT ..."
while [ ! -d "$ROOT/$CHECKPOINT" ] && [ ! -f "$ROOT/$CHECKPOINT" ]; do
    sleep 60
done

echo "Checkpoint found! Running eval ..."
cd "$ROOT"
python scripts/eval_psnr.py "$CHECKPOINT" "$MANIFEST" \
    --frames 20 --device "$DEVICE" --scale "$SCALE" \
    2>&1 | tee "$RESULT"

echo "Results saved to $RESULT"
