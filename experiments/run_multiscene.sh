#!/usr/bin/env bash
# Multi-scene typed-carrier benchmark: Beta (AURA) vs fixed Gaussian on Tanks&Temples
# Truck + Mip-NeRF 360 scenes, same harness/eval per scene. Two arms per scene,
# split across both GPUs. ONLY run when the GPUs are idle (do not contaminate other
# GPU benchmarks). Usage: bash experiments/run_multiscene.sh [ITERS]
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="/tmp/dbs_multiscene"; mkdir -p "$OUT"
ITERS="${1:-7000}"
DBS_PY="$REPO/.dbs_venv/bin/python"
export OMP_NUM_THREADS=4

# (scene_dir, images_subdir) — Mip360 uses images_4 (~1.25k px) for tractable training
SCENES=(
  "data/tanks/truck images"
  "data/mipnerf360/garden images_4"
  "data/mipnerf360/bicycle images_4"
  "data/mipnerf360/bonsai images_2"
  "data/mipnerf360/counter images_2"
  "data/mipnerf360/kitchen images_2"
  "data/mipnerf360/room images_2"
  "data/mipnerf360/stump images_4"
)

beta()  { local g=$1 s=$2 img=$3 name=$4
  CUDA_VISIBLE_DEVICES=$g "$DBS_PY" /tmp/dbs/train.py -s "$REPO/$s" --images "$img" --eval \
    --iterations "$ITERS" --disable_viewer --model_path "$OUT/${name}_beta" > "$OUT/${name}_beta.log" 2>&1; }
gauss() { local g=$1 s=$2 img=$3 name=$4
  UNIFORM_BETA=4 CUDA_VISIBLE_DEVICES=$g "$DBS_PY" experiments/dbs_uniform_beta.py -s "$REPO/$s" --images "$img" --eval \
    --iterations "$ITERS" --sb_number 0 --beta_lr 0 --disable_viewer --model_path "$OUT/${name}_gauss" > "$OUT/${name}_gauss.log" 2>&1; }

i=0
for entry in "${SCENES[@]}"; do
  set -- $entry; sdir=$1; img=$2; name=$(basename "$sdir")
  gpu=$((i % 2))            # alternate GPUs
  ( beta $gpu "$sdir" "$img" "$name"; gauss $gpu "$sdir" "$img" "$name"; echo "done $name" ) &
  i=$((i + 1))
  [ $((i % 2)) -eq 0 ] && wait    # 2 scenes (1 per GPU) at a time
done
wait
echo "=== multi-scene done ==="
"$REPO/.gpu_venv/bin/python" experiments/collect_multiscene.py --out "$OUT"
