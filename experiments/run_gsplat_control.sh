#!/usr/bin/env bash
# B2 headline control: true gsplat-3DGS arm (MCMC strategy) at the SAME fixed
# budget every beta/gauss arm in /tmp/dbs_multiscene used (cap_max = 1,000,000
# primitives), same scenes, same image factors, same every-8th test split
# (DBS llffhold=8 == gsplat test_every=8), LPIPS network forced to vgg to match
# the DBS evaluator. Two numbers are reported per scene:
#   - final-at-30k  (community-standard protocol), and
#   - best-during-training (to match DBS's iteration_best, which selects the
#     checkpoint on TEST metrics — kept only so the comparison is like-for-like).
# Waits for a GPU with >= MIN_FREE MiB free before each scene, so it can be
# launched while other jobs occupy the cards.
# Usage: bash experiments/run_gsplat_control.sh
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GS_EXAMPLES="${AURA_GSPLAT_EXAMPLES:-/tmp/aura_sota_repos/gsplat-main/examples}"
PY="$REPO/.gpu_venv/bin/python"
OUT="${AURA_GSPLAT_CONTROL_OUT:-/tmp/gsplat_control}"
MIN_FREE="${MIN_FREE:-16000}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=4
# The gsplat-main examples need the source-tree gsplat (JIT-built CUDA ops),
# not the pip-installed 1.5.3 — same env as gsplat_main_mcmc_smoke.py, which
# validated this build on 2026-06-25 (cache: /tmp/aura_gsplat_main_3dgs_ext).
export PYTHONPATH="${GS_EXAMPLES%/examples}:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=/tmp/aura_gsplat_main_3dgs_ext
export MAX_JOBS=2 BUILD_3DGS=1 BUILD_3DGUT=0 BUILD_2DGS=0 BUILD_ADAM=1 BUILD_RELOC=1 BUILD_LOSSES=1

# (scene_dir, data_factor) — mirrors experiments/run_multiscene.sh resolutions:
# truck images/1, garden+bicycle+stump images_4, bonsai+counter+kitchen+room images_2.
SCENES=(
  "data/tanks/truck 1"
  "data/mipnerf360/garden 4"
  "data/mipnerf360/bicycle 4"
  "data/mipnerf360/stump 4"
  "data/mipnerf360/bonsai 2"
  "data/mipnerf360/counter 2"
  "data/mipnerf360/kitchen 2"
  "data/mipnerf360/room 2"
)

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v m="$MIN_FREE" '$2>=m {print $1; exit}'
}

for entry in "${SCENES[@]}"; do
  set -- $entry; sdir=$1; factor=$2; name=$(basename "$sdir")
  [ -d "$REPO/$sdir/sparse" ] || { echo "skip $name (no COLMAP model)"; continue; }
  if [ -f "$OUT/$name/stats/val_step29999.json" ]; then echo "skip $name (done)"; continue; fi
  gpu=$(pick_gpu)
  while [ -z "$gpu" ]; do
    echo "waiting for >=${MIN_FREE} MiB free VRAM ($(date +%H:%M))"
    sleep 300
    gpu=$(pick_gpu)
  done
  echo ">>> $name (GPU$gpu, factor $factor)"
  (cd "$GS_EXAMPLES" && CUDA_VISIBLE_DEVICES=$gpu "$PY" simple_trainer.py mcmc \
      --data_dir "$REPO/$sdir" --data_factor "$factor" \
      --result_dir "$OUT/$name" \
      --strategy.cap-max 1000000 \
      --eval_steps 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 \
      --save_steps 30000 --lpips_net vgg \
      --disable_viewer --disable_video) > "$OUT/$name.log" 2>&1
  echo "done $name"
done
echo "=== gsplat control done ==="
"$PY" "$REPO/experiments/collect_gsplat_control.py"
