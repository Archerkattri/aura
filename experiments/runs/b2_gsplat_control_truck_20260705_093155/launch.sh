#!/usr/bin/env bash
# B2 headline control — TRUE gsplat-3DGS (MCMC strategy) matched-budget arm on
# Tanks&Temples Truck. Mirrors experiments/run_gsplat_control.sh (truck row) but:
#   - pinned to CUDA_VISIBLE_DEVICES=1 (GPU1 has the most free VRAM; be a good
#     neighbour to gaussianfeels on the shared cards),
#   - CPU threads capped,
#   - results -> /tmp/gsplat_control/truck  (so experiments/collect_gsplat_control.py
#     finds them at its default AURA_GSPLAT_CONTROL_OUT),
#   - this script's stdout/stderr is captured to ../train.log by the launch wrapper.
# Budget: cap_max = 1,000,000 primitives == the fixed budget every DBS beta/gauss
# arm used in /tmp/dbs_multiscene. Split: test_every=8 (== DBS llffhold=8).
# LPIPS net = vgg to match the DBS evaluator. Reports final-at-30k (standard) and
# best-during-training (matches DBS iteration_best test-metric selection).
set -euo pipefail

REPO=/home/krishi/workspace/brain/workspace/projects/aura/repo
GS_EXAMPLES=/tmp/aura_sota_repos/gsplat-main/examples
PY="$REPO/.gpu_venv/bin/python"
OUT=/tmp/gsplat_control
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# gsplat-main source tree (JIT CUDA ops) — NOT the pip 1.5.3. Same env as
# experiments/gsplat_main_mcmc_smoke.py / run_gsplat_control.sh (build cache reused).
export PYTHONPATH=/tmp/aura_sota_repos/gsplat-main
export TORCH_EXTENSIONS_DIR=/tmp/aura_gsplat_main_3dgs_ext
export MAX_JOBS=2 BUILD_3DGS=1 BUILD_3DGUT=0 BUILD_2DGS=0 BUILD_ADAM=1 BUILD_RELOC=1 BUILD_LOSSES=1

cd "$GS_EXAMPLES"
"$PY" -c "import torch,os;print('device', torch.cuda.get_device_name(0), 'CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES',''))"
exec "$PY" simple_trainer.py mcmc \
  --data_dir "$REPO/data/tanks/truck" --data_factor 1 \
  --result_dir "$OUT/truck" \
  --strategy.cap-max 1000000 \
  --eval_steps 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 \
  --save_steps 30000 --lpips_net vgg \
  --disable_viewer --disable_video
