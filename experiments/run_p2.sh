#!/usr/bin/env bash
# P2 full-resolution + render-loss pipeline for ONE scene.
#
#   experiments/run_p2.sh <scene> <gpu> [iters]
#
# Assumes outputs/<scene>-fr-manifest.json already exists (colmap-to-capture-
# manifest, --image-dir images, --point-seeded --max-seed-regions 120000) and the
# scene images are under data/. Produces, all in outputs/:
#   <scene>-fr.aura/carriers.npz     full-resolution (scale 1.0) train-split carriers
#   <scene>-q.aura/carriers.npz      quarter-resolution (scale 0.25) control carriers
#   reliability_<scene>_fr.npz       full-res colour-agreement label     (P2a)
#   reliability_<scene>_fr_depth.npz full-res occlusion-aware label
#   reliability_<scene>_q.npz        quarter-res colour label (resolution A/B)
#   reliability_renderloss_<scene>_fr.npz  full-res render-loss label     (P2b)
#   rel_<scene>_{fr,fr_depth,q}.json  + rl_<scene>_fr.json  reliability summaries
#   calib_<scene>_{fr,fr_depth,q,renderloss}.json  calibration + certificate reports
#
# Accuracy job — safe on shared GPUs (gpu-usage-policy).
set -euo pipefail
cd "$(dirname "$0")/.."

SCENE="${1:?scene}"; GPU="${2:?gpu}"; ITERS="${3:-5000}"; FRSCALE="${4:-1.0}"
PY=".gpu_venv/bin/python"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # fit shared/loaded 5090s
M="outputs/${SCENE}-fr-manifest.json"
FR="outputs/${SCENE}-fr.aura"
Q="outputs/${SCENE}-q.aura"

# FRSCALE is the carrier TRAINING resolution (1.0 = native images/). Reliability
# labels always project against full-res GT regardless, so only the trained-carrier
# resolution changes. garden's native 17.4MP OOMs under the shared GPU load, so it
# is trained at 0.5 (2593x1680); the other scenes train at native 1.0.
echo "[$SCENE] train full-res (scale $FRSCALE)"
[ -f "$FR/carriers.npz" ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY experiments/train_carriers_p2.py --manifest "$M" --out "$FR" --scale "$FRSCALE" --iterations "$ITERS"
echo "[$SCENE] train quarter-res control (scale 0.25)"
[ -f "$Q/carriers.npz" ] || $PY experiments/train_carriers_p2.py --manifest "$M" --out "$Q" --scale 0.25 --iterations "$ITERS"

echo "[$SCENE] full-res colour label"
$PY experiments/per_carrier_reliability.py --aura "$FR" --manifest "$M" \
    --label color --out "outputs/reliability_${SCENE}_fr.npz" > "outputs/rel_${SCENE}_fr.json"
echo "[$SCENE] full-res occlusion-aware label"
$PY experiments/per_carrier_reliability.py --aura "$FR" --manifest "$M" \
    --label depth_aware --out "outputs/reliability_${SCENE}_fr_depth.npz" > "outputs/rel_${SCENE}_fr_depth.json"
echo "[$SCENE] quarter-res colour label (resolution control)"
$PY experiments/per_carrier_reliability.py --aura "$Q" --manifest "$M" \
    --label color --out "outputs/reliability_${SCENE}_q.npz" > "outputs/rel_${SCENE}_q.json"
echo "[$SCENE] full-res RENDER-LOSS label"
$PY experiments/render_loss_reliability.py --carriers "$FR" --manifest "$M" \
    --color-npz "outputs/reliability_${SCENE}_fr.npz" \
    --out "outputs/reliability_renderloss_${SCENE}_fr.npz" > "outputs/rl_${SCENE}_fr.json"

echo "[$SCENE] calibrate + certify (4 conditions)"
$PY experiments/calibrate_confidence.py --reliability "outputs/reliability_${SCENE}_fr.npz" \
    --scene "$SCENE" --report "outputs/calib_${SCENE}_fr.json" > /dev/null
$PY experiments/calibrate_confidence.py --reliability "outputs/reliability_${SCENE}_fr_depth.npz" \
    --scene "$SCENE" --report "outputs/calib_${SCENE}_fr_depth.json" > /dev/null
$PY experiments/calibrate_confidence.py --reliability "outputs/reliability_${SCENE}_q.npz" \
    --scene "$SCENE" --report "outputs/calib_${SCENE}_q.json" > /dev/null
$PY experiments/calibrate_confidence.py --reliability "outputs/reliability_renderloss_${SCENE}_fr.npz" \
    --scene "$SCENE" --report "outputs/calib_${SCENE}_renderloss.json" > /dev/null
echo "[$SCENE] DONE"
