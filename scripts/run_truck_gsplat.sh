#!/usr/bin/env bash
# Train the truck scene's Gaussian carriers with AURA-Core's gsplat
# differentiable-CUDA-rasterizer backend (densification on), write them back
# into an .aura package, and evaluate with BOTH AURA's own forward renderer
# (cross-renderer fidelity) and gsplat (training-renderer quality).
set -euo pipefail
cd "$(dirname "$0")/.."
source .gpu_venv/bin/activate
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MANIFEST=outputs/truck-pts129k-manifest.json
OUT=outputs/truck-gsplat.aura
SCALE=0.25

echo "[gsplat-train] start $(date)"
python -m aura.cli train-gsplat "$MANIFEST" \
  --output "$OUT" --iterations 7000 --scale "$SCALE" \
  --ssim-weight 0.2 --densify --device cuda --skip-validation
echo "[gsplat-train] done $(date)"

echo "[eval] AURA forward (cuda) renderer — cross-renderer fidelity"
python scripts/eval_psnr.py "$OUT" "$MANIFEST" \
  --frames 5 --device cuda --renderer cuda --scale 0.125 \
  | tee outputs/eval_truck_gsplat_auraforward.txt

echo "[eval] gsplat renderer — training-renderer quality"
python - "$OUT" "$MANIFEST" "$SCALE" <<'PY' | tee outputs/eval_truck_gsplat_gsplatrender.txt
import sys, json
from pathlib import Path
from aura.package import load_package
from aura.gsplat_renderer import render_scene_gsplat
sys.path.insert(0, "scripts")
from eval_psnr import load_jpg_as_rgb, resize_pixels  # reuse identical image IO

pkg_dir, manifest_path, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
scene = load_package(pkg_dir).scene
manifest = json.loads(Path(manifest_path).read_text())
root = Path(manifest["root"]); frames = manifest["frames"]
stride = max(1, len(frames)//5)
sel = frames[::stride][:5]
import math
tot=0.0; n=0
for fr in sel:
    w,h,flat = render_scene_gsplat(scene, fr, scale, device="cuda")
    gw,gh,gt = load_jpg_as_rgb(str(root/fr["image_path"]))
    if (gw,gh)!=(w,h): gt = resize_pixels(gt,gw,gh,w,h)
    mse = sum((a-b)**2 for a,b in zip(flat,gt))/len(flat)
    p = 10*math.log10(1.0/mse) if mse>0 else 99.0
    print(f"  {fr['image_path']}: PSNR={p:.2f} dB"); tot+=p; n+=1
print(f"\nAverage PSNR: {tot/n:.2f} dB  (gsplat renderer, scale {scale})")
PY
echo "[eval] done $(date)"
