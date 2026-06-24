#!/usr/bin/env python3
"""How far can PRISM-native go? Full config (opacity reset + LR decay + clone&split
densification) at increasing iterations, vs the gsplat reference (~18-19 dB @0.25).
Honest measurement of the remaining gap. Runs in .gpu_venv."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "experiments"))
import torch
from prism_ablation import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/truck-pts129k-manifest.json")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--iters", type=int, nargs="+", default=[3000, 7000])
    a = ap.parse_args()
    from aura.cli import load_capture_manifest
    from aura.gsplat_renderer import seed_gaussian_params_from_regions
    from aura.prism import train_carriers_prism, PrismTrainConfig
    mobj = load_capture_manifest(a.manifest, validate=False)
    raw = json.loads(Path(a.manifest).read_text())

    rows = []
    for it in a.iters:
        seed, ctx = seed_gaussian_params_from_regions(mobj.regions, device="cuda")
        cfg = PrismTrainConfig(
            iterations=it, scale=a.scale, densify=True, densify_stop=int(it * 0.7),
            opacity_reset_interval=max(600, it // 4), opacity_reset_to=0.06, prune_opacity=0.02,
            position_lr_final=1.6e-6, split_scale_percentile=0.5, max_per_tile=384,
            log_every=max(1, it // 3), log=lambda s: print(f"[{it}] {s}", flush=True))
        torch.cuda.synchronize(); t0 = time.time()
        scene, _ = train_carriers_prism(seed, ctx, raw, config=cfg, device="cuda", carrier="gaussian")
        torch.cuda.synchronize(); dt = time.time() - t0
        n = len([e for e in scene.elements])
        psnr, ssim = evaluate(scene, raw, a.scale)
        rows.append((it, psnr, ssim, n, dt))
        print(f"[{it}] PSNR={psnr:.2f} SSIM={ssim:.4f} N={n} ({dt:.0f}s)", flush=True)

    print("\n=== PRISM max-push (truck, gaussian, clone+split densify, @%.2f) ===" % a.scale)
    print("  reference: gsplat backend ~18-19 dB @0.25 ; Beta backend 26.4 dB full-res")
    for it, psnr, ssim, n, dt in rows:
        print(f"  {it:6d} iters: PSNR {psnr:.2f}  SSIM {ssim:.4f}  N={n}  ({dt:.0f}s)")


if __name__ == "__main__":
    main()
