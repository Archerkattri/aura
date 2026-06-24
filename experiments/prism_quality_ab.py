#!/usr/bin/env python3
"""A/B: do opacity-reset + position-LR-decay improve PRISM-native quality?

Trains PRISM (gaussian footprint, densify) on the truck twice from the SAME seed —
once baseline, once with the new 3DGS-style stabilisers — and reports PSNR/SSIM.
Honest measurement of whether the gap to gsplat/Beta narrows.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
import torch
from prism_ablation import evaluate  # reuse the PRISM eval


def run(name, raw, regions, iterations, scale, extra):
    from aura.gsplat_renderer import seed_gaussian_params_from_regions
    from aura.prism import train_carriers_prism, PrismTrainConfig
    seed, ctx = seed_gaussian_params_from_regions(regions, device="cuda")
    cfg = PrismTrainConfig(iterations=iterations, scale=scale, densify=True,
                           log_every=max(1, iterations // 4),
                           log=lambda s: print(f"[{name}] {s}", flush=True), **extra)
    torch.cuda.synchronize(); t0 = time.time()
    scene, _ = train_carriers_prism(seed, ctx, raw, config=cfg, device="cuda", carrier="gaussian")
    torch.cuda.synchronize(); dt = time.time() - t0
    psnr, ssim = evaluate(scene, raw, scale)
    print(f"[{name}] PSNR={psnr:.2f} SSIM={ssim:.4f}  ({dt:.0f}s)", flush=True)
    return psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/truck-pts129k-manifest.json")
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--scale", type=float, default=0.25)
    a = ap.parse_args()
    from aura.cli import load_capture_manifest
    mobj = load_capture_manifest(a.manifest, validate=False)
    raw = json.loads(Path(a.manifest).read_text())

    base = run("baseline", raw, mobj.regions, a.iterations, a.scale, {})
    lr = run("lr-decay", raw, mobj.regions, a.iterations, a.scale,
             {"position_lr_final": 1.6e-6})
    full = run("lr-decay+opacity-reset", raw, mobj.regions, a.iterations, a.scale,
               {"opacity_reset_interval": max(600, a.iterations // 4),
                "opacity_reset_to": 0.06,        # ABOVE prune_opacity so reset doesn't
                "prune_opacity": 0.02,           #   mass-extinct on the next densify
                "position_lr_final": 1.6e-6})
    print("\n=== PRISM quality A/B (truck, gaussian, densify, %d iters @%.2f) ===" % (a.iterations, a.scale))
    print(f"baseline                 : PSNR {base[0]:.2f}  SSIM {base[1]:.4f}")
    print(f"lr-decay                 : PSNR {lr[0]:.2f}  SSIM {lr[1]:.4f}   (Δ {lr[0]-base[0]:+.2f})")
    print(f"lr-decay + opacity-reset : PSNR {full[0]:.2f}  SSIM {full[1]:.4f}   (Δ {full[0]-base[0]:+.2f})")


if __name__ == "__main__":
    main()
