#!/usr/bin/env python3
"""DBS <-> AURA bridge (runs INSIDE .dbs_venv, which has the Deformable Beta
Splatting `gsplat` fork + DBS scene code on sys.path).

Two jobs:

  convert  <ply> <out_npz> [--sb-number N] [--sh-degree D]
      Read a DBS-trained `point_cloud.ply` and write AURA's binary carrier
      sidecar (`carriers.npz`) with the FULL typed-carrier state: the usual
      means/scales/quats/opacity/SH *plus* the Beta-kernel shape (`beta`) and
      spherical-Beta colour lobes (`sb`). Lossless round-trip of a Beta carrier.

  eval     <carriers_npz> <manifest> [--scale S] [--device cuda]
      Render the carriers through every manifest frame with the DBS Beta
      rasterizer (betas + sb_params) and report PSNR/SSIM vs the ground-truth
      images. This is the faithful Beta render path — AURA's stock gsplat 1.5.3
      cannot evaluate Beta carriers, the fork can.

The DBS fork installs under the package name `gsplat`; this script must be run
with `.dbs_venv` active, never the main AURA venv.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DBS = Path("/tmp/dbs")
sys.path.insert(0, str(DBS))                      # scene.beta_model

import numpy as np
import torch


def _load_carrier_io():
    """Import aura.carrier_io WITHOUT triggering aura/__init__ (which pulls heavy
    deps like jsonschema that the lean .dbs_venv does not have)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "aura_carrier_io", str(ROOT / "src" / "aura" / "carrier_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_betamodel(ply: str, sb_number: int, sh_degree: int):
    from scene.beta_model import BetaModel
    m = BetaModel(sh_degree, sb_number)
    m.max_sh_degree = sh_degree
    m.active_sh_degree = sh_degree
    m.load_ply(ply)
    return m


def cmd_convert(args):
    save_carriers = _load_carrier_io().save_carriers
    m = _load_betamodel(args.ply, args.sb_number, args.sh_degree)
    with torch.no_grad():
        means = m.get_xyz.detach()                       # [N,3]
        scales = m.get_scaling.detach()                  # [N,3] linear
        quats = m.get_rotation.detach()                  # [N,4] wxyz normalised
        opacity = m.get_opacity.detach().squeeze(-1)     # [N]
        beta = m.get_beta.detach().reshape(-1)           # [N] activated 4*exp(beta)
        sb = m.get_sb_params.detach()                    # [N,L,6]
        # SH: DBS stores _sh0 [N,1,3] (DC) and _shN [N,K-1,3] (higher), already
        # [N, coeffs, channels]. AURA wants [N,K,3] — just concat on the coeff axis.
        sh0 = m._sh0.detach()                            # [N,1,3]
        shN = m._shN.detach() if args.sh_degree else None
        sh = sh0 if shN is None else torch.cat([sh0, shN], dim=1)   # [N,K,3]
    target = save_carriers(
        args.out, means=means, scales=scales, quats=quats, opacity=opacity,
        sh=sh, sh_degree=args.sh_degree, beta=beta, sb=sb,
    )
    print(f"wrote {target}  N={means.shape[0]}  sh_deg={args.sh_degree} "
          f"sb_lobes={sb.shape[1]}  beta[min/med/max]="
          f"{beta.min():.2f}/{beta.median():.2f}/{beta.max():.2f}", flush=True)


def _psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp(min=1e-12)
    return float(-10.0 * torch.log10(mse))


def cmd_eval(args):
    """Render Beta carriers via the fork and score against GT images."""
    from gsplat.rendering import rasterization
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from aura.carrier_io import load_carriers
    from aura.gsplat_renderer import manifest_frame_to_camera
    from eval_psnr import load_jpg_as_rgb, resize_pixels

    dev = args.device
    c = load_carriers(args.carriers, device=dev)
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(manifest["root"])
    sh = c["sh"]                                          # [N,K,3]
    sh_deg = int(c.get("sh_degree", 0))
    betas = c.get("beta")
    sb = c.get("sb")
    sb_number = None if sb is None else int(sb.shape[1])

    psnrs = []
    frames = [f for f in manifest["frames"] if (root / f["image_path"]).exists()]
    if args.max_frames:
        frames = frames[:: max(1, len(frames) // args.max_frames)][: args.max_frames]
    for fr in frames:
        view, k, w, h = manifest_frame_to_camera(fr, args.scale)
        vm = torch.tensor(view, dtype=torch.float32, device=dev).unsqueeze(0)
        K = torch.tensor(k, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            out, _, _ = rasterization(
                means=c["means"], quats=c["quats"], scales=c["scales"],
                opacities=c["opacity"], betas=betas, colors=sh,
                viewmats=vm, Ks=K, width=w, height=h,
                sb_number=sb_number, sb_params=sb, sh_degree=sh_deg,
            )
        pred = out[0, ..., :3].clamp(0, 1)
        gw, gh, gt = load_jpg_as_rgb(str(root / fr["image_path"]))
        if (gw, gh) != (w, h):
            gt = resize_pixels(gt, gw, gh, w, h)
        gt_t = torch.tensor(gt, dtype=torch.float32, device=dev).reshape(h, w, 3)
        psnrs.append(_psnr(pred, gt_t))
    print(json.dumps({"frames": len(psnrs), "psnr_mean": sum(psnrs) / max(1, len(psnrs)),
                      "scale": args.scale}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("ply"); c.add_argument("out")
    c.add_argument("--sb-number", type=int, default=2); c.add_argument("--sh-degree", type=int, default=0)
    c.set_defaults(func=cmd_convert)
    e = sub.add_parser("eval"); e.add_argument("carriers"); e.add_argument("manifest")
    e.add_argument("--scale", type=float, default=1.0); e.add_argument("--device", default="cuda")
    e.add_argument("--max-frames", type=int, default=0)
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
