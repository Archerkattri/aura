#!/usr/bin/env python3
"""Regenerated lineage figure with AURA at its current Beta quality (runs in .dbs_venv).

Renders GT · COLMAP SfM points · NeRF · 3DGS · AURA for the SAME frames/scale, all
through `manifest_frame_to_camera` cameras (matched views). 3DGS = the Gaussian arm,
AURA = the Beta arm — both via the fork. NeRF is a compact from-scratch model.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, "/tmp/dbs")
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from gsplat.rendering import rasterization
from scene import BetaModel
from aura.gsplat_renderer import manifest_frame_to_camera
from aura.ingest.colmap import load_colmap_model
import mini_nerf

_C0 = 0.28209479177387814


def _font(px):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        try: return ImageFont.truetype(p, px)
        except Exception: pass
    return ImageFont.load_default()


def img_of(flat, w, h):
    a = (np.array(flat, "float32").reshape(h, w, 3).clip(0, 1) * 255).astype("uint8")
    return Image.fromarray(a, "RGB")


def load_model(ply, sb_number, sh_degree=0):
    m = BetaModel(sh_degree, sb_number); m.max_sh_degree = sh_degree
    m.load_ply(ply); return m


def render_model(m, view, k, w, h, sb_number):
    vm = torch.tensor(view, dtype=torch.float32, device="cuda").unsqueeze(0)
    K = torch.tensor(k, dtype=torch.float32, device="cuda").unsqueeze(0)
    sh = m.get_shs  # [N, K, 3]
    with torch.no_grad():
        out, _, _ = rasterization(means=m.get_xyz, quats=m.get_rotation, scales=m.get_scaling,
                                  opacities=m.get_opacity.squeeze(-1), betas=m.get_beta.squeeze(),
                                  colors=sh, viewmats=vm, Ks=K, width=w, height=h, sh_degree=0,
                                  sb_number=(sb_number or None),
                                  sb_params=(m.get_sb_params if sb_number else None),
                                  backgrounds=torch.zeros(1, 3, device="cuda"))
    return out[0, ..., :3].clamp(0, 1).reshape(-1).cpu().tolist()


def colmap_points(colmap_dir, view, k, w, h):
    _, _, pts, _ = load_colmap_model(colmap_dir)
    xyz = torch.tensor([list(p.xyz) for p in pts], dtype=torch.float32, device="cuda")
    rgb = torch.tensor([list(p.rgb) for p in pts], dtype=torch.float32, device="cuda").clamp(0, 1)
    R = torch.tensor(view, device="cuda")[:3, :3]; t = torch.tensor(view, device="cuda")[:3, 3]
    K = torch.tensor(k, device="cuda")
    pc = xyz @ R.T + t; z = pc[:, 2]; m = z > 1e-3
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]; v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    ui = u.round().long(); vi = v.round().long()
    inb = m & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    order = torch.argsort(z[inb], descending=True)
    img = torch.zeros(h, w, 3, device="cuda")
    ui2, vi2, rgb2 = ui[inb][order], vi[inb][order], rgb[inb][order]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            img[(vi2 + dy).clamp(0, h - 1), (ui2 + dx).clamp(0, w - 1)] = rgb2
    return img.reshape(-1).cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "outputs/truck-pts129k-manifest.json"))
    ap.add_argument("--colmap", default=str(ROOT / "data/tanks/truck/sparse/0"))
    ap.add_argument("--images", default=str(ROOT / "data/tanks/truck/images"))
    ap.add_argument("--gauss", default="/tmp/dbs_out/truck_gauss/point_cloud/iteration_best/point_cloud.ply")
    ap.add_argument("--beta", default="/tmp/dbs_out/truck_beta/point_cloud/iteration_best/point_cloud.ply")
    ap.add_argument("--out", default=str(ROOT / "docs/lineage_truck.png"))
    ap.add_argument("--frames", type=int, default=3); ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--nerf-iters", type=int, default=6000)
    a = ap.parse_args()

    manifest = json.loads(Path(a.manifest).read_text()); root = Path(manifest["root"])
    fr = manifest["frames"]; stride = max(1, len(fr) // a.frames)
    sel = [f for f in fr[::stride][:a.frames] if (root / f["image_path"]).exists()]
    from PIL import Image as PImage

    print("training NeRF...", flush=True)
    nerf = mini_nerf.train_nerf(a.colmap, a.images, scale=a.scale, iters=a.nerf_iters, log=lambda s: print(s, flush=True))
    gauss = load_model(a.gauss, 0); beta = load_model(a.beta, 2)

    rows = []
    for f in sel:
        view, k, w, h = manifest_frame_to_camera(f, a.scale)
        gt = PImage.open(root / f["image_path"]).convert("RGB").resize((w, h))
        col = img_of(colmap_points(a.colmap, view, k, w, h), w, h)
        nw, nh, nflat = nerf(f, a.scale)
        ner = img_of(nflat, nw, nh).resize((w, h))
        tdgs = img_of(render_model(gauss, view, k, w, h, 0), w, h)
        aura = img_of(render_model(beta, view, k, w, h, 2), w, h)
        rows.append([gt, col, ner, tdgs, aura])

    labels = ["Ground truth", "COLMAP (SfM points)", "NeRF", "vanilla 3DGS", "AURA (Beta)"]
    w, h = rows[0][0].size; band = max(22, w // 16); ncol = 5
    grid = Image.new("RGB", (w * ncol, band + h * len(rows)), (16, 16, 16))
    d = ImageDraw.Draw(grid); f = _font(max(15, w // 14))
    for j, lab in enumerate(labels):
        tw = d.textlength(lab, font=f); d.text((j * w + (w - tw) / 2, 5), lab, font=f, fill=(240, 240, 240))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            grid.paste(im, (j * w, band + i * h))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); grid.save(a.out)
    print(f"wrote {a.out} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
