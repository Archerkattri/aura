#!/usr/bin/env python3
"""Lineage comparison figure: Ground truth | COLMAP (SfM points) | NeRF | vanilla
3DGS | AURA — the Photogrammetry -> NeRF -> 3DGS -> AURA progression, all rendered
through the SAME (correct) COLMAP poses for fairness.
"""
import argparse, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "experiments"))
import torch, numpy as np
from PIL import Image, ImageDraw


def _img(flat, w, h, up=2):
    a = (np.array(flat, dtype="float32").reshape(h, w, 3).clip(0, 1) * 255).astype("uint8")
    im = Image.fromarray(a, "RGB")
    return im.resize((w * up, h * up), Image.NEAREST) if up != 1 else im


def colmap_points_render(colmap_dir, frame, scale, device="cuda"):
    """Render the COLMAP sparse SfM points (photogrammetry) through one frame."""
    from aura.ingest.colmap import load_colmap_model
    from aura.gsplat_renderer import manifest_frame_to_camera
    _, _, points, _ = load_colmap_model(colmap_dir)
    xyz = torch.tensor([list(p.xyz) for p in points], dtype=torch.float32, device=device)
    rgb = torch.tensor([[c/255 for c in p.rgb] for p in points], dtype=torch.float32, device=device)
    view, k, w, h = manifest_frame_to_camera(frame, scale)
    R = torch.tensor(view, device=device)[:3, :3]; t = torch.tensor(view, device=device)[:3, 3]
    K = torch.tensor(k, device=device)
    pc = xyz @ R.T + t
    z = pc[:, 2]; m = z > 1e-3
    u = (K[0, 0] * pc[:, 0] / z + K[0, 2]); v = (K[1, 1] * pc[:, 1] / z + K[1, 2])
    img = torch.zeros(h, w, 3, device=device); zbuf = torch.full((h, w), 1e9, device=device)
    ui = u.round().long(); vi = v.round().long()
    inb = m & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    order = torch.argsort(z[inb], descending=True)  # far first so near overwrites
    ui2, vi2, rgb2, z2 = ui[inb][order], vi[inb][order], rgb[inb][order], z[inb][order]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = (vi2 + dy).clamp(0, h - 1); xx = (ui2 + dx).clamp(0, w - 1)
            img[yy, xx] = rgb2
    return w, h, img.clamp(0, 1).reshape(-1).cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package"); ap.add_argument("manifest")
    ap.add_argument("--colmap", default="data/tanks/truck/sparse/0")
    ap.add_argument("--frames", type=int, default=3); ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--baseline-iters", type=int, default=7000); ap.add_argument("--nerf-iters", type=int, default=6000)
    ap.add_argument("--out", default="docs/lineage_truck.png")
    args = ap.parse_args()
    from aura.package import load_package
    from aura.gsplat_renderer import render_scene_gsplat
    from eval_psnr import load_jpg_as_rgb, resize_pixels
    import run_baseline_3dgs as rb
    import mini_nerf

    manifest = json.loads(Path(args.manifest).read_text()); root = Path(manifest["root"])
    frames = manifest["frames"]; stride = max(1, len(frames) // args.frames)
    sel = [f for f in frames[::stride][:args.frames] if (root / f["image_path"]).exists()]

    print("training NeRF...", flush=True)
    nerf = mini_nerf.train_nerf(args.colmap, str(root / "images"), scale=args.scale, iters=args.nerf_iters, log=lambda s: print(s, flush=True))
    print("training vanilla 3DGS...", flush=True)
    gs = rb.train_baseline(manifest, Path(args.colmap), iterations=args.baseline_iters, device="cuda", scale=args.scale)
    print("loading AURA...", flush=True)
    scene = load_package(args.package).scene

    cols = []
    for fr in sel:
        gw, gh, gt = load_jpg_as_rgb(str(root / fr["image_path"]))
        cw, ch, cflat = colmap_points_render(args.colmap, fr, args.scale)
        nw, nh, nflat = nerf(fr, args.scale)
        bw, bh, bflat = gs(fr, args.scale)
        aw, ah, aflat = render_scene_gsplat(scene, fr, args.scale, device="cuda")
        if (gw, gh) != (aw, ah):
            gt = resize_pixels(gt, gw, gh, aw, ah); gw, gh = aw, ah
        cols.append([_img(gt, gw, gh), _img(cflat, cw, ch), _img(nflat, nw, nh), _img(bflat, bw, bh), _img(aflat, aw, ah)])

    tw, th = cols[0][0].size; hdr = 22; ncol = 5
    grid = Image.new("RGB", (tw * ncol, hdr + th * len(cols)), (18, 18, 18))
    d = ImageDraw.Draw(grid)
    for j, lab in enumerate(["Ground truth", "COLMAP (SfM points)", "NeRF", "vanilla 3DGS", "AURA"]):
        d.text((j * tw + 6, 5), lab, fill=(240, 240, 240))
    for i, row in enumerate(cols):
        for j, im in enumerate(row):
            grid.paste(im, (j * tw, hdr + i * th))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); grid.save(args.out)
    print(f"wrote {args.out} ({grid.size[0]}x{grid.size[1]})", flush=True)


if __name__ == "__main__":
    main()
