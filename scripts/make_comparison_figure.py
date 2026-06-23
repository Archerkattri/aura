#!/usr/bin/env python3
"""Build a GT vs vanilla-3DGS vs AURA/PRISM comparison figure for the README.

Renders the same eval frames three ways — ground truth, an executed vanilla 3DGS
(gsplat) baseline trained on the same scene, and the trained AURA `.aura` package
through PRISM — and tiles them into one labelled PNG.

Usage:
  python scripts/make_comparison_figure.py outputs/truck-gsplat-hq.aura \
      outputs/truck-pts129k-manifest.json --colmap data/tanks/truck/sparse/0 \
      --frames 3 --scale 0.25 --baseline-iters 7000 --out docs/aura_vs_3dgs_truck.png
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _to_img(flat, w, h, upscale=2):
    import numpy as np
    from PIL import Image
    arr = (np.array(flat, dtype="float32").reshape(h, w, 3).clip(0, 1) * 255).astype("uint8")
    img = Image.fromarray(arr, "RGB")
    if upscale != 1:
        img = img.resize((w * upscale, h * upscale), Image.NEAREST)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("manifest")
    ap.add_argument("--colmap", default="data/tanks/truck/sparse/0")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--baseline-iters", type=int, default=7000)
    ap.add_argument("--out", default="docs/aura_vs_3dgs_truck.png")
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    from aura.package import load_package
    from aura.gsplat_renderer import render_scene_gsplat
    from eval_psnr import load_jpg_as_rgb, resize_pixels
    import run_baseline_3dgs as rb

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(manifest["root"])
    frames = manifest["frames"]
    stride = max(1, len(frames) // args.frames)
    sel = [f for f in frames[::stride][:args.frames] if (root / f["image_path"]).exists()]

    print("training vanilla 3DGS baseline (gsplat)...", flush=True)
    render_3dgs = rb.train_baseline(manifest, Path(args.colmap),
                                    iterations=args.baseline_iters, device="cuda", scale=args.scale)
    print("loading AURA package...", flush=True)
    scene = load_package(args.package).scene

    cols = []  # list of (gt_img, gs_img, aura_img)
    for fr in sel:
        gw, gh, gt = load_jpg_as_rgb(str(root / fr["image_path"]))
        bw, bh, bflat = render_3dgs(fr, args.scale)
        aw, ah, aflat = render_scene_gsplat(scene, fr, args.scale, device="cuda")
        if (gw, gh) != (aw, ah):
            gt = resize_pixels(gt, gw, gh, aw, ah); gw, gh = aw, ah
        cols.append((_to_img(gt, gw, gh), _to_img(bflat, bw, bh), _to_img(aflat, aw, ah)))

    tile_w, tile_h = cols[0][0].size
    hdr = 22
    grid = Image.new("RGB", (tile_w * 3, hdr + tile_h * len(cols)), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for j, label in enumerate(["Ground truth", "vanilla 3DGS (gsplat)", "AURA (15.5 dB)"]):
        draw.text((j * tile_w + 6, 5), label, fill=(240, 240, 240))
    for i, (g, b, a) in enumerate(cols):
        y = hdr + i * tile_h
        grid.paste(g, (0, y)); grid.paste(b, (tile_w, y)); grid.paste(a, (2 * tile_w, y))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"wrote {args.out}  ({grid.size[0]}x{grid.size[1]})", flush=True)


if __name__ == "__main__":
    main()
