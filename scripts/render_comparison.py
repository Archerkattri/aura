#!/usr/bin/env python3
"""Render a trained .aura checkpoint and save a GT-vs-AURA comparison image.

Produces a side-by-side PNG (ground-truth | AURA render) for one or more
frames, with the per-frame PSNR labelled, saved to a committed location
(default docs/) for embedding in the README.

Usage:
  python scripts/render_comparison.py outputs/truck-3k-run6.aura \
      outputs/truck-pts129k-manifest.json \
      --out docs/aura_truck_comparison.png \
      --frames 3 --scale 0.25 --renderer cuda --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_psnr import (  # noqa: E402  (reuse the eval renderers + metrics)
    load_jpg_as_rgb,
    mse,
    psnr_from_mse,
    render_frame_cuda,
    render_frame_torch,
    resize_pixels,
)


def _flat_to_image(pixels, w, h):
    from PIL import Image
    buf = bytes(max(0, min(255, int(round(v * 255.0)))) for v in pixels)
    return Image.frombytes("RGB", (w, h), buf)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render GT-vs-AURA comparison image")
    ap.add_argument("package_dir", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/aura_truck_comparison.png"))
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--renderer", choices=["torch", "cuda"], default="cuda")
    ap.add_argument("--ray-batch", type=int, default=128)
    args = ap.parse_args()

    from PIL import Image, ImageDraw
    from aura.package import load_package

    pkg = load_package(args.package_dir)
    scene = pkg.scene
    with open(args.manifest) as f:
        manifest = json.load(f)
    frames = manifest["frames"]
    root = manifest.get("root", str(args.manifest.parent))

    n = len(frames)
    stride = max(1, n // args.frames)
    eval_frames = frames[::stride][: args.frames]
    print(f"Rendering {len(eval_frames)} frames with the {args.renderer} renderer ...")

    rows = []
    pad = 6
    for i, frame in enumerate(eval_frames):
        img_path = Path(root) / frame["image_path"]
        gt_W, gt_H, gt_pixels = load_jpg_as_rgb(str(img_path))
        if args.renderer == "cuda":
            rW, rH, r_pixels = render_frame_cuda(
                scene, frame, device=args.device, scale=args.scale, max_hits=32
            )
        else:
            rW, rH, r_pixels = render_frame_torch(
                scene, frame, device=args.device, scale=args.scale, ray_batch=args.ray_batch
            )
        if (rW, rH) != (gt_W, gt_H):
            gt_pixels = resize_pixels(gt_pixels, gt_W, gt_H, rW, rH)
            gt_W, gt_H = rW, rH
        psnr_val = psnr_from_mse(mse(r_pixels, gt_pixels))
        print(f"  [{i+1}/{len(eval_frames)}] {img_path.name}: PSNR={psnr_val:.2f} dB")

        gt_img = _flat_to_image(gt_pixels, gt_W, gt_H)
        r_img = _flat_to_image(r_pixels, rW, rH)
        row = Image.new("RGB", (gt_W + rW + pad, max(gt_H, rH) + 16), (20, 20, 20))
        row.paste(gt_img, (0, 16))
        row.paste(r_img, (gt_W + pad, 16))
        draw = ImageDraw.Draw(row)
        draw.text((2, 2), f"GT", fill=(230, 230, 230))
        draw.text((gt_W + pad + 2, 2), f"AURA  PSNR={psnr_val:.1f} dB", fill=(230, 230, 230))
        rows.append(row)

    total_h = sum(r.height for r in rows) + pad * (len(rows) - 1)
    total_w = max(r.width for r in rows)
    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + pad

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\nSaved comparison image to {args.out}")


if __name__ == "__main__":
    main()
