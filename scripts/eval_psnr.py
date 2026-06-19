#!/usr/bin/env python3
"""Evaluate PSNR of a trained .aura package against T&T truck ground truth.

Usage:
  python scripts/eval_psnr.py outputs/truck-pts129k-overnight.aura \
      outputs/truck-pts129k-manifest.json \
      --frames 10 --device cuda
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_jpg_as_rgb(path: str) -> tuple[int, int, list[float]]:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    pixels = list(img.getdata())
    flat = [v / 255.0 for rgb in pixels for v in rgb]
    return w, h, flat


def mse(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def psnr_from_mse(mse_val: float) -> float:
    if mse_val <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_val)


def render_frame_torch(scene, frame_data: dict, device: str = "cuda") -> tuple[int, int, list[float]]:
    from aura.torch_renderer import torch_scene_tensors, torch_render_rays

    intr = frame_data["intrinsics"]
    W, H = int(intr["width"]), int(intr["height"])
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])

    origin = frame_data["camera_origin"]
    look_at = frame_data["look_at"]
    up = frame_data.get("up", [0.0, -1.0, 0.0])

    def norm(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def cross(a, b):
        return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

    fwd = norm([look_at[i] - origin[i] for i in range(3)])
    right = norm(cross(fwd, up))
    up_actual = cross(right, fwd)

    dirs = []
    origs = []
    for y_idx in range(H):
        dy = (y_idx - cy) / fy
        for x_idx in range(W):
            dx = (x_idx - cx) / fx
            d = [dx * right[i] + dy * up_actual[i] + fwd[i] for i in range(3)]
            dirs.append(norm(d))
            origs.append(origin)

    total_rays = W * H
    batch_size = 4096
    all_colors: list[float] = []

    st = torch_scene_tensors(scene, device=device)

    for start in range(0, total_rays, batch_size):
        end = min(start + batch_size, total_rays)
        result = torch_render_rays(
            scene,
            ray_origins=origs[start:end],
            ray_directions=dirs[start:end],
            device=device,
            scene_tensors=st,
        )
        for sample in result.samples:
            all_colors.extend(list(sample.color))

    return W, H, all_colors


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR against ground truth")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from aura.package import load_package
    print(f"Loading {args.package_dir}...")
    pkg = load_package(args.package_dir)
    scene = pkg.scene
    print(f"Scene: {len(scene.elements)} elements")

    with open(args.manifest) as f:
        manifest = json.load(f)
    frames = manifest["frames"]
    root = manifest.get("root", str(args.manifest.parent))

    n = len(frames)
    stride = max(1, n // args.frames)
    eval_frames = frames[::stride][:args.frames]
    print(f"Evaluating {len(eval_frames)} frames")

    psnr_values = []
    for i, frame in enumerate(eval_frames):
        img_path = Path(root) / frame["image_path"]
        if not img_path.exists():
            print(f"  Skipping {img_path} (not found)")
            continue

        print(f"  [{i+1}/{len(eval_frames)}] {img_path.name}...", flush=True)
        gt_W, gt_H, gt_pixels = load_jpg_as_rgb(str(img_path))
        render_W, render_H, render_pixels = render_frame_torch(scene, frame, device=args.device)

        if (render_W, render_H) != (gt_W, gt_H):
            print(f"    Size mismatch: {render_W}x{render_H} vs {gt_W}x{gt_H}")
            continue

        mse_val = mse(render_pixels, gt_pixels)
        psnr_val = psnr_from_mse(mse_val)
        psnr_values.append(psnr_val)
        print(f"    PSNR={psnr_val:.2f} dB  MSE={mse_val:.4f}")

    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        print(f"\nAverage PSNR: {avg_psnr:.2f} dB  (3DGS reference: ~25.19 dB)")
    else:
        print("No frames evaluated.")


if __name__ == "__main__":
    main()
