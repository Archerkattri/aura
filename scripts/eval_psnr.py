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


def ssim(pred: list[float], gt: list[float], width: int, height: int) -> float:
    """Compute SSIM between two flat RGB images using an 11x11 Gaussian window.

    pred, gt: flat list of floats in [0,1], length = width * height * 3
    Returns SSIM in [-1, 1] (1.0 = identical).
    """
    import math

    K1, K2, L = 0.01, 0.03, 1.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    # Convert to luminance (grayscale)
    n_pixels = width * height
    lum_pred = [0.2126 * pred[i*3] + 0.7152 * pred[i*3+1] + 0.0722 * pred[i*3+2] for i in range(n_pixels)]
    lum_gt   = [0.2126 * gt[i*3]   + 0.7152 * gt[i*3+1]   + 0.0722 * gt[i*3+2]   for i in range(n_pixels)]

    # Build 11x11 Gaussian kernel weights
    sigma = 1.5
    half = 5
    kernel = []
    for dy in range(-half, half+1):
        for dx in range(-half, half+1):
            kernel.append(math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma)))
    s = sum(kernel)
    kernel = [k / s for k in kernel]

    # Compute windowed statistics over valid pixels
    ssim_vals = []
    for cy in range(half, height - half):
        for cx in range(half, width - half):
            mu_x = mu_y = 0.0
            for ki, (dy, dx) in enumerate((
                (dy, dx) for dy in range(-half, half+1) for dx in range(-half, half+1)
            )):
                idx = (cy + dy) * width + (cx + dx)
                w = kernel[ki]
                mu_x += w * lum_pred[idx]
                mu_y += w * lum_gt[idx]

            sig_x = sig_y = sig_xy = 0.0
            for ki, (dy, dx) in enumerate((
                (dy, dx) for dy in range(-half, half+1) for dx in range(-half, half+1)
            )):
                idx = (cy + dy) * width + (cx + dx)
                w = kernel[ki]
                sig_x  += w * (lum_pred[idx] - mu_x) ** 2
                sig_y  += w * (lum_gt[idx]   - mu_y) ** 2
                sig_xy += w * (lum_pred[idx] - mu_x) * (lum_gt[idx] - mu_y)

            num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
            den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
            ssim_vals.append(num / den if den != 0 else 1.0)

    return sum(ssim_vals) / len(ssim_vals) if ssim_vals else 1.0


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
        for rgb in result.predicted_color:
            all_colors.extend(list(rgb))

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
    ssim_values = []
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

        if render_W * render_H > 256 * 256:
            import warnings
            warnings.warn(f"Pure-python SSIM is slow at {render_W}x{render_H}. Consider downscaling first.")

        mse_val = mse(render_pixels, gt_pixels)
        psnr_val = psnr_from_mse(mse_val)
        ssim_val = ssim(render_pixels, gt_pixels, render_W, render_H)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        print(f"    PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}  MSE={mse_val:.4f}")

    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
        print(f"\nAverage PSNR: {avg_psnr:.2f} dB  Average SSIM: {avg_ssim:.4f}  (3DGS reference: PSNR ~25.19 dB, SSIM ~0.857)")
    else:
        print("No frames evaluated.")


if __name__ == "__main__":
    main()
