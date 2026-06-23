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


def lpips(pred: list[float], gt: list[float], width: int, height: int) -> float | None:
    """Compute LPIPS using the lpips package if available; return None otherwise.

    pred, gt: flat list of floats in [0,1], length = width * height * 3
    Returns LPIPS distance (lower = more similar), or None if lpips is not installed.
    """
    try:
        import torch
        import lpips as lpips_lib
    except ImportError:
        return None

    fn = lpips_lib.LPIPS(net="alex", verbose=False)
    fn.eval()

    # Build CHW tensors in [-1, 1] (lpips convention)
    pred_t = torch.tensor(pred, dtype=torch.float32).reshape(height, width, 3).permute(2, 0, 1)
    gt_t   = torch.tensor(gt,   dtype=torch.float32).reshape(height, width, 3).permute(2, 0, 1)
    pred_t = pred_t * 2.0 - 1.0
    gt_t   = gt_t   * 2.0 - 1.0

    with torch.no_grad():
        dist = fn(pred_t.unsqueeze(0), gt_t.unsqueeze(0))
    return float(dist.squeeze())


def mse(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def psnr_from_mse(mse_val: float) -> float:
    if mse_val <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_val)


def ssim(pred: list[float], gt: list[float], width: int, height: int) -> float:
    """Compute SSIM between two flat RGB images using an 11x11 Gaussian window.

    Uses a vectorised torch (GPU if available) implementation when torch is
    installed — the pure-Python path below is O(W*H*121) and is hundreds of
    times slower at full resolution. Both paths use the same luminance + 11x11
    Gaussian-window SSIM definition, so numbers stay comparable.
    """
    fast = _ssim_torch(pred, gt, width, height)
    if fast is not None:
        return fast

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


def _ssim_torch(pred: list[float], gt: list[float], width: int, height: int) -> float | None:
    """Vectorised luminance SSIM (11x11 Gaussian window) on GPU/CPU via torch.

    Returns None if torch is unavailable so the caller falls back to the pure
    Python implementation. Matches the reference's K1=0.01, K2=0.03, L=1.0 and
    only counts fully-valid windows (conv2d 'valid' region), like the loop below.
    """
    try:
        import torch
    except Exception:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = width * height
    p = torch.tensor(pred, dtype=torch.float32, device=device).reshape(n, 3)
    g = torch.tensor(gt, dtype=torch.float32, device=device).reshape(n, 3)
    w_lum = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32, device=device)
    lp = (p * w_lum).sum(1).reshape(1, 1, height, width)
    lg = (g * w_lum).sum(1).reshape(1, 1, height, width)

    half, sigma = 5, 1.5
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g1 = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    k2d = (g1[:, None] * g1[None, :])
    k2d = (k2d / k2d.sum()).reshape(1, 1, 2 * half + 1, 2 * half + 1)

    def filt(t):
        return torch.nn.functional.conv2d(t, k2d)  # 'valid' region only

    mu_x, mu_y = filt(lp), filt(lg)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = filt(lp * lp) - mu_x2
    sig_y = filt(lg * lg) - mu_y2
    sig_xy = filt(lp * lg) - mu_xy
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
    return float((num / den).mean())


def render_frame_cuda(
    scene,
    frame_data: dict,
    device: str = "cuda",
    scale: float = 1.0,
    max_hits: int = 32,
) -> tuple[int, int, list[float]]:
    """Render one frame using the compiled CUDA extension (no BVH, fastest path)."""
    from aura.cuda_renderer import cuda_render_rays
    import numpy as np

    intr = frame_data["intrinsics"]
    full_W, full_H = int(intr["width"]), int(intr["height"])
    W = max(1, int(full_W * scale))
    H = max(1, int(full_H * scale))
    fx = float(intr["fx"]) * scale
    fy = float(intr["fy"]) * scale
    cx = float(intr["cx"]) * scale
    cy = float(intr["cy"]) * scale

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

    origs_np = np.array(origs, dtype=np.float32)
    dirs_np = np.array(dirs, dtype=np.float32)
    batch = cuda_render_rays(
        scene, origs_np, dirs_np, device=device, use_bvh=False,
        max_hits=max_hits, require_cuda=True,
    )
    # Honesty guard: never report CPU/torch fallback numbers as "cuda". If the
    # compiled extension did not actually run on the GPU, fail loudly so the
    # caller switches to --renderer torch (and gets correctly-labelled numbers)
    # rather than silently mislabelling fallback results.
    if getattr(batch, "backend", None) != "cuda" or not getattr(batch, "production_ready", False):
        raise RuntimeError(
            "cuda renderer did not run on the GPU "
            f"(backend={getattr(batch, 'backend', None)!r}, "
            f"reason={getattr(batch, 'reason', None)!r}); "
            "re-run with --renderer torch for honest CPU numbers."
        )
    all_colors = [v for rgb in batch.color for v in rgb]
    return W, H, all_colors


def render_frame_torch(
    scene,
    frame_data: dict,
    device: str = "cuda",
    scale: float = 1.0,
    ray_batch: int = 128,
) -> tuple[int, int, list[float]]:
    from aura.torch_renderer import torch_scene_tensors, torch_render_rays

    intr = frame_data["intrinsics"]
    full_W, full_H = int(intr["width"]), int(intr["height"])
    W = max(1, int(full_W * scale))
    H = max(1, int(full_H * scale))
    fx = float(intr["fx"]) * scale
    fy = float(intr["fy"]) * scale
    cx = float(intr["cx"]) * scale
    cy = float(intr["cy"]) * scale

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
    all_colors: list[float] = []

    st = torch_scene_tensors(scene, device=device)

    for start in range(0, total_rays, ray_batch):
        end = min(start + ray_batch, total_rays)
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


def resize_pixels(pixels: list[float], src_w: int, src_h: int, dst_w: int, dst_h: int) -> list[float]:
    """Nearest-neighbour downsample a flat RGB pixel list."""
    out = []
    for y in range(dst_h):
        sy = y * src_h // dst_h
        for x in range(dst_w):
            sx = x * src_w // dst_w
            idx = (sy * src_w + sx) * 3
            out.extend(pixels[idx : idx + 3])
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR against ground truth")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Render at this fraction of full resolution (e.g. 0.25 for 1/4). "
                             "GT is downsampled to match. Avoids OOM on large scenes.")
    parser.add_argument("--ray-batch", type=int, default=128,
                        help="Ray batch size passed to the renderer (default 128). "
                             "Increase for faster eval if GPU has headroom.")
    parser.add_argument("--renderer", choices=["torch", "cuda", "gsplat"], default="torch",
                        help="Renderer backend: 'gsplat' (tiled rasterizer, AURA's "
                        "high-fidelity primary-view path), 'torch' (default, batched) or 'cuda' "
                             "(compiled CUDA extension, faster for large scenes).")
    args = parser.parse_args()

    from aura.package import load_package
    print(f"Loading {args.package_dir}...")
    pkg = load_package(args.package_dir)
    scene = pkg.scene
    print(f"Scene: {len(scene.elements)} elements")
    if args.scale != 1.0:
        print(f"Scale: {args.scale:.2f}x  (ray-batch: {args.ray_batch}, renderer: {args.renderer})")

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
    lpips_values = []
    for i, frame in enumerate(eval_frames):
        img_path = Path(root) / frame["image_path"]
        if not img_path.exists():
            print(f"  Skipping {img_path} (not found)")
            continue

        print(f"  [{i+1}/{len(eval_frames)}] {img_path.name}...", flush=True)
        gt_W, gt_H, gt_pixels = load_jpg_as_rgb(str(img_path))
        if args.renderer == "gsplat":
            # Primary-view path: the tiled differentiable rasterizer (full
            # front-to-back alpha compositing of ALL Gaussians) — AURA's
            # high-fidelity render path, matching the training renderer. The
            # "cuda"/"torch" 3D ray renderers below are the secondary-ray/query
            # path and cap hits per ray, so they score lower on primary view.
            from aura.gsplat_renderer import render_scene_gsplat
            render_W, render_H, render_pixels = render_scene_gsplat(
                scene, frame, scale=args.scale, device=args.device
            )
        elif args.renderer == "cuda":
            render_W, render_H, render_pixels = render_frame_cuda(
                scene, frame, device=args.device, scale=args.scale, max_hits=32
            )
        else:
            render_W, render_H, render_pixels = render_frame_torch(
                scene, frame, device=args.device, scale=args.scale, ray_batch=args.ray_batch
            )

        # Downsample GT to rendered resolution when scale < 1.
        if (render_W, render_H) != (gt_W, gt_H):
            gt_pixels = resize_pixels(gt_pixels, gt_W, gt_H, render_W, render_H)
            gt_W, gt_H = render_W, render_H

        if render_W * render_H > 256 * 256:
            import warnings
            warnings.warn(f"Pure-python SSIM is slow at {render_W}x{render_H}. Consider --scale 0.25.")

        mse_val = mse(render_pixels, gt_pixels)
        psnr_val = psnr_from_mse(mse_val)
        ssim_val = ssim(render_pixels, gt_pixels, render_W, render_H)
        lpips_val = lpips(render_pixels, gt_pixels, render_W, render_H)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        if lpips_val is not None:
            lpips_values.append(lpips_val)
        lpips_str = f"  LPIPS={lpips_val:.4f}" if lpips_val is not None else ""
        print(f"    PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}{lpips_str}  MSE={mse_val:.4f}")

    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
        lpips_str = ""
        if lpips_values:
            avg_lpips = sum(lpips_values) / len(lpips_values)
            lpips_str = f"  LPIPS={avg_lpips:.4f}"
        res_note = f" at {args.scale:.2f}x scale" if args.scale != 1.0 else " (full resolution)"
        print(f"\nAverage PSNR: {avg_psnr:.2f} dB  SSIM: {avg_ssim:.4f}{lpips_str}{res_note}")
        print("(3DGS reference: PSNR ~25.19 dB, SSIM ~0.879, LPIPS ~0.148 at full resolution)")
    else:
        print("No frames evaluated.")


if __name__ == "__main__":
    main()
