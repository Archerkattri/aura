#!/usr/bin/env python3
"""Train and evaluate a REAL 3D Gaussian Splatting baseline on the SAME scene
and SAME eval split that AURA uses, so the AURA-vs-3DGS comparison becomes
executed-vs-executed instead of executed-vs-published-table.

Status: implemented; pending GPU run. This script has NOT been executed on the
GPU machine yet. It wraps the open-source ``gsplat`` rasterizer
(https://github.com/nerfstudio-project/gsplat). ``gsplat``, ``torch``, and the
COLMAP-seeded scene are imported/loaded lazily; the script fails loudly with an
actionable message if any is missing. It does NOT download datasets or weights.

Why this is an honest baseline
------------------------------
3DGS baseline numbers were historically copied from
Kerbl et al. 2023 Table 1 (the ``REFERENCE_3DGS`` dict). Those are *published*
numbers, NOT something this repo executed, and they were measured at a different
resolution / eval protocol than AURA's ``eval_psnr.py``. To compare fairly we
must run a real 3DGS optimisation on the identical inputs and evaluate it with
the identical metric code and identical eval frames.

This runner therefore:

1. Loads the SAME AURA capture manifest AURA trains/evals on. The manifest is
   produced by ``aura colmap-to-capture-manifest`` from a COLMAP sparse model;
   its frames carry the exact camera poses, intrinsics, and image paths.
2. Seeds Gaussians from the SAME COLMAP sparse points used to seed AURA's
   carriers (one Gaussian per SfM point — the 3DGS initialisation), located
   under ``--colmap`` (defaults to ``<root>/sparse/0``).
3. Optimises those Gaussians with ``gsplat`` against the manifest's training
   images.
4. Evaluates on the SAME eval frames AURA's ``eval_psnr.py`` selects
   (deterministic ``frames[::stride][:N]``) and computes PSNR / SSIM / LPIPS
   with the SAME functions imported from ``eval_psnr.py`` — at the SAME
   ``--scale``.
5. Prints a summary line in the SAME format as ``eval_psnr.py``
   ("Average PSNR: X dB  SSIM: Y  LPIPS: Z ...") and writes it to
   ``--out`` in a parseable summary format
   like an AURA eval and tabulate executed-vs-executed.

The camera conversion below is the exact inverse of the ray construction in
``eval_psnr.render_frame_torch`` / ``render_frame_cuda``, so the 3DGS render and
the AURA render see pixel-identical cameras.

Usage (on the GPU machine, after fetching the scene; NOT run here):

    python scripts/run_baseline_3dgs.py outputs/truck-pts129k-manifest.json \\
        --colmap data/tanks/truck/sparse/0 \\
        --iterations 30000 \\
        --frames 5 --scale 0.125 --device cuda \\
        --out outputs/eval_truck_baseline3dgs.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # for `import eval_psnr`

# Reuse AURA's exact metric + GT-loading code so the baseline is scored
# identically to AURA. These are pure-python and import without torch.
from eval_psnr import (  # noqa: E402
    load_jpg_as_rgb,
    lpips,
    mse,
    psnr_from_mse,
    resize_pixels,
    ssim,
)


def _require_gsplat():
    """Import torch + gsplat lazily; fail loudly with an actionable message."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "PyTorch is required for the 3DGS baseline. Install with "
            "`pip install -e \".[gpu]\"` on the GPU machine."
        ) from exc
    try:
        import gsplat  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "gsplat is required for the 3DGS baseline but is not installed. "
            "Install it on the GPU machine with `pip install gsplat` "
            "(see https://github.com/nerfstudio-project/gsplat). This script "
            "deliberately does not vendor a splatting implementation."
        ) from exc
    import gsplat
    import torch
    return torch, gsplat


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    if n == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return [c / n for c in v]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def manifest_frame_to_camera(frame: dict, scale: float):
    """Convert one AURA manifest frame to (world-to-camera 4x4, K 3x3, W, H).

    Delegates to the canonical conversion in ``aura.gsplat_renderer`` so the
    baseline uses the SAME (roll-preserving, view_rotation-aware) poses as AURA —
    otherwise the baseline would be hobbled by the look_at/up convention that
    drops camera roll, making the comparison unfair.
    """
    from aura.gsplat_renderer import manifest_frame_to_camera as _canonical
    return _canonical(frame, scale)


def load_colmap_seed(colmap_dir: Path):
    """Load SfM points (xyz + rgb) from the SAME COLMAP model AURA was seeded
    from, reusing AURA's own COLMAP reader so seeding is byte-identical."""
    from aura.ingest.colmap import load_colmap_model

    cameras, images, points, source = load_colmap_model(colmap_dir)
    if not points:
        raise SystemExit(
            f"COLMAP model at {colmap_dir} has no 3D points to seed Gaussians "
            f"({source})."
        )
    xyz = [list(p.xyz) for p in points]
    rgb = [[c / 255.0 for c in p.rgb] for p in points]
    return xyz, rgb


def train_baseline(
    manifest: dict,
    colmap_dir: Path,
    *,
    iterations: int,
    device: str,
    scale: float,
):
    """Train a real 3DGS model with gsplat on the manifest's training images.

    Returns a callable ``render(frame, scale) -> (W, H, flat_rgb_in_[0,1])``
    that rasterizes the optimised Gaussians through the EXACT manifest cameras,
    so eval is pixel-comparable to AURA. NOT executed here — pending GPU run.
    """
    torch, gsplat = _require_gsplat()
    from gsplat import rasterization

    root = Path(manifest.get("root", "."))
    frames = manifest["frames"]

    xyz, rgb = load_colmap_seed(colmap_dir)
    n = len(xyz)
    means = torch.tensor(xyz, dtype=torch.float32, device=device)
    colors = torch.tensor(rgb, dtype=torch.float32, device=device).clamp(0.0, 1.0)

    # Initial isotropic scales from nearest-neighbour spacing (3DGS init).
    with torch.no_grad():
        # Mean nearest-neighbour distance, chunked to bound memory.
        dists = torch.full((n,), 0.01, device=device)
        chunk = 4096
        for start in range(0, n, chunk):
            sl = means[start : start + chunk]
            d2 = torch.cdist(sl, means)  # [chunk, N]
            d2.scatter_(
                1,
                torch.arange(start, min(start + chunk, n), device=device).unsqueeze(1),
                float("inf"),
            )
            dists[start : start + chunk] = d2.min(dim=1).values.clamp(min=1e-6)
    log_scales = torch.log(dists).unsqueeze(1).repeat(1, 3).clone()
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0
    logit_opac = torch.logit(torch.full((n,), 0.1, device=device))

    means = means.requires_grad_(True)
    colors = colors.requires_grad_(True)
    log_scales = log_scales.requires_grad_(True)
    quats = quats.requires_grad_(True)
    logit_opac = logit_opac.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 1.6e-4},
            {"params": [log_scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [logit_opac], "lr": 5e-2},
            {"params": [colors], "lr": 2.5e-3},
        ],
        eps=1e-15,
    )

    def render_gaussians(frame, frame_scale):
        view, k, w, h = manifest_frame_to_camera(frame, frame_scale)
        viewmat = torch.tensor(view, dtype=torch.float32, device=device)
        kmat = torch.tensor(k, dtype=torch.float32, device=device)
        rendered, _alpha, _info = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(log_scales),
            opacities=torch.sigmoid(logit_opac),
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=kmat.unsqueeze(0),
            width=w,
            height=h,
        )
        return rendered[0], w, h  # [H, W, 3]

    # ---- TRAINING LOOP (pending GPU run) ----
    for it in range(iterations):
        frame = frames[it % len(frames)]
        img_path = root / frame["image_path"]
        if not img_path.exists():
            continue
        gt_w, gt_h, gt_pixels = load_jpg_as_rgb(str(img_path))
        rendered, w, h = render_gaussians(frame, scale)
        if (w, h) != (gt_w, gt_h):
            gt_pixels = resize_pixels(gt_pixels, gt_w, gt_h, w, h)
        gt = torch.tensor(gt_pixels, dtype=torch.float32, device=device).reshape(h, w, 3)
        loss = torch.abs(rendered - gt).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (it + 1) % 100 == 0 or it == 0:
            print(f"  [baseline] iter {it + 1}/{iterations}  L1={loss.item():.4f}",
                  file=sys.stderr, flush=True)

    def render(frame, frame_scale):
        with torch.no_grad():
            rendered, w, h = render_gaussians(frame, frame_scale)
        flat = rendered.clamp(0.0, 1.0).reshape(-1).cpu().tolist()
        return w, h, flat

    return render


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train + evaluate a REAL 3DGS baseline on AURA's scene/split"
    )
    ap.add_argument("manifest", type=Path, help="AURA capture manifest AURA trains/evals on")
    ap.add_argument(
        "--colmap",
        type=Path,
        default=None,
        help="COLMAP sparse model dir to seed Gaussians from "
        "(default: <manifest root>/sparse/0)",
    )
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--frames", type=int, default=5, help="Number of eval frames (same selection as eval_psnr.py)")
    ap.add_argument("--scale", type=float, default=0.125)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the eval summary here so downstream tooling can parse it "
        "(e.g. outputs/eval_<scene>_baseline3dgs.txt)",
    )
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    root = Path(manifest.get("root", str(args.manifest.parent)))
    colmap_dir = args.colmap or (root / "sparse" / "0")
    if not colmap_dir.exists():
        raise SystemExit(
            f"COLMAP model not found at {colmap_dir}. Pass --colmap explicitly. "
            f"Fetch the scene first with scripts/fetch_scene.sh."
        )

    print(f"Training 3DGS baseline ({args.iterations} iters) seeded from {colmap_dir} ...")
    render = train_baseline(
        manifest,
        colmap_dir,
        iterations=args.iterations,
        device=args.device,
        scale=args.scale,
    )

    # ---- EVAL on the SAME frames eval_psnr.py selects ----
    frames = manifest["frames"]
    stride = max(1, len(frames) // args.frames)
    eval_frames = frames[::stride][: args.frames]
    print(f"Evaluating 3DGS baseline on {len(eval_frames)} frames (same split as AURA) ...")

    psnr_values, ssim_values, lpips_values = [], [], []
    for i, frame in enumerate(eval_frames):
        img_path = root / frame["image_path"]
        if not img_path.exists():
            print(f"  Skipping {img_path} (not found)")
            continue
        gt_w, gt_h, gt_pixels = load_jpg_as_rgb(str(img_path))
        w, h, pred = render(frame, args.scale)
        if (w, h) != (gt_w, gt_h):
            gt_pixels = resize_pixels(gt_pixels, gt_w, gt_h, w, h)
            gt_w, gt_h = w, h
        mse_val = mse(pred, gt_pixels)
        psnr_val = psnr_from_mse(mse_val)
        ssim_val = ssim(pred, gt_pixels, w, h)
        lpips_val = lpips(pred, gt_pixels, w, h)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        if lpips_val is not None:
            lpips_values.append(lpips_val)
        lpips_str = f"  LPIPS={lpips_val:.4f}" if lpips_val is not None else ""
        print(f"  [{i+1}/{len(eval_frames)}] {img_path.name}: "
              f"PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}{lpips_str}")

    if not psnr_values:
        print("No frames evaluated.")
        return

    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    lpips_summary = ""
    if lpips_values:
        avg_lpips = sum(lpips_values) / len(lpips_values)
        lpips_summary = f"  LPIPS={avg_lpips:.4f}"
    res_note = f" at {args.scale:.2f}x scale" if args.scale != 1.0 else " (full resolution)"
    # SAME format as eval_psnr.py for identical parsing.
    summary = (
        f"\nAverage PSNR: {avg_psnr:.2f} dB  SSIM: {avg_ssim:.4f}{lpips_summary}{res_note}\n"
        f"(real gsplat 3DGS baseline, {args.iterations} iters, "
        f"executed on AURA's scene + eval split)\n"
    )
    print(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary, encoding="utf-8")
        print(f"Wrote baseline eval summary to {args.out}")


if __name__ == "__main__":
    main()
