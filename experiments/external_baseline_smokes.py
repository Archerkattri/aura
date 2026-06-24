#!/usr/bin/env python3
"""Run local same-split smoke baselines for the publication baseline gate.

These are execution/protocol smokes, not final paper-quality external numbers.
They prove that AURA can score additional baseline families on the same scene,
camera split, image scale, and metric code while official 2DGS/ray-traced-GS
baselines remain open.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))

from eval_psnr import load_jpg_as_rgb, lpips, mse, psnr_from_mse, resize_pixels, ssim
from make_lineage_figure import colmap_points_render
import mini_nerf


REQUIRED_BASELINES = ("colmap", "nerf", "3dgs", "2dgs", "ray_traced_gs")


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def merge_smoke_baselines(payload: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = dict(payload)
    existing = dict(payload.get("baselines", {}))
    existing.update(baselines)
    merged["baselines"] = existing
    merged["missingBaselines"] = [name for name in REQUIRED_BASELINES if name not in existing]
    return merged


def evaluate_render(manifest: dict[str, Any], render, *, frames: int, scale: float, device: str) -> dict[str, Any]:
    root = Path(manifest["root"])
    selected = [frame for frame in manifest["frames"][:frames] if (root / frame["image_path"]).exists()]
    psnrs: list[float] = []
    ssims: list[float] = []
    lpips_values: list[float] = []

    for frame in selected:
        gt_w, gt_h, gt = load_jpg_as_rgb(str(root / frame["image_path"]))
        width, height, pred = render(frame, scale)
        if (width, height) != (gt_w, gt_h):
            gt = resize_pixels(gt, gt_w, gt_h, width, height)
        mse_value = mse(pred, gt)
        psnrs.append(psnr_from_mse(mse_value))
        ssims.append(ssim(pred, gt, width, height))
        lpips_value = lpips(pred, gt, width, height, device=device)
        if lpips_value is not None:
            lpips_values.append(lpips_value)

    if not psnrs:
        raise RuntimeError("no frames were evaluated")

    result: dict[str, Any] = {
        "frames": len(psnrs),
        "scale": scale,
        "psnr": round(sum(psnrs) / len(psnrs), 4),
        "ssim": round(sum(ssims) / len(ssims), 6),
    }
    if lpips_values:
        result["lpips"] = round(sum(lpips_values) / len(lpips_values), 6)
    return result


def run_smokes(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(args.manifest.read_text())

    colmap_metrics = evaluate_render(
        manifest,
        lambda frame, scale: colmap_points_render(args.colmap, frame, scale, device=args.device),
        frames=args.frames,
        scale=args.scale,
        device=args.device,
    )

    nerf_render = mini_nerf.train_nerf(
        args.colmap,
        str(Path(manifest["root"]) / "images"),
        scale=args.scale,
        iters=args.nerf_iters,
        device=args.device,
        log=lambda line: print(line, flush=True),
    )
    nerf_metrics = evaluate_render(
        manifest,
        nerf_render,
        frames=args.frames,
        scale=args.scale,
        device=args.device,
    )

    return {
        "format": "AURA_EXTERNAL_BASELINE_SMOKES",
        "scene": args.scene,
        "manifest": repo_path(args.manifest),
        "colmap": colmap_metrics,
        "nerf": {**nerf_metrics, "iterations": args.nerf_iters},
        "device": args.device,
        "seed": args.seed,
        "notes": (
            "Same-split CUDA smoke metrics. COLMAP is sparse SfM point rendering; "
            "NeRF is the repo's compact frequency-encoded MLP smoke, not an official "
            "nerfstudio/SOTA baseline."
        ),
    }


def baseline_entries(smoke: dict[str, Any], artifact: Path) -> dict[str, dict[str, Any]]:
    return {
        "colmap": {
            "label": "same-split COLMAP sparse SfM point smoke",
            "scene": smoke["scene"],
            "sourceArtifact": str(artifact),
            "device": smoke["device"],
            **smoke["colmap"],
            "notes": "Sparse COLMAP point rendering through AURA's held-out camera split; smoke/protocol evidence, not a dense MVS quality baseline.",
        },
        "nerf": {
            "label": "same-split compact NeRF smoke",
            "scene": smoke["scene"],
            "sourceArtifact": str(artifact),
            "device": smoke["device"],
            **smoke["nerf"],
            "notes": "Compact local NeRF MLP trained on CUDA; smoke/protocol evidence, not an official nerfstudio/SOTA quality baseline.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/truck-pts129k-manifest.json")
    parser.add_argument("--colmap", type=Path, default=ROOT / "data/tanks/truck/sparse/0")
    parser.add_argument("--external-json", type=Path, default=ROOT / "experiments/results/external_baselines_2026-06-24.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/external_baseline_smokes_2026-06-24.json")
    parser.add_argument("--scene", default="truck")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--nerf-iters", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    smoke = run_smokes(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(smoke, indent=2) + "\n")

    external = json.loads(args.external_json.read_text()) if args.external_json.exists() else {"baselines": {}}
    merged = merge_smoke_baselines(external, baseline_entries(smoke, Path(repo_path(args.out))))
    args.external_json.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(smoke, indent=2))
    print(f"updated {args.external_json}")


if __name__ == "__main__":
    main()
