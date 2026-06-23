#!/usr/bin/env python3
"""PRISM ablation: train the truck scene under a set of carrier/render configs
(matched iterations + scale + seed) and evaluate PSNR/SSIM, so the post-3DGS
choices (carrier type, mixing, volumetric alpha, densification) are measured
head to head. Writes one JSON per config; aggregate with prism_results.py.

Usage:
  python experiments/prism_ablation.py --configs gaussian,beta --gpu 0 \
      --iterations 1500 --scale 0.25 --out experiments/results
"""
import argparse, json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CONFIGS = {
    "gaussian":            dict(carrier="gaussian"),
    "beta":                dict(carrier="beta"),
    "gabor":               dict(carrier="gabor"),
    "auto":                dict(carrier="auto"),
    "gaussian_volumetric": dict(carrier="gaussian", volumetric=True),
    "gaussian_densify":    dict(carrier="gaussian", densify=True),
    "auto_volumetric":     dict(carrier="auto", volumetric=True),
}


def evaluate(scene, manifest, scale, nframes=5):
    import torch  # noqa
    from aura.prism import render_scene_prism
    from eval_psnr import load_jpg_as_rgb, resize_pixels, ssim as ssim_fn
    root = Path(manifest.get("root", "."))
    frames = manifest["frames"]
    stride = max(1, len(frames) // nframes)
    sel = frames[::stride][:nframes]
    ps, ss = [], []
    for fr in sel:
        ip = root / fr["image_path"]
        if not ip.exists():
            continue
        gw, gh, gt = load_jpg_as_rgb(str(ip))
        w, h, flat = render_scene_prism(scene, fr, scale, device="cuda")
        if (gw, gh) != (w, h):
            gt = resize_pixels(gt, gw, gh, w, h)
        mse = sum((a - b) ** 2 for a, b in zip(flat, gt)) / len(flat)
        ps.append(10 * math.log10(1.0 / mse) if mse > 0 else 99.0)
        ss.append(ssim_fn(flat, gt, w, h))
    return (sum(ps) / len(ps) if ps else 0.0), (sum(ss) / len(ss) if ss else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--manifest", default="outputs/truck-pts129k-manifest.json")
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    import torch
    from aura.cli import load_capture_manifest
    from aura.gsplat_renderer import seed_gaussian_params_from_regions
    from aura.prism import train_carriers_prism, PrismTrainConfig

    manifest_obj = load_capture_manifest(args.manifest, validate=False)
    raw = json.loads(Path(args.manifest).read_text())
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    for name in args.configs.split(","):
        name = name.strip()
        if name not in CONFIGS:
            print(f"skip unknown config {name}", flush=True); continue
        opts = CONFIGS[name]
        seed, ctx = seed_gaussian_params_from_regions(manifest_obj.regions, device="cuda")
        cfg = PrismTrainConfig(iterations=args.iterations, scale=args.scale,
                               log_every=max(1, args.iterations // 5),
                               log=lambda s: print(f"[{name}] {s}", flush=True),
                               **{k: v for k, v in opts.items() if k != "carrier"})
        torch.cuda.synchronize(); t0 = time.time()
        scene, hist = train_carriers_prism(seed, ctx, raw, config=cfg, device="cuda",
                                           carrier=opts["carrier"])
        torch.cuda.synchronize(); train_s = time.time() - t0
        psnr, ssim = evaluate(scene, raw, args.scale)
        result = {
            "config": name, "carrier": opts["carrier"],
            "volumetric": bool(opts.get("volumetric")), "densify": bool(opts.get("densify")),
            "iterations": args.iterations, "scale": args.scale,
            "psnr": round(psnr, 3), "ssim": round(ssim, 4),
            "train_seconds": round(train_s, 1),
            "final_carriers": hist.get("final_gaussian_count"),
            "footprint_counts": hist.get("footprint_counts"),
        }
        (outdir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[{name}] PSNR={psnr:.2f} SSIM={ssim:.4f} N={result['final_carriers']} "
              f"train={train_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
