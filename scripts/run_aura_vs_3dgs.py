#!/usr/bin/env python3
"""AURA vs 3DGS benchmark comparison table.

Reads AURA eval outputs from outputs/ and prints a comparison table against
published 3DGS (Kerbl et al. 2023) reference numbers.

Usage:
    python scripts/run_aura_vs_3dgs.py [--eval-dir outputs]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# 3DGS reference numbers from Kerbl et al. 2023 (Table 1)
# Scene: PSNR / SSIM / LPIPS
REFERENCE_3DGS = {
    "truck":    {"psnr": 25.19, "ssim": 0.879, "lpips": 0.148},
    "train":    {"psnr": 21.10, "ssim": 0.802, "lpips": 0.218},
    "playroom": {"psnr": 29.61, "ssim": 0.900, "lpips": 0.252},
    "drjohnson": {"psnr": 29.14, "ssim": 0.903, "lpips": 0.243},
    "bicycle":  {"psnr": 25.25, "ssim": 0.771, "lpips": 0.205},
    "garden":   {"psnr": 27.41, "ssim": 0.868, "lpips": 0.103},
    "kitchen":  {"psnr": 30.32, "ssim": 0.922, "lpips": 0.129},
}


def parse_eval_txt(path: Path) -> dict:
    """Parse PSNR/SSIM from an eval output text file."""
    text = path.read_text(encoding="utf-8")
    result = {}
    for line in text.splitlines():
        if "Average PSNR:" in line:
            try:
                result["psnr"] = float(line.split("Average PSNR:")[1].split("dB")[0].strip())
            except Exception:
                pass
        if "Average SSIM:" in line:
            try:
                result["ssim"] = float(line.split("Average SSIM:")[1].split()[0].strip())
            except Exception:
                pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA vs 3DGS comparison table")
    parser.add_argument("--eval-dir", default="outputs", type=Path)
    args = parser.parse_args()

    # Find eval txt files
    aura_results = {}
    for txt_path in sorted(args.eval_dir.glob("eval_*.txt")):
        scene = txt_path.stem.replace("eval_", "").split("_")[0]
        metrics = parse_eval_txt(txt_path)
        if metrics:
            aura_results[scene] = metrics

    print("=" * 70)
    print("AURA vs 3DGS Benchmark Comparison")
    print("=" * 70)
    print(f"{'Scene':15s}  {'AURA PSNR':>10}  {'3DGS PSNR':>10}  {'AURA SSIM':>10}  {'3DGS SSIM':>10}")
    print("-" * 70)

    all_scenes = sorted(set(list(aura_results.keys()) + list(REFERENCE_3DGS.keys())))
    for scene in all_scenes:
        aura = aura_results.get(scene, {})
        ref = REFERENCE_3DGS.get(scene, {})
        aura_psnr = f"{aura['psnr']:.2f}" if "psnr" in aura else "N/A"
        ref_psnr = f"{ref['psnr']:.2f}" if "psnr" in ref else "N/A"
        aura_ssim = f"{aura['ssim']:.3f}" if "ssim" in aura else "N/A"
        ref_ssim = f"{ref['ssim']:.3f}" if "ssim" in ref else "N/A"
        print(f"  {scene:13s}  {aura_psnr:>10}  {ref_psnr:>10}  {aura_ssim:>10}  {ref_ssim:>10}")

    print("-" * 70)
    if not aura_results:
        print("\nNo AURA eval results found. Run eval_psnr.py first:")
        print("  python scripts/eval_psnr.py outputs/truck-3k-run5.aura outputs/truck-pts129k-manifest.json")
        print("  (save stdout to outputs/eval_truck_run5.txt for this script to parse)")


if __name__ == "__main__":
    main()
