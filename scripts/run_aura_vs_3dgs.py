#!/usr/bin/env python3
"""AURA vs 3DGS benchmark comparison table.

Reads AURA eval outputs from outputs/ and tabulates them against 3DGS numbers.

IMPORTANT — two kinds of 3DGS columns:
  * "3DGS (published)": the ``REFERENCE_3DGS`` numbers below are copied from
    Kerbl et al. 2023 (Table 1). They were NOT executed by this repo, and were
    measured at a different resolution / eval protocol than AURA's
    ``eval_psnr.py``. They are a sanity reference only — do NOT read them as a
    head-to-head result.
  * "3DGS (real baseline)": if you want an executed-vs-executed comparison, run
    ``scripts/run_baseline_3dgs.py`` first. It trains a real gsplat 3DGS model
    on the SAME COLMAP scene and evaluates it on the SAME eval split with the
    SAME metric code as AURA, writing an eval file this script will pick up
    (any ``eval_*baseline*`` / ``eval_*3dgs*`` file in --eval-dir).

Usage:
    # AURA-vs-published table (reference only):
    python scripts/run_aura_vs_3dgs.py [--eval-dir outputs]
    # AURA-vs-real-baseline (executed-vs-executed):
    python scripts/run_baseline_3dgs.py outputs/<scene>-manifest.json \\
        --out outputs/eval_<scene>_baseline3dgs.txt
    python scripts/run_aura_vs_3dgs.py --eval-dir outputs
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Published 3DGS reference numbers from Kerbl et al. 2023 (Table 1).
# NOT executed by this repo — see the module docstring. Kept only as a sanity
# reference; for a real head-to-head run scripts/run_baseline_3dgs.py.
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
        if "SSIM:" in line:
            # eval_psnr emits the summary as "Average PSNR: .. dB  SSIM: 0.857 .."
            # (the token is "SSIM:", not "Average SSIM:"); per-frame lines use
            # "SSIM=" so they do not false-match here.
            try:
                result["ssim"] = float(line.split("SSIM:")[1].split()[0].strip())
            except Exception:
                pass
    return result


def _is_baseline_file(stem: str) -> bool:
    s = stem.lower()
    return "baseline" in s or "3dgs" in s


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA vs 3DGS comparison table")
    parser.add_argument("--eval-dir", default="outputs", type=Path)
    args = parser.parse_args()

    # Find eval txt files, separating AURA evals from real-baseline evals.
    aura_results: dict[str, dict] = {}
    real_baseline_results: dict[str, dict] = {}
    for txt_path in sorted(args.eval_dir.glob("eval_*.txt")):
        scene = txt_path.stem.replace("eval_", "").split("_")[0]
        metrics = parse_eval_txt(txt_path)
        if not metrics:
            continue
        if _is_baseline_file(txt_path.stem):
            real_baseline_results[scene] = metrics
        else:
            aura_results[scene] = metrics

    print("=" * 92)
    print("AURA vs 3DGS Benchmark Comparison")
    print("=" * 92)
    print(
        f"{'Scene':13s}  {'AURA PSNR':>10}  {'3DGS real':>10}  {'3DGS pub*':>10}  "
        f"{'AURA SSIM':>10}  {'3DGS real':>10}  {'3DGS pub*':>10}"
    )
    print("-" * 92)

    all_scenes = sorted(
        set(aura_results) | set(real_baseline_results) | set(REFERENCE_3DGS)
    )
    for scene in all_scenes:
        aura = aura_results.get(scene, {})
        real = real_baseline_results.get(scene, {})
        ref = REFERENCE_3DGS.get(scene, {})
        aura_psnr = f"{aura['psnr']:.2f}" if "psnr" in aura else "N/A"
        real_psnr = f"{real['psnr']:.2f}" if "psnr" in real else "N/A"
        ref_psnr = f"{ref['psnr']:.2f}" if "psnr" in ref else "N/A"
        aura_ssim = f"{aura['ssim']:.3f}" if "ssim" in aura else "N/A"
        real_ssim = f"{real['ssim']:.3f}" if "ssim" in real else "N/A"
        ref_ssim = f"{ref['ssim']:.3f}" if "ssim" in ref else "N/A"
        print(
            f"  {scene:11s}  {aura_psnr:>10}  {real_psnr:>10}  {ref_psnr:>10}  "
            f"{aura_ssim:>10}  {real_ssim:>10}  {ref_ssim:>10}"
        )

    print("-" * 92)
    print(
        "* '3DGS pub' = published Kerbl et al. 2023 numbers, NOT executed by this "
        "repo and not\n  measured at AURA's resolution/protocol. Only the '3DGS "
        "real' columns are an honest\n  executed-vs-executed comparison; populate "
        "them with scripts/run_baseline_3dgs.py."
    )
    if not real_baseline_results:
        print(
            "\nNo real 3DGS baseline eval found in "
            f"{args.eval_dir}. For an executed-vs-executed comparison run:"
        )
        print(
            "  python scripts/run_baseline_3dgs.py outputs/<scene>-manifest.json "
            "--out outputs/eval_<scene>_baseline3dgs.txt"
        )
    if not aura_results:
        print("\nNo AURA eval results found. Run eval_psnr.py first:")
        print("  python scripts/eval_psnr.py outputs/truck-3k-run6.aura outputs/truck-pts129k-manifest.json")
        print("  (save stdout to outputs/eval_truck_run6.txt for this script to parse)")


if __name__ == "__main__":
    main()
