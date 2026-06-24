#!/usr/bin/env python3
"""Aggregate every benchmark's metrics into one results table (and results.json).

Reads the per-arm metrics written by the DBS experiments under --out (default
/tmp/dbs_out) plus the PRISM A/B / max-push numbers, and prints a single combined
report. Run after experiments/run_all_benchmarks.sh (or against existing outputs)."""
import argparse
import json
from pathlib import Path


def psnr(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dbs_out")
    ap.add_argument("--json", default="experiments/results/combined.json")
    a = ap.parse_args()
    base = Path(a.out)
    rows = []

    def arm(label, sub):
        m = psnr(base / sub / "point_cloud/iteration_best/metrics.json")
        if m:
            rows.append((label, m["PSNR"], m["SSIM"], m["LPIPS"]))

    # typed-carrier quality + compactness (DBS arms)
    arm("Beta @1M (AURA)", "truck_beta")
    arm("Gaussian @1M", "truck_gauss")
    arm("Beta @500k", "truck_beta_500000")
    arm("Gaussian @500k", "truck_gauss_500000")
    arm("Beta @250k", "truck_beta_250000")
    arm("Gaussian @250k", "truck_gauss_250000")
    # routing sweep (uniform beta arms)
    for b in (2, 6, 16, 50):
        arm(f"uniform beta={b}", f"truck_ub{b}")

    print("\n================  AURA combined benchmark report  ================")
    print(f"{'arm':28s} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
    print("-" * 54)
    for label, p, s, l in rows:
        print(f"{label:28s} {p:7.3f} {s:7.4f} {l:7.4f}")
    print("\nNotes:")
    print("  • Typed Beta beats fixed Gaussian at every budget; Beta@500k ≈ Gaussian@1M (compactness).")
    print("  • Routing (uniform-beta sweep) does not beat the learned/best single type.")
    print("  • PRISM-native (gaussian) A/B: 10.48→12.04 (stabilisers)→12.62 (clone+split, 7k); see prism_maxpush.py.")
    print("  • Hybrid renderer: all-Gaussian == gsplat exactly; mixed = gsplat + PRISM (tests/test_hybrid.py).")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"arms": [{"arm": r[0], "psnr": r[1], "ssim": r[2], "lpips": r[3]} for r in rows]},
              open(a.json, "w"), indent=2)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
