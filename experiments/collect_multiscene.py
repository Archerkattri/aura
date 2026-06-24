#!/usr/bin/env python3
"""Aggregate the multi-scene Beta-vs-Gaussian benchmark into a table, a grouped bar
chart, and a per-scene-delta chart. -> docs/multiscene.png, docs/multiscene_delta.png,
experiments/results/multiscene.json"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def metric(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dbs_multiscene")
    a = ap.parse_args()
    base = Path(a.out)
    scenes, beta, gauss, rows = [], [], [], []
    for d in sorted(base.glob("*_beta")):
        name = d.name[:-5]
        b = metric(d / "point_cloud/iteration_best/metrics.json")
        g = metric(base / f"{name}_gauss/point_cloud/iteration_best/metrics.json")
        if not b or not g:
            continue
        scenes.append(name); beta.append(b["PSNR"]); gauss.append(g["PSNR"])
        rows.append({"scene": name, "beta_psnr": b["PSNR"], "gauss_psnr": g["PSNR"],
                     "beta_ssim": b["SSIM"], "gauss_ssim": g["SSIM"],
                     "beta_lpips": b["LPIPS"], "gauss_lpips": g["LPIPS"],
                     "delta_psnr": b["PSNR"] - g["PSNR"]})
    if not scenes:
        print("no completed scene pairs yet"); return

    print(f"\n{'scene':12s} {'Beta':>7} {'Gauss':>7} {'Δ':>6}")
    for r in rows:
        print(f"{r['scene']:12s} {r['beta_psnr']:7.2f} {r['gauss_psnr']:7.2f} {r['delta_psnr']:+6.2f}")
    mean_d = np.mean([r["delta_psnr"] for r in rows])
    print(f"mean Δ PSNR (Beta − Gaussian): {mean_d:+.2f} dB across {len(rows)} scenes")
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    json.dump({"scenes": rows, "mean_delta_psnr": float(mean_d)},
              open("experiments/results/multiscene.json", "w"), indent=2)

    # grouped bar chart
    x = np.arange(len(scenes)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(scenes)), 4.6))
    ax.bar(x - w / 2, gauss, w, label="fixed Gaussian", color="#888")
    ax.bar(x + w / 2, beta, w, label="AURA (adaptive Beta)", color="#1f9d55")
    ax.set_xticks(x); ax.set_xticklabels(scenes, rotation=30, ha="right")
    ax.set_ylabel("PSNR (dB)"); ax.set_ylim(min(min(gauss), min(beta)) - 1, max(max(beta), max(gauss)) + 1)
    ax.set_title(f"AURA typed Beta vs fixed Gaussian — {len(scenes)} scenes (mean +{mean_d:.2f} dB)")
    ax.legend(); ax.grid(axis="y", alpha=0.25); plt.tight_layout()
    plt.savefig(ROOT / "docs/multiscene.png", dpi=130)

    # per-scene delta
    fig2, ax2 = plt.subplots(figsize=(max(7, 1.2 * len(scenes)), 3.6))
    deltas = [r["delta_psnr"] for r in rows]
    ax2.bar(x, deltas, color=["#1f9d55" if d >= 0 else "#bf616a" for d in deltas])
    ax2.axhline(0, color="#bbb", lw=1); ax2.axhline(mean_d, color="#1f9d55", ls=":", lw=1, label=f"mean +{mean_d:.2f}")
    ax2.set_xticks(x); ax2.set_xticklabels(scenes, rotation=30, ha="right")
    ax2.set_ylabel("Δ PSNR (Beta − Gaussian)")
    ax2.set_title("Per-scene quality gain from typed carriers"); ax2.legend(); plt.tight_layout()
    plt.savefig(ROOT / "docs/multiscene_delta.png", dpi=130)
    print("wrote docs/multiscene.png, docs/multiscene_delta.png, experiments/results/multiscene.json")


if __name__ == "__main__":
    main()
