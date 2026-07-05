#!/usr/bin/env python3
"""B2 figure: the true gsplat-3DGS matched-budget control on Truck.

Reads the committed evidence `outputs/gsplat_control.json` and renders a clean
three-bar PSNR comparison at a matched 1M-carrier budget on the held-out Truck
split: true gsplat-3DGS (MCMC, cap_max=1e6, final@30k) vs the frozen-Gaussian
DBS control vs adaptive Beta. The point of the figure is that the adaptive-Beta
win survives against a *real* 3DGS baseline, and that the frozen-Beta control
this repo has used elsewhere is not artificially weak (it lands essentially on
the true 3DGS number).

Honest scope, stated on the figure: Truck only (1 of 8 scenes); the other seven
scenes have no true gsplat-3DGS control yet. Numbers are read verbatim from the
JSON, so the figure regenerates bit-for-bit from the committed artifact.

Usage (CPU is fine — pure matplotlib over a JSON):
    python experiments/make_b2_gsplat_control_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "gsplat_control.json"
OUT = REPO / "assets" / "b2_gsplat_control_truck.png"


def main() -> None:
    data = json.loads(SRC.read_text())
    truck = data["scenes"]["truck"]
    gsplat = float(truck["gsplat_mcmc_1M"]["final"]["psnr"])   # true 3DGS, final@30k
    frozen = float(truck["dbs_frozen_gauss_1M"]["psnr"])       # frozen-Beta control
    beta = float(truck["dbs_beta_1M"]["psnr"])                 # adaptive Beta

    labels = [
        "true gsplat-3DGS\n(MCMC, final@30k)",
        "frozen-β control\n(DBS ablation)",
        "adaptive Beta\n(AURA)",
    ]
    values = [gsplat, frozen, beta]
    # muted grey / muted grey / one accent for the AURA arm
    colors = ["#5b6b7a", "#8a94a0", "#2f7d57"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    bars = ax.bar(labels, values, color=colors, width=0.62, zorder=3,
                  edgecolor="white", linewidth=0.8)

    # value labels on top of each bar
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f} dB",
                ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                color="#222222", zorder=4)

    # zoom the y-axis so the ~0.45 dB spread is legible, but keep a real baseline
    lo = min(values)
    hi = max(values)
    ax.set_ylim(lo - 0.22, hi + 0.42)
    ax.set_ylabel("PSNR (dB), held-out Truck", fontsize=11)
    ax.set_title(
        "B2: adaptive-Beta win holds vs a real gsplat-3DGS control\n"
        "Truck only (1 of 8 scenes), matched 1M-carrier budget",
        fontsize=12, fontweight="bold", pad=10)

    ax.grid(axis="y", color="#d9dde2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=9.5)

    # annotate the two honest comparisons, both placed in clear headroom above the bars
    # compute the delta from the 2-dp bar labels so the annotation is internally
    # consistent with the bars (26.39 - 25.94 = 0.45), not from full precision
    delta_win = round(beta, 2) - round(gsplat, 2)
    ax.annotate(
        f"+{delta_win:.2f} dB vs true 3DGS",
        xy=(2, beta), xytext=(2, hi + 0.30),
        ha="center", fontsize=10.5, color="#2f7d57", fontweight="bold")
    ax.annotate(
        f"frozen control within {abs(frozen - gsplat):.2f} dB of true 3DGS\n"
        "— the frozen-β control was not artificially weak",
        xy=(0.5, frozen), xytext=(0.5, hi + 0.24),
        ha="center", fontsize=8.8, color="#55606b")

    fig.text(0.5, -0.02,
             "Source: outputs/gsplat_control.json (gsplat MCMCStrategy, cap_max=1e6, 30k steps; "
             "every-8th-view test split). 7 of 8 scenes have no true 3DGS control.",
             ha="center", fontsize=7.4, color="#7a828b")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  (gsplat {gsplat:.3f} / frozen {frozen:.3f} / beta {beta:.3f})")


if __name__ == "__main__":
    main()
