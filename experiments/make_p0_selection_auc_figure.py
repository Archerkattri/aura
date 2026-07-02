#!/usr/bin/env python3
"""Render the four-scene P0 selection-AUC figure from committed calib_*.json.

Grouped bar chart: calibrated confidence vs opacity vs oracle ceiling, per scene,
with a recessive random-baseline reference line. Reads the authoritative
`outputs/calib_<scene>.json` reports and writes `assets/p0_selection_auc.png`.

Usage:
    python experiments/make_p0_selection_auc_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
SCENES = [("truck", "Truck"), ("garden", "Garden"), ("kitchen", "Kitchen"), ("room", "Room")]

# Validated palette (dataviz skill, light surface): blue + orange pass all six
# checks (worst adjacent CVD ΔE 96.7). Oracle is a neutral reference, not a
# competing categorical hue; random is recessive chrome.
CAL = "#2a78d6"      # calibrated confidence (our method)
OPAC = "#eb6834"     # opacity (engine pruning default)
ORACLE = "#a6a49c"   # per-scene achievable ceiling (neutral reference)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load():
    rows = []
    for key, label in SCENES:
        d = json.loads((REPO / "outputs" / f"calib_{key}.json").read_text())
        a = d["selection_auc_retained_reliability"]
        rows.append((label, a["calibrated_confidence"], a["opacity"], a["oracle_ceiling"], a["random"]))
    return rows


def main():
    rows = load()
    labels = [r[0] for r in rows]
    cal = [r[1] for r in rows]
    opac = [r[2] for r in rows]
    oracle = [r[3] for r in rows]
    rand = [r[4] for r in rows]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
    n = len(labels)
    group_w = 0.74
    bar_w = group_w / 3.0
    x = list(range(n))

    def positions(offset):
        return [i + offset for i in x]

    p_cal = positions(-bar_w)
    p_opac = positions(0.0)
    p_oracle = positions(bar_w)

    # 2px surface gap between adjacent fills via a light edge.
    common = dict(width=bar_w * 0.92, edgecolor=SURFACE, linewidth=1.6, zorder=3)
    ax.bar(p_cal, cal, color=CAL, label="Calibrated confidence (AURA)", **common)
    ax.bar(p_opac, opac, color=OPAC, label="Opacity (engine default)", **common)
    ax.bar(p_oracle, oracle, color=ORACLE, label="Oracle ceiling", **common)

    # Recessive random-baseline reference over each group.
    for i in x:
        ax.plot([i - bar_w * 1.55, i + bar_w * 1.55], [rand[i], rand[i]],
                color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=4)

    # Direct value labels.
    for xs, ys, col in ((p_cal, cal, CAL), (p_opac, opac, OPAC), (p_oracle, oracle, ORACLE)):
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.008, f"{yi:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)

    ax.set_ylabel("Selection AUC  (mean retained held-out reliability)", fontsize=10.5, color=INK)
    ax.set_ylim(0, 0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=INK)
    ax.set_title("Calibrated confidence is a near-oracle pruning signal; opacity is not",
                 fontsize=12.5, color=INK, pad=12, fontweight="bold")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=1.0)
    ax.xaxis.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.tick_params(axis="x", length=0)

    handles = [
        Patch(facecolor=CAL, label="Calibrated confidence (AURA)"),
        Patch(facecolor=OPAC, label="Opacity (engine default)"),
        Patch(facecolor=ORACLE, label="Oracle ceiling"),
        Line2D([0], [0], color=MUTED, lw=1.4, ls=(0, (4, 3)), label="Random baseline"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
              ncol=2, handlelength=1.4, columnspacing=1.6)

    fig.text(0.5, -0.005,
             "Higher is better. Calibrated confidence lands within 1–4% of the oracle ceiling on every scene "
             "and beats opacity at every pruning budget; opacity sits at or below random.",
             ha="center", fontsize=8.2, color=MUTED)

    out = REPO / "assets" / "p0_selection_auc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
