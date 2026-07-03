#!/usr/bin/env python3
"""README result figures for the P0->P2 hardening arc, from committed data only.

Sources every number from committed artifacts (no training, no GPU):
  * outputs/reliability_<scene>.npz          (P0 per-carrier labels + features)
  * outputs/calib_<scene>.json               (P0 calibration/selection reports)
  * outputs/cross_scene_transfer.json        (P1a transfer matrix)
  * outputs/p2_summary.json                  (P2 full-res + render-loss)

Writes four PNGs into assets/:
  reliability_diagram.png    raw view-count heuristic vs isotonic-calibrated
  selection_curves.png       retained reliability vs pruning budget, 4 scenes
  transfer_ece_heatmap.png   P1a cross-scene calibrator ECE transfer (color+depth)
  proxy_vs_renderloss.png    P2 colour proxy vs render-loss reliability label

Style: DejaVu fonts (avoids the Noto minus-glyph issue), legends never over data,
dataviz-validated blue/orange + neutral palette (matches p0_selection_auc.png).

Usage:  python experiments/make_hardening_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
ASSETS = REPO / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

SCENES = [("truck", "Truck"), ("garden", "Garden"), ("kitchen", "Kitchen"), ("room", "Room")]

# dataviz-validated palette (light surface), consistent across all P0-P2 figures.
CAL = "#2a78d6"       # calibrated confidence (AURA)
OPAC = "#eb6834"      # opacity (engine default)
ORACLE = "#8f8d86"    # oracle ceiling (neutral reference)
RAW = "#c23b57"       # raw view-count heuristic (the miscalibrated "before")
RENDER = "#7048b6"    # render-loss label (P2 stricter label)
RANDOM = "#b7b5ad"    # random baseline (recessive chrome)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#7d7b75"
GRID = "#e4e3dc"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "svg.fonttype": "none",
})


def _style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def load_split(scene, feature="train_agree", seed=0):
    """Reproduce experiments/calibrate_confidence.py's exact cal/eval split."""
    from aura.calibration import IsotonicConfidenceCalibrator

    d = np.load(OUT / f"reliability_{scene}.npz")
    labeled = d["labeled"]
    feat = d[feature][labeled]
    raw_conf = d["raw_conf"][labeled]
    reliability = d["reliability"][labeled]
    opacity = d["opacity"][labeled]
    m = feat.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    half = m // 2
    cal_idx, ev_idx = perm[:half], perm[half:]
    calib = IsotonicConfidenceCalibrator().fit(feat[cal_idx], reliability[cal_idx])
    conf_ev = np.asarray(calib.predict(feat[ev_idx]), dtype="float64")
    return dict(
        conf_ev=conf_ev,
        raw_ev=np.asarray(raw_conf[ev_idx], dtype="float64"),
        rel_ev=np.asarray(reliability[ev_idx], dtype="float64"),
        opa_ev=np.asarray(opacity[ev_idx], dtype="float64"),
        rng=rng,
    )


def quantile_reliability(pred, rel, nbins=10):
    """Equal-count bins on `pred`; return (mean pred, mean reliability) per bin."""
    order = np.argsort(pred)
    pred, rel = pred[order], rel[order]
    edges = np.linspace(0, len(pred), nbins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            xs.append(pred[a:b].mean())
            ys.append(rel[a:b].mean())
    return np.asarray(xs), np.asarray(ys)


# --------------------------------------------------------------------------- #
# Figure A — reliability diagram: raw heuristic vs isotonic-calibrated
# --------------------------------------------------------------------------- #
def fig_reliability_diagram():
    """Per-scene 2x2 reliability diagrams (paper Fig-2 design): in each panel the
    raw shipped heuristic saturates far off the diagonal while the calibrated
    confidence lies on it, with the scene's raw->calibrated ECE in the title."""
    splits = {k: load_split(k) for k, _ in SCENES}
    calibs = {k: json.loads((OUT / f"calib_{k}.json").read_text()) for k, _ in SCENES}

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.9), dpi=190, sharex=True, sharey=True)

    for ax, (skey, slabel) in zip(axes.ravel(), SCENES):
        s = splits[skey]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1.1, ls=(0, (5, 4)), zorder=1)
        xr, yr = quantile_reliability(s["raw_ev"], s["rel_ev"], nbins=10)
        xc, yc = quantile_reliability(s["conf_ev"], s["rel_ev"], nbins=10)
        ax.plot(xr, yr, color=RAW, lw=2.2, marker="s", ms=5.0, mfc=RAW,
                mec=SURFACE, mew=0.7, zorder=3)
        ax.plot(xc, yc, color=CAL, lw=2.2, marker="o", ms=5.0, mfc=CAL,
                mec=SURFACE, mew=0.7, zorder=4)
        raw_ece = calibs[skey]["calibration"]["ece_raw_heuristic"]
        cal_ece = calibs[skey]["calibration"]["ece_calibrated"]
        ax.set_title(f"{slabel}   —   ECE {raw_ece:.2f} → {cal_ece:.4f}",
                     fontsize=14, color=INK, pad=9, fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=11)
        _style(ax)
    for ax in axes[1, :]:
        ax.set_xlabel("Reported confidence", fontsize=12.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical held-out reliability", fontsize=12.5, color=INK)

    handles = [
        Line2D([0], [0], color=RAW, lw=2.2, marker="s", ms=5.0, mec=SURFACE,
               label="raw view-count heuristic (shipped)"),
        Line2D([0], [0], color=CAL, lw=2.2, marker="o", ms=5.0, mec=SURFACE,
               label="isotonic-calibrated confidence (AURA)"),
        Line2D([0], [0], color=MUTED, lw=1.1, ls=(0, (5, 4)),
               label="perfect calibration (y = x)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               frameon=False, fontsize=11.5, ncol=3, columnspacing=1.6)
    fig.suptitle("Isotonic calibration turns the confidence into a trustworthy reliability",
                 fontsize=16, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = ASSETS / "reliability_diagram.png"
    fig.savefig(p, bbox_inches="tight", facecolor=SURFACE, dpi=190)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
# Figure B — selection curves: retained reliability vs pruning budget
# --------------------------------------------------------------------------- #
def fig_selection_curves():
    from aura.calibration import selection_quality_curve

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), dpi=200)
    fracs = np.linspace(0.1, 1.0, 10)
    for ax, (skey, slabel) in zip(axes.ravel(), SCENES):
        s = load_split(skey)
        rel = s["rel_ev"]
        _, cal_c, auc_cal = selection_quality_curve(s["conf_ev"], rel, fracs)
        _, opa_c, auc_opa = selection_quality_curve(s["opa_ev"], rel, fracs)
        _, ora_c, auc_ora = selection_quality_curve(rel, rel, fracs)
        rnd = s["rng"].uniform(size=rel.shape[0])
        _, rnd_c, auc_rnd = selection_quality_curve(rnd, rel, fracs)

        ax.plot(fracs, ora_c, color=ORACLE, lw=1.8, ls=(0, (5, 3)), zorder=2)
        ax.plot(fracs, rnd_c, color=RANDOM, lw=1.4, ls=(0, (1, 2)), zorder=2)
        ax.plot(fracs, opa_c, color=OPAC, lw=2.2, marker="s", ms=4.0, mec=SURFACE, mew=0.6, zorder=3)
        ax.plot(fracs, cal_c, color=CAL, lw=2.4, marker="o", ms=4.4, mec=SURFACE, mew=0.6, zorder=4)

        ax.set_title(f"{slabel}", fontsize=11.5, color=INK, pad=6, fontweight="bold")
        ax.set_xlim(0.1, 1.0)
        ax.set_ylim(min(0.25, opa_c.min() - 0.05), max(cal_c.max(), ora_c.max()) + 0.06)
        ax.invert_xaxis()  # 100%-keep on the left, aggressive pruning to the right
        ax.text(0.03, 0.04,
                f"AUC  cal {auc_cal:.2f}   oracle {auc_ora:.2f}\nopacity {auc_opa:.2f}   random {auc_rnd:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=GRID, lw=0.8))
        _style(ax)
        if ax in (axes[1, 0], axes[1, 1]):
            ax.set_xlabel("Keep fraction (carriers retained)", fontsize=9.5, color=INK)
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Mean retained reliability", fontsize=9.5, color=INK)

    handles = [
        Line2D([0], [0], color=CAL, lw=2.4, marker="o", ms=4.4, mec=SURFACE, label="Calibrated confidence (AURA)"),
        Line2D([0], [0], color=OPAC, lw=2.2, marker="s", ms=4.0, mec=SURFACE, label="Opacity (engine default)"),
        Line2D([0], [0], color=ORACLE, lw=1.8, ls=(0, (5, 3)), label="Oracle ceiling"),
        Line2D([0], [0], color=RANDOM, lw=1.4, ls=(0, (1, 2)), label="Random baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9.4,
               bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Calibrated-confidence pruning tracks the oracle; opacity stays near random",
                 fontsize=13, color=INK, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = ASSETS / "selection_curves.png"
    fig.savefig(p, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
# Figure C — P1a cross-scene calibrator ECE transfer heatmap
# --------------------------------------------------------------------------- #
def fig_transfer_heatmap():
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    d = json.loads((OUT / "cross_scene_transfer.json").read_text())
    order = ["truck", "garden", "kitchen", "room"]
    disp = {"truck": "Truck", "garden": "Garden", "kitchen": "Kitchen", "room": "Room"}

    def matrix(label):
        cells = d["by_label"][label]["matrix"]
        M = np.zeros((4, 4))
        diag = np.zeros((4, 4), dtype=bool)
        for c in cells:
            i, j = order.index(c["source"]), order.index(c["target"])
            M[i, j] = c["ece"]
            diag[i, j] = c["kind"] == "in_scene"
        return M, diag

    cmap = LinearSegmentedColormap.from_list("aura_ece", ["#eef4fc", "#8fb8e6", "#2a78d6", "#12233a"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), dpi=200)
    labels = [("color", "colour-agreement label"), ("depth", "occlusion-aware label")]
    vmax = 0.058
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)

    for ax, (lab, subtitle) in zip(axes, labels):
        M, diag = matrix(lab)
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")
        for i in range(4):
            for j in range(4):
                txt = f"{M[i, j]:.4f}".lstrip("0")
                strong = norm(M[i, j]) > 0.55
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.6,
                        color=("#f4f7fb" if strong else INK),
                        fontweight="bold" if diag[i, j] else "normal")
                if diag[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec="#0b0b0b", lw=1.8, zorder=5))
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([disp[o] for o in order], fontsize=9)
        ax.set_yticklabels([disp[o] for o in order], fontsize=9)
        ax.set_xlabel("target scene (evaluated on)", fontsize=9.5, color=INK)
        ax.set_ylabel("source scene (calibrator fit on)", fontsize=9.5, color=INK)
        ax.set_title(subtitle, fontsize=11, color=INK, pad=8, fontweight="bold")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.038, pad=0.03,
                        ticks=[0.0, 0.005, 0.02, 0.058])
    cbar.ax.set_ylabel("transferred ECE (lower is better)", fontsize=9, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    cbar.outline.set_visible(False)

    fig.suptitle("A single calibrator transfers across scenes (boxed diagonal = in-scene)",
                 fontsize=13, color=INK, fontweight="bold", y=1.02)
    p = ASSETS / "transfer_ece_heatmap.png"
    fig.savefig(p, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
# Figure D — P2 colour proxy vs render-loss label
# --------------------------------------------------------------------------- #
def fig_proxy_vs_renderloss():
    d = json.loads((OUT / "p2_summary.json").read_text())["scenes"]
    keys = [k for k, _ in SCENES]
    disp = [lab for _, lab in SCENES]

    corr_color = [d[k]["fullres_color"]["reliability_summary"]["corr_trainagree_reliability"] for k in keys]
    corr_rloss = [d[k]["fullres_renderloss"]["reliability_summary"]["corr_trainagree_reliability"] for k in keys]

    def auc(k, cond, field):
        return d[k][cond]["selection_auc"][field]

    cal_c = [auc(k, "fullres_color", "calibrated_confidence") for k in keys]
    ora_c = [auc(k, "fullres_color", "oracle_ceiling") for k in keys]
    cal_r = [auc(k, "fullres_renderloss", "calibrated_confidence") for k in keys]
    ora_r = [auc(k, "fullres_renderloss", "oracle_ceiling") for k in keys]
    opa_r = [auc(k, "fullres_renderloss", "opacity") for k in keys]

    fig, (axL, axR) = plt.subplots(2, 1, figsize=(9.2, 10.4), dpi=190)
    x = np.arange(4)

    # Panel 1 — export feature correlation with each label (grouped pair).
    bw = 0.34
    axL.bar(x - bw / 2, corr_color, bw, color=CAL, edgecolor=SURFACE, lw=1.4,
            label="colour-agreement proxy label", zorder=3)
    axL.bar(x + bw / 2, corr_rloss, bw, color=RENDER, edgecolor=SURFACE, lw=1.4,
            label="render-loss label (stricter)", zorder=3)
    for xi, v in zip(x - bw / 2, corr_color):
        axL.text(xi, v + 0.014, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=INK)
    for xi, v in zip(x + bw / 2, corr_rloss):
        axL.text(xi, v + 0.014, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=INK)
    axL.set_ylim(0, 1.22)
    axL.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axL.set_ylabel("corr(export feature, held-out reliability)", fontsize=12, color=INK)
    axL.set_title("The export-time feature still predicts the stricter label", fontsize=14,
                  color=INK, pad=9, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels(disp, fontsize=12.5)
    axL.tick_params(axis="x", length=0)
    axL.tick_params(axis="y", labelsize=11)
    axL.legend(loc="upper center", frameon=False, fontsize=11.5, ncol=2,
               columnspacing=1.6)
    _style(axL)

    # Panel 2 — selection AUC under the render-loss label: opacity vs calibrated
    # vs oracle as three plain bars per scene (no floating markers).
    bw3 = 0.26
    axR.bar(x - bw3, opa_r, bw3, color=OPAC, edgecolor=SURFACE, lw=1.2,
            label="opacity (engine default)", zorder=3)
    axR.bar(x, cal_r, bw3, color=CAL, edgecolor=SURFACE, lw=1.2,
            label="calibrated confidence (AURA)", zorder=3)
    axR.bar(x + bw3, ora_r, bw3, color="#dddbd2", edgecolor=ORACLE, lw=1.2,
            label="oracle ceiling", zorder=3)
    for xi, v in zip(x - bw3, opa_r):
        axR.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, color=INK)
    for xi, v in zip(x, cal_r):
        axR.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5,
                 color=INK, fontweight="bold")
    for xi, v in zip(x + bw3, ora_r):
        axR.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, color=MUTED)
    axR.set_ylim(0, 0.85)
    axR.set_ylabel("Selection AUC under the render-loss label", fontsize=12, color=INK)
    axR.set_title("Calibrated stays between opacity and the oracle;\nthe oracle gap widens to ~6–13%",
                  fontsize=14, color=INK, pad=9, fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels(disp, fontsize=12.5)
    axR.tick_params(axis="x", length=0)
    axR.tick_params(axis="y", labelsize=11)
    axR.legend(loc="upper center", frameon=False, fontsize=11.5, ncol=3,
               columnspacing=1.2, handlelength=1.3)
    _style(axR)

    fig.suptitle("P2: the killer property survives a render-grounded label,\nwith honestly weaker margins",
                 fontsize=15.5, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=2.4)
    p = ASSETS / "proxy_vs_renderloss.png"
    fig.savefig(p, bbox_inches="tight", facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO / "src"))
    fig_reliability_diagram()
    fig_selection_curves()
    fig_transfer_heatmap()
    fig_proxy_vs_renderloss()
