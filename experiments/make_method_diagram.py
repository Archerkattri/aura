#!/usr/bin/env python3
"""Schematic: how COLMAP / NeRF / 3DGS / AURA represent a scene and how they differ.
Pure matplotlib (no GPU). Writes docs/how_it_works.png."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrow, Polygon, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parent.parent
BG = "#0f1115"; FG = "#e6e6e6"; SUB = "#9aa0a8"
rng = np.random.default_rng(3)


def cam(ax, x, y, c="#5e81ac"):
    ax.add_patch(Polygon([[x, y], [x + 0.5, y + 0.28], [x + 0.5, y - 0.28]], closed=True, fc=c, ec="none", alpha=0.9))


def panel(ax, title, how, gives, lacks, draw):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 9.6, boxstyle="round,pad=0.1", fc="#171a21", ec="#2b303b", lw=1.5))
    ax.text(5, 9.3, title, color=FG, fontsize=15, fontweight="bold", ha="center")
    draw(ax)
    ax.text(5, 2.55, how, color=FG, fontsize=8.6, ha="center", style="italic", wrap=True)
    ax.text(0.7, 1.5, "gives:", color="#a3be8c", fontsize=8.2, fontweight="bold")
    ax.text(2.1, 1.5, gives, color=SUB, fontsize=8.2)
    ax.text(0.7, 0.85, "lacks:", color="#bf616a", fontsize=8.2, fontweight="bold")
    ax.text(2.1, 0.85, lacks, color=SUB, fontsize=8.2)


def draw_colmap(ax):
    for cx, cy in [(1.0, 7.8), (1.2, 4.3), (8.6, 7.6), (8.4, 4.1)]:
        cam(ax, cx if cx < 5 else cx - 0.5, cy, "#5e81ac")
    pts = rng.uniform([3.2, 4.0], [6.8, 8.2], size=(70, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#d8dee9", alpha=0.8)
    for cx, cy in [(1.5, 7.8), (1.7, 4.3)]:
        for _ in range(6):
            p = pts[rng.integers(len(pts))]
            ax.plot([cx, p[0]], [cy, p[1]], color="#4c566a", lw=0.3, alpha=0.5)


def draw_nerf(ax):
    ax.add_patch(Circle((5, 6.4), 2.0, fc="#3b4252", ec="none", alpha=0.5))
    cam(ax, 0.8, 6.2, "#5e81ac")
    ts = np.linspace(0, 1, 9)
    ray = np.array([[1.3, 6.4], [8.4, 6.9]])
    pp = ray[0] + ts[:, None] * (ray[1] - ray[0])
    ax.plot(ray[:, 0], ray[:, 1], color="#88c0d0", lw=1.0)
    ax.scatter(pp[:, 0], pp[:, 1], s=14, c="#ebcb8b", zorder=3)
    ax.add_patch(FancyBboxPatch((3.7, 3.4), 2.6, 0.9, boxstyle="round,pad=0.06", fc="#2e3440", ec="#88c0d0"))
    ax.text(5, 3.85, "MLP(x,dir)→(σ,c)", color="#88c0d0", fontsize=8, ha="center")


def draw_3dgs(ax):
    cam(ax, 0.8, 6.2, "#5e81ac")
    for _ in range(16):
        x, y = rng.uniform([3.0, 4.2], [7.4, 8.2])
        e = Ellipse((x, y), rng.uniform(0.5, 1.3), rng.uniform(0.3, 0.7),
                    angle=rng.uniform(0, 180), fc=plt.cm.twilight(rng.random()), ec="none", alpha=0.75)
        ax.add_patch(e)
    ax.text(5, 3.2, "all the same primitive: a Gaussian", color=SUB, fontsize=7.6, ha="center")


def draw_aura(ax):
    cam(ax, 0.8, 6.2, "#5e81ac")
    # mixed typed carriers: ellipse (gaussian), diamond (beta), wavy (gabor)
    ax.add_patch(Ellipse((3.6, 7.4), 1.1, 0.6, angle=20, fc="#81a1c1", ec="none", alpha=0.85))
    ax.add_patch(Polygon([[5.2, 7.8], [5.8, 7.3], [5.2, 6.8], [4.6, 7.3]], fc="#a3be8c", ec="none", alpha=0.9))
    xx = np.linspace(6.2, 7.6, 30); ax.plot(xx, 7.3 + 0.22 * np.sin((xx - 6.2) * 12), color="#ebcb8b", lw=2.2)
    ax.add_patch(Ellipse((4.4, 5.6), 0.9, 0.5, angle=-15, fc="#b48ead", ec="none", alpha=0.85))
    ax.add_patch(Polygon([[6.4, 6.0], [6.9, 5.6], [6.4, 5.2], [5.9, 5.6]], fc="#a3be8c", ec="none", alpha=0.9))
    # tags
    for (tx, ty, t, c) in [(3.0, 4.6, "semantic", "#88c0d0"), (5.0, 4.6, "confidence", "#a3be8c"),
                           (7.0, 4.6, "relight·export", "#ebcb8b")]:
        ax.add_patch(FancyBboxPatch((tx - 0.05, ty - 0.18), 0.1, 0.36, boxstyle="round,pad=0.16", fc="#2e3440", ec=c, lw=1))
        ax.text(tx + 0.05, ty, t, color=c, fontsize=6.8, ha="center", va="center")
    ax.text(5, 3.25, "different carrier TYPES per region + an asset contract", color=SUB, fontsize=7.3, ha="center")


fig, axes = plt.subplots(1, 4, figsize=(18, 5.2)); fig.patch.set_facecolor(BG)
panel(axes[0], "COLMAP  (Photogrammetry)", "feature-match across photos →\ntriangulate camera poses + 3D points",
      "camera poses, sparse geometry", "no surface, no colour/radiance", draw_colmap)
panel(axes[1], "NeRF", "an MLP maps (position, view-dir) →\n(density, colour); integrate along each ray",
      "photoreal continuous field", "slow, implicit, hard to edit/export", draw_nerf)
panel(axes[2], "3D Gaussian Splatting", "millions of anisotropic Gaussians,\nrasterized + alpha-blended in real time",
      "explicit, real-time", "one primitive type; no semantics/\nconfidence/relight; baked lighting", draw_3dgs)
panel(axes[3], "AURA", "adaptive TYPED carriers (Beta/Gabor/\nneural/Gaussian) under one asset contract",
      "typed + compact, queryable,\nrelightable, semantic, engine-export", "(builds on the gsplat engine)", draw_aura)
for ax in axes:
    ax.set_facecolor(BG)
fig.suptitle("How each method represents a scene — and where AURA goes further",
             color=FG, fontsize=17, fontweight="bold", y=1.02)
plt.tight_layout(); out = ROOT / "docs/how_it_works.png"
plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
print(f"wrote {out}")
