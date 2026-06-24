#!/usr/bin/env python3
"""Detailed vertical schematic: how COLMAP / NeRF / 3DGS / AURA represent a scene.
Four tall stacked rows (readable without zooming). Pure matplotlib. -> docs/how_it_works.png"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, Polygon, FancyBboxPatch, Circle,
                                FancyArrowPatch, Rectangle, RegularPolygon)

ROOT = Path(__file__).resolve().parent.parent
BG = "#0f1115"; PANEL = "#171a21"; EDGE = "#2b303b"; FG = "#e9ecef"; SUB = "#aab2bd"
GREEN = "#a3be8c"; RED = "#bf616a"; BLUE = "#88c0d0"; YEL = "#ebcb8b"; PUR = "#b48ead"; ORA = "#d08770"
rng = np.random.default_rng(5)


def cam(ax, x, y, ang=0, s=0.55, c=BLUE, label=None):
    """Camera as a small pyramid + image plane, apex at (x,y), looking +x rotated by ang(deg)."""
    th = np.radians(ang)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = np.array([[0, 0], [1.4 * s, 0.9 * s], [1.4 * s, -0.9 * s]])
    pts = pts @ R.T + [x, y]
    ax.add_patch(Polygon(pts, closed=True, fc=c, ec="white", lw=0.6, alpha=0.92, zorder=5))
    plane = np.array([[1.4 * s, 0.9 * s], [1.4 * s, -0.9 * s]]) @ R.T + [x, y]
    ax.plot(plane[:, 0], plane[:, 1], color="white", lw=1.4, zorder=6)
    if label:
        ax.text(x - 0.2, y - 0.55, label, color=SUB, fontsize=9, ha="right")


def box(ax, x, y, w, h, text, ec=BLUE, fs=11, fc="#222833"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.4, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, color=FG, fontsize=fs, ha="center", va="center", zorder=5)


def arrow(ax, x0, y0, x1, y1, c=SUB, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 color=c, lw=lw, zorder=4))


def header(ax, n, title, color):
    ax.add_patch(Rectangle((0.25, 8.7), 0.12, 1.0, fc=color, ec="none"))
    ax.text(0.6, 9.2, f"{n}  {title}", color=FG, fontsize=20, fontweight="bold", va="center")


def steps(ax, x, lines, y0=7.9, dy=0.62, fs=12.5):
    for i, ln in enumerate(lines):
        ax.text(x, y0 - i * dy, ln, color=SUB, fontsize=fs, va="top")


def verdict(ax, x, y, gives, lacks):
    ax.text(x, y, "✓ gives  ", color=GREEN, fontsize=11.5, fontweight="bold", va="top")
    ax.text(x + 1.5, y, gives, color=FG, fontsize=11.5, va="top")
    ax.text(x, y - 0.6, "✗ lacks  ", color=RED, fontsize=11.5, fontweight="bold", va="top")
    ax.text(x + 1.5, y - 0.6, lacks, color=FG, fontsize=11.5, va="top")


def setup(ax, color):
    ax.set_xlim(0, 20); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.15), 19.8, 9.7, boxstyle="round,pad=0.05",
                                fc=PANEL, ec=EDGE, lw=1.6))
    ax.axvline(9.6, 0.06, 0.9, color=EDGE, lw=1)  # divider: schematic | text


# ---------------------------------------------------------------- rows
def row_colmap(ax):
    setup(ax, BLUE); header(ax, "①", "COLMAP — Structure-from-Motion (photogrammetry)", BLUE)
    # scene points (right region) + 3 cameras around it
    P = rng.uniform([6.3, 3.0], [8.7, 7.4], size=(45, 2))
    cams = [(1.4, 6.6, -18), (1.6, 3.0, 12), (4.6, 1.4, 38)]
    for (cx, cy, a) in cams:
        cam(ax, cx, cy, a, 0.5)
        for p in P[rng.choice(len(P), 7, replace=False)]:
            ax.plot([cx, p[0]], [cy, p[1]], color="#3b4252", lw=0.4, alpha=0.7, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], s=12, c="#d8dee9", zorder=3)
    # feature-match hint between two image planes
    ax.text(3.0, 8.2, "feature matches + triangulation", color=BLUE, fontsize=10, ha="center")
    steps(ax, 10.0, [
        "1.  detect & match keypoints (SIFT) across overlapping photos",
        "2.  bundle-adjust → camera poses + a sparse 3D point cloud",
        "3.  optional dense MVS for a denser point set",
        "",
        "the geometric scaffold every later method builds on",
    ])
    verdict(ax, 10.0, 2.6, "calibrated camera poses, sparse/dense 3D points",
            "no view-dependent radiance — not photoreal, not renderable")


def row_nerf(ax):
    setup(ax, PUR); header(ax, "②", "NeRF — neural volumetric field", PUR)
    cam(ax, 1.0, 5.0, 0, 0.5)
    ray = np.array([[1.6, 5.0], [8.9, 6.2]])
    ax.add_patch(Ellipse((6.2, 5.6), 4.6, 4.0, fc="#2e3440", ec="none", alpha=0.5))
    ax.plot(ray[:, 0], ray[:, 1], color=PUR, lw=1.6, zorder=3)
    ts = np.linspace(0.1, 0.95, 11)
    pp = ray[0] + ts[:, None] * (ray[1] - ray[0])
    sig = np.exp(-((ts - 0.62) ** 2) / 0.01) + 0.15      # density profile (a surface)
    ax.scatter(pp[:, 0], pp[:, 1], s=20 + 130 * sig / sig.max(), c=YEL, zorder=4, edgecolors="none")
    ax.text(4.2, 7.9, "samples along the ray (size ∝ density σ)", color=PUR, fontsize=10)
    box(ax, 3.2, 1.2, 3.2, 1.1, "MLP\nγ(x),γ(d) → (σ, c)", ec=PUR, fs=11)
    arrow(ax, 5.0, 3.0, 4.8, 2.35, c=PUR)
    steps(ax, 10.0, [
        "1.  cast a ray through each pixel; pick sample points along it",
        "2.  an MLP maps (encoded position, view-dir) → (density σ, colour c)",
        "3.  integrate front-to-back:  C = Σ Tᵢ(1−e^(−σᵢδᵢ)) cᵢ",
        "",
        "a continuous, view-dependent radiance field",
    ])
    verdict(ax, 10.0, 2.6, "photoreal, continuous, view-dependent colour",
            "slow (an MLP per sample), implicit — hard to edit / export")


def row_3dgs(ax):
    setup(ax, ORA); header(ax, "③", "3D Gaussian Splatting — explicit real-time splats", ORA)
    cam(ax, 1.0, 5.0, 0, 0.5)
    for _ in range(20):
        x, y = rng.uniform([3.0, 2.3], [6.6, 7.6])
        ax.add_patch(Ellipse((x, y), rng.uniform(0.55, 1.25), rng.uniform(0.28, 0.6),
                     angle=rng.uniform(0, 180), fc=plt.cm.twilight(rng.random()), ec="none", alpha=0.7, zorder=2))
    arrow(ax, 6.9, 5.0, 7.7, 5.0, c=ORA)
    ax.add_patch(Rectangle((7.9, 2.6), 1.4, 4.8, fc="#222833", ec=ORA, lw=1.2))
    for gy in np.linspace(3.1, 6.9, 6):                  # 2D splats on the image plane
        ax.add_patch(Ellipse((8.6, gy), rng.uniform(0.5, 0.9), 0.32, angle=rng.uniform(0, 180),
                     fc=plt.cm.twilight(rng.random()), ec="none", alpha=0.8))
    ax.text(3.0, 8.2, "3D anisotropic Gaussians  μ, Σ=RSSᵀRᵀ, α, SH", color=ORA, fontsize=10)
    ax.text(8.6, 2.2, "image", color=SUB, fontsize=9, ha="center")
    steps(ax, 10.0, [
        "1.  scene = millions of identical anisotropic Gaussians",
        "2.  project each to a 2D splat (EWA), sort by depth per tile",
        "3.  alpha-blend front-to-back:  C = Σ cᵢαᵢ Π(1−αⱼ)",
        "",
        "explicit, differentiable, real-time rasterization",
    ])
    verdict(ax, 10.0, 2.6, "explicit geometry, real-time, trainable",
            "one primitive type; baked lighting; no semantics / confidence")


def row_aura(ax):
    setup(ax, GREEN); header(ax, "④", "AURA — adaptive typed carriers + asset contract", GREEN)
    cam(ax, 1.0, 5.0, 0, 0.5)
    # mixed carrier TYPES placed by region
    ax.add_patch(Ellipse((3.2, 6.6), 1.2, 0.6, angle=15, fc="#81a1c1", ec="white", lw=0.5, alpha=0.9, zorder=3))
    ax.text(3.2, 7.25, "Gaussian", color="#81a1c1", fontsize=8, ha="center")
    ax.add_patch(Polygon([[5.0, 7.0], [5.7, 6.4], [5.0, 5.8], [4.3, 6.4]], fc=GREEN, ec="white", lw=0.5, alpha=0.92, zorder=3))
    ax.text(5.0, 7.3, "Beta (sharp)", color=GREEN, fontsize=8, ha="center")
    xx = np.linspace(6.2, 7.8, 30); ax.plot(xx, 6.4 + 0.22 * np.sin((xx - 6.2) * 13), color=YEL, lw=2.6, zorder=3)
    ax.text(7.0, 6.95, "Gabor (texture)", color=YEL, fontsize=8, ha="center")
    ax.add_patch(RegularPolygon((4.0, 4.4), 6, radius=0.45, fc=PUR, ec="white", lw=0.5, alpha=0.9, zorder=3))
    ax.text(4.0, 3.75, "neural", color=PUR, fontsize=8, ha="center")
    # per-carrier payload chip
    box(ax, 5.7, 3.7, 3.3, 1.0, "payload:  rgb · normal · semantic · confidence", ec=GREEN, fs=9.5)
    # dual engine
    ax.text(2.0, 2.6, "Gaussians → gsplat", color="#81a1c1", fontsize=9.5)
    ax.text(2.0, 2.0, "typed → PRISM", color=YEL, fontsize=9.5)
    ax.text(2.0, 1.4, "→ depth-composited", color=SUB, fontsize=9.5)
    steps(ax, 10.0, [
        "1.  each region gets the carrier TYPE that fits it (Beta/Gabor/neural/Gaussian)",
        "2.  Gaussians render via gsplat (engine); typed carriers via PRISM — composited",
        "3.  every carrier carries colour + normal + semantic + confidence",
        "4.  one contract:  rayQuery → {color,depth,normal,sem,conf} · relight · glTF export",
    ], y0=8.0, dy=0.62)
    verdict(ax, 10.0, 2.6, "typed + compact, queryable, relightable, semantic, exportable",
            "(extends the gsplat engine — uses it for Gaussians, adds the rest)")


fig, axes = plt.subplots(4, 1, figsize=(15, 21)); fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor(BG)
row_colmap(axes[0]); row_nerf(axes[1]); row_3dgs(axes[2]); row_aura(axes[3])
fig.suptitle("Photogrammetry → NeRF → 3D Gaussian Splatting → AURA",
             color=FG, fontsize=24, fontweight="bold", y=0.995)
plt.subplots_adjust(hspace=0.12, top=0.965, bottom=0.01, left=0.01, right=0.99)
out = ROOT / "docs/how_it_works.png"
plt.savefig(out, dpi=120, facecolor=BG)
print(f"wrote {out}")
