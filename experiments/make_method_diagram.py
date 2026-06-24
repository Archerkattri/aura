#!/usr/bin/env python3
"""Generate a README method map: how COLMAP, NeRF, 3DGS, and AURA build a scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BG = "#0f1218"
PANEL = "#171c24"
PANEL2 = "#1d2430"
EDGE = "#303847"
FG = "#f2f4f8"
MUTED = "#a7b0bd"
BLUE = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
RED = "#f85149"


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy, value: str, size=24, fill=FG, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def wrapped(draw: ImageDraw.ImageDraw, xy, value: str, max_width: int, size=22, fill=MUTED, bold=False, gap=7):
    words = value.split()
    active = font(size, bold)
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if draw.textlength(trial, font=active) <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    x, y = xy
    for i, line in enumerate(lines):
        draw.text((x, y + i * (size + gap)), line, font=active, fill=fill)
    return y + len(lines) * (size + gap)


def box(draw, xy, fill=PANEL2, outline=EDGE, width=2, radius=14):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, a, b, color=MUTED, width=5):
    draw.line((a, b), fill=color, width=width)
    ax, ay = a
    bx, by = b
    v = np.array([bx - ax, by - ay], dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-6:
        return
    v = v / n
    p = np.array([-v[1], v[0]])
    tip = np.array([bx, by])
    tail = tip - v * 18
    draw.polygon([tuple(tip), tuple(tail + p * 9), tuple(tail - p * 9)], fill=color)


def image_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tw, th = size
    s = max(tw / w, th / h)
    img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    x = (img.width - tw) // 2
    y = (img.height - th) // 2
    return img.crop((x, y, x + tw, y + th))


def camera(draw, x, y, color=BLUE):
    draw.polygon([(x, y), (x + 55, y - 28), (x + 55, y + 28)], fill=color, outline=FG)
    draw.line((x + 55, y - 28, x + 55, y + 28), fill=FG, width=3)


def point_cloud(draw, x0, y0, w, h, color=FG, seed=2, count=80):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(count, 2))
    pts[:, 0] = x0 + w * (0.5 + 0.22 * pts[:, 0])
    pts[:, 1] = y0 + h * (0.5 + 0.28 * pts[:, 1])
    for x, y in pts:
        if x0 < x < x0 + w and y0 < y < y0 + h:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    return pts


def row_header(draw, x, y, color, title, subtitle):
    draw.rectangle((x, y + 10, x + 10, y + 76), fill=color)
    text(draw, (x + 28, y + 4), title, 32, FG, True)
    wrapped(draw, (x + 28, y + 45), subtitle, 1420, 18, MUTED)


def stage(draw, xy, wh, label, body, color):
    x, y = xy
    w, h = wh
    box(draw, (x, y, x + w, y + h), fill=PANEL2, outline=color, width=3)
    text(draw, (x + 18, y + 14), label, 22, color, True)
    wrapped(draw, (x + 18, y + 50), body, w - 36, 17, FG)


def draw_colmap(canvas, y, thumb):
    d = ImageDraw.Draw(canvas)
    row_header(d, 46, y, BLUE, "1. COLMAP: photos become calibrated sparse geometry", "Feature matches triangulate camera poses and 3D points. This is geometry scaffolding, not a renderable radiance asset.")
    panel = (46, y + 100, 1754, y + 315)
    box(d, panel, fill=PANEL, outline=EDGE, width=2)
    for i, x in enumerate((82, 206, 330)):
        tile = thumb.resize((110, 62), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y + 142 + i * 8))
        d.rectangle((x, y + 142 + i * 8, x + 110, y + 204 + i * 8), outline=FG, width=2)
    arrow(d, (460, y + 175), (565, y + 175), BLUE)
    stage(d, (575, y + 125), (250, 122), "match", "SIFT/keypoint tracks across overlapping photos", BLUE)
    arrow(d, (835, y + 185), (940, y + 185), BLUE)
    camera(d, 958, y + 148, BLUE)
    camera(d, 960, y + 232, BLUE)
    pts = point_cloud(d, 1110, y + 120, 250, 155, seed=4)
    for cx, cy in ((958, y + 148), (960, y + 232)):
        for px, py in pts[::14]:
            d.line((cx, cy, px, py), fill="#2f4258", width=1)
    stage(d, (1420, y + 125), (285, 122), "output", "poses + sparse points; optional MVS dense points", BLUE)


def draw_nerf(canvas, y):
    d = ImageDraw.Draw(canvas)
    row_header(d, 46, y, PURPLE, "2. NeRF: posed rays train a neural volume", "COLMAP poses define rays. An MLP predicts density and color at many samples along every ray; rendering integrates those samples.")
    panel = (46, y + 100, 1754, y + 315)
    box(d, panel, fill=PANEL, outline=EDGE, width=2)
    camera(d, 96, y + 205, PURPLE)
    d.ellipse((360, y + 132, 760, y + 282), fill="#252b36", outline="#394251", width=2)
    for offset, color in ((0, YELLOW), (-34, "#c690ff"), (34, "#c690ff")):
        d.line((154, y + 205, 820, y + 190 + offset), fill=color, width=3)
        for t in np.linspace(0.25, 0.86, 9):
            x = 154 + (820 - 154) * t
            yy = y + 205 + ((y + 190 + offset) - (y + 205)) * t
            r = 4 + 8 * np.exp(-((t - 0.62) ** 2) / 0.012)
            d.ellipse((x - r, yy - r, x + r, yy + r), fill=YELLOW)
    arrow(d, (840, y + 195), (950, y + 195), PURPLE)
    stage(d, (965, y + 132), (275, 140), "MLP", "(x, view dir) -> density sigma + RGB", PURPLE)
    arrow(d, (1255, y + 200), (1360, y + 200), PURPLE)
    stage(d, (1372, y + 132), (310, 140), "integrate", "front-to-back alpha accumulation produces one pixel color", PURPLE)


def draw_3dgs(canvas, y):
    d = ImageDraw.Draw(canvas)
    row_header(d, 46, y, ORANGE, "3. 3DGS: sparse seed points become optimized splats", "A COLMAP point cloud is densified into anisotropic 3D Gaussians. Each Gaussian projects to a 2D ellipse and alpha-composites in tiles.")
    panel = (46, y + 100, 1754, y + 315)
    box(d, panel, fill=PANEL, outline=EDGE, width=2)
    point_cloud(d, 100, y + 130, 230, 150, seed=8, color="#d8dee9", count=42)
    stage(d, (360, y + 130), (225, 135), "seed", "COLMAP points initialize splat centers", ORANGE)
    arrow(d, (600, y + 195), (705, y + 195), ORANGE)
    rng = np.random.default_rng(9)
    for _ in range(42):
        x = rng.uniform(735, 1040)
        yy = rng.uniform(y + 125, y + 280)
        rx = rng.uniform(18, 46)
        ry = rng.uniform(8, 22)
        c = rng.choice([BLUE, GREEN, PURPLE, ORANGE, "#c9d1d9"])
        d.ellipse((x - rx, yy - ry, x + rx, yy + ry), fill=c, outline=None)
    text(d, (720, y + 284), "optimized 3D ellipsoids", 16, MUTED)
    arrow(d, (1085, y + 195), (1190, y + 195), ORANGE)
    box(d, (1210, y + 120, 1360, y + 282), fill="#11161f", outline=ORANGE, width=3)
    for i in range(7):
        yy = y + 135 + i * 20
        d.ellipse((1242, yy, 1328, yy + 16), fill=rng.choice([BLUE, GREEN, PURPLE, "#c9d1d9"]))
    stage(d, (1410, y + 130), (285, 135), "rasterize", "project -> sort/tile -> alpha blend for real-time views", ORANGE)


def draw_aura(canvas, y):
    d = ImageDraw.Draw(canvas)
    row_header(d, 46, y, GREEN, "4. AURA: a radiance field becomes a typed asset", "AURA keeps gsplat/DBS-Beta as primary quality paths, adds PRISM extension footprints, and attaches queryable asset metadata.")
    panel = (46, y + 100, 1754, y + 330)
    box(d, panel, fill=PANEL, outline=EDGE, width=2)
    stage(d, (82, y + 135), (255, 140), "inputs", "capture manifest, COLMAP poses, trained gsplat / DBS-Beta carriers", GREEN)
    arrow(d, (352, y + 205), (450, y + 205), GREEN)
    stage(d, (462, y + 118), (330, 174), "typed carriers", "Gaussian and Beta remain primary; Gabor/neural are PRISM extensions", GREEN)
    for i, (label, color) in enumerate((("G", BLUE), ("B", GREEN), ("Ga", YELLOW), ("N", PURPLE))):
        x = 502 + i * 62
        d.ellipse((x, y + 222, x + 42, y + 264), fill=color)
        text(d, (x + 21, y + 233), label, 13, BG, True, anchor="mm")
    arrow(d, (807, y + 205), (905, y + 205), GREEN)
    stage(d, (918, y + 118), (350, 174), "payload", "color, opacity, normal, material, semantic label, confidence, footprint type", GREEN)
    arrow(d, (1284, y + 205), (1382, y + 205), GREEN)
    stage(d, (1395, y + 118), (300, 174), "asset API", "render, ray query, depth, relight, confidence, semantics, GLB/USD export", GREEN)


def main():
    DOCS.mkdir(exist_ok=True)
    thumb = image_tile(ROOT / "data/tanks/truck/images/000001.jpg", (260, 146))
    canvas = Image.new("RGB", (1800, 1680), BG)
    d = ImageDraw.Draw(canvas)
    text(d, (54, 36), "How a capture becomes a scene representation", 48, FG, True)
    wrapped(d, (58, 94), "Same posed photos, different construction targets: sparse geometry, neural volume, optimized splats, or a typed AURA asset.", 1600, 22, MUTED)
    draw_colmap(canvas, 150, thumb)
    draw_nerf(canvas, 535)
    draw_3dgs(canvas, 920)
    draw_aura(canvas, 1305)
    out = DOCS / "how_it_works.png"
    canvas.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
