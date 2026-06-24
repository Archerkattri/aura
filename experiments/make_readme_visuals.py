#!/usr/bin/env python3
"""Generate README visuals that explain what AURA and PRISM do today.

Inputs are existing local images and benchmark JSON files. No training, no CUDA.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BG = "#101318"
PANEL = "#181d25"
EDGE = "#2c3442"
FG = "#f2f4f8"
MUTED = "#a7b0bd"
GREEN = "#3fb950"
BLUE = "#58a6ff"
YELLOW = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"


def font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def text(draw, xy, value, size=24, fill=FG, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def wrapped_text(draw, xy, value, max_width, size=24, fill=FG, bold=False, line_gap=8):
    words = value.split()
    lines: list[str] = []
    current = ""
    active_font = font(size, bold)
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=active_font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for i, line in enumerate(lines):
        draw.text((x, y + i * (size + line_gap)), line, font=active_font, fill=fill)
    return y + len(lines) * (size + line_gap)


def round_rect(draw, xy, radius=16, fill=PANEL, outline=EDGE, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def fit_media_frame(img: Image.Image, size: tuple[int, int], *, mode: str = "cover", fill: str = "#06080c") -> Image.Image:
    """Fit an image into a fixed README panel.

    `cover` fills the whole panel by cropping after resize. `contain` preserves
    the whole source and pads the remainder. Both modes upscale small GIF frames,
    which prevents README panels from showing tiny postage-stamp animations.
    """
    img = img.convert("RGB")
    w, h = img.size
    tw, th = size
    if w <= 0 or h <= 0:
        raise ValueError("media frame has invalid dimensions")
    if mode not in {"cover", "contain"}:
        raise ValueError(f"unknown fit mode: {mode}")
    s = max(tw / w, th / h) if mode == "cover" else min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if mode == "contain":
        canvas = Image.new("RGB", (tw, th), fill)
        canvas.paste(img, ((tw - nw) // 2, (th - nh) // 2))
        return canvas
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    return img.crop((x0, y0, x0 + tw, y0 + th))


def cover(path: Path, size: tuple[int, int]) -> Image.Image:
    return fit_media_frame(Image.open(path), size, mode="cover")


def first_gif_frame(path: Path) -> Image.Image:
    frames = imageio.mimread(path, memtest=False)
    if frames:
        return Image.fromarray(frames[0]).convert("RGB")
    return Image.open(path).convert("RGB")


def normalize_readme_gifs(target_width: int = 979, max_frames: int = 96) -> list[Path]:
    """Normalize GIF assets without dropping below the source Truck resolution."""
    outputs: list[Path] = []
    for path in (DOCS / "truck_orbit.gif", DOCS / "truck_depth_orbit.gif", DOCS / "relight_sweep.gif"):
        if not path.exists():
            continue
        img = Image.open(path)
        if img.width >= target_width:
            target = img.size
        else:
            scale = target_width / img.width
            target = (target_width, max(1, int(round(img.height * scale))))
        total = getattr(img, "n_frames", 1)
        step = max(1, int(np.ceil(total / max_frames)))
        frames = []
        durations = []
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if i % step:
                continue
            frames.append(frame.convert("RGB").resize(target, Image.Resampling.LANCZOS))
            durations.append(frame.info.get("duration", img.info.get("duration", 80)) * step)
        if frames:
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
            )
            outputs.append(path)
    return outputs


def first_image(scene: str) -> Path:
    if scene == "truck":
        candidates = [ROOT / "data/tanks/truck/images"]
    else:
        candidates = [
            ROOT / f"data/mipnerf360/{scene}/images_4",
            ROOT / f"data/mipnerf360/{scene}/images_2",
            ROOT / f"data/mipnerf360/{scene}/images",
        ]
    for directory in candidates:
        if directory.exists():
            files = sorted([p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            if files:
                return files[len(files) // 2]
    raise FileNotFoundError(scene)


def load_multiscene():
    return json.loads((ROOT / "experiments/results/multiscene.json").read_text())["scenes"]


def dataset_scene_grid():
    rows = load_multiscene()
    rows = sorted(rows, key=lambda r: r["scene"])
    card_w, card_h = 390, 312
    gap = 22
    margin = 34
    cols = 4
    W = margin * 2 + cols * card_w + (cols - 1) * gap
    H = margin * 2 + 86 + 2 * card_h + gap
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    text(d, (margin, 24), "Benchmark scenes: every local scene is tested", 34, bold=True)
    text(d, (margin, 64), "Tanks & Temples Truck + all 7 extracted Mip-NeRF 360 scene roots", 18, MUTED)
    for i, r in enumerate(rows):
        x = margin + (i % cols) * (card_w + gap)
        y = margin + 86 + (i // cols) * (card_h + gap)
        round_rect(d, (x, y, x + card_w, y + card_h), radius=14)
        thumb = cover(first_image(r["scene"]), (card_w - 24, 178))
        canvas.paste(thumb, (x + 12, y + 12))
        text(d, (x + 18, y + 205), r["scene"], 24, bold=True)
        text(d, (x + 18, y + 240), f"Beta {r['beta_psnr']:.2f}  vs  Gaussian {r['gauss_psnr']:.2f}", 17, MUTED)
        pill = (x + 18, y + 270, x + 150, y + 298)
        round_rect(d, pill, radius=14, fill="#10291a", outline=GREEN, width=2)
        text(d, (x + 84, y + 273), f"+{r['delta_psnr']:.2f} dB", 17, GREEN, bold=True, anchor="ma")
    out = DOCS / "dataset_scene_grid.png"
    canvas.save(out)
    return out


def prism_extension_diagram():
    W, H = 1800, 980
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    text(d, (60, 44), "AURA rendering stack: PRISM adds to gsplat/Beta", 42, bold=True)
    text(d, (60, 96), "Production path keeps quality renderers primary; PRISM only contributes extension footprints.", 22, MUTED)

    # Inputs
    round_rect(d, (70, 190, 410, 810), radius=22, fill="#151b23", outline=EDGE)
    text(d, (95, 220), "typed carriers", 30, bold=True)
    carriers = [
        ("Gaussian", BLUE, "quality path"),
        ("Beta", GREEN, "DBS/Beta quality"),
        ("Gabor", YELLOW, "PRISM extension"),
        ("Neural", PURPLE, "PRISM extension"),
    ]
    for i, (name, color, note) in enumerate(carriers):
        y = 292 + i * 110
        d.ellipse((105, y, 165, y + 60), fill=color)
        text(d, (190, y + 4), name, 27, bold=True)
        text(d, (190, y + 40), note, 18, MUTED)

    # Backends
    boxes = [
        (570, 215, 1010, 365, "gsplat", "Gaussian rasterization", BLUE),
        (570, 415, 1010, 565, "DBS / Beta", "high-quality Beta carriers", GREEN),
        (570, 635, 1010, 785, "PRISM", "Gabor / neural extensions only", YELLOW),
    ]
    for x0, y0, x1, y1, title, sub, color in boxes:
        round_rect(d, (x0, y0, x1, y1), radius=22, fill="#1d2430", outline=color, width=4)
        text(d, (x0 + 28, y0 + 30), title, 34, color, bold=True)
        text(d, (x0 + 28, y0 + 82), sub, 22, FG)

    # Arrows
    def line(a, b, color):
        d.line((a[0], a[1], b[0], b[1]), fill=color, width=7)
        d.polygon([(b[0], b[1]), (b[0] - 20, b[1] - 12), (b[0] - 20, b[1] + 12)], fill=color)

    line((410, 325), (570, 290), BLUE)
    line((410, 435), (570, 490), GREEN)
    line((410, 545), (570, 705), YELLOW)
    line((410, 655), (570, 725), PURPLE)

    # Composite + contract
    round_rect(d, (1160, 320, 1700, 515), radius=24, fill="#14231a", outline=GREEN, width=4)
    text(d, (1190, 356), "depth-composited image", 34, GREEN, bold=True)
    text(d, (1190, 412), "primary quality layer + PRISM extension layer", 23, FG)
    line((1010, 292), (1160, 380), BLUE)
    line((1010, 490), (1160, 420), GREEN)
    line((1010, 705), (1160, 470), YELLOW)

    round_rect(d, (1160, 600, 1700, 805), radius=24, fill="#1e1b2b", outline=PURPLE, width=4)
    text(d, (1190, 636), "asset contract", 34, PURPLE, bold=True)
    text(d, (1190, 692), "KHR export  |  ray query  |  relight", 23, FG)
    text(d, (1190, 730), "confidence  |  semantics  |  USD bridge", 23, FG)

    text(d, (70, 890), "Key rule: Beta does not default to PRISM. Beta stays in the DBS/Beta quality path; PRISM is additive.", 24, FG, bold=True)
    out = DOCS / "prism_extension_stack.png"
    canvas.save(out)
    return out


def prism_footprints():
    xs = np.linspace(-3, 3, 180)
    ys = np.linspace(-3, 3, 180)
    X, Y = np.meshgrid(xs, ys)
    R2 = X * X + Y * Y
    gaussian = np.exp(-0.5 * R2)
    beta = np.clip(1 - np.sqrt(R2) / 3, 0, 1) ** 2
    gabor = gaussian * (0.5 + 0.5 * np.cos(9 * X))
    neural = np.exp(-0.35 * R2) * (0.55 + 0.45 * np.sin(5 * X + 3 * Y) ** 2)
    maps = [
        ("Gaussian", gaussian, "primary: gsplat"),
        ("Beta", beta, "primary: DBS/Beta"),
        ("Gabor", gabor, "extension: PRISM"),
        ("Neural", neural, "extension: PRISM"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor=BG)
    for ax, (title, Z, note) in zip(axes, maps):
        ax.imshow(Z, cmap="viridis", origin="lower")
        ax.set_title(title, color=FG, fontsize=20, fontweight="bold")
        ax.text(0.5, -0.08, note, transform=ax.transAxes, ha="center", va="top", color=MUTED, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(EDGE); spine.set_linewidth(2)
    fig.suptitle("Carrier footprint families: quality backends first, PRISM for extensions", color=FG, fontsize=22, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = DOCS / "prism_footprints.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    return out


def capability_board():
    W, H = 1750, 980
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    text(d, (60, 44), "What works now", 42, bold=True)
    text(d, (60, 95), "Evidence-backed AURA + PRISM capability board", 22, MUTED)
    items = [
        ("Quality", "Beta beats Gaussian on 8/8 local scenes", "+0.80 dB mean", GREEN),
        ("Compactness", "Beta@500k beats Gaussian@1M on Truck", "about 2x fewer carriers", GREEN),
        ("Export", "KHR Gaussian Splatting GLB + USD bridge", "engine/DCC ready", BLUE),
        ("Relight", "PPM preview + light-orbit GIF", "editable layer", BLUE),
        ("Confidence", "multi-view support per carrier", "floaters visible", BLUE),
        ("Semantics", "DINOv2 carrier features + CLIP query", "open-vocab wheel query", BLUE),
        ("Ray query", "color/depth/normal/confidence/semantic payload", "carrier_query", BLUE),
        ("PRISM", "Gabor/neural extension layer over gsplat/Beta", "not an alternative", YELLOW),
        ("Validation", "8/8 local publication gates pass", "claim boundary explicit", GREEN),
    ]
    cols = 3
    card_w, card_h = 515, 230
    gap = 32
    x0, y0 = 60, 160
    for i, (title, body, proof, color) in enumerate(items):
        x = x0 + (i % cols) * (card_w + gap)
        y = y0 + (i // cols) * (card_h + gap)
        round_rect(d, (x, y, x + card_w, y + card_h), radius=22, fill=PANEL, outline=color, width=4)
        d.ellipse((x + 30, y + 30, x + 78, y + 78), fill=color)
        text(d, (x + 100, y + 28), title, 30, color, bold=True)
        wrapped_text(d, (x + 30, y + 102), body, card_w - 60, 22, FG)
        text(d, (x + 30, y + 158), proof, 20, MUTED)
    out = DOCS / "capability_board.png"
    canvas.save(out)
    return out


def capability_reel():
    sources = [
        ("reconstruction", DOCS / "truck_orbit.gif"),
        ("depth query", DOCS / "truck_depth_orbit.gif"),
        ("relighting", DOCS / "relight_sweep.gif"),
        ("confidence", DOCS / "confidence_truck.png"),
        ("semantics", DOCS / "semantic_distill_truck.png"),
        ("open-vocab query", DOCS / "semantic_query_truck.png"),
    ]
    W, H = 900, 560
    frames = []
    for title, path in sources:
        img = first_gif_frame(path) if path.suffix == ".gif" else Image.open(path).convert("RGB")
        frame = fit_media_frame(img, (W, H), mode="cover")
        d = ImageDraw.Draw(frame)
        d.rectangle((0, 0, W, 74), fill=(6, 8, 12))
        text(d, (30, 20), f"AURA can do: {title}", 34, bold=True)
        frames.extend([frame] * 10)
    out = DOCS / "aura_capability_reel.gif"
    imageio.mimsave(out, [np.asarray(f) for f in frames], fps=10, loop=0)
    return out


def main():
    DOCS.mkdir(exist_ok=True)
    normalized = normalize_readme_gifs()
    outputs = [
        dataset_scene_grid(),
        prism_extension_diagram(),
        prism_footprints(),
        capability_board(),
        capability_reel(),
    ]
    for out in normalized + outputs:
        print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
