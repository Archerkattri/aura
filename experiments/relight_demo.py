#!/usr/bin/env python3
"""Relighting demo on REAL trained carriers — proves the relightable-asset
capability over a gsplat/DBS-trained scene (not a toy). Renders one view three
ways: flat albedo, lit from the left, lit from the right. The scene responds to
light direction → it is genuinely relightable, unlike vanilla 3DGS.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import json
import numpy as np
from PIL import Image, ImageDraw

from aura.carrier_io import load_carriers
from aura.relight import render_relit, relight_colors
from aura.shading import DirectionalLight


def _img(flat, w, h):
    a = (np.array(flat, "float32").reshape(h, w, 3).clip(0, 1) * 255).astype("uint8")
    return Image.fromarray(a, "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carriers", default="/tmp/dbs_out/truck_smoke/carriers.npz")
    ap.add_argument("--manifest", default=str(ROOT / "outputs/truck-pts129k-manifest.json"))
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "docs/relight_truck.png"))
    a = ap.parse_args()

    c = load_carriers(a.carriers, device="cuda")
    manifest = json.loads(Path(a.manifest).read_text())
    fr = manifest["frames"][a.view % len(manifest["frames"])]

    left = DirectionalLight(direction=(1.0, 0.3, 0.4), color=(1.0, 0.96, 0.9), intensity=1.6)
    right = DirectionalLight(direction=(-1.0, 0.3, 0.4), color=(0.9, 0.95, 1.0), intensity=1.6)

    panels = []
    # flat albedo: ambient 1.0, no directional light
    w, h, alb = render_relit(c, fr, a.scale, [], ambient=1.0)
    panels.append(("albedo (flat)", _img(alb, w, h)))
    w, h, l = render_relit(c, fr, a.scale, [left], ambient=0.15)
    panels.append(("lit from left", _img(l, w, h)))
    w, h, r = render_relit(c, fr, a.scale, [right], ambient=0.15)
    panels.append(("lit from right", _img(r, w, h)))

    hdr = 22
    grid = Image.new("RGB", (w * 3, hdr + h), (18, 18, 18))
    d = ImageDraw.Draw(grid)
    for j, (lab, im) in enumerate(panels):
        d.text((j * w + 6, 5), lab, fill=(240, 240, 240))
        grid.paste(im, (j * w, hdr))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(a.out)
    print(f"wrote {a.out} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
