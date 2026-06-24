#!/usr/bin/env python3
"""Beta-vs-Gaussian comparison figure on a held-out Truck test view.

Runs INSIDE .dbs_venv. Loads the two ablation arms' best models, renders the
SAME held-out (llffhold=8) test camera through each, and writes a side-by-side
GT | frozen-Gaussian | deformable-Beta panel plus a zoom crop, so the
+0.335 dB / lower-LPIPS win is visible, not just tabulated.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/dbs")
import numpy as np
import torch
from PIL import Image, ImageDraw

from argparse import Namespace
from scene import Scene, BetaModel


def load_arm(source, model_path, sb_number, beta_frozen):
    args = Namespace(
        sh_degree=0, sb_number=sb_number, source_path=source, model_path=model_path,
        images="images", resolution=-1, white_background=False, data_device="cuda",
        eval=True, cap_max=1000000, init_type="sfm",
    )
    bm = BetaModel(args.sh_degree, args.sb_number)
    scene = Scene(args, bm, load_iteration=-1, shuffle=False)
    return scene, bm


def to_img(t):
    a = (t.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
    if a.shape[0] in (3, 4):
        a = np.transpose(a[:3], (1, 2, 0))
    return Image.fromarray(a, "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent.parent / "data/tanks/truck"))
    ap.add_argument("--gauss", default="/tmp/dbs_out/truck_gauss")
    ap.add_argument("--beta", default="/tmp/dbs_out/truck_beta")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/beta_vs_gauss_truck.png"))
    ap.add_argument("--view", type=int, default=0)
    a = ap.parse_args()

    sg, bmg = load_arm(a.source, a.gauss, 0, True)
    sb, bmb = load_arm(a.source, a.beta, 2, False)
    cams = sg.getTestCameras()
    cam = cams[a.view % len(cams)]
    cam_b = sb.getTestCameras()[a.view % len(cams)]

    # the sb_number=0 arm initialises an empty background; force black [3] on both
    bmg.background = torch.zeros(3, device="cuda")
    bmb.background = torch.zeros(3, device="cuda")
    with torch.no_grad():
        gauss = bmg.render(cam)["render"]   # [C,H,W]
        beta = bmb.render(cam_b)["render"]
    gt = cam.original_image[:3].cuda()

    from PIL import ImageFont

    def _font(px):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(p, px)
            except Exception:
                continue
        return ImageFont.load_default()

    cols = [("Ground truth", gt), ("fixed Gaussian — 26.02 dB", gauss),
            ("adaptive Beta — 26.35 dB", beta)]
    ims = [to_img(t) for _, t in cols]
    w, h = ims[0].size
    cx, cy, cw = int(w * 0.55), int(h * 0.45), w // 4
    cs = w // 2  # crop display size
    crops = [im.crop((cx, cy, cx + cw, cy + cw)).resize((cs, cs), Image.NEAREST) for im in ims]

    big = _font(max(15, w // 26)); small = _font(max(12, w // 34))
    hdr = big.size + 10                 # title band above the full frames
    cap = small.size + 8                # caption band between rows
    crow_h = cs + small.size + 10       # crop image + its label
    grid = Image.new("RGB", (w * 3, hdr + h + cap + crow_h), (16, 16, 16))
    d = ImageDraw.Draw(grid)

    def _ctext(x_centre, y, text, font, fill=(240, 240, 240)):
        tw = d.textlength(text, font=font)
        d.text((x_centre - tw / 2, y), text, font=font, fill=fill)

    for j, (lab, _) in enumerate(cols):
        _ctext(j * w + w / 2, 6, lab, big)
        grid.paste(ims[j], (j * w, hdr))
    _ctext(w * 1.5, hdr + h + 4, "↓ zoom-in on the same detail region (nearest-neighbour, no smoothing) ↓", small, (170, 170, 170))
    cy0 = hdr + h + cap
    for j, (lab, _) in enumerate(cols):
        grid.paste(crops[j], (j * w + (w - cs) // 2, cy0))
        _ctext(j * w + w / 2, cy0 + cs + 3, lab.split(" — ")[0] + " (detail)", small)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(a.out)
    print(f"wrote {a.out} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
