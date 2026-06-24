#!/usr/bin/env python3
"""Per-carrier confidence heatmap (runs in .dbs_venv).

Colours every carrier by its multi-view observation support (how many training
cameras see it) — red = speculative (seen by few views), green = well-observed —
and renders it through the fork. Visualises the confidence axis of the asset
contract: the floaters AURA is unsure about light up in red.
"""
import argparse
import math
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "/tmp/dbs")
import numpy as np
import torch
import imageio.v2 as imageio
from gsplat.rendering import rasterization
from scene import Scene, BetaModel


def load(source, model_path, sb_number):
    args = Namespace(sh_degree=0, sb_number=sb_number, source_path=source, model_path=model_path,
                     images="images", resolution=-1, white_background=False, data_device="cuda",
                     eval=True, cap_max=1000000, init_type="sfm")
    bm = BetaModel(0, sb_number); sc = Scene(args, bm, load_iteration=-1, shuffle=False)
    return sc, bm


def K_of(cam):
    K = torch.zeros((3, 3), device="cuda")
    K[0, 0] = 0.5 * cam.image_width / math.tan(cam.FoVx / 2)
    K[1, 1] = 0.5 * cam.image_height / math.tan(cam.FoVy / 2)
    K[0, 2] = cam.image_width / 2; K[1, 2] = cam.image_height / 2; K[2, 2] = 1
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent.parent / "data/tanks/truck"))
    ap.add_argument("--model", default="/tmp/dbs_out/truck_beta")
    ap.add_argument("--sb-number", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/confidence_truck.png"))
    ap.add_argument("--view", type=int, default=20)
    ap.add_argument("--saturate", type=float, default=40.0)
    a = ap.parse_args()

    sc, bm = load(a.source, a.model, a.sb_number)
    cams = sorted(sc.getTrainCameras(), key=lambda c: getattr(c, "image_name", "0"))
    means = bm.get_xyz.detach()
    quats = bm.get_rotation.detach(); scales = bm.get_scaling.detach()
    opac = bm.get_opacity.squeeze(-1).detach(); betas = bm.get_beta.squeeze().detach()
    N = means.shape[0]

    # multi-view confidence
    counts = torch.zeros(N, device="cuda")
    homog = torch.cat([means, torch.ones(N, 1, device="cuda")], 1)
    for c in cams:
        K = K_of(c); W2C = c.world_view_transform.transpose(0, 1)
        cam_xyz = (homog @ W2C.T)[:, :3]; z = cam_xyz[:, 2]
        u = K[0, 0] * cam_xyz[:, 0] / z.clamp(min=1e-4) + K[0, 2]
        v = K[1, 1] * cam_xyz[:, 1] / z.clamp(min=1e-4) + K[1, 2]
        inv = (z > 1e-4) & (u >= 0) & (u < c.image_width) & (v >= 0) & (v < c.image_height)
        counts += inv.float()
    conf = 1.0 - torch.exp(-counts / a.saturate)
    print("confidence: min %.3f med %.3f max %.3f  frac<0.3 %.4f"
          % (conf.min(), conf.median(), conf.max(), float((conf < 0.3).float().mean())), flush=True)

    # red (low) -> yellow -> green (high)
    r = (1 - conf).clamp(0, 1); g = conf.clamp(0, 1); b = torch.zeros_like(conf)
    colors = torch.stack([r, g, b], -1)

    cam = cams[a.view]; K = K_of(cam)
    vm = cam.world_view_transform.transpose(0, 1).unsqueeze(0)
    W, H = int(cam.image_width), int(cam.image_height)
    with torch.no_grad():
        out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                                  betas=betas, colors=colors, viewmats=vm, Ks=K.unsqueeze(0),
                                  width=W, height=H, sb_number=None, sb_params=None, sh_degree=None,
                                  backgrounds=torch.zeros(1, 3, device="cuda"))
    arr = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(a.out, arr)
    print(f"wrote {a.out} ({W}x{H})")


if __name__ == "__main__":
    main()
