#!/usr/bin/env python3
"""Semantic grouping visualization (runs in .dbs_venv).

The asset contract reserves a per-carrier `semantic_id`; the query already returns
it. Here we populate it without external labels as an honest first step: cluster
carriers in (position + colour) space into K groups (a spatial-semantic scaffold),
give each group a distinct colour, and render — a segmentation-style view of the
scene. With LangSplat-style features this slot upgrades to real open-vocabulary
semantics; the visualization and the query path are identical.
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
from sklearn.cluster import MiniBatchKMeans
from gsplat.rendering import rasterization
from scene import Scene, BetaModel

_C0 = 0.28209479177387814
_PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.60, 0.90], [0.20, 0.80, 0.30], [0.95, 0.75, 0.10],
    [0.65, 0.25, 0.85], [0.95, 0.45, 0.10], [0.15, 0.85, 0.80], [0.85, 0.20, 0.55],
    [0.50, 0.55, 0.20], [0.30, 0.35, 0.85], [0.75, 0.85, 0.25], [0.55, 0.30, 0.20],
], dtype="float32")


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
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/semantic_truck.png"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--view", type=int, default=20)
    a = ap.parse_args()

    sc, bm = load(a.source, a.model, a.sb_number)
    means = bm.get_xyz.detach(); quats = bm.get_rotation.detach(); scales = bm.get_scaling.detach()
    opac = bm.get_opacity.squeeze(-1).detach(); betas = bm.get_beta.squeeze().detach()
    albedo = (0.5 + _C0 * bm._sh0.detach().squeeze(1)).clamp(0, 1)

    # cluster in normalised (xyz, rgb) space
    xyz = means.cpu().numpy(); rgb = albedo.cpu().numpy()
    xyz = (xyz - xyz.mean(0)) / (xyz.std(0) + 1e-6)
    feat = np.concatenate([xyz, rgb * 2.0], 1).astype("float32")
    labels = MiniBatchKMeans(n_clusters=a.k, random_state=0, n_init=3, batch_size=10000).fit_predict(feat)
    print("semantic groups: k=%d sizes=%s" % (a.k, np.bincount(labels).tolist()), flush=True)
    colors = torch.tensor(_PALETTE[labels % len(_PALETTE)], device="cuda")

    cam = sorted(sc.getTrainCameras(), key=lambda c: getattr(c, "image_name", "0"))[a.view]
    K = K_of(cam); vm = cam.world_view_transform.transpose(0, 1).unsqueeze(0)
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
