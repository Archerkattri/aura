#!/usr/bin/env python3
"""Feature-distilled semantics by multi-view DINOv2 lifting (runs in .dbs_venv).

Upgrades the naive position+colour clustering to real semantic features: project
each carrier into many training views, sample dense DINOv2 patch features at the
projected pixel, aggregate per carrier (visibility-weighted), L2-normalise. The
result is a per-carrier semantic descriptor that respects object boundaries.
KMeans over it gives a coherent segmentation; the features are stored alongside the
carriers for the text-query step (`semantic_query.py`).
"""
import argparse
import math
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "/tmp/dbs")
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from sklearn.cluster import MiniBatchKMeans
from gsplat.rendering import rasterization
from scene import Scene, BetaModel

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.55, 0.90], [0.20, 0.80, 0.30], [0.95, 0.75, 0.10],
    [0.65, 0.25, 0.85], [0.95, 0.45, 0.10], [0.15, 0.85, 0.80], [0.85, 0.20, 0.55],
    [0.50, 0.55, 0.20], [0.35, 0.40, 0.95], [0.75, 0.85, 0.25], [0.55, 0.30, 0.20],
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


@torch.no_grad()
def dino_feature_map(dino, img_chw, patch=14, long_side=518):
    """img_chw [3,H,W] in [0,1] -> (feat [hp,wp,D], (hp,wp))."""
    _, H, W = img_chw.shape
    scale = long_side / max(H, W)
    h = int(round(H * scale / patch)) * patch; w = int(round(W * scale / patch)) * patch
    x = F.interpolate(img_chw.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
    x = (x - _IMAGENET_MEAN.to(x)) / _IMAGENET_STD.to(x)
    out = dino.forward_features(x)["x_norm_patchtokens"][0]   # [hp*wp, D]
    hp, wp = h // patch, w // patch
    return out.reshape(hp, wp, -1), (hp, wp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent.parent / "data/tanks/truck"))
    ap.add_argument("--model", default="/tmp/dbs_out/truck_beta")
    ap.add_argument("--sb-number", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/semantic_distill_truck.png"))
    ap.add_argument("--feat-out", default="/tmp/dbs_out/truck_beta/carrier_features.npz")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--view-stride", type=int, default=4)
    ap.add_argument("--render-view", type=int, default=20)
    a = ap.parse_args()

    sc, bm = load(a.source, a.model, a.sb_number)
    means = bm.get_xyz.detach(); quats = bm.get_rotation.detach(); scales = bm.get_scaling.detach()
    opac = bm.get_opacity.squeeze(-1).detach(); betas = bm.get_beta.squeeze().detach()
    N = means.shape[0]
    cams = sorted(sc.getTrainCameras(), key=lambda c: getattr(c, "image_name", "0"))

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).cuda().eval()
    D = 384
    feat_sum = torch.zeros(N, D, device="cuda"); cnt = torch.zeros(N, device="cuda")
    homog = torch.cat([means, torch.ones(N, 1, device="cuda")], 1)

    used = cams[::a.view_stride]
    for i, c in enumerate(used):
        fmap, (hp, wp) = dino_feature_map(dino, c.original_image[:3].cuda())
        K = K_of(c); W2C = c.world_view_transform.transpose(0, 1)
        cam_xyz = (homog @ W2C.T)[:, :3]; z = cam_xyz[:, 2]
        u = K[0, 0] * cam_xyz[:, 0] / z.clamp(min=1e-4) + K[0, 2]
        v = K[1, 1] * cam_xyz[:, 1] / z.clamp(min=1e-4) + K[1, 2]
        inv = (z > 1e-4) & (u >= 0) & (u < c.image_width) & (v >= 0) & (v < c.image_height)
        pu = (u / c.image_width * (wp - 1)).clamp(0, wp - 1).long()
        pv = (v / c.image_height * (hp - 1)).clamp(0, hp - 1).long()
        idx = torch.nonzero(inv, as_tuple=False).squeeze(-1)
        feat_sum[idx] += fmap[pv[idx], pu[idx]]
        cnt[idx] += 1.0
        if i % 10 == 0:
            print(f"  view {i+1}/{len(used)}", flush=True)

    feat = feat_sum / cnt.clamp(min=1).unsqueeze(-1)
    feat = F.normalize(feat, dim=-1)
    seen = cnt > 0
    print(f"distilled DINO features for {int(seen.sum())}/{N} carriers", flush=True)

    labels = np.full(N, -1)
    km = MiniBatchKMeans(n_clusters=a.k, random_state=0, n_init=3, batch_size=10000)
    labels[seen.cpu().numpy()] = km.fit_predict(feat[seen].cpu().numpy())
    np.savez(a.feat_out, features=feat.cpu().numpy(), labels=labels)
    print(f"sizes={np.bincount(labels[labels>=0]).tolist()}  saved {a.feat_out}", flush=True)

    colors = torch.tensor(_PALETTE[np.where(labels >= 0, labels % len(_PALETTE), 0)], device="cuda")
    colors[~seen] = 0.0
    cam = cams[a.render_view]; K = K_of(cam); vm = cam.world_view_transform.transpose(0, 1).unsqueeze(0)
    W, H = int(cam.image_width), int(cam.image_height)
    with torch.no_grad():
        out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, betas=betas,
                                  colors=colors, viewmats=vm, Ks=K.unsqueeze(0), width=W, height=H,
                                  sb_number=None, sb_params=None, sh_degree=None,
                                  backgrounds=torch.zeros(1, 3, device="cuda"))
    imageio.imwrite(a.out, (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype("uint8"))
    print(f"wrote {a.out} ({W}x{H})")


if __name__ == "__main__":
    main()
