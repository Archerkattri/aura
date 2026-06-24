#!/usr/bin/env python3
"""Clean relighting GIF — sweep a light around the trained scene, render sharply
through the DBS fork (runs in .dbs_venv).

Earlier relight renders were hazy because Beta carriers were rasterized as plain
Gaussians by *standard* gsplat. Here we relight per carrier (normal = Gaussian
short axis, albedo = SH DC colour, Lambertian + ambient) and rasterize through the
FORK with the real Beta kernels, so the result is sharp. A directional light
orbits the scene → moving shading = a genuinely relightable asset.
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

_C0 = 0.28209479177387814


def load(source, model_path, sb_number):
    args = Namespace(sh_degree=0, sb_number=sb_number, source_path=source, model_path=model_path,
                     images="images", resolution=-1, white_background=False, data_device="cuda",
                     eval=True, cap_max=1000000, init_type="sfm")
    bm = BetaModel(0, sb_number); sc = Scene(args, bm, load_iteration=-1, shuffle=False)
    return sc, bm


def normals_from(quats, scales):
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)
    short = torch.argmin(scales, dim=-1)
    n = R.gather(2, short.reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
    return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent.parent / "data/tanks/truck"))
    ap.add_argument("--model", default="/tmp/dbs_out/truck_gauss")   # SH colour = clean albedo
    ap.add_argument("--sb-number", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/relight_sweep.gif"))
    ap.add_argument("--view", type=int, default=20)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--downscale", type=int, default=1)
    a = ap.parse_args()

    sc, bm = load(a.source, a.model, a.sb_number)
    cam = sorted(sc.getTrainCameras(), key=lambda c: getattr(c, "image_name", "0"))[a.view]
    means = bm.get_xyz.detach(); quats = bm.get_rotation.detach(); scales = bm.get_scaling.detach()
    opac = bm.get_opacity.squeeze(-1).detach(); betas = bm.get_beta.squeeze().detach()
    albedo = (0.5 + _C0 * bm._sh0.detach().squeeze(1)).clamp(0, 1)     # [N,3] diffuse
    n = normals_from(quats, scales)

    K = torch.zeros((3, 3), device="cuda")
    fx = 0.5 * cam.image_width / math.tan(cam.FoVx / 2); fy = 0.5 * cam.image_height / math.tan(cam.FoVy / 2)
    K[0, 0] = fx; K[1, 1] = fy; K[0, 2] = cam.image_width / 2; K[1, 2] = cam.image_height / 2; K[2, 2] = 1
    vm = cam.world_view_transform.transpose(0, 1).unsqueeze(0)
    W, H = int(cam.image_width), int(cam.image_height)

    frames = []
    for i in range(a.n):
        th = 2 * math.pi * i / a.n
        L = torch.tensor([math.cos(th), 0.35, math.sin(th)], device="cuda"); L = L / L.norm()
        shade = (n * L).sum(-1).abs().clamp(0, 1)                       # unsigned normals
        lit = (0.18 * albedo + 0.95 * albedo * shade.unsqueeze(-1)).clamp(0, 1)
        with torch.no_grad():
            out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                                      betas=betas, colors=lit, viewmats=vm, Ks=K.unsqueeze(0),
                                      width=W, height=H, sb_number=None, sb_params=None, sh_degree=None,
                                      backgrounds=torch.zeros(1, 3, device="cuda"))
        arr = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        frames.append(arr[::a.downscale, ::a.downscale])
    frames += frames[::-1]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=18, loop=0)
    print(f"wrote {a.out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


if __name__ == "__main__":
    main()
