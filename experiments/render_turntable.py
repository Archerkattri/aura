#!/usr/bin/env python3
"""Smooth fly-through by INTERPOLATING the real camera trajectory (runs in .dbs_venv).

Real T&T-Truck frames are sparse photos (~19% image change between adjacent
captures) so replaying them looks chopped. Here we keep the real capture path
(good framing, always inside the observed volume) but insert interpolated camera
poses between consecutive real views — slerp the rotation, lerp the centre — and
render every in-between frame, so motion is smooth.
"""
import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, "/tmp/dbs")
import numpy as np
import torch
import imageio.v2 as imageio
from argparse import Namespace
from scene import Scene, BetaModel


def load(source, model_path, sb_number):
    args = Namespace(sh_degree=0, sb_number=sb_number, source_path=source, model_path=model_path,
                     images="images", resolution=-1, white_background=False, data_device="cuda",
                     eval=True, cap_max=1000000, init_type="sfm")
    bm = BetaModel(0, sb_number)
    scene = Scene(args, bm, load_iteration=-1, shuffle=False)
    bm.background = torch.zeros(3, device="cuda")
    return scene, bm


def w2c_of(c):
    return c.world_view_transform.transpose(0, 1).detach().cpu().numpy()  # [4,4]


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, u):
    d = np.dot(q0, q1)
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995: q = q0 + u * (q1 - q0); return q / np.linalg.norm(q)
    th0 = np.arccos(d); th = th0 * u
    q2 = q1 - q0 * d; q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def make_camera(R, t, c0, device):
    W2C = np.eye(4); W2C[:3, :3] = R; W2C[:3, 3] = t
    wvt = torch.tensor(W2C.T, dtype=torch.float32, device=device)
    return SimpleNamespace(world_view_transform=wvt, FoVx=float(c0.FoVx), FoVy=float(c0.FoVy),
                           image_width=int(c0.image_width), image_height=int(c0.image_height),
                           projection_matrix=SimpleNamespace(device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent.parent / "data/tanks/truck"))
    ap.add_argument("--model", default="/tmp/dbs_out/truck_beta")
    ap.add_argument("--sb-number", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/truck_turntable.gif"))
    ap.add_argument("--keystride", type=int, default=1, help="use every Nth real camera as a keyframe")
    ap.add_argument("--interp", type=int, default=4, help="interpolated frames between keyframes (more = smoother)")
    ap.add_argument("--start", type=int, default=70, help="first real camera index of the window")
    ap.add_argument("--count", type=int, default=48, help="number of consecutive real cameras to sweep")
    ap.add_argument("--downscale", type=int, default=3)
    ap.add_argument("--render-mode", default="RGB")
    a = ap.parse_args()

    scene, bm = load(a.source, a.model, a.sb_number)
    cams = sorted(scene.getTrainCameras(), key=lambda c: getattr(c, "image_name", "0"))
    window = cams[a.start:a.start + a.count]
    keys = window[::a.keystride]
    c0 = keys[0]
    poses = []
    for c in keys:
        M = w2c_of(c); R = M[:3, :3]; t = M[:3, 3]
        cen = -R.T @ t
        poses.append((rot_to_quat(R), cen))

    frames = []
    with torch.no_grad():
        for k in range(len(poses) - 1):
            q0, c_0 = poses[k]; q1, c_1 = poses[k + 1]
            for j in range(a.interp):
                u = j / a.interp
                q = slerp(q0, q1, u); cen = (1 - u) * c_0 + u * c_1
                R = quat_to_rot(q); t = -R @ cen
                out = bm.render(make_camera(R, t, c0, "cuda"), render_mode=a.render_mode)["render"]
                if a.render_mode in ("Depth", "EDepth"):
                    d = out.squeeze(); d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                    out = d.unsqueeze(0).repeat(3, 1, 1)
                img = out.clamp(0, 1)
                arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                frames.append(arr[::a.downscale, ::a.downscale])
    frames += frames[::-1]  # ping-pong loop
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=20, loop=0)
    print(f"wrote {a.out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


if __name__ == "__main__":
    main()
