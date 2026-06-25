#!/usr/bin/env python3
"""Render showcase GIFs from the trained DBS Beta model (runs in .dbs_venv).

  flythrough : a camera fly-through over real posed views (the reconstruction).
  orbit      : views sorted by azimuth around the scene centre (smooth turntable-ish).

High-quality Beta carriers (truck_beta, 26.35 dB) rendered through the DBS fork.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/dbs")
import numpy as np
import torch
from PIL import Image
from argparse import Namespace
from scene import Scene, BetaModel


def load(source, model_path, sb_number):
    args = Namespace(
        sh_degree=0, sb_number=sb_number, source_path=source, model_path=model_path,
        images="images", resolution=-1, white_background=False, data_device="cuda",
        eval=True, cap_max=1000000, init_type="sfm",
    )
    bm = BetaModel(args.sh_degree, args.sb_number)
    scene = Scene(args, bm, load_iteration=-1, shuffle=False)
    bm.background = torch.zeros(3, device="cuda")
    return scene, bm


def frame_to_img(t, downscale=2):
    a = (t.clamp(0, 1).detach().cpu().numpy() * 255).astype("uint8")
    if a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if downscale > 1:
        a = a[::downscale, ::downscale]
    return a


def write_gif(path: str, frames: list[np.ndarray], fps: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    duration_ms = max(1, round(1000 / fps))
    images = [Image.fromarray(frame) for frame in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent.parent / "data/tanks/truck"))
    ap.add_argument("--model", default="/tmp/dbs_out/truck_beta")
    ap.add_argument("--sb-number", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/truck_flythrough.gif"))
    ap.add_argument("--n", type=int, default=251)
    ap.add_argument("--all-frames", action="store_true", help="render every camera frame instead of selecting an evenly spaced subset")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--mode", choices=["flythrough", "orbit"], default="orbit")
    ap.add_argument("--render-mode", default="RGB", help="RGB | Normal | EDepth | Depth")
    a = ap.parse_args()

    scene, bm = load(a.source, a.model, a.sb_number)
    cams = list(scene.getTrainCameras())
    if a.all_frames:
        cams = [*cams, *list(scene.getTestCameras())]

    if a.mode == "flythrough":
        # capture order = a smooth handheld trajectory; sort by image name so
        # consecutive GIF frames are spatially adjacent (no jump/chop).
        cams = sorted(cams, key=lambda c: getattr(c, "image_name", str(getattr(c, "uid", 0))))
    if a.mode == "orbit":
        centre = bm.get_xyz.mean(dim=0).detach().cpu().numpy()
        def az(c):
            o = (-c.world_view_transform.transpose(0, 1)[:3, :3].T @ c.world_view_transform.transpose(0, 1)[:3, 3]).detach().cpu().numpy()
            d = o - centre
            return float(np.arctan2(d[1], d[0]))
        cams = sorted(cams, key=az)

    if a.all_frames:
        sel = cams
    else:
        step = max(1, len(cams) // a.n)
        sel = cams[::step][: a.n]
    frames = []
    with torch.no_grad():
        for c in sel:
            out = bm.render(c, render_mode=a.render_mode)["render"]
            if a.render_mode in ("Depth", "EDepth"):  # single channel -> normalize to grey
                d = out.squeeze()
                d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                out = d.unsqueeze(0).repeat(3, 1, 1)
            elif a.render_mode == "Normal":
                out = (out * 0.5 + 0.5)  # [-1,1] -> [0,1]
            frames.append(frame_to_img(out, a.downscale))
    if len(frames) > 2:
        frames += frames[-2:0:-1]  # ping-pong without duplicate endpoints, preserving uniform frame delays.
    write_gif(a.out, frames, a.fps)
    print(f"wrote {a.out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, {a.fps}fps target)")


if __name__ == "__main__":
    main()
