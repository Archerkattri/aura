#!/usr/bin/env python3
"""Render local Tanks & Temples scene GIFs from images and COLMAP sparse depth.

This script is intentionally separate from the DBS-Beta showcase renderer:
Truck uses a trained DBS checkpoint, while Train media here is generated from
the local image set plus its COLMAP sparse model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from aura.ingest.colmap import _intrinsics_for_image, _quaternion_to_rotation, load_colmap_model

ROOT = Path(__file__).resolve().parent.parent


def _image_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"{root} does not contain image files")
    return files


def _select_evenly(items, n: int):
    if n <= 0:
        raise ValueError("--n must be positive")
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n, dtype=int)
    return [items[int(i)] for i in idx]


def _write_gif(path: Path, frames: list[np.ndarray], fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames + frames[::-1]
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"wrote {path} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


def _render_rgb(scene_root: Path, n: int, fps: int, out: Path):
    frames = []
    for image_path in _select_evenly(_image_files(scene_root / "images"), n):
        frames.append(np.asarray(Image.open(image_path).convert("RGB")))
    _write_gif(out, frames, fps)


def _render_sparse_depth(scene_root: Path, n: int, fps: int, out: Path):
    colmap_dir = scene_root / "sparse" / "0"
    cameras, images, points, _source = load_colmap_model(colmap_dir)
    if not points:
        raise ValueError(f"{colmap_dir} does not contain sparse points")
    by_name = {image.name: image for image in images}
    selected_paths = _select_evenly(_image_files(scene_root / "images"), n)
    xyz = np.asarray([p.xyz for p in points], dtype=np.float32)
    frames = []
    for image_path in selected_paths:
        image = by_name.get(image_path.name)
        if image is None:
            continue
        camera = cameras[image.camera_id]
        intr = _intrinsics_for_image(camera.intrinsics(), image_path)
        w, h = int(intr["width"]), int(intr["height"])
        rotation = np.asarray(
            _quaternion_to_rotation((image.qw, image.qx, image.qy, image.qz)),
            dtype=np.float32,
        )
        translation = np.asarray((image.tx, image.ty, image.tz), dtype=np.float32)
        cam = xyz @ rotation.T + translation
        z = cam[:, 2]
        valid = z > 1e-5
        u = np.full(z.shape, -1, dtype=np.int32)
        v = np.full(z.shape, -1, dtype=np.int32)
        u[valid] = np.round(intr["fx"] * cam[valid, 0] / z[valid] + intr["cx"]).astype(np.int32)
        v[valid] = np.round(intr["fy"] * cam[valid, 1] / z[valid] + intr["cy"]).astype(np.int32)
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(valid):
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            continue
        uu = u[valid]
        vv = v[valid]
        zz = z[valid]
        lo, hi = np.percentile(zz, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        brightness = np.clip((hi - zz) / (hi - lo), 0.0, 1.0)
        depth = np.zeros((h, w), dtype=np.float32)
        for du, dv in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            x = np.clip(uu + du, 0, w - 1)
            y = np.clip(vv + dv, 0, h - 1)
            np.maximum.at(depth, (y, x), brightness)
        frame = np.stack(
            [
                (depth * 72).astype(np.uint8),
                (depth * 180).astype(np.uint8),
                (depth * 255).astype(np.uint8),
            ],
            axis=-1,
        )
        frames.append(frame)
    if not frames:
        raise ValueError(f"none of the selected images in {scene_root} were registered")
    _write_gif(out, frames, fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-root", type=Path, default=Path("/tmp/tandt_db/tandt/train"))
    ap.add_argument("--n", type=int, default=301)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--rgb-out", type=Path, default=ROOT / "docs/train_orbit.gif")
    ap.add_argument("--depth-out", type=Path, default=ROOT / "docs/train_depth_orbit.gif")
    args = ap.parse_args()

    _render_rgb(args.scene_root, args.n, args.fps, args.rgb_out)
    _render_sparse_depth(args.scene_root, args.n, args.fps, args.depth_out)


if __name__ == "__main__":
    main()
