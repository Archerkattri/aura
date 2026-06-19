"""Monocular depth estimation hook for AURA scene ingest.

Provides a lazy MiDaS inference hook that runs only when torch.hub is
available and the MiDaS model can be loaded. Falls back gracefully to
returning None so callers can use geometric heuristics instead.
"""
from __future__ import annotations


def midas_depth(image_path: str, device: str = "cpu") -> list[float] | None:
    """Run MiDaS monocular depth on an image file.

    Downloads MiDaS_small from torch.hub on first call (cached afterward).

    Parameters
    ----------
    image_path : str
        Path to input image (any PIL-readable format).
    device : str
        Torch device ("cpu" or "cuda").

    Returns
    -------
    list[float] or None
        Flat list of normalized depth values in [0, 1], or None if
        MiDaS is unavailable (no torch, no network, load error).
    """
    try:
        import torch
        import torch.hub
        from PIL import Image
        import numpy as np
    except ImportError:
        return None

    try:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform
    except Exception:
        return None

    try:
        model.eval()
        img = np.array(Image.open(image_path).convert("RGB"))
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        return depth_map.flatten().tolist()
    except Exception:
        return None


def geometric_depth_fallback(
    points_3d: list[tuple[float, float, float]],
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[float]:
    """Estimate depth from 3D point distances to camera origin.

    Parameters
    ----------
    points_3d : list of (x, y, z) tuples
    camera_origin : camera position

    Returns
    -------
    list[float] — normalized depth in [0, 1]
    """
    if not points_3d:
        return []
    import math
    dists = [
        math.sqrt(sum((p[i] - camera_origin[i])**2 for i in range(3)))
        for p in points_3d
    ]
    d_min, d_max = min(dists), max(dists)
    if d_max <= d_min:
        return [0.5] * len(dists)
    return [(d - d_min) / (d_max - d_min) for d in dists]
