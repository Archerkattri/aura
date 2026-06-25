from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VGGT_REPO = Path("/tmp/aura_sota_repos/vggt")
DA3_REPO = Path("/tmp/aura_sota_repos/Depth-Anything-3/src")


def _image_subset(image_dir: Path, count: int) -> list[str]:
    images = sorted(str(path) for path in image_dir.glob("*.jpg"))
    if len(images) < count:
        raise ValueError(f"need at least {count} images in {image_dir}, found {len(images)}")
    if count == 1:
        return [images[0]]
    step = max(1, (len(images) - 1) // (count - 1))
    subset = images[::step][:count]
    if len(subset) < count:
        subset = images[:count]
    return subset


def _finite_fraction(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.numel() == 0:
            return 0.0
        return float(torch.isfinite(tensor).float().mean().item())
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    return float(np.isfinite(array).mean())


def _shape(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    return [int(dim) for dim in np.asarray(value).shape]


def run_vggt(image_paths: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(VGGT_REPO))
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    start = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    seconds = time.perf_counter() - start

    pose_enc = predictions["pose_enc"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    points = unproject_depth_map_to_point_map(
        depth.squeeze(0).detach().cpu().numpy(),
        extrinsics.squeeze(0).detach().cpu().numpy(),
        intrinsics.squeeze(0).detach().cpu().numpy(),
    )

    return {
        "providerId": "vggt_1b",
        "model": "facebook/VGGT-1B",
        "images": float(len(image_paths)),
        "seconds": round(seconds, 4),
        "device": device,
        "validPriorCoverage": 1.0,
        "finiteDepthFraction": _finite_fraction(depth),
        "finiteDepthConfidenceFraction": _finite_fraction(depth_conf),
        "finiteExtrinsicFraction": _finite_fraction(extrinsics),
        "finiteIntrinsicFraction": _finite_fraction(intrinsics),
        "finiteWorldPointFraction": _finite_fraction(points),
        "depthShape": _shape(depth),
        "extrinsicShape": _shape(extrinsics),
        "intrinsicShape": _shape(intrinsics),
        "worldPointShape": _shape(points),
    }


def run_da3(image_paths: list[str], *, process_res: int) -> dict[str, Any]:
    sys.path.insert(0, str(DA3_REPO))
    from depth_anything_3.api import DepthAnything3

    device = "cuda"
    start = time.perf_counter()
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to(device).eval()
    prediction = model.inference(
        image=image_paths,
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_format="mini_npz",
    )
    seconds = time.perf_counter() - start

    depth = prediction.depth
    conf = getattr(prediction, "conf", None)
    extrinsics = getattr(prediction, "extrinsics", None)
    intrinsics = getattr(prediction, "intrinsics", None)
    return {
        "providerId": "depth_anything_3_small",
        "model": "depth-anything/DA3-SMALL",
        "images": float(len(image_paths)),
        "seconds": round(seconds, 4),
        "device": device,
        "processRes": float(process_res),
        "validPriorCoverage": 1.0,
        "finiteDepthFraction": _finite_fraction(depth),
        "finiteConfidenceFraction": _finite_fraction(conf) if conf is not None else 0.0,
        "finiteExtrinsicFraction": _finite_fraction(extrinsics) if extrinsics is not None else 0.0,
        "finiteIntrinsicFraction": _finite_fraction(intrinsics) if intrinsics is not None else 0.0,
        "depthShape": _shape(depth),
        "extrinsicShape": _shape(extrinsics) if extrinsics is not None else [],
        "intrinsicShape": _shape(intrinsics) if intrinsics is not None else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/tanks/truck/images")
    parser.add_argument("--images", type=int, default=12)
    parser.add_argument("--da3-process-res", type=int, default=336)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/geometry_prior_validation_2026-06-25.json")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    image_paths = _image_subset(args.image_dir, args.images)
    report = {
        "format": "AURA_GEOMETRY_PRIOR_VALIDATION",
        "scene": "tanks_truck",
        "imageCount": float(len(image_paths)),
        "imagePaths": image_paths,
        "providers": [
            run_vggt(image_paths),
            run_da3(image_paths, process_res=args.da3_process_res),
        ],
        "claimBoundary": (
            "VGGT and Depth Anything 3 are optional feed-forward geometry priors. "
            "For already-posed Truck data, COLMAP remains the default unless a missing-or-weak-COLMAP repair split is validated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
