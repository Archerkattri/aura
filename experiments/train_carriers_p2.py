"""Train Gaussian carriers for the P2 full-resolution / render-loss study.

Differs from ``aura train-gsplat`` in exactly two ways that P2 needs:

1. **Train/test split.** P0 trained on *all* manifest frames (so the every-8th
   "held-out" views used by the reliability label were actually seen during
   training). P2 holds the every-8th frames OUT of training (``holdout=8``,
   llffhold convention), so the reliability labels — both the P0 colour-agreement
   proxy AND the new render-loss label — are computed on views the carriers never
   saw. This makes the render-loss label a genuine *held-out* signal.
2. **Carriers-only save.** Writes just the fast ``carriers.npz`` sidecar (the
   tensors the reliability + render-loss scripts consume), skipping the multi-
   hundred-MB ``elements.json`` the full ``.aura`` package would emit at full
   resolution.

Otherwise identical to the gsplat training path (same seed, same optimiser, same
DefaultStrategy-off no-densify configuration). ``--scale`` is the resolution knob:
P0 used ``0.25`` (quarter-resolution rendering/eval); ``1.0`` is full resolution.

Accuracy job, safe on shared GPUs (see gpu-usage-policy).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def train_split_carriers(
    manifest_path: str,
    out_dir: str,
    *,
    scale: float = 1.0,
    iterations: int = 5000,
    holdout: int = 8,
    device: str = "cuda",
    ssim_weight: float = 0.2,
    log=lambda m: print(m, file=sys.stderr, flush=True),
) -> dict:
    """Train carriers on the TRAIN split (drop every ``holdout``-th frame) and
    write ``<out_dir>/carriers.npz``. Returns a small metadata dict."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from aura.gsplat_renderer import (
        GsplatTrainConfig,
        gsplat_available,
        seed_gaussian_params_from_regions,
        train_scene_gsplat,
    )
    from aura.ingest.capture import load_capture_manifest
    from aura.carrier_io import save_carriers

    if not gsplat_available():
        raise SystemExit("train-gsplat requires torch + gsplat on the GPU box.")

    manifest_obj = load_capture_manifest(manifest_path, validate=False)
    if not manifest_obj.regions:
        raise SystemExit("no seed regions in manifest")
    seed_params, ctx = seed_gaussian_params_from_regions(manifest_obj.regions, device=device)

    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frames = raw["frames"]
    train_frames = [f for i, f in enumerate(frames) if i % holdout != 0]
    test_frames = [f for i, f in enumerate(frames) if i % holdout == 0]
    raw_train = {**raw, "frames": train_frames}
    log(f"[p2-train] {len(train_frames)} train / {len(test_frames)} held-out "
        f"frames; scale={scale}; iters={iterations}")

    cfg = GsplatTrainConfig(
        iterations=iterations, scale=scale, ssim_weight=ssim_weight,
        densify=False, log=log,
    )
    t0 = time.time()
    _scene, history = train_scene_gsplat(seed_params, ctx, raw_train, config=cfg, device=device)
    dt = time.time() - t0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_carriers(out, **history["carrier_save"])
    meta = {
        "manifest": manifest_path,
        "out": str(out / "carriers.npz"),
        "scale": scale,
        "iterations": iterations,
        "holdout": holdout,
        "train_frames": len(train_frames),
        "test_frames": len(test_frames),
        "carriers": int(history.get("final_gaussian_count", 0)),
        "train_seconds": round(dt, 1),
        "final_loss": history["loss"][-1][1] if history.get("loss") else None,
        "width": int(frames[0]["intrinsics"]["width"]),
        "height": int(frames[0]["intrinsics"]["height"]),
    }
    (out / "p2_train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"[p2-train] done in {dt:.0f}s -> {meta['carriers']} carriers, "
        f"final_loss={meta['final_loss']}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output .aura dir (gets carriers.npz)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = train_split_carriers(
        a.manifest, a.out, scale=a.scale, iterations=a.iterations,
        holdout=a.holdout, device=a.device,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
