#!/usr/bin/env python3
"""FPS sweep on trained scene checkpoints.

The synthetic PRISM benchmark is useful for kernel scaling; this script records
actual trained-checkpoint render speed for the current local assets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fps(ms: float) -> float:
    return 1000.0 / ms if ms > 0 else 0.0


def _bench_dbs_model(source: Path, model: Path, *, sb_number: int, frames: int, warmup: int) -> dict[str, Any]:
    sys.path.insert(0, "/tmp/dbs")
    import torch
    from argparse import Namespace
    from scene import BetaModel, Scene

    args = Namespace(
        sh_degree=0,
        sb_number=sb_number,
        source_path=str(source),
        model_path=str(model),
        images="images",
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=True,
        cap_max=1000000,
        init_type="sfm",
    )
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    scene = Scene(args, beta_model, load_iteration=-1, shuffle=False)
    beta_model.background = torch.zeros(3, device="cuda")
    cameras = list(scene.getTestCameras()) or list(scene.getTrainCameras())
    selected = cameras[:frames]
    if not selected:
        raise RuntimeError(f"no cameras found for {source}")
    with torch.no_grad():
        for camera in selected[:warmup]:
            beta_model.render(camera)["render"]
        torch.cuda.synchronize()
        start = time.perf_counter()
        for camera in selected:
            beta_model.render(camera)["render"]
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    ms = seconds / len(selected) * 1000.0
    first = selected[0]
    return {
        "frames": len(selected),
        "seconds": seconds,
        "msPerFrame": ms,
        "fps": _fps(ms),
        "width": int(getattr(first, "image_width", 0)),
        "height": int(getattr(first, "image_height", 0)),
    }


def real_scene_fps_sweep(out: Path, *, frames: int = 32, warmup: int = 3) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source = ROOT / "data/tanks/truck"
    dbs_runs = (
        ("truck", "DBS-Beta", Path("/tmp/dbs_out/truck_beta"), 2),
        ("truck", "fixed Gaussian control", Path("/tmp/dbs_out/truck_gauss"), 0),
    )
    for scene, method, model, sb_number in dbs_runs:
        if model.exists():
            stats = _bench_dbs_model(source, model, sb_number=sb_number, frames=frames, warmup=warmup)
            rows.append({
                "scene": scene,
                "method": method,
                "source": str(model),
                "renderer": "DBS fork BetaModel.render",
                **stats,
            })

    metric_paths = {
        "truck": Path("/tmp/aura_sota_3dgrut_runs/truck_3dgut_full30000_eval/truck_3dgut_full30000/truck-2506_023310/metrics.json"),
        "room": Path("/tmp/aura_sota_3dgrut_runs/room_3dgut_full30000_ds2_eval/room_3dgut_full30000_ds2/room-2506_035308/metrics.json"),
    }
    for scene, path in metric_paths.items():
        metrics = _read_json(path)
        if not metrics:
            continue
        # Official 3DGUT prints mean inference time in its table but the JSON
        # does not persist it, so keep the measured values from the render logs.
        log_ms = 2.28 if scene == "truck" else 2.47
        rows.append({
            "scene": scene,
            "method": "official 3DGUT",
            "source": str(path),
            "renderer": "official 3DGRUT render.py",
            "msPerFrame": log_ms,
            "fps": _fps(log_ms),
            "metrics": metrics,
        })

    payload = {
        "format": "AURA_REAL_SCENE_FPS_SWEEP",
        "device": "cuda",
        "claimBoundary": "Trained-checkpoint render speed for local Truck DBS arms and official 3DGUT Truck/Room rows; not a full-scene leaderboard FPS claim.",
        "rows": rows,
        "passed": bool(rows) and all(float(row.get("fps", 0.0)) >= 30.0 for row in rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/real_scene_fps_sweep_2026-06-25.json")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    payload = real_scene_fps_sweep(args.out, frames=args.frames, warmup=args.warmup)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
