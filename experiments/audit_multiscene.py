#!/usr/bin/env python3
"""Audit local multi-scene data and benchmark completeness.

This answers a practical question: for the scenes physically downloaded in this
workspace, do we have both Beta and fixed-Gaussian metrics?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _metric(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _local_scenes() -> list[tuple[str, Path]]:
    scenes: list[tuple[str, Path]] = []
    truck = ROOT / "data/tanks/truck"
    if (truck / "sparse").exists():
        scenes.append(("truck", truck))
    mip = ROOT / "data/mipnerf360"
    for scene in sorted(p for p in mip.iterdir() if p.is_dir()) if mip.exists() else []:
        if (scene / "sparse").exists():
            scenes.append((scene.name, scene))
    return scenes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dbs_multiscene", help="benchmark output root")
    ap.add_argument("--json", default="experiments/results/multiscene_audit.json")
    args = ap.parse_args()

    out = Path(args.out)
    rows = []
    missing = []
    for name, path in _local_scenes():
        beta = _metric(out / f"{name}_beta/point_cloud/iteration_best/metrics.json")
        gauss = _metric(out / f"{name}_gauss/point_cloud/iteration_best/metrics.json")
        row = {
            "scene": name,
            "path": str(path.relative_to(ROOT)),
            "has_beta": beta is not None,
            "has_gaussian": gauss is not None,
            "beta_psnr": None if beta is None else beta.get("PSNR"),
            "gaussian_psnr": None if gauss is None else gauss.get("PSNR"),
            "delta_psnr": None if not beta or not gauss else beta.get("PSNR") - gauss.get("PSNR"),
        }
        rows.append(row)
        if not (row["has_beta"] and row["has_gaussian"]):
            missing.append(name)

    complete = len(missing) == 0 and bool(rows)
    payload = {
        "local_scene_count": len(rows),
        "complete": complete,
        "missing": missing,
        "scenes": rows,
    }
    target = ROOT / args.json
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"local scenes: {len(rows)}")
    for row in rows:
        status = "OK" if row["has_beta"] and row["has_gaussian"] else "MISSING"
        delta = row["delta_psnr"]
        print(f"{status:7s} {row['scene']:12s} beta={row['beta_psnr']} gaussian={row['gaussian_psnr']} delta={delta}")
    print(f"complete: {complete}")
    print(f"wrote {target.relative_to(ROOT)}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
