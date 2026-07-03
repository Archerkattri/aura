#!/usr/bin/env python3
"""Collect the B2 gsplat-3DGS matched-budget control results.

Reads the per-scene eval stats written by run_gsplat_control.sh
(<out>/<scene>/stats/val_step*.json) plus the existing DBS beta/gauss arm
metrics (/tmp/dbs_multiscene/<scene>_{beta,gauss}/point_cloud/iteration_best/
metrics.json), and writes outputs/gsplat_control.json with, per scene:

  - gsplat "final" (step 30k, community-standard protocol) and
  - gsplat "best" (best test PSNR over eval steps — matches DBS's
    iteration_best selection rule, which picks on TEST metrics),

alongside the beta/gauss numbers, so the headline table can quote a true
3DGS control at the same 1M-primitive budget.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTROL_OUT = Path(os.environ.get("AURA_GSPLAT_CONTROL_OUT", "/tmp/gsplat_control"))
DBS_OUT = Path(os.environ.get("AURA_DBS_MULTISCENE_OUT", "/tmp/dbs_multiscene"))
RESULT = REPO / "outputs" / "gsplat_control.json"

SCENES = ["truck", "garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room"]


def read_gsplat(scene: str) -> dict | None:
    stats_dir = CONTROL_OUT / scene / "stats"
    if not stats_dir.is_dir():
        return None
    evals = []
    for f in sorted(stats_dir.glob("val_step*.json")):
        step = int(re.search(r"val_step(\d+)", f.name).group(1))
        s = json.loads(f.read_text())
        evals.append({"step": step, **{k: s[k] for k in ("psnr", "ssim", "lpips", "num_GS")}})
    if not evals:
        return None
    final = max(evals, key=lambda e: e["step"])
    best = max(evals, key=lambda e: e["psnr"])
    return {"final": final, "best": best, "n_evals": len(evals)}


def read_dbs(scene: str, arm: str) -> dict | None:
    f = DBS_OUT / f"{scene}_{arm}" / "point_cloud" / "iteration_best" / "metrics.json"
    if not f.is_file():
        return None
    m = json.loads(f.read_text())
    return {"psnr": m["PSNR"], "ssim": m["SSIM"], "lpips": m["LPIPS"]}


def main() -> int:
    table = {}
    for scene in SCENES:
        table[scene] = {
            "gsplat_mcmc_1M": read_gsplat(scene),
            "dbs_beta_1M": read_dbs(scene, "beta"),
            "dbs_frozen_gauss_1M": read_dbs(scene, "gauss"),
        }
    payload = {
        "format": "AURA_B2_GSPLAT_CONTROL",
        "protocol": {
            "budget": 1_000_000,
            "strategy": "gsplat MCMCStrategy, cap_max=1e6, 30k steps",
            "split": "every 8th view test (llffhold=8 == test_every=8)",
            "lpips": "vgg (matches DBS evaluator)",
            "selection_note": (
                "DBS iteration_best selects the checkpoint on TEST metrics; "
                "'best' mirrors that rule for gsplat, 'final' is the standard "
                "final-at-30k protocol. Quote 'final' by default and state the "
                "selection rule whenever 'best' is quoted."
            ),
        },
        "scenes": table,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2) + "\n")

    hdr = f"{'scene':<9} {'gsplat final':>13} {'gsplat best':>12} {'beta best':>10} {'gauss best':>11}"
    print(hdr)
    print("-" * len(hdr))
    def fmt(d: dict | None, *keys: str) -> str:
        for k in keys:
            d = d[k] if d else None
        return f"{d:.2f}" if d is not None else "—"

    for scene, row in table.items():
        g = row["gsplat_mcmc_1M"]
        b, q = row["dbs_beta_1M"], row["dbs_frozen_gauss_1M"]
        print(
            f"{scene:<9} {fmt(g, 'final', 'psnr'):>13} {fmt(g, 'best', 'psnr'):>12} "
            f"{fmt(b, 'psnr'):>10} {fmt(q, 'psnr'):>11}"
        )
    print(f"\nwrote {RESULT}")
    missing = [s for s, r in table.items() if r["gsplat_mcmc_1M"] is None]
    if missing:
        print(f"NOTE: no gsplat results yet for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
