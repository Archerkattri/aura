#!/usr/bin/env python3
"""Smoke-test gsplat-main source MCMC strategy on CUDA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GSPLAT_MAIN = Path("/tmp/aura_sota_repos/gsplat-main")


def run_gsplat_main_mcmc_smoke(out: Path) -> dict:
    code = r"""
import json
import torch
from gsplat.strategy import MCMCStrategy

device = "cuda"
torch.manual_seed(3)
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(torch.randn(64, 3, device=device) * 0.05),
    "scales": torch.nn.Parameter(torch.full((64, 3), -4.0, device=device)),
    "quats": torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(64, 4, device=device), dim=-1)),
    "opacities": torch.nn.Parameter(torch.full((64,), -2.0, device=device)),
})
optimizers = {key: torch.optim.Adam([params[key]], lr=1e-4) for key in params}
strategy = MCMCStrategy(cap_max=96, refine_start_iter=0, refine_stop_iter=2, refine_every=1, noise_injection_stop_iter=1)
strategy.check_sanity(params, optimizers)
state = strategy.initialize_state()
loss = sum((value.float() ** 2).mean() for value in params.values())
loss.backward()
strategy.step_post_backward(params, optimizers, state, 1, {}, lr=1e-4)
torch.cuda.synchronize()
print(json.dumps({
    "strategy": strategy.__class__.__name__,
    "device": torch.cuda.get_device_name(0),
    "gaussianCount": int(params["means"].shape[0]),
    "capMax": int(strategy.cap_max),
    "finiteMeans": bool(torch.isfinite(params["means"]).all().item()),
    "finiteScales": bool(torch.isfinite(params["scales"]).all().item()),
}))
"""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(GSPLAT_MAIN),
            "TORCH_EXTENSIONS_DIR": "/tmp/aura_gsplat_main_3dgs_ext",
            "MAX_JOBS": "2",
            "BUILD_3DGS": "1",
            "BUILD_3DGUT": "0",
            "BUILD_2DGS": "0",
            "BUILD_ADAM": "1",
            "BUILD_RELOC": "1",
            "BUILD_LOSSES": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    smoke = json.loads(result.stdout)
    payload = {
        "format": "AURA_GSPLAT_MAIN_MCMC_SMOKE",
        "date": "2026-06-25",
        "repository": str(GSPLAT_MAIN),
        "commit": subprocess.check_output(["git", "-C", str(GSPLAT_MAIN), "rev-parse", "HEAD"], text=True).strip(),
        "buildArtifact": "experiments/results/gsplat_main_build_probe_2026-06-25.json",
        "smoke": smoke,
        "passed": smoke["strategy"] == "MCMCStrategy" and smoke["finiteMeans"] and smoke["finiteScales"],
        "leaderboardImpact": "CUDA strategy smoke only; not a quality benchmark or promotion row.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/gsplat_main_mcmc_smoke_2026-06-25.json")
    args = parser.parse_args()
    payload = run_gsplat_main_mcmc_smoke(args.out)
    print(json.dumps({"passed": payload["passed"], "smoke": payload["smoke"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
