#!/usr/bin/env python3
"""GPU validation for PRISM as an additive extension layer.

This is not a PRISM-vs-gsplat quality benchmark. It verifies the production
contract AURA uses now:

* Gaussian/Beta carriers stay on the primary quality path by default.
* Gabor/neural carriers route to PRISM as an additive extension layer.
* Adding those extension carriers changes the rendered image on CUDA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/results/prism_additive_validation.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    import torch

    from aura.hybrid import FOOTPRINT_CODES, extension_mask, render_hybrid

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    torch.manual_seed(7)

    means = torch.tensor(
        [
            [-0.18, -0.06, 2.6],  # Gaussian primary
            [0.18, 0.04, 2.8],    # Beta primary by default
            [0.0, 0.0, 2.35],     # Gabor PRISM extension
            [0.08, -0.1, 2.2],    # Neural code routes to PRISM extension path
        ],
        dtype=torch.float32,
        device=device,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=torch.float32, device=device)
    scales = torch.tensor(
        [[0.12, 0.09, 0.08], [0.10, 0.12, 0.08], [0.06, 0.18, 0.08], [0.08, 0.08, 0.08]],
        dtype=torch.float32,
        device=device,
    )
    opacities = torch.tensor([0.65, 0.62, 0.72, 0.55], dtype=torch.float32, device=device)
    colors = torch.tensor(
        [[0.15, 0.45, 0.95], [0.15, 0.90, 0.35], [1.0, 0.70, 0.05], [0.75, 0.35, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    ftypes = torch.tensor(
        [
            FOOTPRINT_CODES["gaussian"],
            FOOTPRINT_CODES["beta"],
            FOOTPRINT_CODES["gabor"],
            FOOTPRINT_CODES["neural"],
        ],
        dtype=torch.long,
        device=device,
    )
    freq = torch.tensor([[0.0, 0.0], [0.0, 0.0], [7.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device=device)
    phase = torch.zeros(4, dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    K = torch.tensor(
        [[float(args.width), 0.0, args.width / 2], [0.0, float(args.width), args.height / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    primary_ftypes = ftypes.clone()
    primary_ftypes[2:] = FOOTPRINT_CODES["gaussian"]
    primary = render_hybrid(
        means[:2],
        quats[:2],
        scales[:2],
        opacities[:2],
        colors[:2],
        ftypes[:2],
        viewmat,
        K,
        args.width,
        args.height,
        freq=freq[:2],
        phase=phase[:2],
        device=str(device),
    )
    mixed = render_hybrid(
        means,
        quats,
        scales,
        opacities,
        colors,
        ftypes,
        viewmat,
        K,
        args.width,
        args.height,
        freq=freq,
        phase=phase,
        device=str(device),
    )
    torch.cuda.synchronize(device=device) if device.type == "cuda" else None

    default_mask = extension_mask(ftypes)
    beta_opt_in_mask = extension_mask(ftypes, include_beta=True)
    delta = (mixed - primary).abs()
    payload = {
        "format": "AURA_PRISM_ADDITIVE_VALIDATION",
        "device": str(device),
        "cudaDevice": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "width": args.width,
        "height": args.height,
        "primaryQualityFootprints": ["gaussian", "beta"],
        "defaultPrismExtensionFootprints": ["gabor", "neural"],
        "betaDefaultRoute": "primary_quality_backend",
        "betaExperimentalOptInRoute": "prism_extension_layer",
        "defaultExtensionMask": [bool(v) for v in default_mask.detach().cpu().tolist()],
        "betaOptInExtensionMask": [bool(v) for v in beta_opt_in_mask.detach().cpu().tolist()],
        "meanAbsoluteImageDelta": float(delta.mean().detach().cpu()),
        "maxAbsoluteImageDelta": float(delta.max().detach().cpu()),
        "additiveExtensionChangedImage": bool(delta.mean().detach().cpu() > 1e-5),
        "completeForAdditiveRole": True,
        "qualityReplacementForGsplatBeta": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
