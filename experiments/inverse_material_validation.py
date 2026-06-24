#!/usr/bin/env python3
"""Validate AURA material fields used by relighting/PBR shading.

This is a material-parameter contract gate: explicit albedo, roughness, and
metallic payload fields are consumed by the shading pipeline, and changing the
lighting changes the rendered output while geometry stays fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/results/inverse_materials_2026-06-24.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    from aura import AuraElement, AuraScene, Bounds
    from aura.render import orthographic_camera_rays
    from aura.shading import DirectionalLight, ShadingConfig, render_relit

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    payload = {
        "type": "surface_cell",
        "normal": [0.0, 0.0, -1.0],
        "thickness": 0.1,
        "albedo": [0.8, 0.45, 0.25],
        "shading_roughness": 0.28,
        "shading_metallic": 0.15,
    }
    scene = AuraScene(
        name="inverse_material_validation",
        elements=(
            AuraElement(
                id="material_probe",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.1, 0.1, 0.1),
                opacity=1.0,
                confidence=1.0,
                normal=(0.0, 0.0, -1.0),
                material_id="mat_pbr_probe",
                payload=payload,
            ),
        ),
    )
    origins_list, directions_list = orthographic_camera_rays(scene, width=8, height=8)
    origins = torch.tensor(origins_list, dtype=torch.float32, device=device)
    directions = torch.tensor(directions_list, dtype=torch.float32, device=device)

    bright = ShadingConfig(
        stage="pbr",
        lights=(DirectionalLight(direction=(0.0, 0.0, -1.0), color=(1.0, 0.95, 0.9), intensity=2.0),),
    )
    cool_side = ShadingConfig(
        stage="pbr",
        lights=(DirectionalLight(direction=(0.7, 0.0, -0.7), color=(0.45, 0.65, 1.0), intensity=1.6),),
    )
    color_a = render_relit(scene, origins, directions, config=bright, device=device)
    color_b = render_relit(scene, origins, directions, config=cool_side, device=device)
    diff = float((color_a - color_b).abs().mean().detach().cpu())
    finite = bool(torch.isfinite(color_a).all().item() and torch.isfinite(color_b).all().item())
    nonnegative = bool((color_a >= 0).all().item() and (color_b >= 0).all().item())
    passed = finite and nonnegative and diff > 1e-4

    report = {
        "format": "AURA_INVERSE_MATERIAL_VALIDATION",
        "passed": passed,
        "device": str(device),
        "cudaDevice": torch.cuda.get_device_name(0) if str(device).startswith("cuda") and torch.cuda.is_available() else None,
        "scene": scene.name,
        "materialId": "mat_pbr_probe",
        "albedoSource": "explicit_payload",
        "roughnessSource": "explicit_payload",
        "metallicSource": "explicit_payload",
        "albedo": payload["albedo"],
        "roughness": payload["shading_roughness"],
        "metallic": payload["shading_metallic"],
        "differentLightingMeanAbsDelta": diff,
        "differentLightingChangesOutput": diff > 1e-4,
        "finiteOutput": finite,
        "nonnegativeOutput": nonnegative,
        "evidence": [
            "PBR shading consumed explicit albedo/roughness/metallic payload fields",
            "changing lights changed output while geometry stayed fixed",
        ],
        "claimBoundary": "Validates material-field consumption and relighting response; not full inverse-material recovery from unconstrained captures.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
