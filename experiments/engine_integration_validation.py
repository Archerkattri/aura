#!/usr/bin/env python3
"""Validate AURA's viewer/engine export bridge.

This writes concrete engine-facing artifacts:

* KHR_gaussian_splatting GLB for browser/game-engine splat viewers.
* USD ASCII bridge for DCC/engine metadata pipelines.
* Native runtime export report for query/chunk/object contracts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _carriers() -> dict:
    means = np.array(
        [
            [-0.20, -0.10, 2.40],
            [0.00, 0.08, 2.20],
            [0.22, -0.04, 2.55],
            [0.08, 0.20, 2.35],
        ],
        dtype="float32",
    )
    return {
        "means": means,
        "scales": np.array(
            [[0.08, 0.06, 0.05], [0.06, 0.09, 0.05], [0.10, 0.05, 0.06], [0.05, 0.05, 0.08]],
            dtype="float32",
        ),
        "quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (means.shape[0], 1)),
        "opacity": np.array([0.72, 0.65, 0.69, 0.58], dtype="float32"),
        "colors": np.array(
            [[0.12, 0.48, 0.92], [0.20, 0.82, 0.36], [0.95, 0.55, 0.10], [0.82, 0.30, 0.88]],
            dtype="float32",
        ),
        "confidence": np.array([0.98, 0.91, 0.87, 0.83], dtype="float32"),
        "sh_degree": 0,
    }


def validate_engine_exports(out: Path, artifact_dir: Path) -> dict:
    from aura.cli import native_demo_scene
    from aura.gltf_splat import write_splat_glb
    from aura.package import load_package, package_scene
    from aura.runtime_export import runtime_export_report
    from aura.usd_writer import write_usda

    artifact_dir.mkdir(parents=True, exist_ok=True)
    package_dir = artifact_dir / "native_demo_package"
    glb_path = artifact_dir / "aura_splat.glb"
    usd_path = artifact_dir / "aura_scene.usda"

    scene = native_demo_scene()
    package_scene(scene, fallbacks={"splat": str(glb_path.name), "usd": str(usd_path.name)}).write(package_dir)
    package = load_package(package_dir)
    write_splat_glb(_carriers(), glb_path)
    write_usda(scene, usd_path)
    runtime = runtime_export_report(package).to_dict()

    glb_bytes = glb_path.read_bytes()
    usd_text = usd_path.read_text(encoding="utf-8")
    glb_magic = glb_bytes[:4] == b"glTF"
    glb_has_extension = b"KHR_gaussian_splatting" in glb_bytes
    usd_has_points = 'def Points "GaussianCarriers"' in usd_text
    usd_has_count = "custom:aura:carrierCount" in usd_text

    payload = {
        "format": "AURA_ENGINE_INTEGRATION_VALIDATION",
        "passed": bool(
            glb_magic
            and glb_has_extension
            and usd_text.startswith("#usda 1.0")
            and usd_has_points
            and usd_has_count
            and runtime["engineWorkflow"]["nativeRuntimeReady"]
            and runtime["engineWorkflow"]["gltfPreviewReady"]
            and runtime["engineWorkflow"]["usdMetadataReady"]
            and runtime["engineWorkflow"]["chunkedStreamingReady"]
        ),
        "artifacts": {
            "package": str(package_dir),
            "gltfSplatGlb": str(glb_path),
            "usdBridge": str(usd_path),
        },
        "gltf": {
            "glbMagicValid": glb_magic,
            "usesKHRGaussianSplatting": glb_has_extension,
            "bytes": glb_path.stat().st_size,
        },
        "usd": {
            "usdaMagicValid": usd_text.startswith("#usda 1.0"),
            "hasGaussianPointsPrim": usd_has_points,
            "hasAuraCarrierMetadata": usd_has_count,
            "bytes": usd_path.stat().st_size,
        },
        "runtime": runtime,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/engine_integration_validation_2026-06-25.json")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "docs/engine_exports")
    args = parser.parse_args()

    payload = validate_engine_exports(args.out, args.artifact_dir)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
