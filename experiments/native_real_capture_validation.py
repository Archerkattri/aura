#!/usr/bin/env python3
"""Validate AURA native-carrier evidence on locally downloaded real captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORMAT = "AURA_NATIVE_REAL_CAPTURE_VALIDATION"
ROOT = Path(__file__).resolve().parent.parent


def summarize_native_gate(
    *,
    audit: dict[str, Any],
    multiscene: dict[str, Any],
    min_scene_count: int,
) -> dict[str, Any]:
    """Summarize local real-capture native-carrier validation evidence."""

    audit_scenes = audit.get("scenes") or []
    metric_scenes = multiscene.get("scenes") or []
    audit_names = {str(row.get("scene")) for row in audit_scenes}
    metric_names = {str(row.get("scene")) for row in metric_scenes}
    missing = list(audit.get("missing") or [])
    complete_rows = [
        row
        for row in audit_scenes
        if row.get("has_beta") is True
        and row.get("has_gaussian") is True
        and row.get("delta_psnr") is not None
    ]
    positive_delta_rows = [
        row
        for row in metric_scenes
        if row.get("delta_psnr") is not None and float(row["delta_psnr"]) > 0.0
    ]
    scene_count = int(audit.get("local_scene_count") or len(audit_scenes))
    mean_delta = float(multiscene.get("mean_delta_psnr", 0.0))

    failures = []
    if not bool(audit.get("complete")) or missing:
        failures.append("not all locally downloaded scenes have complete Beta/Gaussian metrics")
    if scene_count < min_scene_count:
        failures.append("local real-capture scene count is below the required floor")
    if len(complete_rows) != scene_count:
        failures.append("one or more local scene rows lacks complete native-carrier metrics")
    if audit_names != metric_names:
        failures.append("multi-scene metric rows do not match the audited local scenes")
    if len(positive_delta_rows) != len(metric_scenes) or mean_delta <= 0.0:
        failures.append("typed Beta carrier did not improve PSNR over fixed Gaussian on every audited scene")

    return {
        "format": FORMAT,
        "passed": not failures,
        "allLocalScenesComplete": bool(audit.get("complete")) and not missing,
        "sceneCount": scene_count,
        "requiredSceneCount": int(min_scene_count),
        "missing": missing,
        "validatedCarrierFamilies": ["beta", "gaussian"],
        "completeMetricSceneCount": len(complete_rows),
        "meanDeltaPsnr": mean_delta,
        "positiveDeltaSceneCount": len(positive_delta_rows),
        "sceneNames": sorted(audit_names),
        "failures": failures,
        "claimBoundary": (
            "Validates native typed Beta carriers against fixed Gaussian fallback on all locally "
            "downloaded real captures; it does not claim every official upstream dataset scene is present."
        ),
        "audit": audit,
        "multiscene": multiscene,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=ROOT / "experiments/results/multiscene_audit.json")
    parser.add_argument("--multiscene", type=Path, default=ROOT / "experiments/results/multiscene.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/native_real_capture_validation_2026-06-24.json")
    parser.add_argument("--min-scene-count", type=int, default=8)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    multiscene = json.loads(args.multiscene.read_text(encoding="utf-8"))
    payload = summarize_native_gate(audit=audit, multiscene=multiscene, min_scene_count=args.min_scene_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
