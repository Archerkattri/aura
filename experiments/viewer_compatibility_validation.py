#!/usr/bin/env python3
"""Strict local compatibility checks for exported viewer/engine artifacts.

This is a structural validator for the files AURA hands to external viewers. It
does not claim that a particular GUI application rendered the asset; it verifies
that the GLB and USDA files expose the contracts those viewers need.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KHR = "KHR_gaussian_splatting"


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes, list[str]]:
    errors: list[str] = []
    raw = path.read_bytes()
    if len(raw) < 20:
        return {}, b"", ["GLB is too short"]
    magic, version, total = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        errors.append("invalid GLB magic")
    if version != 2:
        errors.append("GLB version is not 2")
    if total != len(raw):
        errors.append("GLB total length does not match file size")

    offset = 12
    chunks: dict[int, bytes] = {}
    while offset + 8 <= len(raw):
        length, chunk_type = struct.unpack("<II", raw[offset: offset + 8])
        offset += 8
        chunks[chunk_type] = raw[offset: offset + length]
        offset += length
    json_chunk = chunks.get(0x4E4F534A, b"").rstrip(b" \t\r\n\x00")
    bin_chunk = chunks.get(0x004E4942, b"")
    try:
        gltf = json.loads(json_chunk.decode("utf-8"))
    except Exception as exc:
        return {}, bin_chunk, [*errors, f"invalid JSON chunk: {exc}"]
    return gltf, bin_chunk, errors


def validate_viewer_compatibility(glb_path: Path, usd_path: Path, out: Path) -> dict[str, Any]:
    gltf, bin_chunk, errors = _read_glb(glb_path)
    primitive = (((gltf.get("meshes") or [{}])[0].get("primitives") or [{}])[0]) if gltf else {}
    attributes = primitive.get("attributes") or {}
    accessors = gltf.get("accessors") or []
    buffers = gltf.get("buffers") or []
    extensions = set(gltf.get("extensionsUsed") or [])

    required_attrs = {
        "POSITION",
        "COLOR_0",
        f"{KHR}:ROTATION",
        f"{KHR}:SCALE",
        f"{KHR}:OPACITY",
    }
    missing_attrs = sorted(required_attrs - set(attributes))
    if missing_attrs:
        errors.append(f"missing required KHR attributes: {', '.join(missing_attrs)}")
    if KHR not in extensions:
        errors.append("KHR_gaussian_splatting is not declared in extensionsUsed")
    if primitive.get("mode") != 0:
        errors.append("KHR splat primitive must use POINTS mode")
    if KHR not in (primitive.get("extensions") or {}):
        errors.append("KHR_gaussian_splatting primitive extension block is missing")
    if buffers and int(buffers[0].get("byteLength", -1)) != len(bin_chunk):
        errors.append("buffer byteLength does not match BIN chunk length")

    accessor_types = {
        name: accessors[index].get("type") if isinstance(index, int) and 0 <= index < len(accessors) else None
        for name, index in attributes.items()
    }
    expected_types = {
        "POSITION": "VEC3",
        "COLOR_0": "VEC4",
        f"{KHR}:ROTATION": "VEC4",
        f"{KHR}:SCALE": "VEC3",
        f"{KHR}:OPACITY": "SCALAR",
    }
    for name, expected in expected_types.items():
        if name in accessor_types and accessor_types[name] != expected:
            errors.append(f"{name} accessor type is {accessor_types[name]}, expected {expected}")

    usd_text = usd_path.read_text(encoding="utf-8")
    usd_checks = {
        "startsWithUsdaMagic": usd_text.startswith("#usda 1.0"),
        "hasDefaultPrim": 'defaultPrim = "AURAScene"' in usd_text,
        "hasPointsPrim": 'def Points "GaussianCarriers"' in usd_text,
        "hasDisplayColorPrimvar": "primvars:displayColor" in usd_text,
        "hasAuraCarrierMetadata": "custom:aura:carrierCount" in usd_text,
        "balancedBraces": usd_text.count("{") == usd_text.count("}"),
        "balancedBrackets": usd_text.count("[") == usd_text.count("]"),
    }
    for name, ok in usd_checks.items():
        if not ok:
            errors.append(f"USD check failed: {name}")

    external_tools = {
        "node": shutil.which("node"),
        "npx": shutil.which("npx"),
        "blender": shutil.which("blender"),
        "usdchecker": shutil.which("usdchecker"),
        "usdcat": shutil.which("usdcat"),
    }
    payload = {
        "format": "AURA_VIEWER_COMPATIBILITY_VALIDATION",
        "passed": not errors,
        "claimBoundary": "Local structural compatibility only; no third-party GUI viewer render is claimed unless an external tool is installed and recorded here.",
        "artifacts": {"gltfSplatGlb": str(glb_path), "usdBridge": str(usd_path)},
        "gltf": {
            "assetVersion": (gltf.get("asset") or {}).get("version"),
            "usesKHRGaussianSplatting": KHR in extensions,
            "primitiveMode": primitive.get("mode"),
            "requiredAttributesPresent": not missing_attrs,
            "accessorTypes": accessor_types,
            "bufferBytes": len(bin_chunk),
        },
        "usd": usd_checks,
        "externalTools": external_tools,
        "errors": errors,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, default=ROOT / "docs/engine_exports/aura_splat.glb")
    parser.add_argument("--usd", type=Path, default=ROOT / "docs/engine_exports/aura_scene.usda")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/viewer_compatibility_validation_2026-06-25.json")
    args = parser.parse_args()

    payload = validate_viewer_compatibility(args.glb, args.usd, args.out)
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
