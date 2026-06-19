"""glTF 2.0 writer for AURA gaussian carrier scenes.

Exports gaussian splats as a POINTS mesh (positions + colors as vertex
attributes). This produces a valid glTF file viewable in Babylon.js,
three.js, and Blender, though full gaussian splatting display requires
a custom renderer extension.

Output format: .gltf (JSON) + .bin (binary buffer)
The .bin file is placed alongside the .gltf file.

Usage:
    from aura.gltf_writer import write_gltf
    write_gltf(scene, Path("outputs/model.gltf"))
"""
from __future__ import annotations
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.scene import AuraScene


def write_gltf(scene: "AuraScene", output_path: "str | Path") -> Path:
    """Write scene's gaussian carriers to glTF 2.0 POINTS format.

    Parameters
    ----------
    scene : AuraScene
        The loaded scene to export.
    output_path : str or Path
        Output .gltf file path. A matching .bin file is written alongside it.

    Returns
    -------
    Path to the written .gltf file.
    """
    output_path = Path(output_path)
    bin_path = output_path.with_suffix(".bin")

    # Collect gaussian element positions and colors.
    # AuraElement stores position as a bounding box; we use its center.
    # Gaussian-specific payload may also carry an explicit "mean" field.
    positions = []
    colors = []
    for element in scene.elements:
        if element.carrier_id != "gaussian":
            continue

        # Prefer an explicit mean from the payload (gaussian_fallback payloads),
        # then fall back to attributes named mean/position, then use bounds center.
        pos = None
        if hasattr(element, "mean") and element.mean is not None:
            pos = tuple(float(v) for v in element.mean)
        elif hasattr(element, "position") and element.position is not None:
            pos = tuple(float(v) for v in element.position)
        elif (
            hasattr(element, "payload")
            and isinstance(element.payload, dict)
            and element.payload.get("mean") is not None
        ):
            raw = element.payload["mean"]
            pos = tuple(float(v) for v in raw)
        elif hasattr(element, "bounds") and element.bounds is not None:
            b = element.bounds
            pos = tuple(
                (float(b.min_corner[i]) + float(b.max_corner[i])) * 0.5
                for i in range(3)
            )
        else:
            continue

        positions.append(pos)

        if hasattr(element, "color") and element.color is not None:
            colors.append(
                tuple(min(1.0, max(0.0, float(v))) for v in element.color[:3])
            )
        else:
            colors.append((0.8, 0.8, 0.8))

    if not positions:
        # No gaussian elements — write empty scene
        _write_empty_gltf(output_path)
        return output_path

    n = len(positions)

    # Build binary buffer: positions (float32 xyz) + colors (float32 rgb)
    pos_bytes = bytearray()
    for xyz in positions:
        pos_bytes += struct.pack("<fff", *xyz)
    col_bytes = bytearray()
    for rgb in colors:
        col_bytes += struct.pack("<fff", *rgb)

    # Pad to 4-byte alignment (already aligned for float32 triples, but be safe)
    while len(pos_bytes) % 4:
        pos_bytes += b"\x00"
    while len(col_bytes) % 4:
        col_bytes += b"\x00"

    bin_data = bytes(pos_bytes) + bytes(col_bytes)
    bin_path.write_bytes(bin_data)

    pos_offset = 0
    col_offset = len(pos_bytes)
    pos_length = n * 12  # 3 * float32 = 12 bytes per point
    col_length = n * 12

    # Compute bounding box for POSITION accessor (required by glTF spec)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    pos_min = [min(xs), min(ys), min(zs)]
    pos_max = [max(xs), max(ys), max(zs)]

    gltf = {
        "asset": {"version": "2.0", "generator": "AURA glTF writer v1"},
        "scene": 0,
        "scenes": [{"name": "AURA Scene", "nodes": [0]}],
        "nodes": [{"name": "GaussianCarriers", "mesh": 0}],
        "meshes": [
            {
                "name": "GaussianSplats",
                "primitives": [
                    {
                        "mode": 0,  # POINTS
                        "attributes": {
                            "POSITION": 0,
                            "COLOR_0": 1,
                        },
                    }
                ],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": n,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": n,
                "type": "VEC3",
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_length,
                "target": 34962,  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": col_offset,
                "byteLength": col_length,
                "target": 34962,
            },
        ],
        "buffers": [
            {
                "uri": bin_path.name,
                "byteLength": len(bin_data),
            }
        ],
    }

    output_path.write_text(json.dumps(gltf, indent=2) + "\n", encoding="utf-8")
    return output_path


def _write_empty_gltf(output_path: Path) -> None:
    """Write a minimal valid glTF file when there are no gaussian elements."""
    gltf = {
        "asset": {"version": "2.0", "generator": "AURA glTF writer v1"},
        "scene": 0,
        "scenes": [{"name": "Empty", "nodes": []}],
    }
    output_path.write_text(json.dumps(gltf, indent=2) + "\n", encoding="utf-8")
