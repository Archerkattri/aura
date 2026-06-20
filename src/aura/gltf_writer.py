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


def _collect_gaussian_points(
    scene: "AuraScene",
) -> "tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]":
    """Collect (positions, colors) for the scene's gaussian carriers.

    Shared by the .gltf (text) and .glb (binary) writers so both export an
    identical point set. AuraElement stores position as a bounding box; the
    center is used. Gaussian-specific payloads may carry an explicit "mean".
    """
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
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

    return positions, colors


def _build_point_buffer(
    positions: "list[tuple[float, float, float]]",
    colors: "list[tuple[float, float, float]]",
) -> "tuple[bytes, int]":
    """Pack positions then colors as little-endian float32 triples.

    Returns (buffer_bytes, position_byte_length). The position block precedes
    the color block; both are 12 bytes per point (always 4-byte aligned).
    """
    pos_bytes = bytearray()
    for xyz in positions:
        pos_bytes += struct.pack("<fff", *xyz)
    col_bytes = bytearray()
    for rgb in colors:
        col_bytes += struct.pack("<fff", *rgb)
    while len(pos_bytes) % 4:  # pragma: no cover — float32 triples are always 12-byte aligned
        pos_bytes += b"\x00"
    while len(col_bytes) % 4:  # pragma: no cover
        col_bytes += b"\x00"
    return bytes(pos_bytes) + bytes(col_bytes), len(pos_bytes)


def _build_gltf_dict(
    positions: "list[tuple[float, float, float]]",
    pos_block_length: int,
    buffer_length: int,
    *,
    buffer_uri: "str | None",
) -> dict:
    """Assemble the glTF 2.0 JSON for a POINTS mesh with POSITION + COLOR_0.

    ``buffer_uri`` is the external .bin filename for .gltf output, or None for
    a GLB self-contained binary buffer (no ``uri`` key, per the GLB spec).
    """
    n = len(positions)
    pos_length = n * 12  # 3 * float32 = 12 bytes per point
    col_length = n * 12
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    buffer: dict = {"byteLength": buffer_length}
    if buffer_uri is not None:
        buffer["uri"] = buffer_uri
    return {
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
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
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
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
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
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_length, "target": 34962},
            {"buffer": 0, "byteOffset": pos_block_length, "byteLength": col_length, "target": 34962},
        ],
        "buffers": [buffer],
    }


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

    positions, colors = _collect_gaussian_points(scene)
    if not positions:
        # No gaussian elements — write empty scene
        _write_empty_gltf(output_path)
        return output_path

    bin_data, pos_block_length = _build_point_buffer(positions, colors)
    bin_path.write_bytes(bin_data)

    gltf = _build_gltf_dict(
        positions, pos_block_length, len(bin_data), buffer_uri=bin_path.name
    )
    output_path.write_text(json.dumps(gltf, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_glb(scene: "AuraScene", output_path: "str | Path") -> Path:
    """Write scene's gaussian carriers to a binary glTF (.glb) container.

    A .glb is a single self-contained file: a 12-byte header followed by a JSON
    chunk and a BIN chunk (the glTF 2.0 GLB layout), so it needs no sidecar
    .bin. This is the pure-stdlib binary companion to ``write_gltf``; both
    export the same POINTS mesh (POSITION + COLOR_0 float32).

    Returns the Path to the written .glb file.
    """
    output_path = Path(output_path)

    positions, colors = _collect_gaussian_points(scene)
    if not positions:
        _write_empty_glb(output_path)
        return output_path

    bin_data, pos_block_length = _build_point_buffer(positions, colors)
    gltf = _build_gltf_dict(positions, pos_block_length, len(bin_data), buffer_uri=None)

    _write_glb_container(output_path, gltf, bin_data)
    return output_path


# GLB constants (glTF 2.0 binary container spec)
_GLB_MAGIC = 0x46546C67  # "glTF" little-endian
_GLB_VERSION = 2
_GLB_CHUNK_JSON = 0x4E4F534A  # "JSON"
_GLB_CHUNK_BIN = 0x004E4942  # "BIN\0"


def _write_glb_container(output_path: Path, gltf: dict, bin_data: bytes) -> None:
    """Pack a glTF JSON dict + binary buffer into a .glb file (stdlib only)."""
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # JSON chunk is padded with spaces (0x20) to a 4-byte boundary; the BIN
    # chunk is padded with zeros — both required by the GLB spec.
    json_pad = (-len(json_bytes)) % 4
    json_bytes += b"\x20" * json_pad
    bin_pad = (-len(bin_data)) % 4
    bin_data = bin_data + b"\x00" * bin_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    with open(output_path, "wb") as fh:
        # 12-byte header: magic, version, total length
        fh.write(struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total_length))
        # JSON chunk: length, type, data
        fh.write(struct.pack("<II", len(json_bytes), _GLB_CHUNK_JSON))
        fh.write(json_bytes)
        # BIN chunk: length, type, data
        fh.write(struct.pack("<II", len(bin_data), _GLB_CHUNK_BIN))
        fh.write(bin_data)


def _write_empty_glb(output_path: Path) -> None:
    """Write a minimal valid .glb with an empty scene and no BIN chunk."""
    gltf = {
        "asset": {"version": "2.0", "generator": "AURA glTF writer v1"},
        "scene": 0,
        "scenes": [{"name": "Empty", "nodes": []}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b"\x20" * ((-len(json_bytes)) % 4)
    total_length = 12 + 8 + len(json_bytes)
    with open(output_path, "wb") as fh:
        fh.write(struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total_length))
        fh.write(struct.pack("<II", len(json_bytes), _GLB_CHUNK_JSON))
        fh.write(json_bytes)


def _write_empty_gltf(output_path: Path) -> None:
    """Write a minimal valid glTF file when there are no gaussian elements."""
    gltf = {
        "asset": {"version": "2.0", "generator": "AURA glTF writer v1"},
        "scene": 0,
        "scenes": [{"name": "Empty", "nodes": []}],
    }
    output_path.write_text(json.dumps(gltf, indent=2) + "\n", encoding="utf-8")
