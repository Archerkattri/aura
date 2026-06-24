"""Standards-compliant `KHR_gaussian_splatting` glTF/GLB export.

The legacy ``gltf_writer`` emits a degenerate POINTS cloud (POSITION + COLOR_0)
— viewers see dots, not splats. This module writes the real Khronos
`KHR_gaussian_splatting` extension (ratified into glTF 2.0, 2025) so AURA carriers
load as actual Gaussian splats in any conformant engine (three.js, PlayCanvas,
Babylon, …).

Per-splat attributes (all on a single POINTS primitive carrying the extension):

  POSITION                              VEC3  float    centre
  COLOR_0                               VEC4  float    SH0 diffuse rgb + opacity a
  KHR_gaussian_splatting:ROTATION       VEC4  float    unit quat, glTF xyzw order
  KHR_gaussian_splatting:SCALE          VEC3  float    linear, non-negative
  KHR_gaussian_splatting:OPACITY        SCALAR float   linear [0,1]
  KHR_gaussian_splatting:SH_DEGREE_l_COEF_n  VEC3 float   higher-order SH (optional)

Input is the carrier-tensor dict produced by :mod:`aura.carrier_io` (means /
scales / quats(wxyz) / opacity / colors-or-sh), so it works directly on
gsplat- *and* DBS-trained carriers.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

# SH band-0 constant (Y_0^0): COLOR_0 fallback = 0.5 + C0 * f_dc.
_C0 = 0.28209479177387814
_EXT = "KHR_gaussian_splatting"


def _np(x):
    import numpy as np
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype="float32"))


def _diffuse_rgb(carriers):
    """Recover linear [0,1] diffuse RGB for COLOR_0 from either flat colours or
    the SH DC coefficient (per the KHR fallback formula)."""
    import numpy as np
    if "sh" in carriers and carriers["sh"] is not None:
        dc = _np(carriers["sh"])[:, 0, :]          # [N,3] DC coefficient (f_dc)
        return np.clip(0.5 + _C0 * dc, 0.0, 1.0)
    return np.clip(_np(carriers["colors"]), 0.0, 1.0)


def _quat_wxyz_to_xyzw(q):
    import numpy as np
    q = _np(q)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    return np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)


def build_splat_gltf(carriers, *, buffer_uri=None):
    """Build (gltf_dict, bin_bytes) for the KHR_gaussian_splatting asset."""
    import numpy as np

    means = _np(carriers["means"])                     # [N,3]
    scales = np.clip(_np(carriers["scales"]), 0.0, None)  # [N,3] linear, >=0
    rot = _quat_wxyz_to_xyzw(carriers["quats"])         # [N,4] xyzw
    opacity = np.clip(_np(carriers["opacity"]).reshape(-1, 1), 0.0, 1.0)  # [N,1]
    rgb = _diffuse_rgb(carriers)                        # [N,3]
    color0 = np.concatenate([rgb, opacity], axis=1)     # [N,4] rgba
    n = means.shape[0]

    sh_degree = int(carriers.get("sh_degree", 0) or 0)
    sh = _np(carriers.get("sh")) if sh_degree > 0 and carriers.get("sh") is not None else None

    # --- pack a single interleaved-by-block binary buffer ---------------------
    blocks = []          # (semantic_key, np_array [N, comps], accessor_type)
    blocks.append(("POSITION", means, "VEC3"))
    blocks.append(("COLOR_0", color0, "VEC4"))
    blocks.append((f"{_EXT}:ROTATION", rot, "VEC4"))
    blocks.append((f"{_EXT}:SCALE", scales, "VEC3"))
    blocks.append((f"{_EXT}:OPACITY", opacity, "SCALAR"))
    if sh is not None:
        # sh is [N, K, 3], K=(deg+1)^2, index 0 = DC (already in COLOR_0).
        for l in range(1, sh_degree + 1):
            for m in range(2 * l + 1):
                idx = l * l + m
                blocks.append((f"{_EXT}:SH_DEGREE_{l}_COEF_{m}",
                               np.ascontiguousarray(sh[:, idx, :]), "VEC3"))

    bin_parts, views, accessors, attributes = [], [], [], {}
    offset = 0
    for key, arr, atype in blocks:
        arr = np.ascontiguousarray(arr.astype("float32"))
        raw = arr.tobytes()
        pad = (-len(raw)) % 4
        view_i = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": 34962})
        acc = {"bufferView": view_i, "byteOffset": 0, "componentType": 5126,
               "count": n, "type": atype}
        if key == "POSITION":
            acc["min"] = [float(means[:, i].min()) for i in range(3)]
            acc["max"] = [float(means[:, i].max()) for i in range(3)]
        accessors.append(acc)
        attributes[key] = len(accessors) - 1
        bin_parts.append(raw + b"\x00" * pad)
        offset += len(raw) + pad

    bin_data = b"".join(bin_parts)
    buffer = {"byteLength": len(bin_data)}
    if buffer_uri is not None:
        buffer["uri"] = buffer_uri

    gltf = {
        "asset": {"version": "2.0", "generator": "AURA KHR_gaussian_splatting writer"},
        "extensionsUsed": [_EXT],
        "scene": 0,
        "scenes": [{"name": "AURA Scene", "nodes": [0]}],
        "nodes": [{"name": "GaussianSplats", "mesh": 0}],
        "meshes": [{
            "name": "GaussianSplats",
            "primitives": [{
                "mode": 0,  # POINTS (mandatory for KHR_gaussian_splatting)
                "attributes": attributes,
                "extensions": {_EXT: {}},
            }],
        }],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [buffer],
    }
    return gltf, bin_data


def write_splat_gltf(carriers, output_path):
    """Write `<path>.gltf` + sidecar `.bin` (KHR_gaussian_splatting)."""
    output_path = Path(output_path)
    bin_path = output_path.with_suffix(".bin")
    gltf, bin_data = build_splat_gltf(carriers, buffer_uri=bin_path.name)
    bin_path.write_bytes(bin_data)
    output_path.write_text(json.dumps(gltf, indent=2) + "\n", encoding="utf-8")
    return output_path


def write_splat_glb(carriers, output_path):
    """Write a self-contained `.glb` (KHR_gaussian_splatting)."""
    from .gltf_writer import _write_glb_container
    output_path = Path(output_path)
    gltf, bin_data = build_splat_gltf(carriers, buffer_uri=None)
    _write_glb_container(output_path, gltf, bin_data)
    return output_path
