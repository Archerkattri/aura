"""Tests for standards-compliant KHR_gaussian_splatting export."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura import gltf_splat as G  # noqa: E402

_EXT = "KHR_gaussian_splatting"


def _carriers(n=12, sh_degree=1):
    rng = np.random.default_rng(0)
    return dict(
        means=rng.standard_normal((n, 3)).astype("float32"),
        scales=np.abs(rng.standard_normal((n, 3))).astype("float32"),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype("float32"),
        opacity=rng.random(n).astype("float32"),
        sh=(rng.standard_normal((n, 4, 3)) * 0.1).astype("float32"),
        sh_degree=sh_degree,
    )


def test_gltf_declares_extension_and_required_attrs(tmp_path):
    c = _carriers()
    p = G.write_splat_gltf(c, tmp_path / "a.gltf")
    g = json.loads(p.read_text())
    assert g["extensionsUsed"] == [_EXT]
    prim = g["meshes"][0]["primitives"][0]
    assert prim["mode"] == 0  # POINTS mandatory
    assert _EXT in prim["extensions"]
    for req in ("POSITION", "COLOR_0", f"{_EXT}:ROTATION", f"{_EXT}:SCALE", f"{_EXT}:OPACITY"):
        assert req in prim["attributes"], req
    # sidecar bin exists and matches buffer length
    assert (tmp_path / "a.bin").stat().st_size == g["buffers"][0]["byteLength"]


def test_color0_is_vec4_and_sh_higher_orders_present(tmp_path):
    c = _carriers(sh_degree=1)
    p = G.write_splat_gltf(c, tmp_path / "a.gltf")
    g = json.loads(p.read_text())
    prim = g["meshes"][0]["primitives"][0]
    acc = g["accessors"]
    assert acc[prim["attributes"]["COLOR_0"]]["type"] == "VEC4"
    assert acc[prim["attributes"][f"{_EXT}:SCALE"]]["type"] == "VEC3"
    assert acc[prim["attributes"][f"{_EXT}:OPACITY"]]["type"] == "SCALAR"
    # degree-1 SH has 3 coefficients
    for m in range(3):
        assert f"{_EXT}:SH_DEGREE_1_COEF_{m}" in prim["attributes"]


def test_glb_container_is_valid(tmp_path):
    c = _carriers()
    p = G.write_splat_glb(c, tmp_path / "a.glb")
    raw = p.read_bytes()
    magic, ver, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67 and ver == 2 and total == len(raw)
    jl, jt = struct.unpack("<II", raw[12:20])
    assert jt == 0x4E4F534A  # JSON chunk
    gj = json.loads(raw[20:20 + jl])
    assert gj["extensionsUsed"] == [_EXT]


def test_quat_wxyz_converted_to_xyzw_normalized():
    # identity quat wxyz [1,0,0,0] -> xyzw [0,0,0,1]
    out = G._quat_wxyz_to_xyzw(np.array([[1.0, 0, 0, 0]], dtype="float32"))
    assert np.allclose(out[0], [0, 0, 0, 1])


def test_flat_color_path(tmp_path):
    c = dict(
        means=np.zeros((3, 3), "float32"), scales=np.ones((3, 3), "float32"),
        quats=np.tile([1.0, 0, 0, 0], (3, 1)).astype("float32"),
        opacity=np.full(3, 0.5, "float32"),
        colors=np.full((3, 3), 0.7, "float32"), sh_degree=0,
    )
    g, _ = G.build_splat_gltf(c)
    prim = g["meshes"][0]["primitives"][0]
    assert "COLOR_0" in prim["attributes"]
    # no SH higher-order attrs when degree 0
    assert not any("SH_DEGREE" in k for k in prim["attributes"])
