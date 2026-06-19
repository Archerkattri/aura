"""Tests for aura.gltf_writer — glTF 2.0 export of gaussian carrier points."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aura.gltf_writer import write_gltf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene(n: int = 10, carrier_id: str = "gaussian") -> MagicMock:
    """Build a minimal mock AuraScene with *n* gaussian elements."""
    scene = MagicMock()
    elements = []
    for i in range(n):
        element = MagicMock()
        element.carrier_id = carrier_id
        # Use bounds (min_corner / max_corner) so the writer derives the center.
        bounds = MagicMock()
        bounds.min_corner = (float(i), float(i), float(i))
        bounds.max_corner = (float(i) + 1.0, float(i) + 1.0, float(i) + 1.0)
        element.bounds = bounds
        # No explicit mean / position attributes on the mock
        del element.mean
        del element.position
        element.payload = {}
        element.color = (0.8, 0.5, 0.2)
        elements.append(element)
    scene.elements = elements
    return scene


def _make_scene_with_mean(n: int = 10) -> MagicMock:
    """Build a mock scene where elements have an explicit .mean attribute."""
    scene = MagicMock()
    elements = []
    for i in range(n):
        element = MagicMock()
        element.carrier_id = "gaussian"
        element.mean = (float(i) * 0.1, float(i) * 0.2, float(i) * 0.3)
        element.color = (0.8, 0.5, 0.2)
        element.payload = {}
        elements.append(element)
    scene.elements = elements
    return scene


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_write_gltf_creates_files(tmp_path: Path) -> None:
    """write_gltf should produce both a .gltf and a .bin file."""
    scene = _make_scene(10)
    gltf_path = tmp_path / "model.gltf"
    write_gltf(scene, gltf_path)
    assert gltf_path.exists(), ".gltf file must be created"
    assert gltf_path.with_suffix(".bin").exists(), ".bin file must be created"


def test_gltf_is_valid_json(tmp_path: Path) -> None:
    """The .gltf file must be valid JSON."""
    scene = _make_scene(5)
    gltf_path = tmp_path / "model.gltf"
    write_gltf(scene, gltf_path)
    content = gltf_path.read_text(encoding="utf-8")
    parsed = json.loads(content)  # raises if not valid JSON
    assert isinstance(parsed, dict)


def test_gltf_has_required_fields(tmp_path: Path) -> None:
    """The glTF JSON must contain the required top-level keys."""
    scene = _make_scene(5)
    gltf_path = tmp_path / "model.gltf"
    write_gltf(scene, gltf_path)
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    for key in ("asset", "scene", "scenes", "nodes", "meshes"):
        assert key in data, f"Required glTF key '{key}' missing"
    assert data["asset"]["version"] == "2.0"


def test_bin_size_matches_n_gaussians(tmp_path: Path) -> None:
    """Binary buffer size must be exactly n * 12 * 2 bytes (positions + colors)."""
    n = 129_531
    scene = _make_scene(n)
    gltf_path = tmp_path / "model.gltf"
    write_gltf(scene, gltf_path)
    bin_path = gltf_path.with_suffix(".bin")
    expected = n * 12 * 2  # 3 float32 per point × 2 attribute arrays
    assert bin_path.stat().st_size == expected, (
        f"Expected {expected} bytes, got {bin_path.stat().st_size}"
    )


def test_points_mode(tmp_path: Path) -> None:
    """The primitive mode must be 0 (POINTS)."""
    scene = _make_scene(4)
    gltf_path = tmp_path / "model.gltf"
    write_gltf(scene, gltf_path)
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    primitive = data["meshes"][0]["primitives"][0]
    assert primitive["mode"] == 0, "Primitive mode must be 0 (POINTS)"


def test_accessor_count_matches_elements(tmp_path: Path) -> None:
    """Both accessors must report count equal to the number of gaussian elements."""
    n = 42
    scene = _make_scene(n)
    gltf_path = tmp_path / "model.gltf"
    write_gltf(scene, gltf_path)
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    for acc in data["accessors"]:
        assert acc["count"] == n, (
            f"Accessor count {acc['count']} does not match element count {n}"
        )


def test_empty_scene_writes_valid_gltf(tmp_path: Path) -> None:
    """A scene with no gaussian elements should produce a minimal valid glTF (no .bin)."""
    # Elements with a different carrier_id are not gaussian
    scene = _make_scene(5, carrier_id="mesh")
    gltf_path = tmp_path / "empty.gltf"
    write_gltf(scene, gltf_path)

    assert gltf_path.exists(), "gltf file must still be written for empty scene"
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    assert data["asset"]["version"] == "2.0"
    # No .bin produced for empty scenes
    assert not gltf_path.with_suffix(".bin").exists(), (
        ".bin must NOT be written for an empty scene"
    )


def test_write_gltf_returns_path(tmp_path: Path) -> None:
    """write_gltf must return the Path to the written .gltf file."""
    scene = _make_scene(3)
    gltf_path = tmp_path / "output.gltf"
    result = write_gltf(scene, gltf_path)
    assert isinstance(result, Path), "Return value must be a Path"
    assert result == gltf_path, "Returned path must match the requested output path"
