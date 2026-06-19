"""Tests for aura.usd_writer — USD ASCII export of gaussian carrier points."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aura.usd_writer import write_usda


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene(n: int = 5, carrier_id: str = "gaussian") -> MagicMock:
    """Build a minimal mock AuraScene with *n* gaussian elements."""
    scene = MagicMock()
    scene.asset = MagicMock()
    scene.asset.name = "test-scene"
    elements = []
    for i in range(n):
        element = MagicMock()
        element.carrier_id = carrier_id
        element.id = f"carrier-{i}"
        element.mean = (float(i) * 0.1, float(i) * 0.2, float(i) * 0.3)
        element.color = (0.8, 0.5, 0.2)
        element.payload = {}
        elements.append(element)
    scene.elements = elements
    return scene


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_write_usda_creates_file(tmp_path: Path) -> None:
    """write_usda should produce a .usda file on disk."""
    scene = _make_scene(5)
    out = tmp_path / "model.usda"
    write_usda(scene, out)
    assert out.exists(), ".usda file must be created"


def test_usda_starts_with_magic(tmp_path: Path) -> None:
    """First line of the .usda file must be '#usda 1.0'."""
    scene = _make_scene(3)
    out = tmp_path / "model.usda"
    write_usda(scene, out)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "#usda 1.0", f"Expected '#usda 1.0', got {first_line!r}"


def test_usda_has_points_prim(tmp_path: Path) -> None:
    """The .usda file must contain a 'def Points' primitive declaration."""
    scene = _make_scene(4)
    out = tmp_path / "model.usda"
    write_usda(scene, out)
    content = out.read_text(encoding="utf-8")
    assert "def Points" in content, "'def Points' primitive not found in .usda"


def test_usda_contains_correct_point_count(tmp_path: Path) -> None:
    """The points block should contain exactly n position entries."""
    n = 7
    scene = _make_scene(n)
    out = tmp_path / "model.usda"
    write_usda(scene, out)
    content = out.read_text(encoding="utf-8")

    # Extract the points = [...] block and count lines that look like vec3 tuples
    in_points_block = False
    count = 0
    for line in content.splitlines():
        stripped = line.strip()
        if "point3f[] points = [" in stripped:
            in_points_block = True
            continue
        if in_points_block:
            if stripped == "]":
                break
            if stripped.startswith("("):
                count += 1
    assert count == n, f"Expected {n} point entries, found {count}"


def test_empty_scene_writes_valid_usda(tmp_path: Path) -> None:
    """A scene with no gaussian elements should still produce a valid .usda file."""
    scene = _make_scene(5, carrier_id="mesh")
    out = tmp_path / "empty.usda"
    write_usda(scene, out)

    assert out.exists(), ".usda file must be created even for an empty scene"
    content = out.read_text(encoding="utf-8")
    assert content.startswith("#usda 1.0"), "Empty-scene file must still start with USD magic"
    assert "def Points" in content, "'def Points' must still be present for an empty scene"

    # No position tuples inside points block
    in_points_block = False
    count = 0
    for line in content.splitlines():
        stripped = line.strip()
        if "point3f[] points = [" in stripped:
            in_points_block = True
            continue
        if in_points_block:
            if stripped == "]":
                break
            if stripped.startswith("("):
                count += 1
    assert count == 0, f"Expected 0 point entries for empty scene, found {count}"


def test_write_usda_returns_path(tmp_path: Path) -> None:
    """write_usda must return a Path object pointing to the written file."""
    scene = _make_scene(3)
    out = tmp_path / "output.usda"
    result = write_usda(scene, out)
    assert isinstance(result, Path), "Return value must be a Path"
    assert result.exists(), "Returned path must point to an existing file"


def test_suffix_corrected_to_usda(tmp_path: Path) -> None:
    """Passing 'model.txt' as output path should produce 'model.usda' instead."""
    scene = _make_scene(3)
    out = tmp_path / "model.txt"
    result = write_usda(scene, out)
    assert result.suffix == ".usda", f"Expected .usda suffix, got {result.suffix!r}"
    assert result.exists(), "Corrected .usda file must exist on disk"
    assert not out.exists(), "Original .txt path must NOT be created"
