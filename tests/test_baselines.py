import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aura import discover_3dgs_export, load_package, package_3dgs_export

FIXTURE_PLY = Path(__file__).parent / "fixtures" / "tiny_3dgs_export.ply"


def test_discover_3dgs_export_accepts_direct_ply_file():
    export = discover_3dgs_export(FIXTURE_PLY, name="fixture")

    assert export.baseline == "3dgs"
    assert export.splat_path == FIXTURE_PLY
    assert export.scene_name == "fixture"


def test_discover_3dgs_export_chooses_latest_common_iteration_layout(tmp_path):
    root = tmp_path / "garden"
    older = root / "point_cloud" / "iteration_1000" / "point_cloud.ply"
    newer = root / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_PLY, older)
    shutil.copyfile(FIXTURE_PLY, newer)

    export = discover_3dgs_export(root)

    assert export.splat_path == newer
    assert export.scene_name == "garden"


def test_discover_3dgs_export_rejects_ambiguous_recursive_plys(tmp_path):
    root = tmp_path / "ambiguous"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    shutil.copyfile(FIXTURE_PLY, root / "a" / "one.ply")
    shutil.copyfile(FIXTURE_PLY, root / "b" / "two.ply")

    with pytest.raises(ValueError, match="multiple PLY files"):
        discover_3dgs_export(root)


def test_package_3dgs_export_records_baseline_fallbacks(tmp_path):
    package = package_3dgs_export(FIXTURE_PLY, name="fixture")
    out = package.write(tmp_path / "fixture.aura")
    loaded = load_package(out)

    assert loaded.asset.name == "fixture"
    assert loaded.asset.fallbacks["baseline"] == "3dgs"
    assert loaded.asset.fallbacks["splat"].endswith("tiny_3dgs_export.ply")


def test_import_3dgs_cli_writes_valid_package(tmp_path):
    output_dir = tmp_path / "imported.aura"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "import-3dgs",
            str(FIXTURE_PLY),
            "--name",
            "fixture",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert str(output_dir) in result.stdout
    assert manifest["name"] == "fixture"
    assert manifest["fallbacks"]["baseline"] == "3dgs"
