import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aura import (
    AURA_FORMAT,
    AURA_SCHEMA_VERSION,
    AuraElement,
    AuraScene,
    Bounds,
    load_package,
    package_scene,
)
from aura.cli import demo_scene

pytestmark = pytest.mark.integration

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC) + (os.pathsep + existing if existing else "")
    return env


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aura.cli", *arguments],
        check=True,
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
        env=_cli_env(),
    )


def test_package_writer_persists_manifest_and_exchange_documents(tmp_path):
    package = package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb"})
    package.write(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    exchange = json.loads((tmp_path / "exchange.json").read_text(encoding="utf-8"))

    assert manifest["format"] == AURA_FORMAT
    assert manifest["version"] == AURA_SCHEMA_VERSION
    assert manifest["capabilities"]["rayQuery"] is True
    assert manifest["fallbacks"]["mesh"] == "fallback/preview.glb"
    assert manifest["exchange"] == "exchange.json"
    assert exchange["gltfFallback"]["supports_ray_query"] is False
    assert exchange["usdBridge"]["supports_typed_carriers"] is True


def test_package_writer_and_loader_round_trip_empty_chunk_manifest(tmp_path):
    scene = AuraScene(
        name="unchunked",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
            ),
        ),
    )
    package_scene(scene).write(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    loaded = load_package(tmp_path)

    assert manifest["chunks"] == []
    assert loaded.scene.chunks == ()
    assert loaded.scene.elements[0].id == "surface"


def test_package_loader_round_trips_scene_and_manifest(tmp_path):
    package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb"}).write(tmp_path)

    loaded = load_package(tmp_path)

    assert loaded.asset.name == "demo"
    assert loaded.asset.fallbacks["mesh"] == "fallback/preview.glb"
    assert loaded.exchange["asset"] == "demo"
    assert loaded.exchange["native"].startswith(".aura package")
    assert loaded.scene.elements[0].id == "wall_patch"
    assert loaded.scene.elements[0].payload["type"] == "surface_cell"
    assert loaded.scene.semantic_graph.nodes == ()


def test_package_loader_round_trips_confidence_and_edit_metadata(tmp_path):
    scene = AuraScene(
        name="editable",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                confidence_map={"geometry": 0.9},
                edit={"editable": True, "group": "wall"},
            ),
        ),
    )
    package_scene(scene).write(tmp_path)

    loaded = load_package(tmp_path)

    assert loaded.scene.elements[0].confidence_map == {"geometry": 0.9}
    assert loaded.scene.elements[0].edit == {"editable": True, "group": "wall"}


def test_validate_package_cli_reports_package_summary(tmp_path):
    package_scene(demo_scene()).write(tmp_path)

    result = _run_cli("validate-package", str(tmp_path))

    assert "valid AURA package: demo" in result.stdout
    assert f"version {AURA_SCHEMA_VERSION}" in result.stdout
    assert "1 elements" in result.stdout
    assert "1 chunks" in result.stdout


def test_inspect_package_cli_reports_json_summary(tmp_path):
    package_scene(demo_scene()).write(tmp_path)

    result = _run_cli("inspect-package", str(tmp_path))
    payload = json.loads(result.stdout)

    assert payload == {
        "format": AURA_FORMAT,
        "version": AURA_SCHEMA_VERSION,
        "name": "demo",
        "carriers": ["surface"],
        "elementCount": 1,
        "chunkCount": 1,
        "semanticObjectCount": 0,
        "exchangeTargets": ["asset", "gltfFallback", "native", "usdBridge"],
        "migration": {
            "current_version": AURA_SCHEMA_VERSION,
            "target_version": AURA_SCHEMA_VERSION,
            "supported": True,
            "actions": ["none"],
        },
    }
