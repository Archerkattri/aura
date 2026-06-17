import json
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from aura import (
    AURA_FORMAT,
    AURA_SCHEMA_VERSION,
    AuraAsset,
    AuraChunk,
    AuraElement,
    AuraPackage,
    AuraScene,
    Bounds,
    Ray,
    load_package,
    package_scene,
    validate_package,
)
from aura.cli import demo_scene
from aura.exchange import exchange_plan

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "docs" / "schemas"


def test_scene_ray_query_hits_front_element():
    scene = AuraScene(
        name="fixture",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.5,
                confidence=0.8,
            ),
        ),
    )

    result = scene.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert result.provenance == "surface"
    assert result.depth == 1.0
    assert result.opacity == 0.5


def test_scene_ray_query_miss_returns_empty_result():
    result = demo_scene().ray_query(Ray(origin=(3.0, 3.0, -2.0), direction=(0.0, 0.0, 1.0)))

    assert result.provenance == "miss"
    assert result.transmittance == 1.0


def test_package_writer_outputs_manifest(tmp_path):
    package = package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb"})
    package.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["format"] == AURA_FORMAT
    assert manifest["version"] == AURA_SCHEMA_VERSION
    assert manifest["capabilities"]["rayQuery"] is True
    assert manifest["fallbacks"]["mesh"] == "fallback/preview.glb"


def test_package_loader_round_trips_scene_and_manifest(tmp_path):
    package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb"}).write(tmp_path)

    package = load_package(tmp_path)

    assert package.asset.name == "demo"
    assert package.asset.fallbacks["mesh"] == "fallback/preview.glb"
    assert package.scene.elements[0].id == "wall_patch"
    assert package.scene.ray_query(Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0))).provenance == "wall_patch"


def test_package_validation_rejects_unknown_chunk_element():
    scene = AuraScene(
        name="bad",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),),
        chunks=(),
    )
    package = package_scene(scene)
    bad_package = type(package)(
        asset=package.asset,
        scene=AuraScene(
            name="bad",
            elements=scene.elements,
            chunks=(AuraChunk(id="root", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("missing",)),),
        ),
    )

    with pytest.raises(ValueError, match="unknown elements"):
        validate_package(bad_package)


def test_package_loader_rejects_manifest_chunk_mismatch(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"] = ["missing"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest chunks"):
        load_package(tmp_path)


def test_package_loader_rejects_missing_manifest_keys(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["carrierIds"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.schema.json validation failed"):
        load_package(tmp_path)


def test_package_loader_rejects_malformed_manifest_version(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "dev"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.schema.json validation failed at version"):
        load_package(tmp_path)


def test_package_loader_rejects_element_schema_violation(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    elements_path = tmp_path / "elements.json"
    elements = json.loads(elements_path.read_text(encoding="utf-8"))
    elements[0]["opacity"] = 1.5
    elements_path.write_text(json.dumps(elements), encoding="utf-8")

    with pytest.raises(ValueError, match="elements.schema.json validation failed"):
        load_package(tmp_path)


def test_package_loader_rejects_unsupported_major_version(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported AURA major version"):
        load_package(tmp_path)


def test_validate_package_rejects_unsupported_asset_version():
    scene = demo_scene()
    package = AuraPackage(asset=AuraAsset(name="future", carrier_ids=scene.carrier_ids(), version="99.0"), scene=scene)

    with pytest.raises(ValueError, match="unsupported AURA major version"):
        validate_package(package)


def test_validate_package_cli_reports_version_and_counts(tmp_path):
    package_scene(demo_scene()).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "validate-package", str(tmp_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "valid AURA package: demo" in result.stdout
    assert f"version {AURA_SCHEMA_VERSION}" in result.stdout
    assert "1 elements" in result.stdout
    assert "1 chunks" in result.stdout


def test_inspect_package_cli_reports_stable_json_summary(tmp_path):
    package_scene(demo_scene()).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "inspect-package", str(tmp_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "format": AURA_FORMAT,
        "version": AURA_SCHEMA_VERSION,
        "name": "demo",
        "carriers": ["surface"],
        "elementCount": 1,
        "chunkCount": 1,
    }


def test_json_schema_documents_are_parseable_and_versioned():
    schema_names = {"manifest.schema.json", "elements.schema.json", "chunks.schema.json"}
    found = {path.name for path in SCHEMA_DIR.glob("*.schema.json")}

    assert found == schema_names
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert f"/{AURA_SCHEMA_VERSION}/" in schema["$id"]
        assert schema["type"] in {"object", "array"}


def test_json_schema_documents_are_packaged_runtime_resources():
    schema_names = {"manifest.schema.json", "elements.schema.json", "chunks.schema.json"}
    package_files = resources.files("aura.schemas")
    found = {path.name for path in package_files.iterdir() if path.name.endswith(".schema.json")}

    assert found == schema_names
    for name in schema_names:
        packaged = json.loads(package_files.joinpath(name).read_text(encoding="utf-8"))
        documented = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        assert packaged == documented


def test_exchange_plan_keeps_native_contract_distinct():
    package = package_scene(demo_scene())
    plan = exchange_plan(package.asset)

    assert ".aura" in plan["native"]
    assert plan["gltfFallback"]["supports_ray_query"] is False
