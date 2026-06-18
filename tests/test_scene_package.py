import json
from math import exp
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
    SemanticGraph,
    SemanticNode,
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


def test_scene_traversal_reports_ordered_multi_carrier_hits():
    scene = AuraScene(
        name="trace_fixture",
        elements=(
            AuraElement(
                id="front_surface",
                carrier_id="surface",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
                payload={"type": "surface_cell", "normal": [0.0, 0.0, -1.0]},
                color=(1.0, 0.0, 0.0),
                opacity=0.25,
                confidence=0.9,
                material_id="mat_surface",
            ),
            AuraElement(
                id="rear_volume",
                carrier_id="volume",
                bounds=Bounds((-1.0, -1.0, 0.2), (1.0, 1.0, 0.7)),
                payload={"type": "volume_cell", "density": 1.0},
                color=(0.0, 0.0, 1.0),
                opacity=0.8,
                confidence=0.6,
                semantic_id="mist",
            ),
        ),
    )

    traversal = scene.traverse_ray(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    payload = traversal.to_dict()

    assert traversal.hit_count == 2
    assert [hit.element_id for hit in traversal.ordered_hits] == ["front_surface", "rear_volume"]
    assert [hit.carrier_id for hit in traversal.ordered_hits] == ["surface", "volume"]
    assert traversal.result.depth == 1.0
    assert traversal.result.normal == (0.0, 0.0, -1.0)
    assert traversal.result.material_id == "mat_surface"
    assert traversal.result.provenance == "front_surface,rear_volume"
    assert traversal.result.transmittance == pytest.approx(0.75 * exp(-0.5))
    assert payload["orderedHits"][0]["elementId"] == "front_surface"
    assert payload["orderedHits"][1]["carrierId"] == "volume"
    assert payload["compositing"]["mode"] == "front_to_back"
    assert payload["compositing"]["provenanceOrder"] == ["front_surface", "rear_volume"]


def test_scene_ray_query_miss_returns_empty_result():
    result = demo_scene().ray_query(Ray(origin=(3.0, 3.0, -2.0), direction=(0.0, 0.0, 1.0)))

    assert result.provenance == "miss"
    assert result.transmittance == 1.0


def test_package_writer_outputs_manifest(tmp_path):
    package = package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb"})
    package.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    exchange = json.loads((tmp_path / "exchange.json").read_text())

    assert manifest["format"] == AURA_FORMAT
    assert manifest["version"] == AURA_SCHEMA_VERSION
    assert manifest["capabilities"]["rayQuery"] is True
    assert manifest["fallbacks"]["mesh"] == "fallback/preview.glb"
    assert manifest["exchange"] == "exchange.json"
    assert exchange["gltfFallback"]["supports_ray_query"] is False
    assert exchange["usdBridge"]["supports_typed_carriers"] is True


def test_package_writer_lists_chunk_documents_not_element_defaults(tmp_path):
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
    package = load_package(tmp_path)

    assert manifest["chunks"] == []
    assert package.scene.chunks == ()


def test_package_loader_round_trips_scene_and_manifest(tmp_path):
    package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb"}).write(tmp_path)

    package = load_package(tmp_path)

    assert package.asset.name == "demo"
    assert package.asset.fallbacks["mesh"] == "fallback/preview.glb"
    assert package.exchange["asset"] == "demo"
    assert package.exchange["native"].startswith(".aura package")
    assert package.scene.elements[0].id == "wall_patch"
    assert package.scene.elements[0].payload["type"] == "surface_cell"
    assert package.scene.semantic_graph.nodes == ()
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


def test_package_validation_rejects_unused_manifest_carrier():
    scene = AuraScene(
        name="bad",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),),
    )
    package = package_scene(scene)
    bad_package = type(package)(
        asset=type(package.asset)(
            name=package.asset.name,
            carrier_ids=("surface", "gaussian"),
            version=package.asset.version,
            units=package.asset.units,
            coordinate_system=package.asset.coordinate_system,
            fallbacks=package.asset.fallbacks,
        ),
        scene=scene,
        exchange=package.exchange,
    )

    with pytest.raises(ValueError, match="unused manifest carriers: gaussian"):
        validate_package(bad_package)


def test_package_validation_rejects_missing_manifest_carrier():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(id="surface", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            AuraElement(id="gaussian", carrier_id="gaussian", bounds=Bounds((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))),
        ),
    )
    package = package_scene(scene)
    bad_package = type(package)(
        asset=type(package.asset)(
            name=package.asset.name,
            carrier_ids=("surface",),
            version=package.asset.version,
            units=package.asset.units,
            coordinate_system=package.asset.coordinate_system,
            fallbacks=package.asset.fallbacks,
        ),
        scene=scene,
        exchange=package.exchange,
    )

    with pytest.raises(ValueError, match="missing scene carriers: gaussian"):
        validate_package(bad_package)


def test_package_validation_rejects_duplicate_chunk_ids():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                chunk_id="root",
            ),
        ),
        chunks=(
            AuraChunk(id="root", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("surface",)),
            AuraChunk(id="root", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("surface",)),
        ),
    )

    with pytest.raises(ValueError, match="duplicate chunk ids"):
        validate_package(package_scene(scene))


def test_package_validation_rejects_duplicate_chunk_element_ids():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                chunk_id="root",
            ),
        ),
        chunks=(AuraChunk(id="root", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("surface", "surface")),),
    )

    with pytest.raises(ValueError, match="duplicate elements: surface"):
        validate_package(package_scene(scene))


def test_package_validation_rejects_chunk_lod_mismatch():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="detail",
                carrier_id="beta",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                lod=1,
                chunk_id="detail",
                payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [1.0, 1.0, 1.0]},
            ),
        ),
        chunks=(AuraChunk(id="detail", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("detail",), lod=0),),
    )

    with pytest.raises(ValueError, match="lod 0 does not match member element lod 1"):
        validate_package(package_scene(scene))


def test_package_validation_accepts_gabor_carrier_normal_payload(tmp_path):
    scene = AuraScene(
        name="gabor_normal",
        elements=(
            AuraElement(
                id="frequency",
                carrier_id="gabor",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.8, 0.6, 0.2),
                payload={
                    "type": "gabor_frequency",
                    "frequency": [1.0, 0.0, 0.0],
                    "bandwidth": 0.5,
                    "phase": 0.0,
                    "normal": [0.0, 0.0, -1.0],
                    "plane_point": [0.0, 0.0, 0.0],
                },
            ),
        ),
    )

    output = package_scene(scene).write(tmp_path / "gabor-normal.aura")

    assert load_package(output).scene.elements[0].payload["normal"] == [0.0, 0.0, -1.0]


def test_package_validation_accepts_volume_opacity_payload(tmp_path):
    scene = AuraScene(
        name="volume_opacity",
        elements=(
            AuraElement(
                id="fog",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0)),
                color=(0.2, 0.4, 0.8),
                opacity=0.4,
                payload={"type": "volume_cell", "density": 0.7, "opacity": 0.4, "phase_anisotropy": 0.0},
            ),
        ),
    )

    output = package_scene(scene).write(tmp_path / "volume-opacity.aura")

    assert load_package(output).scene.elements[0].payload["opacity"] == 0.4


def test_package_validation_rejects_element_chunk_id_without_chunk():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                chunk_id="missing",
            ),
        ),
        chunks=(AuraChunk(id="root", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=()),),
    )

    with pytest.raises(ValueError, match="references unknown chunk"):
        validate_package(package_scene(scene))


def test_package_validation_rejects_chunk_membership_mismatch():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                chunk_id="right",
            ),
        ),
        chunks=(
            AuraChunk(id="left", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("surface",)),
            AuraChunk(id="right", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=()),
        ),
    )

    with pytest.raises(ValueError, match="assigned to other chunks"):
        validate_package(package_scene(scene))


def test_package_validation_rejects_chunk_omitting_assigned_element():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                chunk_id="root",
            ),
        ),
        chunks=(AuraChunk(id="root", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=()),),
    )

    with pytest.raises(ValueError, match="omits assigned elements"):
        validate_package(package_scene(scene))


def test_package_validation_rejects_chunk_bounds_that_do_not_contain_elements():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
                chunk_id="root",
            ),
        ),
        chunks=(AuraChunk(id="root", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), element_ids=("surface",)),),
    )

    with pytest.raises(ValueError, match="bounds do not contain elements"):
        validate_package(package_scene(scene))


def test_package_validation_rejects_payload_carrier_mismatch():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                payload={"type": "gaussian_fallback"},
            ),
        ),
    )
    package = package_scene(scene)

    with pytest.raises(ValueError, match="payload type"):
        validate_package(package)


def test_package_validation_rejects_malformed_in_memory_typed_payload():
    scene = AuraScene(
        name="bad",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 0.0],
                    "covariance": [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                    "source": "test",
                },
            ),
        ),
    )

    with pytest.raises(ValueError, match="malformed 'gaussian_fallback' payload"):
        validate_package(package_scene(scene))


def test_package_validation_accepts_well_formed_in_memory_typed_payloads():
    scene = AuraScene(
        name="typed",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                payload={"type": "surface_cell", "normal": [0.0, 0.0, 1.0], "thickness": 0.02, "roughness": 0.5},
            ),
            AuraElement(
                id="semantic",
                carrier_id="semantic",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                payload={"type": "semantic_feature", "label": "fixture", "confidence": 0.9, "feature_refs": []},
            ),
        ),
    )

    validate_package(package_scene(scene))


def test_package_loader_rejects_manifest_chunk_mismatch(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"] = ["missing"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest chunks"):
        load_package(tmp_path)


def test_package_loader_rejects_unmanifested_chunk_document(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    chunks_path = tmp_path / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks.append(
        {
            "id": "ghost",
            "bounds": {"min": [2.0, 2.0, 2.0], "max": [3.0, 3.0, 3.0]},
            "element_ids": [],
            "lod": 0,
        }
    )
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

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


def test_package_loader_rejects_corrupted_json_document(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    (tmp_path / "elements.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="elements.json must contain valid JSON"):
        load_package(tmp_path)


def test_package_loader_rejects_corrupted_manifest_capabilities(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capabilities"]["rayQuery"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest capabilities"):
        load_package(tmp_path)


def test_package_loader_rejects_element_schema_violation(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    elements_path = tmp_path / "elements.json"
    elements = json.loads(elements_path.read_text(encoding="utf-8"))
    elements[0]["opacity"] = 1.5
    elements_path.write_text(json.dumps(elements), encoding="utf-8")

    with pytest.raises(ValueError, match="elements.schema.json validation failed"):
        load_package(tmp_path)


def test_package_loader_rejects_malformed_typed_payload(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    elements_path = tmp_path / "elements.json"
    elements = json.loads(elements_path.read_text(encoding="utf-8"))
    del elements[0]["payload"]["normal"]
    elements_path.write_text(json.dumps(elements), encoding="utf-8")

    with pytest.raises(ValueError, match="elements.schema.json validation failed"):
        load_package(tmp_path)


def test_package_loader_round_trips_confidence_map_and_edit_metadata(tmp_path):
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

    package = load_package(tmp_path)

    assert package.scene.elements[0].confidence_map == {"geometry": 0.9}
    assert package.scene.elements[0].edit == {"editable": True, "group": "wall"}


def test_package_loader_rejects_unknown_semantic_graph_element(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    graph_path = tmp_path / "semantic_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "object:bad",
                        "label": "bad",
                        "element_ids": ["missing"],
                        "confidence": 1.0,
                        "attributes": {},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic node"):
        load_package(tmp_path)


def test_package_validation_rejects_duplicate_semantic_node_elements():
    scene = AuraScene(
        name="bad_semantic",
        elements=(
            AuraElement(id="surface", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))),
        ),
        semantic_graph=SemanticGraph(
            nodes=(
                SemanticNode(
                    id="object:wall",
                    label="wall",
                    element_ids=("surface", "surface"),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="duplicate elements: surface"):
        validate_package(package_scene(scene))


def test_package_validation_rejects_multi_node_semantic_ownership():
    scene = AuraScene(
        name="bad_semantic",
        elements=(
            AuraElement(id="surface", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))),
        ),
        semantic_graph=SemanticGraph(
            nodes=(
                SemanticNode(id="object:wall", label="wall", element_ids=("surface",)),
                SemanticNode(id="object:paint", label="paint", element_ids=("surface",)),
            ),
        ),
    )

    with pytest.raises(ValueError, match="assigns elements to multiple nodes: surface"):
        validate_package(package_scene(scene))


def test_package_loader_rejects_manifest_exchange_mismatch(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exchange"] = "wrong.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.schema.json validation failed"):
        load_package(tmp_path)


def test_package_loader_rejects_malformed_exchange_metadata(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    exchange_path = tmp_path / "exchange.json"
    exchange = json.loads(exchange_path.read_text(encoding="utf-8"))
    del exchange["gltfFallback"]["notes"]
    exchange_path.write_text(json.dumps(exchange), encoding="utf-8")

    with pytest.raises(ValueError, match="exchange.schema.json validation failed"):
        load_package(tmp_path)


def test_package_loader_rejects_exchange_asset_mismatch(tmp_path):
    package_scene(demo_scene()).write(tmp_path)
    exchange_path = tmp_path / "exchange.json"
    exchange = json.loads(exchange_path.read_text(encoding="utf-8"))
    exchange["asset"] = "other"
    exchange_path.write_text(json.dumps(exchange), encoding="utf-8")

    with pytest.raises(ValueError, match="exchange asset"):
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
        "semanticObjectCount": 0,
        "exchangeTargets": ["asset", "gltfFallback", "native", "usdBridge"],
        "migration": {
            "current_version": AURA_SCHEMA_VERSION,
            "target_version": AURA_SCHEMA_VERSION,
            "supported": True,
            "actions": ["none"],
        },
    }


def test_json_schema_documents_are_parseable_and_versioned():
    schema_names = {
        "manifest.schema.json",
        "elements.schema.json",
        "chunks.schema.json",
        "semantic_graph.schema.json",
        "exchange.schema.json",
        "training_dataset.schema.json",
        "capture_manifest.schema.json",
    }
    found = {path.name for path in SCHEMA_DIR.glob("*.schema.json")}

    assert found == schema_names
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert f"/{AURA_SCHEMA_VERSION}/" in schema["$id"]
        assert schema["type"] in {"object", "array"}


def test_json_schema_documents_are_packaged_runtime_resources():
    schema_names = {
        "manifest.schema.json",
        "elements.schema.json",
        "chunks.schema.json",
        "semantic_graph.schema.json",
        "exchange.schema.json",
        "training_dataset.schema.json",
        "capture_manifest.schema.json",
    }
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
