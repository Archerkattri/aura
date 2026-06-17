import subprocess
import sys

from aura import load_package, package_scene, validate_package
from aura.cli import native_demo_scene


def test_native_demo_scene_exposes_strong_carrier_metadata(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)
    package = load_package(tmp_path)
    validate_package(package)

    by_carrier = {element.carrier_id: element for element in package.scene.elements}

    assert set(by_carrier) == {"surface", "volume", "beta", "gabor", "neural", "semantic", "gaussian"}
    assert all(element.metadata["demo_role"] for element in package.scene.elements)
    assert all(element.metadata["query_contract"] for element in package.scene.elements)
    assert all(element.metadata["export_proxy"] for element in package.scene.elements)
    assert all("assignment" in element.confidence_map for element in package.scene.elements)
    assert by_carrier["surface"].confidence_map["collision"] == 0.88
    assert by_carrier["volume"].confidence_map["transmittance"] == 0.78
    assert by_carrier["gabor"].metadata["query_contract"] == "carrier_color_modulation"
    assert by_carrier["neural"].residual is True
    assert by_carrier["semantic"].edit["operation"] == "object_level_edit"
    assert by_carrier["beta"].edit["operation"] == "local_support_move"
    assert by_carrier["gaussian"].payload["source"] == "native-demo-low-structure-evidence"


def test_native_demo_semantic_graph_records_interaction_relationship():
    scene = native_demo_scene()
    edge = scene.semantic_graph.edges[0]
    node_by_id = {node.id: node for node in scene.semantic_graph.nodes}

    assert edge.source == "object:fixture_object"
    assert edge.target == "object:wall"
    assert edge.relation == "occluded_by"
    assert node_by_id["object:wall"].attributes["editGroup"] == "room_shell"
    assert node_by_id["object:fixture_object"].attributes["exportTarget"] == "usd_object_metadata"


def test_build_native_demo_writes_validation_and_export_fallback_metadata(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "build-native-demo",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    package = load_package(tmp_path)
    validate_package(package)

    assert package.asset.fallbacks == {
        "mesh": "fallback/native-preview.glb",
        "preview": "fallback/native-demo.ppm",
        "usd": "fallback/native-scene.usda",
    }
    assert package.summary()["semanticObjectCount"] == 2
