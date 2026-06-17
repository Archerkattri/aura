import json
import math
import subprocess
import sys

from aura import (
    AuraElement,
    Bounds,
    EvidenceSample,
    Ray,
    RegionEvidence,
    decompose_evidence,
    load_package,
    package_scene,
    validate_package,
)
from aura.cli import native_demo_scene


def test_native_demo_scene_is_mixed_aura_first_fixture():
    scene = native_demo_scene()

    assert scene.name == "native_demo"
    assert scene.carrier_ids() == ["beta", "gabor", "gaussian", "neural", "semantic", "surface", "volume"]
    assert scene.chunk_ids() == [
        "base_surface_lod0",
        "base_volume_lod0",
        "detail_beta_lod1",
        "detail_gabor_lod1",
        "fallback_gaussian_lod2",
        "residual_neural_lod1",
        "semantic_object_lod0",
    ]
    assert all(element.payload for element in scene.elements)
    assert {element.id: element.semantic_id for element in scene.elements}["surface_wall"] == "wall"
    assert {node.label for node in scene.semantic_graph.nodes} == {"wall", "fixture_object"}


def test_decompose_evidence_produces_mixed_native_carriers(tmp_path):
    samples = (
        EvidenceSample(
            id="surface_wall",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
            evidence=RegionEvidence(geometry_confidence=0.9, material_confidence=0.7, edit_need=0.6),
            normal=(0.0, 0.0, -1.0),
            confidence_map={"geometry": 0.9, "material": 0.7},
            edit={"editable": True, "group": "wall"},
        ),
        EvidenceSample(
            id="fog_volume",
            bounds=Bounds((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            evidence=RegionEvidence(fuzzy_confidence=0.9, geometry_confidence=0.2),
            opacity=0.4,
        ),
        EvidenceSample(
            id="weave_detail",
            bounds=Bounds((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
            evidence=RegionEvidence(high_frequency=0.9),
        ),
        EvidenceSample(
            id="view_residual",
            bounds=Bounds((3.0, 0.0, 0.0), (4.0, 1.0, 1.0)),
            evidence=RegionEvidence(view_dependent=0.9, material_confidence=0.2),
        ),
        EvidenceSample(
            id="semantic_tooth",
            bounds=Bounds((4.0, 0.0, 0.0), (5.0, 1.0, 1.0)),
            evidence=RegionEvidence(semantic_confidence=0.95),
            semantic_label="tooth_12",
        ),
        EvidenceSample(
            id="compact_chip",
            bounds=Bounds((5.0, 0.0, 0.0), (5.5, 0.5, 0.5)),
            evidence=RegionEvidence(compact_detail=0.9),
        ),
        EvidenceSample(
            id="fallback_sample",
            bounds=Bounds((6.0, 0.0, 0.0), (7.0, 1.0, 1.0)),
            evidence=RegionEvidence(image_error=0.05, geometry_confidence=0.3),
        ),
    )

    scene = decompose_evidence(samples, name="mixed_native")
    package = package_scene(scene)
    validate_package(package)
    package.write(tmp_path)
    loaded = load_package(tmp_path)

    assert loaded.scene.carrier_ids() == ["beta", "gabor", "gaussian", "neural", "semantic", "surface", "volume"]
    assert [element.payload["type"] for element in loaded.scene.elements] == [
        "surface_cell",
        "volume_cell",
        "gabor_frequency",
        "neural_residual",
        "semantic_feature",
        "beta_kernel",
        "gaussian_fallback",
    ]
    assert loaded.scene.elements[0].confidence_map == {"assignment": 1.0, "geometry": 0.9, "material": 0.7}
    assert loaded.scene.elements[0].edit == {"source": "adaptive-decomposition", "editable": True, "group": "wall"}
    chunk_by_id = {chunk.id: chunk for chunk in loaded.scene.chunks}
    assert chunk_by_id["base_surface_lod0"].element_ids == ("surface_wall",)
    assert chunk_by_id["base_volume_lod0"].element_ids == ("fog_volume",)
    assert chunk_by_id["detail_gabor_lod1"].element_ids == ("weave_detail",)
    assert chunk_by_id["residual_neural_lod1"].element_ids == ("view_residual",)
    assert chunk_by_id["semantic_object_lod0"].element_ids == ("semantic_tooth",)
    assert chunk_by_id["detail_beta_lod1"].element_ids == ("compact_chip",)
    assert chunk_by_id["fallback_gaussian_lod2"].element_ids == ("fallback_sample",)
    assert {element.id: element.lod for element in loaded.scene.elements} == {
        "surface_wall": 0,
        "fog_volume": 0,
        "weave_detail": 1,
        "view_residual": 1,
        "semantic_tooth": 0,
        "compact_chip": 1,
        "fallback_sample": 2,
    }
    assert [node.label for node in loaded.scene.semantic_graph.nodes] == ["tooth_12"]
    assert loaded.scene.semantic_graph.nodes[0].element_ids == ("semantic_tooth",)


def test_adaptive_decomposition_records_native_evidence_and_gaussian_fallback(tmp_path):
    samples = (
        EvidenceSample(
            id="semantic_region",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            evidence=RegionEvidence(semantic_confidence=0.9, image_error=0.9),
            semantic_label="crown",
        ),
        EvidenceSample(
            id="volume_region",
            bounds=Bounds((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            evidence=RegionEvidence(fuzzy_confidence=0.85, geometry_confidence=0.2, image_error=0.8),
        ),
        EvidenceSample(
            id="gabor_region",
            bounds=Bounds((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
            evidence=RegionEvidence(high_frequency=0.9, image_error=0.8),
        ),
        EvidenceSample(
            id="neural_region",
            bounds=Bounds((3.0, 0.0, 0.0), (4.0, 1.0, 1.0)),
            evidence=RegionEvidence(view_dependent=0.9, material_confidence=0.1, image_error=0.8),
        ),
        EvidenceSample(
            id="surface_region",
            bounds=Bounds((4.0, 0.0, 0.0), (5.0, 1.0, 0.1)),
            evidence=RegionEvidence(geometry_confidence=0.9, edit_need=0.6, image_error=0.8),
        ),
        EvidenceSample(
            id="beta_region",
            bounds=Bounds((5.0, 0.0, 0.0), (5.5, 0.5, 0.5)),
            evidence=RegionEvidence(compact_detail=0.9, image_error=0.8),
        ),
        EvidenceSample(
            id="weak_region",
            bounds=Bounds((6.0, 0.0, 0.0), (7.0, 1.0, 1.0)),
            evidence=RegionEvidence(image_error=0.1, geometry_confidence=0.3),
        ),
    )

    package_scene(decompose_evidence(samples, name="metadata_contract")).write(tmp_path)
    loaded = load_package(tmp_path)
    by_id = {element.id: element for element in loaded.scene.elements}

    assert {element.carrier_id for element in loaded.scene.elements if element.id != "weak_region"} == {
        "semantic",
        "volume",
        "gabor",
        "neural",
        "surface",
        "beta",
    }
    assert all(
        by_id[element_id].metadata["decomposition_role"] == "native"
        for element_id in by_id
        if element_id != "weak_region"
    )
    assert by_id["semantic_region"].metadata["selected_carrier"] == "semantic"
    assert by_id["semantic_region"].metadata["selection_reason"] == "semantic_confidence>=0.80"
    assert by_id["semantic_region"].metadata["selection_evidence"] == "semantic_confidence=0.900"
    assert "image_error=0.900" in by_id["semantic_region"].metadata["evidence_summary"]
    assert by_id["volume_region"].metadata["selection_reason"] == (
        "fuzzy_confidence>=0.70 and geometry_confidence<0.60"
    )
    assert by_id["gabor_region"].metadata["selection_reason"] == "high_frequency>=0.80"
    assert by_id["neural_region"].metadata["selection_reason"] == (
        "view_dependent>=0.75 and material_confidence<0.50"
    )
    assert by_id["surface_region"].metadata["selection_reason"] == (
        "geometry_confidence>=0.75 and edit_need>=0.40"
    )
    assert by_id["beta_region"].metadata["selection_reason"] == "compact_detail>=0.75"
    assert by_id["weak_region"].carrier_id == "gaussian"
    assert by_id["weak_region"].metadata["selected_carrier"] == "gaussian"
    assert by_id["weak_region"].metadata["decomposition_role"] == "fallback"
    assert by_id["weak_region"].metadata["fallback_label"] == "gaussian_fallback"
    assert by_id["weak_region"].metadata["selection_reason"] == "no native carrier threshold met"


def test_inspect_package_reports_mixed_native_scene(tmp_path):
    scene = decompose_evidence(
        (
            EvidenceSample(
                id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)),
                evidence=RegionEvidence(geometry_confidence=0.9, edit_need=0.6),
            ),
            EvidenceSample(
                id="volume",
                bounds=Bounds((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
                evidence=RegionEvidence(fuzzy_confidence=0.9, geometry_confidence=0.2),
            ),
        ),
        name="inspect_mixed",
    )
    package_scene(scene).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "inspect-package", str(tmp_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["name"] == "inspect_mixed"
    assert payload["carriers"] == ["surface", "volume"]
    assert payload["elementCount"] == 2
    assert payload["semanticObjectCount"] == 0
    assert payload["exchangeTargets"] == ["asset", "gltfFallback", "native", "usdBridge"]
    assert payload["migration"]["actions"] == ["none"]


def test_write_native_demo_cli_writes_mixed_aura_package(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "write-native-demo-package",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    package = load_package(tmp_path)

    assert str(tmp_path) in result.stdout
    assert package.asset.name == "native_demo"
    assert package.scene.carrier_ids() == ["beta", "gabor", "gaussian", "neural", "semantic", "surface", "volume"]


def test_build_native_demo_cli_alias_writes_mixed_aura_package(tmp_path):
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

    assert package.asset.name == "native_demo"
    assert len(package.scene.semantic_graph.nodes) == 2


def test_query_demo_cli_uses_native_scene():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "query-demo", "--x", "-0.5", "--y", "-0.5"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "surface_wall" in result.stdout


def test_surface_payload_supplies_reference_query_normal():
    element = AuraElement(
        id="surface",
        carrier_id="surface",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
        payload={"type": "surface_cell", "normal": [0.0, 0.0, -1.0], "thickness": 0.1, "roughness": 0.5},
    )

    result = element.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert result is not None
    assert result.normal == (0.0, 0.0, -1.0)


def test_volume_payload_controls_reference_query_attenuation():
    element = AuraElement(
        id="volume",
        carrier_id="volume",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
        opacity=0.1,
        payload={"type": "volume_cell", "density": 1.0, "phase_anisotropy": 0.0},
    )

    result = element.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert result is not None
    assert result.transmittance == math.exp(-2.0)


def test_semantic_and_neural_payloads_affect_query_contract():
    semantic = AuraElement(
        id="semantic",
        carrier_id="semantic",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "semantic_feature", "label": "tooth_12", "confidence": 0.8, "feature_refs": []},
    )
    neural = AuraElement(
        id="neural",
        carrier_id="neural",
        bounds=Bounds((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
        payload={"type": "neural_residual", "latent_dim": 16, "residual_scale": 0.8, "model_ref": None},
    )

    semantic_result = semantic.ray_query(Ray(origin=(0.5, 0.5, -1.0), direction=(0.0, 0.0, 1.0)))
    neural_result = neural.ray_query(Ray(origin=(2.5, 0.5, -1.0), direction=(0.0, 0.0, 1.0)))

    assert semantic_result is not None
    assert semantic_result.semantic_id == "tooth_12"
    assert semantic_result.confidence == 0.8
    assert neural_result is not None
    assert neural_result.residual is True


def test_beta_and_gabor_payloads_affect_query_contract():
    beta = AuraElement(
        id="beta",
        carrier_id="beta",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        opacity=0.8,
        payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [0.5, 0.5, 0.1]},
    )
    gabor = AuraElement(
        id="gabor",
        carrier_id="gabor",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        color=(1.0, 1.0, 1.0),
        payload={"type": "gabor_frequency", "frequency": [1.0, 0.0, 0.0], "bandwidth": 1.0, "phase": 0.0},
    )

    beta_center = beta.ray_query(Ray(origin=(0.5, 0.5, -1.0), direction=(0.0, 0.0, 1.0)))
    beta_edge = beta.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    gabor_peak = gabor.ray_query(Ray(origin=(0.25, 0.5, -1.0), direction=(0.0, 0.0, 1.0)))
    gabor_trough = gabor.ray_query(Ray(origin=(0.75, 0.5, -1.0), direction=(0.0, 0.0, 1.0)))

    assert beta_center is not None
    assert beta_edge is not None
    assert beta_center.transmittance < beta_edge.transmittance
    assert gabor_peak is not None
    assert gabor_trough is not None
    assert gabor_peak.color[0] > gabor_trough.color[0]
