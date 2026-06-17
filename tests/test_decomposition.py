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
    assert loaded.scene.chunks[0].element_ids == tuple(sample.id for sample in samples)


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
