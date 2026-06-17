import json
from math import log, sqrt
import struct
from pathlib import Path

import pytest

from aura import Ray, load_3dgs_export, load_3dgs_scene, package_scene
from aura.decomposition import EvidenceSample


JSON_FIXTURE = Path(__file__).parent / "fixtures" / "tiny_3dgs_export.json"
PLY_FIXTURE = Path(__file__).parent / "fixtures" / "tiny_3dgs_export.ply"


def test_load_3dgs_export_reads_means_covariances_and_opacities():
    samples = load_3dgs_export(JSON_FIXTURE)

    assert [sample.id for sample in samples] == ["red_front", "blue_back"]
    assert samples[0].mean == (0.0, 0.0, 0.0)
    assert samples[0].covariance[0][0] == pytest.approx(0.01)
    assert samples[1].covariance[2][2] == pytest.approx(0.0049)
    assert samples[0].opacity == pytest.approx(0.65)


def test_3dgs_splat_exports_evidence_before_aura_decomposition():
    sample = load_3dgs_export(JSON_FIXTURE)[0]

    evidence = sample.to_evidence_sample(radius_sigma=2.0)

    assert isinstance(evidence, EvidenceSample)
    assert evidence.metadata["source"] == "3dgs-export"
    assert evidence.edit["source"] == "aura-ingest:3dgs"
    assert evidence.fallback_source == "3dgs-ingest"
    assert evidence.gaussian_covariance == sample.covariance


def test_load_3dgs_scene_builds_gaussian_aura_elements_with_bounds():
    scene = load_3dgs_scene(JSON_FIXTURE, radius_sigma=2.0)

    assert scene.name == "tiny_splats"
    assert scene.carrier_ids() == ["gaussian"]
    assert scene.chunk_ids() == ["root"]
    assert len(scene.elements) == 2
    assert scene.elements[0].bounds.min_corner == pytest.approx((-0.2, -0.2, -0.1))
    assert scene.elements[0].bounds.max_corner == pytest.approx((0.2, 0.2, 0.1))
    assert scene.chunks[0].element_ids == ("red_front", "blue_back")
    assert scene.elements[0].metadata["decomposition"] == "evidence-v0"
    assert scene.elements[0].metadata["source"] == "3dgs-export"


def test_splat_scene_ray_query_reports_first_hit_depth_and_transmittance():
    scene = load_3dgs_scene(JSON_FIXTURE, name="fixture", radius_sigma=2.0)

    result = scene.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert result.depth == pytest.approx(0.9)
    assert result.transmittance == pytest.approx((1.0 - 0.65) * (1.0 - 0.4))
    assert result.provenance == "red_front,blue_back"
    assert result.confidence > 0.0


def test_package_writer_preserves_splat_fixture_contract(tmp_path):
    scene = load_3dgs_scene(JSON_FIXTURE, name="fixture", radius_sigma=2.0)
    package = package_scene(scene, fallbacks={"splat": str(JSON_FIXTURE)})

    package.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    elements = json.loads((tmp_path / "elements.json").read_text(encoding="utf-8"))

    assert manifest["carrierIds"] == ["gaussian"]
    assert manifest["fallbacks"]["splat"].endswith("tiny_3dgs_export.json")
    assert elements[0]["metadata"]["source"] == "3dgs-export"
    assert elements[0]["payload"]["type"] == "gaussian_fallback"
    assert elements[0]["payload"]["source"] == "3dgs-ingest"


def test_load_3dgs_export_reads_ascii_ply_splats():
    samples = load_3dgs_export(PLY_FIXTURE)

    assert [sample.id for sample in samples] == ["ply_splat_0000", "ply_splat_0001"]
    assert samples[0].mean == (0.0, 0.0, 0.0)
    assert samples[0].covariance[0][0] == pytest.approx(0.01)
    assert samples[0].color == pytest.approx((1.0, 32.0 / 255.0, 16.0 / 255.0))
    assert samples[0].metadata["source_format"] == "ply"
    assert samples[0].metadata["rotation_quaternion"] == "[1.0, 0.0, 0.0, 0.0]"
    assert samples[1].opacity == pytest.approx(0.4)


def test_load_3dgs_export_reads_binary_little_endian_ply_splats(tmp_path):
    path = tmp_path / "tiny_binary_3dgs_export.ply"
    c = sqrt(0.5)
    path.write_bytes(
        _binary_ply_header(vertex_count=2)
        + struct.pack(
            "<fffffffffffBBB",
            0.0,
            0.0,
            0.0,
            log(0.1),
            log(0.1),
            log(0.05),
            1.0,
            0.0,
            0.0,
            0.0,
            0.70,
            255,
            32,
            16,
        )
        + struct.pack(
            "<fffffffffffBBB",
            0.0,
            0.0,
            0.3,
            log(0.2),
            log(0.1),
            log(0.05),
            c,
            0.0,
            0.0,
            c,
            -0.4054651081,
            32,
            64,
            255,
        )
    )

    samples = load_3dgs_export(path)

    assert [sample.id for sample in samples] == ["ply_splat_0000", "ply_splat_0001"]
    assert samples[0].covariance[0][0] == pytest.approx(0.01)
    assert samples[0].color == pytest.approx((1.0, 32.0 / 255.0, 16.0 / 255.0))
    assert samples[1].covariance[0][0] == pytest.approx(0.01)
    assert samples[1].covariance[1][1] == pytest.approx(0.04)
    assert samples[1].opacity == pytest.approx(0.4)


def test_load_3dgs_scene_uses_ply_stem_and_ray_query_contract():
    scene = load_3dgs_scene(PLY_FIXTURE, radius_sigma=2.0)

    assert scene.name == "tiny_3dgs_export"
    assert scene.carrier_ids() == ["gaussian"]
    result = scene.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    assert result.depth == pytest.approx(0.9)
    assert result.transmittance == pytest.approx((1.0 - 0.7) * (1.0 - 0.4))


def test_package_writer_accepts_ply_splat_fallback(tmp_path):
    scene = load_3dgs_scene(PLY_FIXTURE, radius_sigma=2.0)
    package = package_scene(scene, fallbacks={"splat": str(PLY_FIXTURE)})

    package.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "tiny_3dgs_export"
    assert manifest["fallbacks"]["splat"].endswith("tiny_3dgs_export.ply")


def _binary_ply_header(*, vertex_count: int) -> bytes:
    return f"""ply
format binary_little_endian 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property uchar red
property uchar green
property uchar blue
end_header
""".encode("ascii")
