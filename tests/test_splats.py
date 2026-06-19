import json
from math import log, sqrt
import struct
from pathlib import Path

import pytest

from aura import Ray, load_3dgs_export, load_3dgs_scene, package_scene
from aura.decomposition import EvidenceSample
from aura.ingest.splats import (
    GaussianSplatSample,
    PlyProperty,
    _color_from_ply_row,
    _logistic,
    _matrix3,
    _rotation_matrix_from_ply_row,
    _union_bounds,
    _unit,
    _vec3,
    load_3dgs_ply,
    splats_to_scene,
)
from aura.elements import Bounds


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
    assert scene.chunk_ids() == ["fallback_gaussian_lod2"]
    assert len(scene.elements) == 2
    assert scene.elements[0].bounds.min_corner == pytest.approx((-0.2, -0.2, -0.1))
    assert scene.elements[0].bounds.max_corner == pytest.approx((0.2, 0.2, 0.1))
    assert scene.chunks[0].element_ids == ("red_front", "blue_back")
    assert scene.elements[0].metadata["decomposition"] == "evidence-v1"
    assert scene.elements[0].metadata["decomposition_role"] == "fallback"
    assert scene.elements[0].metadata["fallback_label"] == "gaussian_fallback"
    assert scene.elements[0].metadata["source"] == "3dgs-export"


def test_splat_scene_ray_query_reports_first_hit_depth_and_transmittance():
    scene = load_3dgs_scene(JSON_FIXTURE, name="fixture", radius_sigma=2.0)

    result = scene.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    # Contribution-weighted expected depth across the blended splats.
    assert result.depth == pytest.approx(0.9549367088607595)
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
    # Contribution-weighted expected depth across the blended splats.
    assert result.depth == pytest.approx(0.9439024390691747)
    assert result.transmittance == pytest.approx((1.0 - 0.7) * (1.0 - 0.4))


def test_package_writer_accepts_ply_splat_fallback(tmp_path):
    scene = load_3dgs_scene(PLY_FIXTURE, radius_sigma=2.0)
    package = package_scene(scene, fallbacks={"splat": str(PLY_FIXTURE)})

    package.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "tiny_3dgs_export"
    assert manifest["fallbacks"]["splat"].endswith("tiny_3dgs_export.ply")


# ---------------------------------------------------------------------------
# PlyProperty: struct_format and byte_size with unsupported types (lines 51, 56-58)
# ---------------------------------------------------------------------------


def test_ply_property_struct_format_raises_for_unsupported_type():
    prop = PlyProperty(name="x", type_name="unknown_type")
    with pytest.raises(ValueError, match="unsupported PLY property type"):
        _ = prop.struct_format


def test_ply_property_byte_size_raises_for_unsupported_type():
    prop = PlyProperty(name="x", type_name="unknown_type")
    with pytest.raises(ValueError, match="unsupported PLY property type"):
        _ = prop.byte_size


def test_ply_property_byte_size_returns_correct_size():
    prop = PlyProperty(name="x", type_name="float")
    assert prop.byte_size == 4


# ---------------------------------------------------------------------------
# _vec3: wrong length (line 73)
# ---------------------------------------------------------------------------


def test_vec3_raises_for_wrong_length():
    with pytest.raises(ValueError, match="mean must have exactly three values"):
        _vec3("mean", [1.0, 2.0])


# ---------------------------------------------------------------------------
# _unit: out of range (line 80)
# ---------------------------------------------------------------------------


def test_unit_raises_for_out_of_range():
    with pytest.raises(ValueError, match="opacity must be in"):
        _unit("opacity", 1.5)


# ---------------------------------------------------------------------------
# _matrix3: not 3×3 (line 86) and non-positive diagonal (line 90)
# ---------------------------------------------------------------------------


def test_matrix3_raises_for_non_3x3():
    with pytest.raises(ValueError, match="covariance must be a 3x3 matrix"):
        _matrix3("covariance", [[1.0, 0.0], [0.0, 1.0]])


def test_matrix3_raises_for_non_positive_diagonal():
    with pytest.raises(ValueError, match="covariance diagonal entries must be positive"):
        _matrix3("covariance", [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# _logistic: negative branch (lines 96-97)
# ---------------------------------------------------------------------------


def test_logistic_negative_value():
    result = _logistic(-5.0)
    assert 0.0 < result < 0.5


def test_logistic_positive_value():
    result = _logistic(5.0)
    assert 0.5 < result < 1.0


# ---------------------------------------------------------------------------
# GaussianSplatSample.from_dict: missing covariance keys (line 122)
# ---------------------------------------------------------------------------


def test_gaussian_splat_from_dict_raises_without_covariance():
    payload = {"mean": [0.0, 0.0, 0.0], "opacity": 0.5}
    with pytest.raises(ValueError, match="splat must define covariance or covariance_diag"):
        GaussianSplatSample.from_dict(payload, 0)


def test_gaussian_splat_from_dict_uses_covariance_diag():
    payload = {
        "mean": [1.0, 2.0, 3.0],
        "opacity": 0.5,
        "covariance_diag": [0.01, 0.04, 0.09],
    }
    sample = GaussianSplatSample.from_dict(payload, 0)
    assert sample.covariance[0][0] == pytest.approx(0.01)
    assert sample.covariance[1][1] == pytest.approx(0.04)
    assert sample.covariance[2][2] == pytest.approx(0.09)
    assert sample.covariance[0][1] == 0.0


# ---------------------------------------------------------------------------
# GaussianSplatSample.to_element (lines 141-144/155)
# ---------------------------------------------------------------------------


def test_gaussian_splat_to_element_builds_aura_element():
    sample = GaussianSplatSample(
        id="test_splat",
        mean=(0.0, 0.0, 0.0),
        covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        opacity=0.5,
        color=(1.0, 0.0, 0.0),
    )
    element = sample.to_element(radius_sigma=2.0, chunk_id="root")
    assert element.id == "test_splat"
    assert element.carrier_id == "gaussian"
    assert element.chunk_id == "root"
    assert element.payload["type"] == "gaussian_fallback"
    assert element.bounds.min_corner == pytest.approx((-0.2, -0.2, -0.2))


# ---------------------------------------------------------------------------
# GaussianSplatSample.to_evidence_sample: radius_sigma <= 0 (line 160)
# ---------------------------------------------------------------------------


def test_to_evidence_sample_raises_for_non_positive_radius_sigma():
    sample = GaussianSplatSample(
        id="s",
        mean=(0.0, 0.0, 0.0),
        covariance=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        opacity=0.5,
        color=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="radius_sigma must be positive"):
        sample.to_evidence_sample(radius_sigma=0.0)


# ---------------------------------------------------------------------------
# _read_3dgs_payload: not a dict (line 218)
# ---------------------------------------------------------------------------


def test_read_3dgs_payload_raises_for_non_dict(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="3DGS export must be a JSON object"):
        load_3dgs_export(path)


# ---------------------------------------------------------------------------
# _samples_from_payload: not a list or empty (line 225)
# ---------------------------------------------------------------------------


def test_samples_from_payload_raises_for_missing_splats_key(tmp_path):
    path = tmp_path / "no_splats.json"
    path.write_text(json.dumps({"scene": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="3DGS export must contain a non-empty splats list"):
        load_3dgs_export(path)


def test_samples_from_payload_raises_for_empty_splats(tmp_path):
    path = tmp_path / "empty_splats.json"
    path.write_text(json.dumps({"splats": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="3DGS export must contain a non-empty splats list"):
        load_3dgs_export(path)


# ---------------------------------------------------------------------------
# load_3dgs_ply: unsupported format (line 245)
# This requires _parse_ply_header to return a header with unsupported format_name.
# We'll patch _parse_ply_header to test load_3dgs_ply's third branch.
# ---------------------------------------------------------------------------


def test_load_3dgs_ply_raises_for_unsupported_format(tmp_path):
    # The only way to get an unsupported format through _parse_ply_header is
    # to craft one — but the parser would reject it. We test via monkeypatching.
    from unittest.mock import patch
    from aura.ingest.splats import PlyHeader, PlyProperty

    fake_header = PlyHeader(
        format_name="binary_big_endian",
        properties=(),
        vertex_count=0,
        data_start=0,
    )
    path = tmp_path / "fake.ply"
    path.write_bytes(b"ply\n")
    with patch("aura.ingest.splats._parse_ply_header", return_value=fake_header):
        with pytest.raises(ValueError, match="unsupported PLY format: binary_big_endian"):
            load_3dgs_ply(path)


# ---------------------------------------------------------------------------
# _load_ascii_ply_samples: too few values (line 258) and wrong vertex count (line 262)
# ---------------------------------------------------------------------------


def _ascii_ply(vertex_count: int, body: str) -> bytes:
    header = f"""ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float opacity
end_header
""".encode("ascii")
    return header + body.encode("ascii")


def test_load_ascii_ply_raises_for_too_few_values_in_row(tmp_path):
    # Row has only 3 values but header expects 7
    data = _ascii_ply(1, "0.0 0.0 0.0\n")
    path = tmp_path / "bad.ply"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="has 3 values for 7 properties"):
        load_3dgs_ply(path)


def test_load_ascii_ply_raises_for_wrong_vertex_count(tmp_path):
    # Declare 2 vertices but only provide 1
    data = _ascii_ply(2, "0.0 0.0 0.0 -2.3 -2.3 -2.3 0.7\n")
    path = tmp_path / "short.ply"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="PLY expected 2 vertices but parsed 1"):
        load_3dgs_ply(path)


def test_load_ascii_ply_skips_blank_lines_in_body(tmp_path):
    """Blank line within vertex_count lines is skipped (line 255), causing count mismatch."""
    # vertex_count=2 but first line is blank → only 1 sample → mismatch error
    data = _ascii_ply(2, "\n0.0 0.0 0.0 -2.3 -2.3 -2.3 0.7\n")
    path = tmp_path / "blank_line.ply"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="PLY expected 2 vertices but parsed 1"):
        load_3dgs_ply(path)


# ---------------------------------------------------------------------------
# _load_binary_little_endian_ply_samples: too short (line 271)
# ---------------------------------------------------------------------------


def test_load_binary_ply_raises_for_too_short_body(tmp_path):
    path = tmp_path / "truncated.ply"
    # Header says 2 vertices but binary body is empty
    path.write_bytes(_binary_ply_header(vertex_count=2))
    with pytest.raises(ValueError, match="PLY binary body is too short"):
        load_3dgs_ply(path)


# ---------------------------------------------------------------------------
# _parse_ply_header: various error branches
# ---------------------------------------------------------------------------


def test_parse_ply_header_raises_for_missing_end_header(tmp_path):
    path = tmp_path / "no_end.ply"
    path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\n")
    with pytest.raises(ValueError, match="PLY export missing end_header"):
        load_3dgs_ply(path)


def test_parse_ply_header_no_newline_after_end_header(tmp_path):
    """end_header with no trailing newline — data_start = marker_offset + len(marker).

    data.find(b'\n', marker_offset) returns -1 when no newline exists anywhere
    after the end_header marker in the entire file. We achieve this by appending
    vertex data with no trailing newline immediately after 'end_header'.
    """
    header = (
        b"ply\n"
        b"format ascii 1.0\n"
        b"element vertex 1\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property float scale_0\n"
        b"property float scale_1\n"
        b"property float scale_2\n"
        b"property float opacity\n"
    )
    # Append end_header and vertex data with NO newline anywhere after end_header
    data = header + b"end_header" + b"0.0 0.0 0.0 -2.3 -2.3 -2.3 0.7"
    path = tmp_path / "no_newline.ply"
    path.write_bytes(data)
    samples = load_3dgs_ply(path)
    assert len(samples) == 1


def test_parse_ply_header_raises_for_non_ply_start(tmp_path):
    path = tmp_path / "bad_start.ply"
    path.write_bytes(b"notply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(ValueError, match="PLY export must start with a ply header"):
        load_3dgs_ply(path)


def test_parse_ply_header_skips_empty_lines(tmp_path):
    """Empty lines in the header body are silently skipped (line 307)."""
    data = b"""ply
format ascii 1.0

element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float opacity
end_header
0.0 0.0 0.0 -2.3 -2.3 -2.3 0.7
"""
    path = tmp_path / "empty_lines.ply"
    path.write_bytes(data)
    samples = load_3dgs_ply(path)
    assert len(samples) == 1


def test_parse_ply_header_raises_for_bad_format_version(tmp_path):
    path = tmp_path / "bad_version.ply"
    path.write_bytes(b"ply\nformat ascii 2.0\nend_header\n")
    with pytest.raises(ValueError, match="PLY export must use format version 1.0"):
        load_3dgs_ply(path)


def test_parse_ply_header_raises_for_list_property(tmp_path):
    path = tmp_path / "list_prop.ply"
    path.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar int vertex_index\nend_header\n"
    )
    with pytest.raises(ValueError, match="PLY vertex list properties are not supported"):
        load_3dgs_ply(path)


def test_parse_ply_header_raises_for_no_vertex_count(tmp_path):
    """end_header reached but no element vertex line → vertex_count is None."""
    path = tmp_path / "no_vertex.ply"
    path.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(ValueError, match="PLY export must define an element vertex count"):
        load_3dgs_ply(path)


def test_parse_ply_header_raises_for_no_format(tmp_path):
    """end_header reached but no format line → format_name is None."""
    path = tmp_path / "no_format.ply"
    path.write_bytes(b"ply\nelement vertex 1\nend_header\n")
    with pytest.raises(ValueError, match="PLY export must define a format"):
        load_3dgs_ply(path)


def test_parse_ply_header_raises_for_unsupported_format_name(tmp_path):
    """format line is present with unsupported name → error before missing-props check."""
    path = tmp_path / "unsupported_fmt.ply"
    path.write_bytes(b"ply\nformat binary_big_endian 1.0\nelement vertex 1\nend_header\n")
    with pytest.raises(ValueError, match="unsupported PLY format: binary_big_endian"):
        load_3dgs_ply(path)


def test_parse_ply_header_raises_for_missing_required_properties(tmp_path):
    """All required properties except scale_* missing."""
    path = tmp_path / "missing_props.ply"
    path.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with pytest.raises(ValueError, match="PLY export missing required properties"):
        load_3dgs_ply(path)


def test_parse_ply_header_raises_when_loop_ends_without_end_header_token(tmp_path):
    """Lines loop exits without hitting end_header token (never reached via parts[0])."""
    # Create a file where 'end_header' appears in the data but the loop processes
    # it as the first token in a line. Actually the loop DOES hit it — the "raise
    # ValueError('PLY export missing end_header')" at line 344 is the fallback
    # when the loop ends without returning. We get there if the header has no
    # end_header LINE at all (the marker search via data.find finds a comment or data).
    # Simpler: use a file where the header marker is embedded in a value, not a line.
    # But the easiest is: the `data.find(b"end_header")` finds it but `lines[0]` is not "ply".
    # We already test that. The line 344 is reached when `parts[0] == "end_header"` never
    # matches during the loop — which means we need `end_header` to not be a line token.
    # Create a header where end_header is inside a comment-like line (no `end_header` as first token).
    # Actually the simplest way: include no end_header line at all but have it in binary body.
    data = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nmore data end_header here\n"
    # data.find(b"end_header") will find it in the last line,
    # but when we split that line it won't start with "end_header" as first token.
    path = tmp_path / "no_end_token.ply"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="PLY export missing end_header"):
        load_3dgs_ply(path)


# ---------------------------------------------------------------------------
# _rotation_matrix_from_ply_row: no rotation keys (identity, line 373)
# and zero quaternion (line 377)
# ---------------------------------------------------------------------------


def test_rotation_matrix_from_ply_row_identity_when_no_rot_keys():
    row = {"x": 0.0, "y": 0.0, "z": 0.0, "scale_0": -1.0, "scale_1": -1.0, "scale_2": -1.0, "opacity": 0.5}
    result = _rotation_matrix_from_ply_row(row)
    assert result == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_rotation_matrix_from_ply_row_raises_for_zero_quaternion():
    row = {"rot_0": 0.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0}
    with pytest.raises(ValueError, match="PLY rotation quaternion must be non-zero"):
        _rotation_matrix_from_ply_row(row)


# ---------------------------------------------------------------------------
# _color_from_ply_row: f_dc path (lines 399-405) and fallback (line 406)
# ---------------------------------------------------------------------------


def test_color_from_ply_row_uses_f_dc_coefficients():
    sh_c0 = 0.28209479177387814
    row = {"f_dc_0": 1.0, "f_dc_1": 0.0, "f_dc_2": -1.0}
    r, g, b = _color_from_ply_row(row)
    assert r == pytest.approx(min(1.0, max(0.0, 0.5 + sh_c0 * 1.0)))
    assert g == pytest.approx(0.5)
    assert b == pytest.approx(max(0.0, 0.5 + sh_c0 * -1.0))


def test_color_from_ply_row_fallback_to_white():
    row = {"x": 0.0}  # no color keys
    r, g, b = _color_from_ply_row(row)
    assert (r, g, b) == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# splats_to_scene: empty samples (line 434)
# ---------------------------------------------------------------------------


def test_splats_to_scene_raises_for_empty_samples():
    with pytest.raises(ValueError, match="samples must be non-empty"):
        splats_to_scene([])


# ---------------------------------------------------------------------------
# _union_bounds (line 453)
# ---------------------------------------------------------------------------


def test_union_bounds_computes_bounding_box():
    b1 = Bounds(min_corner=(-1.0, -1.0, -1.0), max_corner=(1.0, 1.0, 1.0))
    b2 = Bounds(min_corner=(0.0, -2.0, 0.5), max_corner=(2.0, 0.5, 3.0))
    result = _union_bounds([b1, b2])
    assert result.min_corner == (-1.0, -2.0, -1.0)
    assert result.max_corner == (2.0, 1.0, 3.0)


# ---------------------------------------------------------------------------
# load_3dgs_scene with JSON: name from scene key (line 448)
# ---------------------------------------------------------------------------


def test_load_3dgs_scene_uses_scene_key_for_name(tmp_path):
    payload = {
        "scene": "my_scene_name",
        "splats": [
            {
                "id": "s0",
                "mean": [0.0, 0.0, 0.0],
                "covariance": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                "opacity": 0.5,
            }
        ],
    }
    path = tmp_path / "named_scene.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    scene = load_3dgs_scene(path)
    assert scene.name == "my_scene_name"


def test_load_3dgs_scene_uses_stem_when_no_scene_key(tmp_path):
    payload = {
        "splats": [
            {
                "id": "s0",
                "mean": [0.0, 0.0, 0.0],
                "covariance": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                "opacity": 0.5,
            }
        ]
    }
    path = tmp_path / "my_file.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    scene = load_3dgs_scene(path)
    assert scene.name == "my_file"


# ---------------------------------------------------------------------------
# load_3dgs_scene with PLY: explicit name override (line 444)
# ---------------------------------------------------------------------------


def test_load_3dgs_scene_ply_uses_explicit_name():
    scene = load_3dgs_scene(PLY_FIXTURE, name="custom_name")
    assert scene.name == "custom_name"


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
