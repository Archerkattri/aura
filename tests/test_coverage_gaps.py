"""Targeted tests to close coverage gaps in semantic, ray, elements, and baselines modules."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aura.elements import AuraElement, AuraChunk, Bounds
from aura.ray import Ray, RayQueryResult
from aura.semantic import SemanticEdge, SemanticGraph, SemanticNode, decode_semantic_feature


# ---------------------------------------------------------------------------
# ray.py — missing lines 12, 60
# ---------------------------------------------------------------------------


def test_check_vec3_rejects_wrong_length():
    """Line 12 ray.py: _check_vec3 raises ValueError when length != 3."""
    from aura.ray import _check_vec3
    with pytest.raises(ValueError, match="three values"):
        _check_vec3("origin", (1.0, 2.0))


def test_ray_query_result_rejects_negative_depth():
    """Line 60 ray.py: RayQueryResult raises ValueError for negative depth."""
    with pytest.raises(ValueError, match="depth"):
        RayQueryResult(
            color=(0.0, 0.0, 0.0),
            transmittance=1.0,
            confidence=0.0,
            depth=-1.0,
        )


# ---------------------------------------------------------------------------
# semantic.py — missing lines 23, 25, 27, 57, 59, 61, 90, 93, 140, 144, 147
# ---------------------------------------------------------------------------


def test_semantic_node_rejects_empty_id():
    """Line 23 semantic.py: SemanticNode raises ValueError for empty id."""
    with pytest.raises(ValueError, match="id"):
        SemanticNode(id="", label="room")


def test_semantic_node_rejects_empty_label():
    """Line 25 semantic.py: SemanticNode raises ValueError for empty label."""
    with pytest.raises(ValueError, match="label"):
        SemanticNode(id="n1", label="")


def test_semantic_node_rejects_confidence_out_of_range():
    """Line 27 semantic.py: SemanticNode raises ValueError for confidence > 1."""
    with pytest.raises(ValueError, match="confidence"):
        SemanticNode(id="n1", label="room", confidence=1.5)


def test_semantic_edge_rejects_empty_source_or_target():
    """Line 57 semantic.py: SemanticEdge raises ValueError for empty source/target."""
    with pytest.raises(ValueError, match="source and target"):
        SemanticEdge(source="", target="n2", relation="contains")
    with pytest.raises(ValueError, match="source and target"):
        SemanticEdge(source="n1", target="", relation="contains")


def test_semantic_edge_rejects_empty_relation():
    """Line 59 semantic.py: SemanticEdge raises ValueError for empty relation."""
    with pytest.raises(ValueError, match="relation"):
        SemanticEdge(source="n1", target="n2", relation="")


def test_semantic_edge_rejects_confidence_out_of_range():
    """Line 61 semantic.py: SemanticEdge raises ValueError for confidence < 0."""
    with pytest.raises(ValueError, match="confidence"):
        SemanticEdge(source="n1", target="n2", relation="contains", confidence=-0.1)


def test_semantic_graph_rejects_duplicate_node_ids():
    """Line 90 semantic.py: SemanticGraph raises ValueError for duplicate node ids."""
    n1 = SemanticNode(id="n1", label="room")
    n2 = SemanticNode(id="n1", label="hall")  # same id
    with pytest.raises(ValueError, match="duplicate node ids"):
        SemanticGraph(nodes=(n1, n2))


def test_semantic_graph_rejects_edge_to_unknown_node():
    """Line 93 semantic.py: SemanticGraph raises ValueError for edge referencing unknown node."""
    n1 = SemanticNode(id="n1", label="room")
    edge = SemanticEdge(source="n1", target="n999", relation="adjacent")
    with pytest.raises(ValueError, match="unknown node"):
        SemanticGraph(nodes=(n1,), edges=(edge,))


def test_decode_semantic_feature_dense_path_returns_none():
    """Line 130 semantic.py: decode_semantic_feature returns None for dense path."""
    result = decode_semantic_feature({"use_sparse_codebook": False})
    assert result is None

    # Default (no key) also takes dense path
    result = decode_semantic_feature({})
    assert result is None


def test_decode_semantic_feature_sparse_no_indices_returns_zero_vector():
    """Line 140 semantic.py: returns zero vector when sparse_indices/weights are None."""
    result = decode_semantic_feature(
        {"use_sparse_codebook": True, "codebook_dim": 4},
        codebook=None,
    )
    assert result == [0.0, 0.0, 0.0, 0.0]


def test_decode_semantic_feature_sparse_no_codebook_returns_zero_vector():
    """Line 144 semantic.py: returns zero vector when codebook is None but indices present."""
    result = decode_semantic_feature(
        {
            "use_sparse_codebook": True,
            "sparse_indices": [0],
            "sparse_weights": [1.0],
            "codebook_dim": 3,
        },
        codebook=None,
    )
    assert result == [0.0, 0.0, 0.0]


def test_decode_semantic_feature_sparse_length_mismatch_raises():
    """Line 147 semantic.py: raises ValueError when sparse_indices and sparse_weights differ in length."""
    with pytest.raises(ValueError, match="length"):
        decode_semantic_feature(
            {
                "use_sparse_codebook": True,
                "sparse_indices": [0, 1],
                "sparse_weights": [1.0],
                "codebook_dim": 2,
            },
            codebook=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_decode_semantic_feature_sparse_weighted_sum():
    """Lines 153-158 semantic.py: correctly computes weighted sum from codebook."""
    codebook = [[1.0, 0.0], [0.0, 1.0]]
    result = decode_semantic_feature(
        {
            "use_sparse_codebook": True,
            "sparse_indices": [0, 1],
            "sparse_weights": [0.5, 0.5],
            "codebook_dim": 2,
        },
        codebook=codebook,
    )
    assert result == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# elements.py — missing lines 20, 71, 73, 75, 77, 80, 194, 197, 205, 221
# ---------------------------------------------------------------------------


def test_bounds_rejects_inverted_corners():
    """Line 20 elements.py: Bounds raises ValueError when min > max on any axis."""
    with pytest.raises(ValueError, match="min <= max"):
        Bounds((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))


def test_aura_element_rejects_empty_id():
    """Line 71 elements.py: AuraElement raises ValueError for empty id."""
    with pytest.raises(ValueError, match="element id"):
        AuraElement(
            id="",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )


def test_aura_element_rejects_empty_carrier_id():
    """Line 73 elements.py: AuraElement raises ValueError for empty carrier_id."""
    with pytest.raises(ValueError, match="carrier_id"):
        AuraElement(
            id="elem1",
            carrier_id="",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )


def test_aura_element_rejects_opacity_out_of_range():
    """Line 75 elements.py: AuraElement raises ValueError for opacity > 1."""
    with pytest.raises(ValueError, match="opacity"):
        AuraElement(
            id="elem1",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=1.5,
        )


def test_aura_element_rejects_confidence_out_of_range():
    """Line 77 elements.py: AuraElement raises ValueError for confidence < 0."""
    with pytest.raises(ValueError, match="confidence"):
        AuraElement(
            id="elem1",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            confidence=-0.1,
        )


def test_aura_element_rejects_confidence_map_out_of_range():
    """Line 80 elements.py: AuraElement raises ValueError for confidence_map value > 1."""
    with pytest.raises(ValueError, match="confidence_map"):
        AuraElement(
            id="elem1",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            confidence_map={"view_0": 1.5},
        )


def _make_element(**kwargs) -> AuraElement:
    defaults = dict(
        id="test_elem",
        carrier_id="surface",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
        color=(1.0, 1.0, 1.0),
        opacity=1.0,
    )
    defaults.update(kwargs)
    return AuraElement(**defaults)


def _forward_ray() -> Ray:
    return Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0))


def test_element_payload_surface_cell_extracts_normal():
    """Lines 99-100 elements.py: surface_cell payload provides normal from payload when element normal is None."""
    elem = _make_element(
        payload={"type": "surface_cell", "normal": [0.0, 1.0, 0.0]},
        normal=None,
    )
    result = elem.ray_query(_forward_ray())
    assert result is not None
    assert result.normal == (0.0, 1.0, 0.0)


def test_element_payload_volume_cell_computes_transmittance():
    """Lines 101-106 elements.py: volume_cell payload computes alpha from density/path_length."""
    elem = _make_element(
        payload={"type": "volume_cell", "density": 1.0, "opacity": 1.0},
        opacity=0.5,
    )
    result = elem.ray_query(_forward_ray())
    assert result is not None
    # transmittance should be < 1 due to volume integration
    assert result.transmittance < 1.0


def test_element_payload_beta_kernel_modulates_transmittance():
    """Lines 107-110 elements.py: beta_kernel payload computes weight-based transmittance."""
    elem = _make_element(
        payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0},
        opacity=0.8,
    )
    result = elem.ray_query(_forward_ray())
    assert result is not None
    assert 0.0 <= result.transmittance <= 1.0


def test_element_payload_gabor_frequency_modulates_color():
    """Lines 111-115 elements.py: gabor_frequency payload modulates color with wave pattern."""
    elem = _make_element(
        payload={"type": "gabor_frequency", "frequency": [1.0, 0.0, 0.0], "phase": 0.0, "bandwidth": 0.5},
        color=(0.8, 0.8, 0.8),
    )
    result = elem.ray_query(_forward_ray())
    assert result is not None
    # Color should be in [0, 1] after modulation
    for channel in result.color:
        assert 0.0 <= channel <= 1.0


def test_element_payload_neural_residual_sets_residual_flag():
    """Lines 116-118 elements.py: neural_residual payload sets residual=True."""
    elem = _make_element(
        payload={"type": "neural_residual", "residual_scale": 0.5},
    )
    result = elem.ray_query(_forward_ray())
    assert result is not None
    assert result.residual is True


def test_element_payload_semantic_feature_sets_semantic_id():
    """Lines 119-121 elements.py: semantic_feature payload provides semantic_id from label."""
    elem = _make_element(
        payload={"type": "semantic_feature", "label": "chair", "confidence": 0.9},
        semantic_id=None,
    )
    result = elem.ray_query(_forward_ray())
    assert result is not None
    assert result.semantic_id == "chair"
    assert result.confidence == pytest.approx(0.9)


def test_element_payload_gaussian_fallback_computes_weight():
    """Lines 122-125 elements.py: gaussian_fallback payload computes transmittance via Gaussian weight."""
    mean = [0.0, 0.0, 0.1]
    covariance = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    elem = _make_element(
        payload={"type": "gaussian_fallback", "mean": mean, "covariance": covariance},
        opacity=0.9,
    )
    result = elem.ray_query(_forward_ray())
    assert result is not None
    assert 0.0 <= result.transmittance <= 1.0


def test_element_to_dict_structure():
    """Lines 139-145 elements.py: to_dict returns proper nested bounds structure."""
    elem = _make_element()
    d = elem.to_dict()
    assert "bounds" in d
    assert d["bounds"]["min"] == list(elem.bounds.min_corner)
    assert d["bounds"]["max"] == list(elem.bounds.max_corner)


def test_aura_chunk_to_dict():
    """AuraChunk.to_dict returns correct serialization."""
    b = Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    chunk = AuraChunk(id="c1", bounds=b, element_ids=("e1", "e2"), lod=1)
    d = chunk.to_dict()
    assert d["id"] == "c1"
    assert d["element_ids"] == ["e1", "e2"]
    assert d["lod"] == 1


# ---------------------------------------------------------------------------
# ingest/baselines.py — missing lines 22, 42, 74, 80, 90, 92
# ---------------------------------------------------------------------------

FIXTURE_PLY = Path(__file__).parent / "fixtures" / "tiny_3dgs_export.ply"


def test_baseline_export_to_dict():
    """Line 22 baselines.py: BaselineExport.to_dict serializes all fields."""
    from aura.ingest.baselines import BaselineExport
    export = BaselineExport(
        baseline="3dgs",
        source_path=Path("/data/scene"),
        splat_path=Path("/data/scene/point_cloud.ply"),
        scene_name="my_scene",
    )
    d = export.to_dict()
    assert d["baseline"] == "3dgs"
    assert d["sourcePath"] == "/data/scene"
    assert d["splatPath"] == "/data/scene/point_cloud.ply"
    assert d["sceneName"] == "my_scene"


def test_discover_3dgs_export_rejects_missing_path(tmp_path):
    """Line 42 baselines.py: discover_3dgs_export raises FileNotFoundError for non-existent path."""
    from aura.ingest.baselines import discover_3dgs_export
    with pytest.raises(FileNotFoundError):
        discover_3dgs_export(tmp_path / "does_not_exist")


def test_require_supported_splat_file_rejects_unsupported_extension(tmp_path):
    """Line 74 baselines.py: _require_supported_splat_file raises ValueError for .txt file."""
    from aura.ingest.baselines import discover_3dgs_export
    bad_file = tmp_path / "export.txt"
    bad_file.write_text("not a splat")
    with pytest.raises(ValueError, match="unsupported"):
        discover_3dgs_export(bad_file)


def test_discover_3dgs_export_finds_direct_ply(tmp_path):
    """Line 80 baselines.py: _discover_3dgs_ply finds point_cloud.ply in root directory."""
    from aura.ingest.baselines import discover_3dgs_export
    ply = tmp_path / "point_cloud.ply"
    shutil.copyfile(FIXTURE_PLY, ply)
    export = discover_3dgs_export(tmp_path, name="scene")
    assert export.splat_path == ply


def test_discover_3dgs_export_finds_single_recursive_ply(tmp_path):
    """Line 90 baselines.py: _discover_3dgs_ply returns the only .ply found recursively."""
    from aura.ingest.baselines import discover_3dgs_export
    subdir = tmp_path / "custom_output"
    subdir.mkdir()
    ply = subdir / "my_scene.ply"
    shutil.copyfile(FIXTURE_PLY, ply)
    export = discover_3dgs_export(tmp_path)
    assert export.splat_path == ply


def test_discover_3dgs_export_rejects_empty_directory(tmp_path):
    """Line 92 baselines.py: _discover_3dgs_ply raises FileNotFoundError when no PLY found."""
    from aura.ingest.baselines import discover_3dgs_export
    with pytest.raises(FileNotFoundError, match="no 3DGS PLY"):
        discover_3dgs_export(tmp_path)
