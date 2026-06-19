import json
import subprocess
import sys

from aura import (
    Bounds,
    DepthEvidencePoint,
    SemanticMaskRegion,
    SparsePointPrior,
    depth_points_to_evidence,
    semantic_masks_to_evidence,
    sparse_points_to_evidence,
    supported_ingest_adapters,
)


def test_supported_ingest_adapters_are_evidence_based():
    adapters = supported_ingest_adapters()
    ids = {adapter.id for adapter in adapters}

    assert {"3dgs", "depth-prior", "semantic-mask", "colmap-sparse", "pixelsplat", "idesplat"}.issubset(ids)
    assert all(adapter.output == "EvidenceSample" for adapter in adapters)


def test_depth_semantic_and_sparse_priors_convert_to_evidence_samples():
    depth = depth_points_to_evidence(
        (DepthEvidencePoint(id="depth_1", position=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), confidence=0.9),)
    )
    semantic = semantic_masks_to_evidence(
        (
            SemanticMaskRegion(
                id="mask_1",
                label="chair",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                confidence=0.8,
            ),
        )
    )
    sparse = sparse_points_to_evidence((SparsePointPrior(id="pt_1", position=(0.1, 0.2, 0.3), confidence=0.7),))

    assert depth[0].metadata["source"] == "depth-prior"
    assert depth[0].normal == (0.0, 0.0, 1.0)
    assert semantic[0].metadata["source"] == "semantic-mask"
    assert semantic[0].semantic_label == "chair"
    assert sparse[0].metadata["source"] == "colmap-sparse-prior"


def test_ingest_adapters_cli_prints_contracts():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "ingest-adapters"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload[0]["id"] == "3dgs"
    assert all(item["output"] == "EvidenceSample" for item in payload)


def test_ingest_adapter_spec_to_dict():
    """Line 24: IngestAdapterSpec.to_dict() returns dict with expected keys."""
    from aura.ingest.evidence import IngestAdapterSpec
    spec = IngestAdapterSpec(id="test", status="implemented", notes="note")
    d = spec.to_dict()
    assert d["id"] == "test"
    assert d["status"] == "implemented"
    assert d["notes"] == "note"
    assert d["output"] == "EvidenceSample"


def test_depth_evidence_point_rejects_non_positive_radius():
    """Line 39: DepthEvidencePoint.to_evidence_sample() raises ValueError when radius <= 0."""
    import pytest
    from aura import DepthEvidencePoint
    point = DepthEvidencePoint(id="p1", position=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), confidence=0.9, radius=0.0)
    with pytest.raises(ValueError, match="radius"):
        point.to_evidence_sample()


def test_bounds_around_rejects_non_positive_radius():
    """Line 127: _bounds_around raises ValueError when radius <= 0."""
    import pytest
    from aura.ingest.evidence import _bounds_around
    with pytest.raises(ValueError, match="radius"):
        _bounds_around((0.0, 0.0, 0.0), -1.0)
