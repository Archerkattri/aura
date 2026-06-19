import aura.scene as scene_module
from aura import AuraChunk, AuraElement, AuraScene, Bounds, Ray, RayTraversal, package_scene, run_reference_benchmark


def test_scene_ray_query_uses_chunk_traversal_candidates():
    scene = _two_chunk_scene()

    traversal = scene.traverse_ray(Ray(origin=(-0.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    result = scene.ray_query(Ray(origin=(-0.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert isinstance(traversal, RayTraversal)
    assert result.provenance == "left_surface"
    assert traversal.result.provenance == "left_surface"
    assert traversal.tested_chunk_ids == ("left_chunk",)
    assert traversal.tested_element_ids == ("left_surface",)
    assert traversal.hit_count == 1
    assert traversal.skipped_element_count == 1


def test_scene_traversal_miss_skips_all_elements_when_chunks_miss():
    scene = _two_chunk_scene()

    traversal = scene.traverse_ray(Ray(origin=(4.0, 4.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert traversal.result.provenance == "miss"
    assert traversal.tested_chunk_ids == ()
    assert traversal.tested_element_ids == ()
    assert traversal.hit_count == 0
    assert traversal.skipped_element_count == 2


def test_scene_traversal_keeps_unchunked_elements_visible():
    scene = _two_chunk_scene()
    orphan = AuraElement(
        id="orphan_surface",
        carrier_id="surface",
        bounds=Bounds((4.0, -0.5, 0.0), (5.0, 0.5, 0.1)),
        color=(0.0, 1.0, 0.0),
        opacity=1.0,
    )
    scene = AuraScene(name=scene.name, elements=(*scene.elements, orphan), chunks=scene.chunks)

    traversal = scene.traverse_ray(Ray(origin=(4.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert traversal.result.provenance == "orphan_surface"
    assert traversal.tested_chunk_ids == ()
    assert traversal.tested_element_ids == ("orphan_surface",)
    assert traversal.hit_count == 1
    assert traversal.skipped_element_count == 2


def test_scene_traversal_uses_bvh_for_multi_chunk_scenes():
    scene = _four_chunk_scene()

    traversal = scene.traverse_ray(Ray(origin=(3.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    acceleration = scene.traversal_acceleration().to_dict()

    assert traversal.result.provenance == "surface_3"
    assert traversal.traversal_mode == "bvh"
    assert traversal.tested_bvh_node_count > 0
    assert traversal.tested_bvh_leaf_count == 1
    assert traversal.tested_bvh_chunk_bounds_count == 1
    assert traversal.tested_chunk_ids == ("chunk_3",)
    assert traversal.tested_element_ids == ("surface_3",)
    assert traversal.skipped_element_count == 3
    assert traversal.to_dict()["traversalMode"] == "bvh"
    assert traversal.to_dict()["testedBvhNodeCount"] == traversal.tested_bvh_node_count
    assert traversal.to_dict()["acceleration"]["bvh"]["testedLeafCount"] == 1
    assert acceleration["activeTraversalMode"] == "bvh"
    assert acceleration["chunkedElementCoverageRate"] == 1.0
    assert acceleration["bvhNodeCount"] == 7
    assert acceleration["bvhLeafCount"] == 4
    assert acceleration["bvhMaxDepth"] == 3
    assert acceleration["bvhLeafChunkCounts"] == [1, 1, 1, 1]


def test_scene_bvh_traversal_miss_reports_node_visits_without_element_tests():
    scene = _four_chunk_scene()

    traversal = scene.traverse_ray(Ray(origin=(12.0, 12.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert traversal.result.provenance == "miss"
    assert traversal.traversal_mode == "bvh"
    assert traversal.tested_bvh_node_count == 1
    assert traversal.tested_bvh_leaf_count == 0
    assert traversal.tested_bvh_chunk_bounds_count == 0
    assert traversal.tested_chunk_ids == ()
    assert traversal.tested_element_ids == ()
    assert traversal.skipped_element_count == 4


def test_scene_chunk_traversal_deduplicates_overlapping_candidate_elements():
    bounds = Bounds((0.0, -0.5, 0.0), (1.0, 0.5, 0.1))
    element = AuraElement(
        id="shared_surface",
        carrier_id="surface",
        bounds=bounds,
        color=(1.0, 0.0, 0.0),
        opacity=0.5,
        chunk_id="near",
    )
    scene = AuraScene(
        name="overlap",
        elements=(element,),
        chunks=(
            AuraChunk(id="near", bounds=bounds, element_ids=("shared_surface",)),
            AuraChunk(id="far", bounds=bounds, element_ids=("shared_surface",)),
        ),
    )

    traversal = scene.traverse_ray(Ray(origin=(0.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert traversal.tested_chunk_ids == ("near", "far")
    assert traversal.tested_element_ids == ("shared_surface",)
    assert traversal.hit_count == 1
    assert traversal.ordered_hits[0].element_id == "shared_surface"


def test_scene_bvh_is_cached_across_repeated_queries(monkeypatch):
    scene = _four_chunk_scene()
    build_count = 0
    original_build_bvh = scene_module._build_bvh

    def counting_build_bvh(chunks):
        nonlocal build_count
        build_count += 1
        return original_build_bvh(chunks)

    monkeypatch.setattr(scene_module, "_build_bvh", counting_build_bvh)

    traversal = scene.traverse_ray(Ray(origin=(3.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    assert traversal.result.provenance == "surface_3"
    first_query_build_count = build_count
    assert first_query_build_count > 0

    for _index in range(2):
        traversal = scene.traverse_ray(Ray(origin=(3.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
        assert traversal.result.provenance == "surface_3"

    assert build_count == first_query_build_count


def test_reference_benchmark_reports_chunk_traversal_metrics():
    package = package_scene(_two_chunk_scene())

    payload = run_reference_benchmark(package, render_width=2, render_height=2)

    traversal = payload["rayQuery"]["chunkTraversal"]
    assert traversal["enabled"] is True
    assert traversal["probeCount"] == 2
    assert traversal["modes"] == ["chunk_linear"]
    assert traversal["testedBvhNodeCount"] == 0
    assert traversal["testedChunkCount"] == 2
    assert traversal["testedElementCount"] == 2
    assert traversal["skippedElementCount"] == 2


def _two_chunk_scene() -> AuraScene:
    left_bounds = Bounds((-1.0, -0.5, 0.0), (0.0, 0.5, 0.1))
    right_bounds = Bounds((1.0, -0.5, 0.0), (2.0, 0.5, 0.1))
    return AuraScene(
        name="two_chunk_scene",
        elements=(
            AuraElement(
                id="left_surface",
                carrier_id="surface",
                bounds=left_bounds,
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                chunk_id="left_chunk",
            ),
            AuraElement(
                id="right_surface",
                carrier_id="surface",
                bounds=right_bounds,
                color=(0.0, 0.0, 1.0),
                opacity=1.0,
                chunk_id="right_chunk",
            ),
        ),
        chunks=(
            AuraChunk(id="left_chunk", bounds=left_bounds, element_ids=("left_surface",)),
            AuraChunk(id="right_chunk", bounds=right_bounds, element_ids=("right_surface",)),
        ),
    )


def test_scene_chunk_bvh_cached_property_matches_traversal_index():
    """Line 93 scene.py: _chunk_bvh cached property returns the BVH root from traversal index."""
    scene = _four_chunk_scene()
    # Access the cached property directly to cover line 93
    bvh = scene._chunk_bvh
    traversal_bvh = scene._traversal_index.bvh_root
    # Both must be the same object (both are cached)
    assert bvh is traversal_bvh
    # For a 4-chunk scene BVH_CHUNK_THRESHOLD=3, so BVH should exist
    assert bvh is not None


def test_scene_carrier_ids_and_chunk_ids():
    """Lines 85-89 scene.py: carrier_ids and chunk_ids return sorted unique values."""
    scene = _four_chunk_scene()
    carrier_ids = scene.carrier_ids()
    chunk_ids = scene.chunk_ids()
    assert carrier_ids == ["surface"]
    assert chunk_ids == sorted(chunk_ids)
    assert len(chunk_ids) == 4


def test_bvh_node_with_none_child_branch():
    """Line 432 scene.py: BVH traversal skips None children in internal BVH nodes."""
    from aura.scene import _BvhNode, _candidate_chunks_bvh, _BvhTraversalStats
    from aura.elements import Bounds, AuraChunk

    b = Bounds((0.0, -0.5, 0.0), (1.0, 0.5, 0.1))
    chunk = AuraChunk(id="only_chunk", bounds=b, element_ids=())
    # Build a node where right child is explicitly None (only one side)
    leaf = _BvhNode(bounds=b, chunks=(chunk,))
    internal_with_one_child = _BvhNode(bounds=b, chunks=(), left=leaf, right=None)

    ray = Ray(origin=(0.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0))
    found, stats = _candidate_chunks_bvh(ray, internal_with_one_child)
    # Should traverse left child (leaf) but skip None right child
    assert chunk in found


def _four_chunk_scene() -> AuraScene:
    elements = []
    chunks = []
    for index in range(4):
        bounds = Bounds((float(index), -0.5, 0.0), (float(index) + 0.5, 0.5, 0.1))
        element_id = f"surface_{index}"
        chunk_id = f"chunk_{index}"
        elements.append(
            AuraElement(
                id=element_id,
                carrier_id="surface",
                bounds=bounds,
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                chunk_id=chunk_id,
            )
        )
        chunks.append(AuraChunk(id=chunk_id, bounds=bounds, element_ids=(element_id,)))
    return AuraScene(name="four_chunk_scene", elements=tuple(elements), chunks=tuple(chunks))
