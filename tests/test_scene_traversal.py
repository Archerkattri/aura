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

    assert traversal.result.provenance == "surface_3"
    assert traversal.traversal_mode == "bvh"
    assert traversal.tested_bvh_node_count > 0
    assert traversal.tested_chunk_ids == ("chunk_3",)
    assert traversal.tested_element_ids == ("surface_3",)
    assert traversal.skipped_element_count == 3
    assert traversal.to_dict()["traversalMode"] == "bvh"
    assert traversal.to_dict()["testedBvhNodeCount"] == traversal.tested_bvh_node_count


def test_scene_bvh_traversal_miss_reports_node_visits_without_element_tests():
    scene = _four_chunk_scene()

    traversal = scene.traverse_ray(Ray(origin=(12.0, 12.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert traversal.result.provenance == "miss"
    assert traversal.traversal_mode == "bvh"
    assert traversal.tested_bvh_node_count == 1
    assert traversal.tested_chunk_ids == ()
    assert traversal.tested_element_ids == ()
    assert traversal.skipped_element_count == 4


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
