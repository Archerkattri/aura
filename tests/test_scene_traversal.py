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


def test_reference_benchmark_reports_chunk_traversal_metrics():
    package = package_scene(_two_chunk_scene())

    payload = run_reference_benchmark(package, render_width=2, render_height=2)

    traversal = payload["rayQuery"]["chunkTraversal"]
    assert traversal["enabled"] is True
    assert traversal["probeCount"] == 2
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
