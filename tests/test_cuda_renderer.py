import importlib.util

import pytest

from aura import AuraElement, AuraScene, Bounds, Ray
from aura.cuda_renderer import cuda_render_rays, cuda_renderer_boundary_report, cuda_renderer_launch_config


def test_cuda_renderer_launch_config_validates_and_computes_grid():
    config = cuda_renderer_launch_config(
        257,
        threads_per_block=128,
        max_hits=4,
        fallback_backend="cpu",
        device="cuda:0",
    )

    assert config.block_count == 3
    assert config.to_dict() == {
        "rayCount": 257,
        "threadsPerBlock": 128,
        "blockCount": 3,
        "maxHits": 4,
        "fallbackBackend": "cpu",
        "device": "cuda:0",
        "requireCuda": False,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"ray_count": 0}, "ray_count must be positive"),
        ({"ray_count": 1, "threads_per_block": 0}, "threads_per_block must be positive"),
        ({"ray_count": 1, "threads_per_block": 2048}, "threads_per_block must be <= 1024"),
        ({"ray_count": 1, "max_hits": 0}, "max_hits must be positive"),
        ({"ray_count": 1, "fallback_backend": "fake"}, "fallback_backend must be one of"),
    ),
)
def test_cuda_renderer_launch_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cuda_renderer_launch_config(**kwargs)


def test_cuda_render_rays_cpu_fallback_matches_aura_ray_query_contract():
    scene = AuraScene(
        name="cuda_cpu_fallback_scene",
        elements=(
            AuraElement(
                id="fog",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.4)),
                color=(0.1, 0.3, 0.8),
                opacity=0.5,
                confidence=0.6,
                payload={"type": "volume_cell", "density": 0.25},
            ),
            AuraElement(
                id="panel",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.2), (0.5, 0.5, 0.3)),
                color=(0.9, 0.2, 0.1),
                opacity=0.75,
                confidence=0.95,
                normal=(0.0, 0.0, -1.0),
                material_id="enamel",
                payload={"type": "surface_cell"},
            ),
        ),
    )

    batch = cuda_render_rays(
        scene,
        ray_origins=((0.0, 0.0, -1.0), (2.0, 0.0, -1.0)),
        ray_directions=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        fallback_backend="cpu",
        threads_per_block=64,
        max_hits=1,
    )
    payload = batch.to_dict()
    expected_hit = scene.traverse_ray(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    expected_miss = scene.traverse_ray(Ray(origin=(2.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert payload["format"] == "AURA_CUDA_RENDERER_BATCH"
    assert payload["productionReady"] is False
    assert payload["available"] is False
    assert payload["backend"] == "cpu"
    assert payload["reason"] == "cuda_extension_unavailable_cpu_fallback"
    assert payload["launchConfig"]["rayCount"] == 2
    assert payload["launchConfig"]["threadsPerBlock"] == 64
    assert payload["launchConfig"]["blockCount"] == 1
    assert payload["extension"]["buildAttempted"] is False
    assert payload["extension"]["reason"] == "build_not_attempted"

    assert batch.color[0] == pytest.approx(expected_hit.result.color)
    assert batch.transmittance[0] == pytest.approx(expected_hit.result.transmittance)
    assert batch.opacity[0] == pytest.approx(expected_hit.result.opacity)
    assert batch.depth[0] == pytest.approx(expected_hit.result.depth)
    assert batch.normal[0] == expected_hit.result.normal
    assert batch.confidence[0] == pytest.approx(expected_hit.result.confidence)
    assert batch.material_ids[0] == expected_hit.result.material_id
    assert batch.semantic_ids[0] == expected_hit.result.semantic_id
    assert batch.residual[0] is expected_hit.result.residual
    assert batch.provenance[0] == expected_hit.result.provenance
    assert batch.element_ids[0] == expected_hit.ordered_hits[0].element_id
    assert batch.carrier_ids[0] == expected_hit.ordered_hits[0].carrier_id
    assert batch.ordered_hits[0][0]["elementId"] == expected_hit.ordered_hits[0].element_id
    assert batch.ordered_hit_overflow[0] is True

    assert batch.color[1] == pytest.approx(expected_miss.result.color)
    assert batch.transmittance[1] == pytest.approx(1.0)
    assert batch.depth[1] is None
    assert batch.element_ids[1] is None
    assert batch.carrier_ids[1] is None
    assert batch.provenance[1] == "miss"
    assert batch.ordered_hits[1] == ()
    assert batch.ordered_hit_overflow[1] is False


def test_cuda_renderer_boundary_report_distinguishes_callable_fallback_from_production_cuda():
    scene = AuraScene(
        name="cuda_boundary_report_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.8, 0.1, 0.2),
                opacity=0.7,
                confidence=0.9,
                payload={"type": "surface_cell"},
            ),
        ),
    )

    report = cuda_renderer_boundary_report(scene, fallback_backend="cpu", max_hits=4)

    assert report["format"] == "AURA_CUDA_RENDERER_BOUNDARY_REPORT"
    assert report["apiName"] == "aura.cuda_renderer.cuda_render_rays"
    assert report["callableBoundaryAvailable"] is True
    assert report["available"] is False
    assert report["productionReady"] is False
    assert report["fallbackProbe"]["executed"] is True
    assert report["fallbackProbe"]["backend"] == "cpu"
    assert report["fallbackProbe"]["rayCount"] == 1
    assert report["fallbackProbe"]["maxHits"] == 4
    assert set(report["fallbackProbe"]["outputFields"]).issuperset(
        {"color", "transmittance", "depth", "normal", "confidence", "orderedHits"}
    )
    assert "compiled_cuda_renderer_dispatch_missing" in report["productionBlockers"]
    assert "not production CUDA acceleration" in report["notes"]


def test_cuda_renderer_boundary_report_without_scene_is_metadata_only():
    report = cuda_renderer_boundary_report()

    assert report["callableBoundaryAvailable"] is True
    assert report["productionReady"] is False
    assert report["fallbackProbe"] is None


def test_cuda_render_rays_rejects_invalid_ray_batches_before_fallback():
    scene = AuraScene(name="invalid_cuda_batch_scene", elements=())

    with pytest.raises(ValueError, match="does not match"):
        cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        )

    with pytest.raises(ValueError, match="ray_directions must contain 3D ray vectors"):
        cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0),),
        )


def test_cuda_render_rays_refuses_to_fallback_when_cuda_is_required():
    scene = AuraScene(name="require_cuda_scene", elements=())

    with pytest.raises(RuntimeError, match="CUDA renderer extension is unavailable"):
        cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0, 1.0),),
            require_cuda=True,
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_torch_fallback_matches_aura_ray_query_contract():
    scene = AuraScene(
        name="cuda_torch_fallback_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.8, 0.1, 0.2),
                opacity=0.7,
                confidence=0.9,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    ray = Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0))

    batch = cuda_render_rays(
        scene,
        ray_origins=(ray.origin,),
        ray_directions=(ray.direction,),
        fallback_backend="torch",
        device="cpu",
    )
    expected = scene.traverse_ray(ray)

    assert batch.backend == "torch"
    assert batch.device == "cpu"
    assert batch.reason == "cuda_extension_unavailable_torch_fallback"
    assert batch.color[0] == pytest.approx(expected.result.color)
    assert batch.transmittance[0] == pytest.approx(expected.result.transmittance)
    assert batch.depth[0] == pytest.approx(expected.result.depth)
    assert batch.normal[0] == expected.result.normal
    assert batch.element_ids[0] == expected.ordered_hits[0].element_id
    assert batch.ordered_hits[0][0]["elementId"] == expected.ordered_hits[0].element_id
