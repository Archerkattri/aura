import importlib.util

import pytest

from aura import AuraElement, AuraScene, Bounds, Ray, RenderTarget, torch_render_targets


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")


def test_torch_render_targets_matches_cpu_ray_query_for_miss():
    scene = AuraScene(
        name="parity_miss_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.2, 0.4, 0.6),
                opacity=0.75,
            ),
        ),
    )

    _assert_torch_matches_cpu_ray_query(
        scene,
        Ray(origin=(2.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
    )


def test_torch_render_targets_matches_cpu_ray_query_for_parallel_axis_outside():
    scene = AuraScene(
        name="parity_parallel_outside_scene",
        elements=(
            AuraElement(
                id="volume",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5)),
                color=(0.1, 0.7, 0.3),
                opacity=0.8,
                payload={"type": "volume_cell", "density": 0.5},
            ),
        ),
    )

    _assert_torch_matches_cpu_ray_query(
        scene,
        Ray(origin=(0.75, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
    )


def test_torch_render_targets_matches_cpu_ray_query_from_inside_volume():
    scene = AuraScene(
        name="parity_inside_volume_scene",
        elements=(
            AuraElement(
                id="inside_volume",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
                color=(0.6, 0.2, 0.1),
                opacity=0.4,
                confidence=0.7,
                payload={"type": "volume_cell", "density": 0.8},
            ),
        ),
    )

    _assert_torch_matches_cpu_ray_query(
        scene,
        Ray(origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0)),
    )


def test_torch_render_targets_matches_cpu_ray_query_for_overlapping_surface_and_volume():
    scene = AuraScene(
        name="parity_overlapping_scene",
        elements=(
            AuraElement(
                id="fog",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.3)),
                color=(0.2, 0.6, 0.8),
                opacity=0.4,
                confidence=0.5,
                payload={"type": "volume_cell", "density": 0.5},
            ),
            AuraElement(
                id="panel",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.05), (0.5, 0.5, 0.1)),
                color=(0.9, 0.1, 0.2),
                opacity=0.6,
                confidence=0.9,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    _assert_torch_matches_cpu_ray_query(
        scene,
        Ray(origin=(0.0, 0.0, -2.0), direction=(0.0, 0.0, 1.0)),
    )


def test_torch_render_targets_matches_cpu_ray_query_for_semantic_payload():
    scene = AuraScene(
        name="parity_semantic_scene",
        elements=(
            AuraElement(
                id="semantic_region",
                carrier_id="semantic",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.3, 0.3, 0.7),
                opacity=0.5,
                material_id="mat_semantic",
                payload={"type": "semantic_feature", "label": "crown", "confidence": 0.85},
            ),
        ),
    )

    _assert_torch_matches_cpu_ray_query(
        scene,
        Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
        target_semantic_id="crown",
        target_material_id="mat_semantic",
    )


def test_torch_render_targets_matches_cpu_ray_query_for_gaussian_fallback_at_entry_mean():
    scene = AuraScene(
        name="parity_gaussian_scene",
        elements=(
            AuraElement(
                id="gaussian",
                carrier_id="gaussian",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                color=(0.8, 0.4, 0.2),
                opacity=0.6,
                confidence=0.75,
                payload={
                    "type": "gaussian_fallback",
                    "mean": (0.0, 0.0, 0.0),
                    "covariance": ((0.25, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 0.25)),
                },
            ),
        ),
    )

    _assert_torch_matches_cpu_ray_query(
        scene,
        Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
    )


def _assert_torch_matches_cpu_ray_query(
    scene: AuraScene,
    ray: Ray,
    *,
    target_semantic_id: str | None = None,
    target_material_id: str | None = None,
) -> None:
    traversal = scene.traverse_ray(ray)
    cpu_result = scene.ray_query(ray)
    assert cpu_result == traversal.result

    batch = torch_render_targets(
        scene,
        (
            RenderTarget(
                frame_id="parity_frame",
                ray=ray,
                target_color=cpu_result.color,
                target_depth=cpu_result.depth if cpu_result.depth is not None and cpu_result.depth > 0.0 else 1.0,
                target_semantic_id=target_semantic_id,
                target_material_id=target_material_id,
                target_normal=cpu_result.normal,
            ),
        ),
        device="cpu",
    )

    first_hit = traversal.ordered_hits[0] if traversal.ordered_hits else None
    assert batch.element_ids == ((first_hit.element_id if first_hit else None),)
    assert batch.carrier_ids == ((first_hit.carrier_id if first_hit else None),)
    assert batch.provenance == ((cpu_result.provenance or "miss"),)
    assert batch.predicted_color[0] == pytest.approx(cpu_result.color)
    if cpu_result.depth is None:
        assert batch.predicted_depth[0] is None
    else:
        assert batch.predicted_depth[0] == pytest.approx(cpu_result.depth)
    assert batch.transmittance[0] == pytest.approx(cpu_result.transmittance)
    assert batch.opacity[0] == pytest.approx(cpu_result.opacity)
    assert batch.confidence[0] == pytest.approx(cpu_result.confidence)
    assert batch.normal == (cpu_result.normal,)
    assert batch.material_ids == (cpu_result.material_id,)
    assert batch.semantic_ids == (cpu_result.semantic_id,)
    assert batch.residual == (cpu_result.residual,)
    assert batch.query_loss == (0.0,)

    assert len(batch.ordered_hits[0]) == len(traversal.ordered_hits)
    for torch_hit, cpu_hit in zip(batch.ordered_hits[0], traversal.ordered_hits):
        assert torch_hit["elementId"] == cpu_hit.element_id
        assert torch_hit["carrierId"] == cpu_hit.carrier_id
        assert torch_hit["depth"] == pytest.approx(cpu_hit.result.depth)
        assert torch_hit["transmittance"] == pytest.approx(cpu_hit.result.transmittance)
        assert torch_hit["opacity"] == pytest.approx(cpu_hit.result.opacity)
        assert torch_hit["provenance"] == cpu_hit.result.provenance
