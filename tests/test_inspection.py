import json
import subprocess
import sys

import pytest

from aura import native_demo_interaction_probes, package_scene
from aura.cli import native_demo_scene
from aura.inspection import inspect_scene_rays
from aura import AuraElement, AuraScene, Bounds


def test_native_demo_interaction_probes_report_occlusion_shadow_and_reflection():
    probes = {probe.label: probe for probe in native_demo_interaction_probes(native_demo_scene())}

    occlusion = probes["inserted_object_occlusion"]
    reflection = probes["reflection_ready_surface"]
    empty = probes["empty_space_control"]

    assert occlusion.first_hit is True
    assert occlusion.hit_point == (-0.5, -0.5, 0.0)
    assert occlusion.occluded is True
    assert occlusion.material_id == "mat_wall_plaster"
    assert occlusion.shadow_ready is True
    assert occlusion.shadow_direction == (0.0, 0.0, -1.0)
    assert occlusion.shadow_transmittance == 1.0
    assert occlusion.shadow_occluded is False
    assert occlusion.transmittance < 1.0
    assert reflection.reflection_ready is True
    assert reflection.reflection_direction == (0.0, 0.0, -1.0)
    assert reflection.reflection_hit is False
    assert reflection.collision_proxy_ready is True
    assert reflection.collision_distance == 2.0
    assert empty.first_hit is False
    assert empty.hit_point is None
    assert empty.occluded is False
    assert empty.shadow_transmittance is None
    assert empty.reflection_direction is None


def test_inspect_rays_cli_reports_native_demo_probe_json(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "inspect-rays", str(tmp_path), "--native-demo-probes"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    labels = {item["label"] for item in payload}

    assert "inserted_object_occlusion" in labels
    assert any(item["shadowReady"] for item in payload)
    assert any(item["shadowTransmittance"] == 1.0 for item in payload)
    assert any(item["reflectionReady"] for item in payload)
    assert any(item["reflectionDirection"] == [0.0, 0.0, -1.0] for item in payload)
    assert any(item["collisionDistance"] == 2.0 for item in payload)


def test_inspect_scene_rays_returns_empty_for_empty_scene():
    """Lines 120-121 inspection.py: inspect_scene_rays returns empty tuple for scene with no elements."""
    scene = AuraScene(name="empty", elements=())
    result = inspect_scene_rays(scene)
    assert result == ()


def test_inspect_scene_rays_rejects_non_positive_max_rays():
    """Lines 118-119 inspection.py: inspect_scene_rays raises ValueError for max_rays <= 0."""
    scene = AuraScene(name="empty", elements=())
    with pytest.raises(ValueError, match="max_rays"):
        inspect_scene_rays(scene, max_rays=0)


def test_inspect_scene_rays_returns_one_inspection_per_element():
    """Lines 122-128 inspection.py: inspect_scene_rays casts one ray per element up to max_rays."""
    elements = tuple(
        AuraElement(
            id=f"elem_{i}",
            carrier_id="surface",
            bounds=Bounds((float(i), -0.5, 0.0), (float(i) + 0.5, 0.5, 0.1)),
            color=(1.0, 0.0, 0.0),
            opacity=1.0,
        )
        for i in range(3)
    )
    scene = AuraScene(name="multi", elements=elements)
    result = inspect_scene_rays(scene, max_rays=8)
    assert len(result) == 3
    labels = {insp.label for insp in result}
    assert labels == {"elem_0", "elem_1", "elem_2"}


def test_inspect_scene_rays_respects_max_rays_limit():
    """Lines 124 inspection.py: inspect_scene_rays caps at max_rays elements."""
    elements = tuple(
        AuraElement(
            id=f"elem_{i}",
            carrier_id="surface",
            bounds=Bounds((float(i), -0.5, 0.0), (float(i) + 0.5, 0.5, 0.1)),
            color=(1.0, 0.0, 0.0),
            opacity=1.0,
        )
        for i in range(5)
    )
    scene = AuraScene(name="many", elements=elements)
    result = inspect_scene_rays(scene, max_rays=2)
    assert len(result) == 2
