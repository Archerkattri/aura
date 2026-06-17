import json
import subprocess
import sys

from aura import native_demo_interaction_probes, package_scene
from aura.cli import native_demo_scene


def test_native_demo_interaction_probes_report_occlusion_shadow_and_reflection():
    probes = {probe.label: probe for probe in native_demo_interaction_probes(native_demo_scene())}

    occlusion = probes["inserted_object_occlusion"]
    reflection = probes["reflection_ready_surface"]
    empty = probes["empty_space_control"]

    assert occlusion.first_hit is True
    assert occlusion.occluded is True
    assert occlusion.material_id == "mat_wall_plaster"
    assert occlusion.shadow_ready is True
    assert occlusion.transmittance < 1.0
    assert reflection.reflection_ready is True
    assert reflection.collision_proxy_ready is True
    assert empty.first_hit is False
    assert empty.occluded is False


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
    assert any(item["reflectionReady"] for item in payload)
