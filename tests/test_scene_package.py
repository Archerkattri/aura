import json

from aura import AuraElement, AuraScene, Bounds, Ray, package_scene
from aura.cli import demo_scene
from aura.exchange import exchange_plan


def test_scene_ray_query_hits_front_element():
    scene = AuraScene(
        name="fixture",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=0.5,
                confidence=0.8,
            ),
        ),
    )

    result = scene.ray_query(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert result.provenance == "surface"
    assert result.depth == 1.0
    assert result.opacity == 0.5


def test_scene_ray_query_miss_returns_empty_result():
    result = demo_scene().ray_query(Ray(origin=(3.0, 3.0, -2.0), direction=(0.0, 0.0, 1.0)))

    assert result.provenance == "miss"
    assert result.transmittance == 1.0


def test_package_writer_outputs_manifest(tmp_path):
    package = package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb"})
    package.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["format"] == "AURA"
    assert manifest["capabilities"]["rayQuery"] is True
    assert manifest["fallbacks"]["mesh"] == "fallback/preview.glb"


def test_exchange_plan_keeps_native_contract_distinct():
    package = package_scene(demo_scene())
    plan = exchange_plan(package.asset)

    assert ".aura" in plan["native"]
    assert plan["gltfFallback"]["supports_ray_query"] is False

