import json
import subprocess
import sys

from aura import load_package, package_scene, runtime_export_report
from aura.cli import native_demo_scene


def test_runtime_export_report_separates_native_contract_from_fallbacks(tmp_path):
    package_scene(native_demo_scene(), fallbacks={"mesh": "fallback/native-preview.glb", "splat": "fallback/native-preview.splat"}).write(
        tmp_path
    )
    package = load_package(tmp_path)

    report = runtime_export_report(package).to_dict()

    assert report["format"] == "AURA_RUNTIME_EXPORT_REPORT"
    assert report["nativeContract"]["typedCarriers"] is True
    assert report["nativeContract"]["rayQuery"] is True
    assert report["nativeContract"]["semanticGraph"] is True
    assert report["fallbackTargets"]["gltfFallback"]["supportsRayQuery"] is False
    assert "runtime_ray_query" in report["fallbackTargets"]["gltfFallback"]["losses"]
    assert "adaptive_native_carriers" in report["fallbackTargets"]["gltfFallback"]["losses"]
    assert report["engineWorkflow"]["nativeRuntimeReady"] is True
    assert report["engineWorkflow"]["gltfPreviewReady"] is True
    assert report["engineWorkflow"]["requiresNativeAuraForQueries"] is True
    by_carrier = {item["carrierId"]: item for item in report["carrierExport"]}
    assert by_carrier["gabor"]["gltfFallback"] == "texture_metadata_only_without_native_runtime"
    assert by_carrier["semantic"]["usdBridge"] == "typed_metadata_no_native_ray_query"


def test_runtime_export_report_uses_exchange_plan_for_in_memory_packages():
    package = package_scene(native_demo_scene(), fallbacks={"mesh": "fallback/native-preview.glb"})

    report = runtime_export_report(package).to_dict()

    assert report["fallbackTargets"]["gltfFallback"]["name"] == "glTF/KHR_gaussian_splatting fallback"
    assert report["fallbackTargets"]["usdBridge"]["supportsTypedCarriers"] is True
    assert report["engineWorkflow"]["usdMetadataReady"] is True


def test_export_report_cli_prints_runtime_export_json(tmp_path):
    package_scene(native_demo_scene(), fallbacks={"mesh": "fallback/native-preview.glb"}).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "export-report", str(tmp_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_RUNTIME_EXPORT_REPORT"
    assert payload["asset"] == "native_demo"
    assert payload["fallbackTargets"]["usdBridge"]["supportsTypedCarriers"] is True
