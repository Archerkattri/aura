import json
import subprocess
import sys

from aura import AuraElement, AuraScene, Bounds, load_package, package_scene, runtime_export_report
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
    assert report["engineWorkflow"]["chunkedStreamingReady"] is True
    assert report["engineWorkflow"]["objectQueriesReady"] is True
    assert report["engineWorkflow"]["objectEditGroupsReady"] is True
    assert report["engineWorkflow"]["requiresNativeAuraForQueries"] is True
    assert report["rayQueryContract"]["fields"] == [
        "firstHit",
        "depth",
        "normal",
        "transmittance",
        "opacity",
        "semanticId",
        "materialId",
        "confidence",
        "residual",
        "provenance",
        "orderedHits",
    ]
    assert report["rayQueryContract"]["supportsOrderedHitTrace"] is True
    assert report["rayQueryContract"]["supportsCompositing"] is True
    assert report["accelerationContract"]["bvhChunkThreshold"] == 3
    assert report["accelerationContract"]["activeTraversalMode"] == "bvh"
    assert report["accelerationContract"]["elementCount"] == 7
    assert report["accelerationContract"]["chunkCount"] == 7
    assert report["accelerationContract"]["chunkedElementCount"] == 7
    assert report["accelerationContract"]["orphanElementCount"] == 0
    assert report["accelerationContract"]["chunkedElementCoverageRate"] == 1.0
    assert report["accelerationContract"]["supportsCachedBvh"] is True
    assert report["accelerationContract"]["supportsOrderedFrontToBackCandidates"] is True
    assert report["accelerationContract"]["supportsUnchunkedElementFallback"] is True
    assert report["accelerationContract"]["candidateOrdering"] == "front_to_back_chunks_then_unchunked_elements"
    assert report["accelerationContract"]["bvhNodeCount"] > 0
    assert report["accelerationContract"]["bvhLeafCount"] > 0
    assert report["accelerationContract"]["bvhMaxDepth"] > 0
    assert report["accelerationContract"]["bvhLeafChunkCounts"]
    assert report["accelerationContract"]["serializedAccelerationMetadataReady"] is True
    assert report["accelerationContract"]["productionGpuTraversalReady"] is False
    assert report["semanticObjectContract"]["objectCount"] == 2
    assert report["semanticObjectContract"]["ownedElementCount"] == 2
    assert report["semanticObjectContract"]["unownedElementCount"] == 5
    assert report["semanticObjectContract"]["supportsUniqueElementOwnership"] is True
    assert report["semanticObjectContract"]["supportsObjectRayQuery"] is True
    assert report["semanticObjectContract"]["supportsObjectEditGroups"] is True
    assert report["semanticObjectContract"]["supportsObjectExportMetadata"] is True
    by_object = {item["nodeId"]: item for item in report["semanticObjectContract"]["objects"]}
    assert by_object["object:wall"]["carrierIds"] == ["surface"]
    assert by_object["object:wall"]["editableElementIds"] == ["surface_wall"]
    assert by_object["object:fixture_object"]["carrierIds"] == ["semantic"]
    assert by_object["object:fixture_object"]["editableElementIds"] == ["semantic_object"]
    by_carrier = {item["carrierId"]: item for item in report["carrierExport"]}
    assert by_carrier["gabor"]["gltfFallback"] == "texture_metadata_only_without_native_runtime"
    assert by_carrier["semantic"]["usdBridge"] == "typed_metadata_no_native_ray_query"
    by_chunk = {item["chunkId"]: item for item in report["chunkExport"]}
    assert by_chunk["base_surface_lod0"]["lod"] == 0
    assert by_chunk["detail_gabor_lod1"]["carrierIds"] == ["gabor"]
    assert by_chunk["fallback_gaussian_lod2"]["carrierIds"] == ["gaussian"]
    assert by_chunk["fallback_gaussian_lod2"]["requiresNativeRuntime"] is True


def test_runtime_export_report_uses_exchange_plan_for_in_memory_packages():
    package = package_scene(native_demo_scene(), fallbacks={"mesh": "fallback/native-preview.glb"})

    report = runtime_export_report(package).to_dict()

    assert report["fallbackTargets"]["gltfFallback"]["name"] == "glTF/KHR_gaussian_splatting fallback"
    assert report["fallbackTargets"]["usdBridge"]["supportsTypedCarriers"] is True
    assert report["accelerationContract"]["activeTraversalMode"] == "bvh"
    assert report["engineWorkflow"]["usdMetadataReady"] is True


def test_runtime_export_report_marks_unchunked_linear_traversal():
    scene = AuraScene(
        name="linear_scene",
        elements=(AuraElement(id="surface", carrier_id="surface", bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 0.1))),),
    )

    report = runtime_export_report(package_scene(scene)).to_dict()

    assert report["accelerationContract"]["activeTraversalMode"] == "element_linear"
    assert report["accelerationContract"]["elementCount"] == 1
    assert report["accelerationContract"]["chunkCount"] == 0
    assert report["accelerationContract"]["chunkedElementCount"] == 0
    assert report["accelerationContract"]["orphanElementCount"] == 1
    assert report["accelerationContract"]["chunkedElementCoverageRate"] == 0.0
    assert report["accelerationContract"]["supportsChunkCulling"] is False
    assert report["accelerationContract"]["supportsCachedBvh"] is False
    assert report["accelerationContract"]["bvhNodeCount"] == 0
    assert report["accelerationContract"]["bvhLeafCount"] == 0
    assert report["accelerationContract"]["bvhMaxDepth"] == 0


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
