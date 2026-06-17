from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aura.exchange import exchange_plan
from aura.package import AuraPackage
from aura.scene import BVH_CHUNK_THRESHOLD


@dataclass(frozen=True)
class RuntimeExportReport:
    asset: str
    native_contract: dict[str, bool]
    fallback_targets: dict[str, dict[str, Any]]
    carrier_export: tuple[dict[str, Any], ...]
    chunk_export: tuple[dict[str, Any], ...]
    ray_query_contract: dict[str, Any]
    acceleration_contract: dict[str, Any]
    semantic_object_contract: dict[str, Any]
    engine_workflow: dict[str, bool]

    def to_dict(self) -> dict:
        return {
            "format": "AURA_RUNTIME_EXPORT_REPORT",
            "asset": self.asset,
            "nativeContract": self.native_contract,
            "fallbackTargets": self.fallback_targets,
            "carrierExport": list(self.carrier_export),
            "chunkExport": list(self.chunk_export),
            "rayQueryContract": self.ray_query_contract,
            "accelerationContract": self.acceleration_contract,
            "semanticObjectContract": self.semantic_object_contract,
            "engineWorkflow": self.engine_workflow,
        }


def runtime_export_report(package: AuraPackage) -> RuntimeExportReport:
    scene = package.scene
    exchange = dict(package.exchange or exchange_plan(package.asset))
    carriers = tuple(scene.carrier_ids())
    native_contract = {
        "typedCarriers": True,
        "rayQuery": True,
        "confidenceMaps": any(element.confidence_map for element in scene.elements),
        "editMetadata": any(element.edit for element in scene.elements),
        "semanticGraph": bool(scene.semantic_graph.nodes),
        "lodChunks": bool(scene.chunks),
        "fallbacks": bool(package.asset.fallbacks),
    }
    fallback_targets = {
        "gltfFallback": _target_report(exchange.get("gltfFallback", {}), package.asset.fallbacks, carriers),
        "usdBridge": _target_report(exchange.get("usdBridge", {}), package.asset.fallbacks, carriers),
    }
    carrier_export = tuple(_carrier_export_entry(carrier_id) for carrier_id in carriers)
    chunk_export = _chunk_export_entries(scene)
    semantic_object_contract = _semantic_object_contract(scene)
    return RuntimeExportReport(
        asset=package.asset.name,
        native_contract=native_contract,
        fallback_targets=fallback_targets,
        carrier_export=carrier_export,
        chunk_export=chunk_export,
        ray_query_contract=_ray_query_contract(),
        acceleration_contract=_acceleration_contract(scene),
        semantic_object_contract=semantic_object_contract,
        engine_workflow={
            "nativeRuntimeReady": native_contract["typedCarriers"] and native_contract["rayQuery"],
            "gltfPreviewReady": bool(package.asset.fallbacks.get("mesh") or package.asset.fallbacks.get("splat")),
            "usdMetadataReady": bool(exchange.get("usdBridge", {}).get("supports_typed_carriers")),
            "chunkedStreamingReady": bool(chunk_export),
            "objectQueriesReady": semantic_object_contract["supportsObjectRayQuery"],
            "objectEditGroupsReady": semantic_object_contract["supportsObjectEditGroups"],
            "requiresNativeAuraForQueries": True,
        },
    )


def _target_report(target: dict[str, Any], fallbacks: dict[str, str], carriers: tuple[str, ...]) -> dict[str, Any]:
    name = str(target.get("name", "unknown"))
    supports_typed = bool(target.get("supports_typed_carriers", False))
    supports_ray = bool(target.get("supports_ray_query", False))
    return {
        "name": name,
        "supportsTypedCarriers": supports_typed,
        "supportsRayQuery": supports_ray,
        "availableFallbacks": dict(fallbacks),
        "losses": _target_losses(name, supports_typed, supports_ray, carriers),
    }


def _target_losses(name: str, supports_typed: bool, supports_ray: bool, carriers: tuple[str, ...]) -> tuple[str, ...]:
    losses = []
    if not supports_typed:
        losses.append("typed_carrier_semantics")
    if not supports_ray:
        losses.append("runtime_ray_query")
    if "semantic" in carriers and "USD" not in name:
        losses.append("semantic_object_graph")
    if any(carrier in carriers for carrier in ("beta", "gabor", "neural", "volume")) and not supports_typed:
        losses.append("adaptive_native_carriers")
    return tuple(losses)


def _carrier_export_entry(carrier_id: str) -> dict[str, Any]:
    fallback_status = {
        "surface": "mesh_or_metadata_proxy",
        "volume": "metadata_only_without_native_runtime",
        "beta": "metadata_only_without_native_runtime",
        "gabor": "texture_metadata_only_without_native_runtime",
        "neural": "metadata_only_without_native_runtime",
        "semantic": "object_metadata_bridge",
        "gaussian": "splat_fallback_possible",
    }.get(carrier_id, "unknown")
    return {
        "carrierId": carrier_id,
        "nativeAura": "full_contract",
        "gltfFallback": fallback_status,
        "usdBridge": "typed_metadata_no_native_ray_query",
        "requiresNativeRuntimeForRayQuery": carrier_id in {"volume", "beta", "gabor", "neural", "semantic", "gaussian", "surface"},
    }


def _chunk_export_entries(scene: Any) -> tuple[dict[str, Any], ...]:
    element_by_id = {element.id: element for element in scene.elements}
    entries = []
    for chunk in scene.chunks:
        elements = tuple(element_by_id[element_id] for element_id in chunk.element_ids if element_id in element_by_id)
        entries.append(
            {
                "chunkId": chunk.id,
                "lod": chunk.lod,
                "elementIds": list(chunk.element_ids),
                "elementCount": len(chunk.element_ids),
                "carrierIds": sorted({element.carrier_id for element in elements}),
                "bounds": {
                    "min": list(chunk.bounds.min_corner),
                    "max": list(chunk.bounds.max_corner),
                },
                "requiresNativeRuntime": any(
                    element.carrier_id in {"volume", "beta", "gabor", "neural", "semantic", "gaussian", "surface"} for element in elements
                ),
            }
        )
    return tuple(entries)


def _ray_query_contract() -> dict[str, Any]:
    fields = (
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
    )
    return {
        "fields": list(fields),
        "supportsFirstHit": True,
        "supportsOrderedHitTrace": True,
        "supportsCompositing": True,
        "requiresNativeAuraRuntime": True,
    }


def _acceleration_contract(scene: Any) -> dict[str, Any]:
    chunk_count = len(scene.chunks)
    active_mode = "bvh" if chunk_count >= BVH_CHUNK_THRESHOLD else "chunk_linear" if chunk_count else "element_linear"
    return {
        "chunkCount": chunk_count,
        "bvhChunkThreshold": BVH_CHUNK_THRESHOLD,
        "activeTraversalMode": active_mode,
        "supportedTraversalModes": ["element_linear", "chunk_linear", "bvh"],
        "supportsChunkCulling": chunk_count > 0,
        "supportsCachedBvh": chunk_count >= BVH_CHUNK_THRESHOLD,
        "supportsOrderedFrontToBackCandidates": True,
        "supportsUnchunkedElementFallback": True,
        "productionGpuTraversalReady": False,
    }


def _semantic_object_contract(scene: Any) -> dict[str, Any]:
    element_by_id = {element.id: element for element in scene.elements}
    node_entries = []
    owned_element_ids: set[str] = set()
    for node in scene.semantic_graph.nodes:
        element_ids = tuple(str(element_id) for element_id in node.element_ids)
        owned_element_ids.update(element_ids)
        node_elements = tuple(element_by_id[element_id] for element_id in element_ids if element_id in element_by_id)
        editable_elements = [element.id for element in node_elements if element.edit]
        node_entries.append(
            {
                "nodeId": node.id,
                "label": node.label,
                "elementIds": list(element_ids),
                "elementCount": len(element_ids),
                "carrierIds": sorted({element.carrier_id for element in node_elements}),
                "editableElementIds": editable_elements,
                "confidence": node.confidence,
            }
        )
    return {
        "objectCount": len(scene.semantic_graph.nodes),
        "edgeCount": len(scene.semantic_graph.edges),
        "ownedElementCount": len(owned_element_ids),
        "unownedElementCount": max(0, len(scene.elements) - len(owned_element_ids)),
        "supportsUniqueElementOwnership": True,
        "supportsObjectRayQuery": bool(scene.semantic_graph.nodes),
        "supportsObjectEditGroups": any(entry["editableElementIds"] for entry in node_entries),
        "supportsObjectExportMetadata": bool(scene.semantic_graph.nodes),
        "objects": node_entries,
    }
