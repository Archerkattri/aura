"""AURA package assembly, loading, and schema validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aura.asset import AuraAsset
from aura.carrier_payloads import (
    BetaKernelPayload,
    GaussianFallbackPayload,
    GaborFrequencyPayload,
    NeuralResidualPayload,
    SemanticFeaturePayload,
    SurfaceCellPayload,
    VolumeCellPayload,
)
from aura.carriers import default_registry
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.exchange import exchange_plan
from aura.migration import migration_report
from aura.schema import (
    AURA_FORMAT,
    AURA_SCHEMA_VERSION,
    AURA_SUPPORTED_MAJOR_VERSIONS,
    parse_aura_schema_version,
)
from aura.scene import AuraScene
from aura.semantic import SemanticGraph

PAYLOAD_TYPE_BY_CARRIER = {
    "surface": "surface_cell",
    "volume": "volume_cell",
    "beta": "beta_kernel",
    "gabor": "gabor_frequency",
    "neural": "neural_residual",
    "gaussian": "gaussian_fallback",
    "semantic": "semantic_feature",
}

PAYLOAD_CLASS_BY_TYPE = {
    "surface_cell": SurfaceCellPayload,
    "volume_cell": VolumeCellPayload,
    "beta_kernel": BetaKernelPayload,
    "gabor_frequency": GaborFrequencyPayload,
    "neural_residual": NeuralResidualPayload,
    "gaussian_fallback": GaussianFallbackPayload,
    "semantic_feature": SemanticFeaturePayload,
}


@dataclass(frozen=True)
class AuraPackage:
    """An assembled AURA package: asset manifest, scene, and exchange metadata."""

    asset: AuraAsset
    scene: AuraScene
    exchange: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "format": AURA_FORMAT,
            "version": self.asset.version,
            "name": self.asset.name,
            "carriers": list(self.asset.carrier_ids),
            "elementCount": len(self.scene.elements),
            "chunkCount": len(self.scene.chunks),
            "semanticObjectCount": len(self.scene.semantic_graph.nodes),
            "exchangeTargets": sorted((self.exchange or exchange_plan(self.asset)).keys()),
            "migration": migration_report(self.asset.version).to_dict(),
        }

    def manifest(self) -> dict:
        registry = default_registry()
        return {
            "format": AURA_FORMAT,
            "version": self.asset.version or AURA_SCHEMA_VERSION,
            "name": self.asset.name,
            "units": self.asset.units,
            "coordinateSystem": self.asset.coordinate_system,
            "carrierIds": list(self.asset.carrier_ids),
            "capabilities": self.asset.capabilities(registry),
            "fallbacks": dict(self.asset.fallbacks),
            "chunks": [chunk.id for chunk in self.scene.chunks],
            "semanticGraph": "semantic_graph.json",
            "exchange": "exchange.json",
        }

    def write(self, output_dir: Path | str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "elements.json").write_text(
            json.dumps([element.to_dict() for element in self.scene.elements], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in self.scene.chunks], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out / "semantic_graph.json").write_text(
            json.dumps(self.scene.semantic_graph.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out / "exchange.json").write_text(
            json.dumps(self.exchange or exchange_plan(self.asset), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return out


def package_scene(scene: AuraScene, *, name: str | None = None, fallbacks: dict[str, str] | None = None) -> AuraPackage:
    """Wrap an :class:`~aura.scene.AuraScene` in an :class:`AuraPackage` with optional fallback hints."""
    asset = AuraAsset(name=name or scene.name, carrier_ids=scene.carrier_ids(), fallbacks=fallbacks or {})
    return AuraPackage(asset=asset, scene=scene)


def load_package(package_dir: Path | str) -> AuraPackage:
    """Load and validate an on-disk ``.aura`` package directory into an :class:`AuraPackage`."""
    root = Path(package_dir)
    manifest = _read_json_object(root / "manifest.json")
    elements_payload = _read_json_list(root / "elements.json")
    chunks_payload = _read_json_list(root / "chunks.json")
    semantic_graph_payload = _read_json_object(root / "semantic_graph.json")
    exchange_payload = _read_json_object(root / "exchange.json")

    validate_package_documents(manifest, elements_payload, chunks_payload, semantic_graph_payload, exchange_payload)
    _validate_manifest_shape(manifest)
    elements = tuple(_element_from_dict(item) for item in elements_payload)
    chunks = tuple(_chunk_from_dict(item) for item in chunks_payload)
    semantic_graph = SemanticGraph.from_dict(semantic_graph_payload)
    scene = AuraScene(name=str(manifest["name"]), elements=elements, chunks=chunks, semantic_graph=semantic_graph)
    asset = AuraAsset(
        name=str(manifest["name"]),
        carrier_ids=tuple(str(item) for item in manifest["carrierIds"]),
        version=str(manifest["version"]),
        units=str(manifest["units"]),
        coordinate_system=str(manifest["coordinateSystem"]),
        fallbacks={str(key): str(value) for key, value in manifest.get("fallbacks", {}).items()},
    )
    package = AuraPackage(asset=asset, scene=scene, exchange=exchange_payload)
    validate_package(package, manifest=manifest)
    return package


def validate_package_documents(
    manifest: dict,
    elements: list,
    chunks: list,
    semantic_graph: dict | None = None,
    exchange: dict | None = None,
) -> None:
    """Validate raw JSON documents against the AURA package JSON Schemas."""
    _validate_json_schema("manifest.schema.json", manifest)
    _validate_json_schema("elements.schema.json", elements)
    _validate_json_schema("chunks.schema.json", chunks)
    if semantic_graph is not None:
        _validate_json_schema("semantic_graph.schema.json", semantic_graph)
    if exchange is not None:
        _validate_json_schema("exchange.schema.json", exchange)


def validate_package(package: AuraPackage, *, manifest: dict | None = None) -> None:
    """Validate an :class:`AuraPackage` for semantic consistency beyond JSON Schema.

    Checks carrier, chunk, element payload, semantic ownership, and manifest
    cross-reference contracts. Raises :class:`ValueError` on the first violation.
    """
    _validate_schema_version(package.asset.version)
    registry = default_registry()
    expected_capabilities = package.asset.capabilities(registry)
    if manifest is not None and manifest.get("capabilities") != expected_capabilities:
        raise ValueError("manifest capabilities do not match declared carrierIds")
    if package.exchange:
        exchange_asset = package.exchange.get("asset")
        if exchange_asset is not None and exchange_asset != package.asset.name:
            raise ValueError(f"exchange asset {exchange_asset!r} does not match manifest name {package.asset.name!r}")
    carrier_ids = set(package.asset.carrier_ids)
    scene_carrier_ids = set(package.scene.carrier_ids())
    if carrier_ids != scene_carrier_ids:
        missing = sorted(scene_carrier_ids.difference(carrier_ids))
        extra = sorted(carrier_ids.difference(scene_carrier_ids))
        details = []
        if missing:
            details.append(f"missing scene carriers: {', '.join(missing)}")
        if extra:
            details.append(f"unused manifest carriers: {', '.join(extra)}")
        raise ValueError(f"manifest carrierIds do not match scene carriers ({'; '.join(details)})")
    element_ids = {element.id for element in package.scene.elements}
    if len(element_ids) != len(package.scene.elements):
        raise ValueError("package contains duplicate element ids")
    chunk_ids = {chunk.id for chunk in package.scene.chunks}
    if len(chunk_ids) != len(package.scene.chunks):
        raise ValueError("package contains duplicate chunk ids")
    elements_by_id = {element.id: element for element in package.scene.elements}
    for element in package.scene.elements:
        if element.carrier_id not in carrier_ids:
            raise ValueError(f"element {element.id} uses carrier not declared in manifest: {element.carrier_id}")
        if chunk_ids and element.chunk_id not in chunk_ids:
            raise ValueError(f"element {element.id} references unknown chunk: {element.chunk_id}")
        _validate_element_payload(element.id, element.carrier_id, element.payload)
    for chunk in package.scene.chunks:
        chunk_element_ids = tuple(str(element_id) for element_id in chunk.element_ids)
        duplicates = sorted(_duplicate_values(chunk_element_ids))
        if duplicates:
            raise ValueError(f"chunk {chunk.id} contains duplicate elements: {', '.join(duplicates)}")
        missing = sorted(set(chunk_element_ids).difference(element_ids))
        if missing:
            raise ValueError(f"chunk {chunk.id} references unknown elements: {', '.join(missing)}")
        mismatched = sorted(
            element_id
            for element_id in chunk_element_ids
            if elements_by_id[element_id].chunk_id != chunk.id
        )
        if mismatched:
            raise ValueError(f"chunk {chunk.id} contains elements assigned to other chunks: {', '.join(mismatched)}")
        assigned = sorted(element.id for element in package.scene.elements if element.chunk_id == chunk.id)
        omitted = sorted(set(assigned).difference(chunk_element_ids))
        if omitted:
            raise ValueError(f"chunk {chunk.id} omits assigned elements: {', '.join(omitted)}")
        member_lods = sorted({elements_by_id[element_id].lod for element_id in chunk_element_ids})
        if len(member_lods) > 1:
            raise ValueError(f"chunk {chunk.id} mixes element lods: {', '.join(str(lod) for lod in member_lods)}")
        if member_lods and chunk.lod != member_lods[0]:
            raise ValueError(f"chunk {chunk.id} lod {chunk.lod} does not match member element lod {member_lods[0]}")
        outside = sorted(
            element_id
            for element_id in chunk_element_ids
            if not _bounds_contain(chunk.bounds, elements_by_id[element_id].bounds)
        )
        if outside:
            raise ValueError(f"chunk {chunk.id} bounds do not contain elements: {', '.join(outside)}")
    semantic_element_owners: dict[str, str] = {}
    for node in package.scene.semantic_graph.nodes:
        node_element_ids = tuple(str(element_id) for element_id in node.element_ids)
        unique_node_element_ids = set(node_element_ids)
        if len(unique_node_element_ids) != len(node_element_ids):
            duplicates = sorted(_duplicate_values(node_element_ids))
            raise ValueError(f"semantic node {node.id} contains duplicate elements: {', '.join(duplicates)}")
        missing = sorted(unique_node_element_ids.difference(element_ids))
        if missing:
            raise ValueError(f"semantic node {node.id} references unknown elements: {', '.join(missing)}")
        duplicate_owners = sorted(
            element_id
            for element_id in unique_node_element_ids
            if element_id in semantic_element_owners
        )
        if duplicate_owners:
            details = ", ".join(
                f"{element_id} ({semantic_element_owners[element_id]}, {node.id})"
                for element_id in duplicate_owners
            )
            raise ValueError(f"semantic graph assigns elements to multiple nodes: {details}")
        for element_id in unique_node_element_ids:
            semantic_element_owners[element_id] = node.id
    if manifest is not None and "chunks" in manifest:
        manifest_chunks = sorted(str(item) for item in manifest["chunks"])
        chunk_document_ids = sorted(chunk.id for chunk in package.scene.chunks)
        if manifest_chunks != chunk_document_ids:
            raise ValueError("manifest chunks do not match chunks.json")
    if manifest is not None and manifest.get("semanticGraph") != "semantic_graph.json":
        raise ValueError("manifest semanticGraph must reference semantic_graph.json")
    if manifest is not None and manifest.get("exchange") != "exchange.json":
        raise ValueError("manifest exchange must reference exchange.json")


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _read_json_list(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    return payload


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} must contain valid JSON: {exc.msg}") from exc


def _validate_manifest_shape(manifest: dict) -> None:
    required = ("format", "version", "name", "units", "coordinateSystem", "carrierIds")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"manifest missing required keys: {', '.join(missing)}")
    if manifest["format"] != AURA_FORMAT:
        raise ValueError(f"manifest format must be {AURA_FORMAT}")
    _validate_schema_version(str(manifest["version"]))
    if not isinstance(manifest["carrierIds"], list) or not manifest["carrierIds"]:
        raise ValueError("manifest carrierIds must be a non-empty list")
    if "chunks" in manifest and not isinstance(manifest["chunks"], list):
        raise ValueError("manifest chunks must be a list")
    if "fallbacks" in manifest and not isinstance(manifest["fallbacks"], dict):
        raise ValueError("manifest fallbacks must be an object")


def _validate_schema_version(version: str) -> None:
    major = parse_aura_schema_version(version, label="manifest version").major
    if major not in AURA_SUPPORTED_MAJOR_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(AURA_SUPPORTED_MAJOR_VERSIONS))
        raise ValueError(f"unsupported AURA major version {major}; supported major versions: {supported}")


def _bounds_contain(outer: Bounds, inner: Bounds) -> bool:
    return all(
        outer.min_corner[index] <= inner.min_corner[index] and inner.max_corner[index] <= outer.max_corner[index]
        for index in range(3)
    )


def _duplicate_values(values: tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _validate_element_payload(element_id: str, carrier_id: str, payload: dict) -> None:
    if not payload:
        return
    expected = PAYLOAD_TYPE_BY_CARRIER.get(carrier_id)
    actual = payload.get("type")
    if expected is None:
        raise ValueError(f"element {element_id} uses unknown carrier payload mapping: {carrier_id}")
    if actual != expected:
        raise ValueError(f"element {element_id} payload type {actual!r} does not match carrier {carrier_id!r}; expected {expected!r}")
    payload_class = PAYLOAD_CLASS_BY_TYPE.get(actual)
    if payload_class is None:
        raise ValueError(f"element {element_id} uses unknown typed payload: {actual!r}")
    try:
        payload_class.from_dict(payload).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"element {element_id} malformed {actual!r} payload: {exc}") from exc


def _validate_json_schema(schema_name: str, payload: object) -> None:
    schema = _load_schema(schema_name)
    validator = Draft202012Validator(schema)
    try:
        validator.validate(payload)
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.absolute_path)
        location = f" at {path}" if path else ""
        raise ValueError(f"{schema_name} validation failed{location}: {exc.message}") from exc


def _load_schema(schema_name: str) -> dict:
    schema_path = resources.files("aura.schemas").joinpath(schema_name)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _bounds_from_dict(payload: dict) -> Bounds:
    if not isinstance(payload, dict) or "min" not in payload or "max" not in payload:
        raise ValueError("bounds must contain min and max")
    return Bounds(min_corner=tuple(payload["min"]), max_corner=tuple(payload["max"]))  # type: ignore[arg-type]


def _element_from_dict(payload: dict) -> AuraElement:
    if not isinstance(payload, dict):
        raise ValueError("element entry must be an object")
    return AuraElement(
        id=str(payload["id"]),
        carrier_id=str(payload["carrier_id"]),
        bounds=_bounds_from_dict(payload["bounds"]),
        color=tuple(payload.get("color", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
        opacity=float(payload.get("opacity", 1.0)),
        confidence=float(payload.get("confidence", 1.0)),
        normal=tuple(payload["normal"]) if payload.get("normal") is not None else None,  # type: ignore[arg-type]
        material_id=payload.get("material_id"),
        semantic_id=payload.get("semantic_id"),
        residual=bool(payload.get("residual", False)),
        lod=int(payload.get("lod", 0)),
        chunk_id=str(payload.get("chunk_id", "root")),
        metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
        confidence_map={str(key): float(value) for key, value in payload.get("confidence_map", {}).items()},
        edit=dict(payload.get("edit", {})),
        payload=dict(payload.get("payload", {})),
    )


def _chunk_from_dict(payload: dict) -> AuraChunk:
    if not isinstance(payload, dict):
        raise ValueError("chunk entry must be an object")
    return AuraChunk(
        id=str(payload["id"]),
        bounds=_bounds_from_dict(payload["bounds"]),
        element_ids=tuple(str(item) for item in payload.get("element_ids", ())),
        lod=int(payload.get("lod", 0)),
    )
