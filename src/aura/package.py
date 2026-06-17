from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aura.asset import AuraAsset
from aura.carriers import default_registry
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.schema import AURA_FORMAT, AURA_SCHEMA_VERSION, AURA_SUPPORTED_MAJOR_VERSIONS
from aura.scene import AuraScene


@dataclass(frozen=True)
class AuraPackage:
    asset: AuraAsset
    scene: AuraScene

    def summary(self) -> dict:
        return {
            "format": AURA_FORMAT,
            "version": self.asset.version,
            "name": self.asset.name,
            "carriers": list(self.asset.carrier_ids),
            "elementCount": len(self.scene.elements),
            "chunkCount": len(self.scene.chunks),
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
            "chunks": self.scene.chunk_ids(),
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
        return out


def package_scene(scene: AuraScene, *, name: str | None = None, fallbacks: dict[str, str] | None = None) -> AuraPackage:
    asset = AuraAsset(name=name or scene.name, carrier_ids=scene.carrier_ids(), fallbacks=fallbacks or {})
    return AuraPackage(asset=asset, scene=scene)


def load_package(package_dir: Path | str) -> AuraPackage:
    root = Path(package_dir)
    manifest = _read_json_object(root / "manifest.json")
    elements_payload = _read_json_list(root / "elements.json")
    chunks_payload = _read_json_list(root / "chunks.json")

    validate_package_documents(manifest, elements_payload, chunks_payload)
    _validate_manifest_shape(manifest)
    elements = tuple(_element_from_dict(item) for item in elements_payload)
    chunks = tuple(_chunk_from_dict(item) for item in chunks_payload)
    scene = AuraScene(name=str(manifest["name"]), elements=elements, chunks=chunks)
    asset = AuraAsset(
        name=str(manifest["name"]),
        carrier_ids=tuple(str(item) for item in manifest["carrierIds"]),
        version=str(manifest["version"]),
        units=str(manifest["units"]),
        coordinate_system=str(manifest["coordinateSystem"]),
        fallbacks={str(key): str(value) for key, value in manifest.get("fallbacks", {}).items()},
    )
    package = AuraPackage(asset=asset, scene=scene)
    validate_package(package, manifest=manifest)
    return package


def validate_package_documents(manifest: dict, elements: list, chunks: list) -> None:
    _validate_json_schema("manifest.schema.json", manifest)
    _validate_json_schema("elements.schema.json", elements)
    _validate_json_schema("chunks.schema.json", chunks)


def validate_package(package: AuraPackage, *, manifest: dict | None = None) -> None:
    _validate_schema_version(package.asset.version)
    registry = default_registry()
    package.asset.capabilities(registry)
    carrier_ids = set(package.asset.carrier_ids)
    element_ids = {element.id for element in package.scene.elements}
    if len(element_ids) != len(package.scene.elements):
        raise ValueError("package contains duplicate element ids")
    for element in package.scene.elements:
        if element.carrier_id not in carrier_ids:
            raise ValueError(f"element {element.id} uses carrier not declared in manifest: {element.carrier_id}")
    for chunk in package.scene.chunks:
        missing = sorted(set(chunk.element_ids).difference(element_ids))
        if missing:
            raise ValueError(f"chunk {chunk.id} references unknown elements: {', '.join(missing)}")
    if manifest is not None and "chunks" in manifest:
        manifest_chunks = sorted(str(item) for item in manifest["chunks"])
        scene_chunks = package.scene.chunk_ids()
        if manifest_chunks != scene_chunks:
            raise ValueError("manifest chunks do not match chunks.json")


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _read_json_list(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    return payload


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
    parts = version.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"manifest version must use major.minor format: {version}")
    major = int(parts[0])
    if major not in AURA_SUPPORTED_MAJOR_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(AURA_SUPPORTED_MAJOR_VERSIONS))
        raise ValueError(f"unsupported AURA major version {major}; supported major versions: {supported}")


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
