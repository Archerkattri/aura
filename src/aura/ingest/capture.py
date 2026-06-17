from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from aura.assignment import RegionEvidence
from aura.core import TrainingDataset, TrainingFrame, TrainingRegion
from aura.elements import Bounds
from aura.ray import Vec3


@dataclass(frozen=True)
class CaptureManifest:
    """A real-capture ingest contract before images are loaded or optimized."""

    root: str
    frames: tuple[TrainingFrame, ...]
    regions: tuple[TrainingRegion, ...]

    def to_training_dataset(self) -> TrainingDataset:
        return TrainingDataset(frames=self.frames, regions=self.regions)

    def to_dict(self) -> dict:
        return {
            "format": "AURA_CAPTURE_MANIFEST",
            "root": self.root,
            "frames": [frame.to_dict() for frame in self.frames],
            "regions": [region.to_dict() for region in self.regions],
        }


def load_capture_manifest(path: Path | str) -> CaptureManifest:
    """Load an AURA capture manifest and convert it to training contracts.

    This intentionally does not read image pixels. The GPU-side implementation
    can replace target_color/depth summaries with real differentiable image
    sampling while keeping the same manifest and frame identifiers.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capture manifest JSON must be an object")
    validate_capture_manifest_document(payload)

    root = str(payload.get("root") or manifest_path.parent)
    frames = tuple(_frame_from_capture_payload(item) for item in payload["frames"])
    regions = tuple(_region_from_capture_payload(item) for item in payload["regions"])
    dataset = TrainingDataset(frames=frames, regions=regions)
    _validate_manifest_links(dataset)
    return CaptureManifest(root=root, frames=frames, regions=regions)


def write_capture_manifest_template(path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(capture_manifest_template(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def capture_manifest_template() -> dict:
    return {
        "format": "AURA_CAPTURE_MANIFEST",
        "root": "data/custom-captures/example-scene",
        "frames": [
            {
                "id": "frame_000001",
                "image_path": "images/frame_000001.png",
                "depth_path": "depth/frame_000001.exr",
                "mask_path": "masks/frame_000001.png",
                "camera_model": "pinhole",
                "intrinsics": {"fx": 1200.0, "fy": 1200.0, "cx": 960.0, "cy": 540.0, "width": 1920.0, "height": 1080.0},
                "camera_origin": [0.0, 0.0, -2.0],
                "look_at": [0.0, 0.0, 0.0],
                "target_color": [0.72, 0.68, 0.61],
                "target_depth": 2.0,
                "semantic_label": "room",
            }
        ],
        "regions": [
            {
                "id": "wall_surface_000001",
                "frame_id": "frame_000001",
                "bounds": {"min": [-0.6, -0.4, 0.0], "max": [0.6, 0.4, 0.1]},
                "evidence": {"geometry_confidence": 0.9, "material_confidence": 0.7, "ray_need": 0.8, "edit_need": 0.5},
                "color": [0.72, 0.68, 0.61],
                "opacity": 0.9,
                "confidence": 0.85,
                "normal": [0.0, 0.0, -1.0],
                "material_id": "mat_wall",
                "semantic_label": "wall",
                "fallback_source": "capture-manifest",
            }
        ],
    }


def validate_capture_manifest_document(payload: dict) -> None:
    schema_path = resources.files("aura.schemas").joinpath("capture_manifest.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    try:
        validator.validate(payload)
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.absolute_path)
        location = f" at {path}" if path else ""
        raise ValueError(f"capture_manifest.schema.json validation failed{location}: {exc.message}") from exc


def _frame_from_capture_payload(payload: dict[str, Any]) -> TrainingFrame:
    return TrainingFrame(
        id=str(payload["id"]),
        camera_origin=_vec3(payload["camera_origin"], "camera_origin"),
        look_at=_vec3(payload["look_at"], "look_at"),
        target_color=_vec3(payload["target_color"], "target_color"),
        target_depth=float(payload["target_depth"]),
        semantic_label=str(payload["semantic_label"]) if payload.get("semantic_label") is not None else None,
        image_path=str(payload["image_path"]),
        depth_path=str(payload["depth_path"]) if payload.get("depth_path") is not None else None,
        mask_path=str(payload["mask_path"]) if payload.get("mask_path") is not None else None,
        camera_model=str(payload["camera_model"]) if payload.get("camera_model") is not None else None,
        intrinsics={key: float(value) for key, value in payload["intrinsics"].items()}
        if payload.get("intrinsics") is not None
        else None,
    )


def _region_from_capture_payload(payload: dict[str, Any]) -> TrainingRegion:
    bounds = payload["bounds"]
    evidence = payload.get("evidence", {})
    return TrainingRegion(
        id=str(payload["id"]),
        frame_id=str(payload["frame_id"]),
        bounds=Bounds(min_corner=_vec3(bounds["min"], "bounds.min"), max_corner=_vec3(bounds["max"], "bounds.max")),
        evidence=RegionEvidence(**{key: float(value) for key, value in evidence.items()}),
        color=_vec3(payload["color"], "color") if payload.get("color") is not None else None,
        opacity=float(payload.get("opacity", 1.0)),
        confidence=float(payload.get("confidence", 1.0)),
        normal=_vec3(payload["normal"], "normal") if payload.get("normal") is not None else None,
        material_id=str(payload["material_id"]) if payload.get("material_id") is not None else None,
        semantic_label=str(payload["semantic_label"]) if payload.get("semantic_label") is not None else None,
        fallback_source=str(payload.get("fallback_source", "capture-manifest")),
    )


def _validate_manifest_links(dataset: TrainingDataset) -> None:
    frame_ids = {frame.id for frame in dataset.frames}
    if len(frame_ids) != len(dataset.frames):
        raise ValueError("capture manifest contains duplicate frame ids")
    region_ids = {region.id for region in dataset.regions}
    if len(region_ids) != len(dataset.regions):
        raise ValueError("capture manifest contains duplicate region ids")
    missing = sorted({region.frame_id for region in dataset.regions}.difference(frame_ids))
    if missing:
        raise ValueError(f"capture regions reference unknown frame ids: {', '.join(missing)}")


def _vec3(payload: object, name: str) -> Vec3:
    if not isinstance(payload, list | tuple) or len(payload) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return (float(payload[0]), float(payload[1]), float(payload[2]))
