from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aura.asset import AuraAsset
from aura.carriers import default_registry
from aura.scene import AuraScene


@dataclass(frozen=True)
class AuraPackage:
    asset: AuraAsset
    scene: AuraScene

    def manifest(self) -> dict:
        registry = default_registry()
        return {
            "format": "AURA",
            "version": self.asset.version,
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

