from __future__ import annotations

from dataclasses import dataclass

from aura.asset import AuraAsset


@dataclass(frozen=True)
class ExchangeTarget:
    name: str
    supports_typed_carriers: bool
    supports_ray_query: bool
    notes: str


GLTF_GAUSSIAN_FALLBACK = ExchangeTarget(
    name="glTF/KHR_gaussian_splatting fallback",
    supports_typed_carriers=False,
    supports_ray_query=False,
    notes="Stores preview/fallback splats or mesh, not the full AURA contract.",
)

USD_ASSET_BRIDGE = ExchangeTarget(
    name="OpenUSD bridge",
    supports_typed_carriers=True,
    supports_ray_query=False,
    notes="Carries scene graph, geometry proxies, materials, and metadata; runtime ray-query data remains native AURA.",
)


def exchange_plan(asset: AuraAsset) -> dict:
    return {
        "asset": asset.name,
        "native": ".aura package preserves carrier registry, ray-query data, confidence maps, chunks, and residual metadata.",
        "gltfFallback": GLTF_GAUSSIAN_FALLBACK.__dict__,
        "usdBridge": USD_ASSET_BRIDGE.__dict__,
    }

