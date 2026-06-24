#!/usr/bin/env python3
"""Validate AURA secondary-ray/reflection query behavior.

This gate checks the live scene-query path, not just static docs:
primary hits expose shadow transmittance and reflection vectors, and the probe
results are serialized as a publication-validation artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rate(values):
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/results/secondary_ray_reflection_2026-06-24.json"))
    args = parser.parse_args()

    from aura.cli import native_demo_scene
    from aura.inspection import native_demo_interaction_probes

    probes = tuple(native_demo_interaction_probes(native_demo_scene()))
    hits = tuple(probe for probe in probes if probe.first_hit)
    shadow_ready_rate = _rate(probe.shadow_transmittance is not None for probe in hits)
    shadow_bounds_rate = _rate(
        0.0 <= probe.shadow_transmittance <= 1.0
        for probe in hits
        if probe.shadow_transmittance is not None
    )
    reflection_vector_rate = _rate(probe.reflection_direction is not None for probe in hits)
    collision_rate = _rate(probe.collision_proxy_ready for probe in hits)
    passed = bool(hits) and shadow_ready_rate == 1.0 and shadow_bounds_rate == 1.0 and reflection_vector_rate > 0.0

    payload = {
        "format": "AURA_SECONDARY_RAY_REFLECTION_VALIDATION",
        "passed": passed,
        "scene": "native_demo",
        "probeCount": len(probes),
        "hitProbeCount": len(hits),
        "shadowTransmittanceReadyRate": shadow_ready_rate,
        "shadowTransmittanceWithinBoundsRate": shadow_bounds_rate,
        "reflectionVectorReadyRate": reflection_vector_rate,
        "collisionProxyReadyRate": collision_rate,
        "evidence": [
            "native_demo_interaction_probes casts secondary shadow and reflection rays from live ray-query hits",
            f"{len(hits)} primary-hit probes expose bounded shadow transmittance",
            "at least one primary-hit probe exposes a reflection vector",
        ],
        "claimBoundary": "Validates secondary ray-query readiness; not a photorealistic reflected-image benchmark.",
        "probes": [probe.to_dict() for probe in probes],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
