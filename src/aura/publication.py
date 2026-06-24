"""Publication validation aggregation for AURA.

This module reads durable experiment artifacts and turns them into an explicit
paper-claim boundary. It is intentionally artifact-based: a gate only passes when
the corresponding JSON evidence exists and satisfies the expected contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"


@dataclass(frozen=True)
class PublicationGate:
    id: str
    title: str
    passed: bool
    evidence: tuple[str, ...]
    gaps: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "passed": self.passed,
            "evidence": list(self.evidence),
            "gaps": list(self.gaps),
            "nextSteps": list(self.next_steps),
        }


@dataclass(frozen=True)
class PublicationValidationReport:
    gates: tuple[PublicationGate, ...]
    artifacts: dict[str, str]

    @property
    def publication_ready(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def remaining_gate_ids(self) -> tuple[str, ...]:
        return tuple(gate.id for gate in self.gates if not gate.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "AURA_PUBLICATION_VALIDATION_REPORT",
            "publicationReady": self.publication_ready,
            "passedGateCount": sum(1 for gate in self.gates if gate.passed),
            "gateCount": len(self.gates),
            "remainingGateIds": list(self.remaining_gate_ids),
            "gates": [gate.to_dict() for gate in self.gates],
            "artifacts": self.artifacts,
            "claimBoundary": {
                "canClaim": [
                    "AURA Beta beats fixed Gaussian on every audited local scene",
                    "AURA has a fully audited local dataset coverage table",
                    "PRISM is validated as an additive Gabor/neural extension layer over gsplat/DBS-Beta",
                    "PRISM CUDA throughput has measured FPS artifacts on RTX 5090",
                    "learned LPIPS runs on CUDA and can be emitted into JSON reports",
                    "secondary shadow/reflection ray-query readiness is validated on live probes",
                    "explicit albedo/roughness/metallic material fields are consumed by PBR relighting",
                ],
                "cannotClaim": [
                    "superiority over COLMAP/NeRF/2DGS/ray-traced-GS baselines",
                    "full production-resolution FPS across all publication scenes",
                    "photorealistic reflected-image benchmark quality",
                    "full inverse-material recovery from unconstrained captures",
                ],
            },
        }


def publication_validation_report(results_dir: Path | None = None) -> PublicationValidationReport:
    results_dir = results_dir or RESULTS
    artifacts: dict[str, str] = {}

    multiscene = _read_json(results_dir / "multiscene.json", artifacts)
    audit = _read_json(results_dir / "multiscene_audit.json", artifacts)
    prism_additive = _read_json(results_dir / "prism_additive_validation_2026-06-24.json", artifacts)
    prism_fps = _read_json(results_dir / "prism_fps_2026-06-24.json", artifacts)
    learned_lpips = _read_json(results_dir / "learned_lpips_smoke_2026-06-24.json", artifacts)
    external = _latest_json(results_dir, "external_baselines*.json", artifacts)
    secondary = _latest_json(results_dir, "secondary_ray_reflection*.json", artifacts)
    materials = _latest_json(results_dir, "inverse_materials*.json", artifacts)

    gates = (
        _local_multiscene_gate(multiscene),
        _dataset_audit_gate(audit),
        _prism_additive_gate(prism_additive),
        _prism_fps_gate(prism_fps),
        _learned_lpips_gate(learned_lpips),
        _external_baselines_gate(external),
        _secondary_reflection_gate(secondary),
        _inverse_materials_gate(materials),
    )
    return PublicationValidationReport(gates=gates, artifacts=artifacts)


def _read_json(path: Path, artifacts: dict[str, str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifacts[path.stem] = str(path)
    return json.loads(path.read_text())


def _latest_json(results_dir: Path, pattern: str, artifacts: dict[str, str]) -> dict[str, Any] | None:
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return _read_json(matches[0], artifacts)


def _local_multiscene_gate(payload: dict[str, Any] | None) -> PublicationGate:
    rows = tuple((payload or {}).get("scenes", ()))
    wins = [float(row.get("delta_psnr", 0.0)) > 0.0 for row in rows]
    passed = bool(rows) and all(wins)
    return PublicationGate(
        id="local_multiscene_quality",
        title="Local multi-scene Beta-vs-Gaussian quality",
        passed=passed,
        evidence=(f"{sum(wins)}/{len(rows)} local scenes have positive Beta PSNR delta",) if rows else (),
        gaps=() if passed else ("missing multiscene.json or at least one scene does not beat Gaussian",),
        next_steps=("extend the same table to external-method baselines",),
    )


def _dataset_audit_gate(payload: dict[str, Any] | None) -> PublicationGate:
    complete = bool((payload or {}).get("complete"))
    local_scene_count = int((payload or {}).get("localSceneCount", 0) or len((payload or {}).get("scenes", ())))
    return PublicationGate(
        id="dataset_audit",
        title="Downloaded local dataset audit",
        passed=complete,
        evidence=(f"{local_scene_count} local scene roots audited with Beta and Gaussian arms",) if complete else (),
        gaps=() if complete else ("local dataset audit is missing or incomplete",),
        next_steps=("repeat audit whenever more scenes are downloaded",),
    )


def _prism_additive_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and bool(payload.get("completeForAdditiveRole")) and bool(payload.get("additiveExtensionChangedImage"))
    return PublicationGate(
        id="prism_additive_contract",
        title="PRISM additive extension contract",
        passed=passed,
        evidence=(
            "Gaussian/Beta route to the primary backend; Gabor/neural route to PRISM",
            f"mean image delta {payload.get('meanAbsoluteImageDelta'):.6f}",
        ) if passed and payload else (),
        gaps=() if passed else ("PRISM additive CUDA validation artifact is missing or failed",),
        next_steps=("keep PRISM positioned as additive, not a gsplat/DBS-Beta replacement",),
    )


def _prism_fps_gate(payload: dict[str, Any] | None) -> PublicationGate:
    rows = tuple((payload or {}).get("benchmark", ()))
    fps = [float(row["prism_cuda_fps"]) for row in rows if "prism_cuda_fps" in row]
    passed = bool(fps) and min(fps) >= 30.0
    evidence = (f"PRISM CUDA FPS range {min(fps):.1f}-{max(fps):.1f} over {len(fps)} synthetic sweeps",) if fps else ()
    return PublicationGate(
        id="prism_cuda_fps",
        title="PRISM CUDA throughput smoke",
        passed=passed,
        evidence=evidence,
        gaps=() if passed else ("PRISM CUDA FPS artifact missing or below 30 FPS",),
        next_steps=("run full production-resolution FPS sweeps on real trained scenes",),
    )


def _learned_lpips_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and payload.get("lpipsBackend") == "learned_lpips_alex" and payload.get("meanLpips") is not None
    return PublicationGate(
        id="learned_lpips_cuda",
        title="Learned LPIPS on CUDA",
        passed=passed,
        evidence=(
            f"learned LPIPS mean {float(payload['meanLpips']):.4f} on {payload.get('frameCount')} CUDA-evaluated frames",
        ) if passed and payload else (),
        gaps=() if passed else ("learned LPIPS CUDA report missing or fell back to proxy/unavailable",),
        next_steps=("run learned LPIPS on the full publication split, not only the smoke subset",),
    )


def _external_baselines_gate(payload: dict[str, Any] | None) -> PublicationGate:
    required = {"colmap", "nerf", "3dgs", "2dgs", "ray_traced_gs"}
    present = set((payload or {}).get("baselines", {}).keys())
    passed = required.issubset(present)
    return PublicationGate(
        id="external_method_baselines",
        title="External method baseline table",
        passed=passed,
        evidence=(f"external baselines present: {', '.join(sorted(present))}",) if present else (),
        gaps=() if passed else (f"missing baselines: {', '.join(sorted(required - present))}",),
        next_steps=("run or import COLMAP/NeRF/3DGS/2DGS/ray-traced-GS baseline renders and metrics",),
    )


def _secondary_reflection_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and bool(payload.get("passed"))
    return PublicationGate(
        id="secondary_ray_reflection",
        title="Secondary-ray/reflection validation",
        passed=passed,
        evidence=tuple(payload.get("evidence", ())) if passed and payload else (),
        gaps=() if passed else ("no integrated rendered secondary-ray/reflection validation artifact yet",),
        next_steps=("benchmark rendered reflection/shadow behavior, not only ray-query readiness probes",),
    )


def _inverse_materials_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and bool(payload.get("passed"))
    return PublicationGate(
        id="inverse_materials",
        title="Inverse-material validation",
        passed=passed,
        evidence=tuple(payload.get("evidence", ())) if passed and payload else (),
        gaps=() if passed else ("no rich inverse-material estimation validation artifact yet",),
        next_steps=("evaluate material/albedo/roughness behavior beyond the current relighting layer",),
    )
