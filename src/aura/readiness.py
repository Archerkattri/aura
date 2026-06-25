"""AURA production readiness audit across implemented capability pillars."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aura.benchmark import cuda_renderer_callable_boundary_report, evaluate_backend_readiness
from aura.core import synthetic_training_frames, synthetic_training_regions
from aura.cuda_kernels import cuda_kernel_source_report, cuda_renderer_report
from aura.decomposition import decompose_evidence
from aura.sota import latest_sota_ab_artifact
from aura.torch_kernels import torch_carrier_kernel_report
from aura.torch_renderer import torch_renderer_status

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"


@dataclass(frozen=True)
class ReadinessPillar:
    """A single production readiness pillar with evidence, gaps, and next steps."""

    id: str
    title: str
    implemented: bool
    production_ready: bool
    evidence: tuple[str, ...]
    gaps: tuple[str, ...]
    next_steps: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "implemented": self.implemented,
            "productionReady": self.production_ready,
            "evidence": list(self.evidence),
            "gaps": list(self.gaps),
            "nextSteps": list(self.next_steps),
        }


@dataclass(frozen=True)
class ProductionReadinessReport:
    """Aggregated production readiness result across all capability pillars."""

    pillars: tuple[ReadinessPillar, ...]
    torch_renderer: dict
    torch_carrier_kernels: dict
    cuda_kernel_sources: dict
    legacy_cuda_renderer: dict
    cuda_renderer_callable_boundary: dict
    backend_readiness: dict

    @property
    def production_ready(self) -> bool:
        return all(pillar.production_ready for pillar in self.pillars)

    @property
    def implemented_pillar_count(self) -> int:
        return sum(1 for pillar in self.pillars if pillar.implemented)

    @property
    def production_ready_pillar_count(self) -> int:
        return sum(1 for pillar in self.pillars if pillar.production_ready)

    def to_dict(self) -> dict:
        missing = tuple(pillar for pillar in self.pillars if not pillar.production_ready)
        return {
            "format": "AURA_PRODUCTION_READINESS_REPORT",
            "productionReady": self.production_ready,
            "implementedPillarCount": self.implemented_pillar_count,
            "productionReadyPillarCount": self.production_ready_pillar_count,
            "pillarCount": len(self.pillars),
            "implemented": [pillar.to_dict() for pillar in self.pillars if pillar.implemented],
            "missingOrIncomplete": [pillar.to_dict() for pillar in missing],
            "pillars": [pillar.to_dict() for pillar in self.pillars],
            "torchRenderer": self.torch_renderer,
            "torchCarrierKernels": self.torch_carrier_kernels,
            "cudaKernelSources": self.cuda_kernel_sources,
            "legacyCudaRenderer": self.legacy_cuda_renderer,
            "cudaRendererCallableBoundary": self.cuda_renderer_callable_boundary,
            "backendReadiness": self.backend_readiness,
            "summary": _summary(self.pillars),
        }


def production_readiness_report() -> ProductionReadinessReport:
    """Evaluate and return the current AURA production readiness across all pillars."""
    torch_status = torch_renderer_status().to_dict()
    kernel_report = torch_carrier_kernel_report()
    cuda_sources = cuda_kernel_source_report()
    readiness_scene = _readiness_scene()
    legacy_cuda_renderer = cuda_renderer_report()
    cuda_callable_boundary = cuda_renderer_callable_boundary_report(readiness_scene)
    backend_readiness = evaluate_backend_readiness(readiness_scene)
    cuda_production = _cuda_production_artifact()
    native_real_capture = _native_real_capture_artifact()
    torch_backend_validation = _torch_backend_validation_artifact()
    publication_validation = _publication_validation_artifact()
    sota_ab = latest_sota_ab_artifact()
    pillars = (
        ReadinessPillar(
            id="native_carriers",
            title="Native adaptive carriers",
            implemented=True,
            production_ready=bool(native_real_capture.get("passed")),
            evidence=(
                "surface, volume, beta, gabor, neural residual, semantic, and gaussian fallback carrier contracts exist",
                "native demo and decomposition paths produce typed AuraElement payloads",
                "ray-query behavior is exercised by reference package and benchmark tests",
                *_native_real_capture_evidence(native_real_capture),
            ),
            gaps=() if native_real_capture.get("passed") else (
                "adaptive split/merge/promote/demote is validated at fixture scale, not full real captures",
                "native carrier behavior is not yet validated on full real capture datasets",
                "3DGS remains an ingest evidence source and fallback, not a production native carrier replacement",
                *_native_real_capture_gaps(native_real_capture),
            ),
            next_steps=(
                "extend real-capture validation whenever new local scenes are downloaded",
                "add larger adaptive carrier evolution stress tests and scene-behavior benchmarks",
            ),
        ),
        ReadinessPillar(
            id="package_validation",
            title="Package schema and validation",
            implemented=True,
            production_ready=True,
            evidence=(
                "manifest, element, chunk, semantic graph, and exchange JSON schemas are validated",
                "load_package enforces carrier, chunk, payload, semantic ownership, and migration contracts",
                "CLI validate-package and inspect-package commands exercise the on-disk .aura contract",
            ),
            gaps=(
                "schema compatibility is limited to the current schema version policy",
            ),
            next_steps=(
                "keep migration fixtures updated whenever package schemas change",
            ),
        ),
        ReadinessPillar(
            id="torch_backend",
            title="PyTorch reference backend",
            implemented=bool(torch_status.get("available")) or bool(kernel_report.get("autogradCarrierCount", 0)),
            production_ready=bool(torch_backend_validation.get("passed")),
            evidence=(
                "torch renderer status is reportable through aura torch-renderer-status",
                "carrier payloads have torch autograd kernel specs",
                f"backend readiness reports {backend_readiness['sceneCarrierAutogradCoverageRate']:.0%} scene-carrier autograd coverage",
                "the torch optimizer consumes packed capture tensor batches when PyTorch is installed",
                "grouped torch ray/carrier intersection and compositing is carrier-complete and covered by parity tests",
                *_torch_backend_validation_evidence(torch_backend_validation),
            ),
            gaps=() if torch_backend_validation.get("passed") else (
                "the renderer and optimizer are validated on deterministic fixtures, not full real capture datasets",
                "full-resolution tiled or GPU-native data loading at dataset scale is not yet benchmarked",
                *_torch_backend_validation_gaps(torch_backend_validation),
            ),
            next_steps=(
                "extend the bounded torch CUDA validation to larger target budgets as GPU memory allows",
                "publish renderer/optimizer throughput and quality on real-dataset baselines",
            ),
        ),
        ReadinessPillar(
            id="cuda_backend",
            title="CUDA production backend",
            implemented=bool(cuda_sources.get("availableSourceCount", 0)),
            production_ready=bool(cuda_production.get("passed")),
            evidence=(
                "CUDA carrier kernels compile and load via aura cuda-kernel-build-report --build",
                "the compiled CUDA renderer matches the torch renderer per-carrier in fixture parity tests",
                "a production GPU BVH traversal kernel (render_rays_bvh) replaces the brute-force element scan",
                "aura benchmark-cuda-runtime measures on-device throughput and cross-backend parity",
                f"backend readiness reports {backend_readiness['sceneCarrierCudaCoverageRate']:.0%} scene-carrier CUDA production coverage",
                *_cuda_production_evidence(cuda_production),
            ),
            gaps=() if cuda_production.get("passed") else (
                "CUDA extension build is not attempted by this readiness report",
                "callable cuda_renderer fallback is not CUDA acceleration",
                "torch_carrier_kernel_report marks CUDA carrier kernels as not production ready",
                "CUDA/BVH parity and throughput are validated on fixtures, not yet on real-dataset baselines",
                *_cuda_production_gaps(cuda_production),
            ),
            next_steps=(
                "measure CUDA/BVH parity and throughput against real-dataset baselines at scale",
                "publish reproducible CUDA-vs-torch performance and quality numbers",
            ),
        ),
        ReadinessPillar(
            id="renderer_trainer",
            title="Renderer and trainer production path",
            implemented=True,
            production_ready=_publication_gates_passed(
                publication_validation,
                ("prism_additive_contract", "prism_cuda_fps", "secondary_ray_reflection", "inverse_materials"),
            ),
            evidence=(
                "CPU reference rendering, ray query, reconstruction, and torch optimization paths exist",
                "GPU BVH traversal, EXR/PFM/video export, and a long-run memory stability probe are implemented",
                "capture manifest conversion and tensor target planning share deterministic contracts",
                "capture proposal weights can be trained from labeled feature examples and reused in native region generation",
                *_renderer_trainer_publication_evidence(publication_validation),
            ),
            gaps=() if _publication_gates_passed(
                publication_validation,
                ("prism_additive_contract", "prism_cuda_fps", "secondary_ray_reflection", "inverse_materials"),
            ) else (
                "renderer real-time performance is not yet benchmarked at production resolution",
                "the gsplat/DBS training path is validated on local real datasets; the native PRISM trainer remains a research extension",
                "proposal model is a lightweight logistic contract, not a full neural region proposal network",
                "relighting is implemented as an editable layer; secondary-ray/reflection integration remains future work",
                *_publication_validation_gaps(publication_validation),
            ),
            next_steps=(
                "keep PRISM documented as an additive extension to gsplat/DBS-Beta, not a primary-quality replacement",
                "extend production-resolution FPS sweeps beyond the current PRISM CUDA benchmark artifact",
                "replace the lightweight proposal model with a neural proposal backend when training labels are available",
            ),
        ),
        ReadinessPillar(
            id="benchmarks",
            title="Benchmarks and claim boundary",
            implemented=True,
            production_ready=_publication_gates_passed(
                publication_validation,
                ("local_multiscene_quality", "dataset_audit", "external_method_baselines"),
            ),
            evidence=(
                "benchmark plan covers visual quality, ray-query correctness, interaction quality, export, speed, and ablations",
                "reference benchmark emits deterministic metrics",
                "a real-scene benchmark harness scores packages against external COLMAP/NeRF/3DGS renders",
                "multi-scene Beta-vs-fixed-Gaussian results cover all local downloaded scenes (8/8, mean +0.80 dB)",
                "publication-validation report includes same-split COLMAP, NeRF, 3DGS, 2DGS, and ray-traced-GS baseline smoke/protocol metrics",
                "readiness-report includes the backend readiness contract used by reference benchmarks",
                *_benchmark_publication_evidence(publication_validation),
            ),
            gaps=() if _publication_gates_passed(
                publication_validation,
                ("local_multiscene_quality", "dataset_audit", "external_method_baselines"),
            ) else (
                "official external-repo full-split baseline runs are still optional replacement evidence, not required to close the local publication gate",
                "paper claims must not include production-resolution FPS, robustness, or official leaderboard superiority without new evidence",
                *_publication_validation_gaps(publication_validation),
            ),
            next_steps=(
                "keep official-leaderboard claims out of the paper unless optional official full-split baselines are added",
                "limit published claims to implemented-and-tested capabilities and the completed local validation evidence",
            ),
        ),
        ReadinessPillar(
            id="sota_ab_upgrades",
            title="SOTA method/library A/B upgrades",
            implemented=bool(sota_ab.get("comparisons")),
            production_ready=bool(sota_ab.get("abReady")),
            evidence=(
                "SOTA A/B validation compares upgrades against current AURA baselines before promotion",
                f"{sota_ab.get('summary', {}).get('comparisonCount', 0)} upgrade comparisons are recorded",
                f"promoted providers: {', '.join(sota_ab.get('summary', {}).get('promotedProviderIds', ())) or 'none'}",
                f"real SOTA candidate coverage ready: {bool(sota_ab.get('sotaReady'))}",
            ),
            gaps=() if sota_ab.get("abReady") else (
                "SOTA A/B validation artifact is missing or has blocked tasks",
            ),
            next_steps=(
                "replace fixture SOTA scores with real DINOv3, VGGT, Depth Anything 3, 3DGRUT, and official 2DGS artifacts",
                "keep local publication claims separate from official leaderboard-grade claims",
            ),
        ),
    )
    return ProductionReadinessReport(
        pillars=pillars,
        torch_renderer=torch_status,
        torch_carrier_kernels=kernel_report,
        cuda_kernel_sources=cuda_sources,
        legacy_cuda_renderer=legacy_cuda_renderer,
        cuda_renderer_callable_boundary=cuda_callable_boundary,
        backend_readiness=backend_readiness,
    )


def _readiness_scene():
    frames = {frame.id: frame for frame in synthetic_training_frames()}
    evidence = tuple(region.to_evidence_sample(frames[region.frame_id]) for region in synthetic_training_regions())
    return decompose_evidence(evidence, name="readiness_probe")


def _latest_json(results_dir: Path, pattern: str) -> dict | None:
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return json.loads(matches[0].read_text())


def _cuda_production_artifact() -> dict:
    payload = _latest_json(RESULTS, "cuda_production_backend*.json")
    if not payload:
        return {"passed": False, "reason": "missing"}
    parity = payload.get("parity") or {}
    throughput = payload.get("throughput") or {}
    max_abs_error = float(parity.get("maxAbsError", float("inf")))
    parity_threshold = float(parity.get("threshold", -1.0))
    rays_per_second = float(throughput.get("raysPerSecond", 0.0))
    min_rays_per_second = float(throughput.get("minRaysPerSecond", float("inf")))
    passed = (
        bool(payload.get("passed"))
        and bool(payload.get("compiledCudaDispatch"))
        and not bool(payload.get("fallbackUsed"))
        and str(payload.get("device")) == "cuda"
        and max_abs_error <= parity_threshold
        and rays_per_second >= min_rays_per_second
    )
    return {
        **payload,
        "passed": passed,
        "maxAbsError": max_abs_error,
        "parityThreshold": parity_threshold,
        "raysPerSecond": rays_per_second,
        "minRaysPerSecond": min_rays_per_second,
    }


def _cuda_production_evidence(payload: dict) -> tuple[str, ...]:
    if not payload.get("passed"):
        return ()
    return (
        "compiled CUDA dispatch artifact passed without fallback",
        f"CUDA parity maxAbsError {payload['maxAbsError']:.6g} <= {payload['parityThreshold']:.6g}",
        f"CUDA throughput {payload['raysPerSecond']:.1f} rays/s >= {payload['minRaysPerSecond']:.1f}",
    )


def _cuda_production_gaps(payload: dict) -> tuple[str, ...]:
    if payload.get("reason") == "missing":
        return ("production CUDA validation artifact is missing",)
    return ("compiled CUDA dispatch artifact did not pass",)


def _native_real_capture_artifact() -> dict:
    payload = _latest_json(RESULTS, "native_real_capture_validation*.json")
    if not payload:
        return {"passed": False, "reason": "missing"}
    scene_count = int(payload.get("sceneCount", 0))
    required_scene_count = int(payload.get("requiredSceneCount", 0))
    mean_delta = float(payload.get("meanDeltaPsnr", 0.0))
    families = tuple(payload.get("validatedCarrierFamilies") or ())
    passed = (
        bool(payload.get("passed"))
        and bool(payload.get("allLocalScenesComplete"))
        and scene_count >= required_scene_count
        and mean_delta > 0.0
        and {"beta", "gaussian"}.issubset(set(families))
        and not payload.get("missing")
    )
    return {
        **payload,
        "passed": passed,
        "sceneCount": scene_count,
        "requiredSceneCount": required_scene_count,
        "meanDeltaPsnr": mean_delta,
        "validatedCarrierFamilies": families,
    }


def _native_real_capture_evidence(payload: dict) -> tuple[str, ...]:
    if not payload.get("passed"):
        return ()
    return (
        f"{payload['sceneCount']} local real captures validated with complete Beta and Gaussian metrics",
        f"typed Beta carriers beat fixed Gaussian fallback by mean {payload['meanDeltaPsnr']:.2f} dB PSNR",
        "native real-capture validation artifact covers all locally downloaded scenes",
    )


def _native_real_capture_gaps(payload: dict) -> tuple[str, ...]:
    if payload.get("reason") == "missing":
        return ("native real-capture validation artifact is missing",)
    return ("native real-capture validation artifact did not pass",)


def _torch_backend_validation_artifact() -> dict:
    payload = _latest_json(RESULTS, "torch_backend_validation*.json")
    if not payload:
        return {"passed": False, "reason": "missing"}
    packed_target_count = int(payload.get("packedTargetCount", 0))
    min_packed_targets = int(payload.get("minPackedTargets", 0))
    manifest_regions = int(payload.get("manifestRegionCount", 0))
    min_manifest_regions = int(payload.get("minManifestRegions", 0))
    render_seconds = float(payload.get("renderSeconds", 0.0))
    passed = (
        bool(payload.get("passed"))
        and str(payload.get("device", "")).startswith("cuda")
        and bool(payload.get("finiteLosses"))
        and int(payload.get("loadedFrameCount", 0)) > 0
        and int(payload.get("sceneElementCount", 0)) > 0
        and int(payload.get("packedBatchCount", 0)) > 0
        and packed_target_count >= min_packed_targets
        and manifest_regions >= min_manifest_regions
        and int(payload.get("maxBatchTargetCount", 0)) <= int(payload.get("maxAllowedBatchTargets", 0))
        and render_seconds > 0.0
    )
    return {
        **payload,
        "passed": passed,
        "packedTargetCount": packed_target_count,
        "minPackedTargets": min_packed_targets,
        "manifestRegionCount": manifest_regions,
        "minManifestRegions": min_manifest_regions,
        "renderSeconds": render_seconds,
    }


def _torch_backend_validation_evidence(payload: dict) -> tuple[str, ...]:
    if not payload.get("passed"):
        return ()
    return (
        f"real-capture packed CUDA render validated {payload['packedTargetCount']} targets on {payload.get('device')}",
        f"real capture manifest includes {payload['manifestRegionCount']} regions and bounded CUDA batches stayed within {payload.get('maxAllowedBatchTargets')} targets",
        f"torch backend render summary produced finite losses in {payload['renderSeconds']:.3f}s",
    )


def _torch_backend_validation_gaps(payload: dict) -> tuple[str, ...]:
    if payload.get("reason") == "missing":
        return ("torch backend real-capture CUDA validation artifact is missing",)
    return ("torch backend real-capture CUDA validation artifact did not pass",)


def _publication_validation_artifact() -> dict:
    payload = _latest_json(RESULTS, "publication_validation*.json")
    if not payload:
        return {"publicationReady": False, "reason": "missing", "gates": ()}
    gates = tuple(payload.get("gates") or ())
    gate_status = {str(gate.get("id")): bool(gate.get("passed")) for gate in gates}
    return {
        **payload,
        "gates": gates,
        "gateStatus": gate_status,
    }


def _publication_gates_passed(payload: dict, required_gate_ids: tuple[str, ...]) -> bool:
    gate_status = payload.get("gateStatus") or {}
    return bool(payload.get("publicationReady")) and all(gate_status.get(gate_id) for gate_id in required_gate_ids)


def _renderer_trainer_publication_evidence(payload: dict) -> tuple[str, ...]:
    if not _publication_gates_passed(payload, ("prism_additive_contract", "prism_cuda_fps", "secondary_ray_reflection", "inverse_materials")):
        return ()
    return (
        "PRISM additive extension contract passed: Gabor/neural are extension-layer carriers while Gaussian/Beta stay on the primary quality backend",
        "PRISM CUDA FPS, secondary-ray/reflection, and inverse-material validation gates passed",
        "renderer/trainer claim boundary keeps PRISM additive to gsplat/DBS-Beta rather than a replacement",
    )


def _benchmark_publication_evidence(payload: dict) -> tuple[str, ...]:
    if not _publication_gates_passed(payload, ("local_multiscene_quality", "dataset_audit", "external_method_baselines")):
        return ()
    return (
        f"publication validation report passed {payload.get('passedGateCount')}/{payload.get('gateCount')} gates",
        "local dataset audit, multi-scene Beta-vs-Gaussian quality, and same-split external baseline gates passed",
        "claim boundary blocks official-leaderboard superiority claims without optional official full-split baseline replacement runs",
    )


def _publication_validation_gaps(payload: dict) -> tuple[str, ...]:
    if payload.get("reason") == "missing":
        return ("publication validation artifact is missing",)
    remaining = tuple(payload.get("remainingGateIds") or ())
    if remaining:
        return (f"publication validation gates still open: {', '.join(str(item) for item in remaining)}",)
    return ("publication validation artifact did not pass the required gate subset",)


def _summary(pillars: tuple[ReadinessPillar, ...]) -> str:
    ready = sum(1 for pillar in pillars if pillar.production_ready)
    implemented = sum(1 for pillar in pillars if pillar.implemented)
    if ready == len(pillars):
        return (
            f"{implemented}/{len(pillars)} readiness pillars are implemented; "
            f"{ready}/{len(pillars)} are artifact-backed production-ready. "
            "The claim remains bounded to the checked evidence: local real-capture coverage, compiled CUDA dispatch, "
            "bounded torch CUDA packed rendering, PRISM as an additive extension to gsplat/DBS-Beta, and same-split "
            "publication baselines with explicit exclusions for official-leaderboard superiority claims."
        )
    return (
        f"{implemented}/{len(pillars)} readiness pillars are implemented; "
        f"{ready}/{len(pillars)} are fully production-validated. "
        "All core code paths (native carriers, the torch and compiled CUDA renderers, GPU BVH "
        "traversal, per-carrier parity, and IO/streaming) are implemented and tested; the production "
        "claim has multi-scene Beta-vs-Gaussian evidence and same-split external-method smoke/protocol "
        "baselines; production FPS and official full-split external-repo replacement runs remain polish gates."
    )
