"""AURA production readiness audit across implemented capability pillars."""

from __future__ import annotations

from dataclasses import dataclass

from aura.benchmark import cuda_renderer_callable_boundary_report, evaluate_backend_readiness
from aura.core import synthetic_training_frames, synthetic_training_regions
from aura.cuda_kernels import cuda_kernel_source_report, cuda_renderer_report
from aura.decomposition import decompose_evidence
from aura.torch_kernels import torch_carrier_kernel_report
from aura.torch_renderer import torch_renderer_status


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
    pillars = (
        ReadinessPillar(
            id="native_carriers",
            title="Native adaptive carriers",
            implemented=True,
            production_ready=False,
            evidence=(
                "surface, volume, beta, gabor, neural residual, semantic, and gaussian fallback carrier contracts exist",
                "native demo and decomposition paths produce typed AuraElement payloads",
                "ray-query behavior is exercised by reference package and benchmark tests",
            ),
            gaps=(
                "adaptive split/merge/promote/demote is validated at fixture scale, not full real captures",
                "native carrier behavior is not yet validated on full real capture datasets",
                "3DGS remains an ingest evidence source and fallback, not a production native carrier replacement",
            ),
            next_steps=(
                "validate mixed-carrier reconstruction on real capture manifests",
                "add larger adaptive carrier evolution tests and scene-behavior benchmarks",
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
            production_ready=False,
            evidence=(
                "torch renderer status is reportable through aura torch-renderer-status",
                "carrier payloads have torch autograd kernel specs",
                f"backend readiness reports {backend_readiness['sceneCarrierAutogradCoverageRate']:.0%} scene-carrier autograd coverage",
                "the torch optimizer consumes packed capture tensor batches when PyTorch is installed",
                "grouped torch ray/carrier intersection and compositing is carrier-complete and covered by parity tests",
            ),
            gaps=(
                "the renderer and optimizer are validated on deterministic fixtures, not full real capture datasets",
                "full-resolution tiled or GPU-native data loading at dataset scale is not yet benchmarked",
            ),
            next_steps=(
                "run torch optimization on real captures with memory-bounded batching",
                "publish renderer/optimizer throughput and quality on real-dataset baselines",
            ),
        ),
        ReadinessPillar(
            id="cuda_backend",
            title="CUDA production backend",
            implemented=bool(cuda_sources.get("availableSourceCount", 0)),
            production_ready=False,
            evidence=(
                "CUDA carrier kernels compile and load via aura cuda-kernel-build-report --build",
                "the compiled CUDA renderer matches the torch renderer per-carrier in fixture parity tests",
                "a production GPU BVH traversal kernel (render_rays_bvh) replaces the brute-force element scan",
                "aura benchmark-cuda-runtime measures on-device throughput and cross-backend parity",
                f"backend readiness reports {backend_readiness['sceneCarrierCudaCoverageRate']:.0%} scene-carrier CUDA production coverage",
            ),
            gaps=(
                "CUDA extension build is not attempted by this readiness report",
                "callable cuda_renderer fallback is not CUDA acceleration",
                "torch_carrier_kernel_report marks CUDA carrier kernels as not production ready",
                "CUDA/BVH parity and throughput are validated on fixtures, not yet on real-dataset baselines",
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
            production_ready=False,
            evidence=(
                "CPU reference rendering, ray query, reconstruction, and torch optimization paths exist",
                "GPU BVH traversal, EXR/PFM/video export, and a long-run memory stability probe are implemented",
                "capture manifest conversion and tensor target planning share deterministic contracts",
                "capture proposal weights can be trained from labeled feature examples and reused in native region generation",
            ),
            gaps=(
                "renderer real-time performance is not yet benchmarked at production resolution",
                "trainer is not yet validated as a full reconstruction system on large real datasets",
                "proposal model is a lightweight logistic contract, not a full neural region proposal network",
                "secondary rays and relighting are not yet implemented",
            ),
            next_steps=(
                "replace CPU reference loops with GPU renderer/trainer implementations",
                "train proposal weights on real capture labels and replace the logistic model with a neural proposal backend",
                "add full-scene performance, memory, and correctness gates",
            ),
        ),
        ReadinessPillar(
            id="benchmarks",
            title="Benchmarks and claim boundary",
            implemented=True,
            production_ready=False,
            evidence=(
                "benchmark plan covers visual quality, ray-query correctness, interaction quality, export, speed, and ablations",
                "reference benchmark emits deterministic metrics",
                "a real-scene benchmark harness scores packages against external COLMAP/NeRF/3DGS renders",
                "readiness-report includes the backend readiness contract used by reference benchmarks",
            ),
            gaps=(
                "no production benchmark results against COLMAP, NeRF/nerfstudio, 3DGS, 2DGS, or ray-traced GS baselines",
                "LPIPS is currently a deterministic proxy rather than a learned LPIPS backend",
                "paper claims must not include real-time performance, robustness, or better PSNR without new evidence",
            ),
            next_steps=(
                "run reproducible real-dataset baselines and publish PSNR/SSIM/LPIPS/FPS plus scene-behavior metrics",
                "limit published claims to implemented-and-tested capabilities until real-dataset baselines are released",
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


def _summary(pillars: tuple[ReadinessPillar, ...]) -> str:
    ready = sum(1 for pillar in pillars if pillar.production_ready)
    implemented = sum(1 for pillar in pillars if pillar.implemented)
    return (
        f"{implemented}/{len(pillars)} readiness pillars are implemented; "
        f"{ready}/{len(pillars)} are fully production-validated. "
        "All core code paths (native carriers, the torch and compiled CUDA renderers, GPU BVH "
        "traversal, per-carrier parity, and IO/streaming) are implemented and tested; the production "
        "claim is pending reproducible real-dataset baseline benchmarks "
        "(PSNR/SSIM/LPIPS on Mip-NeRF 360 / Tanks and Temples)."
    )
