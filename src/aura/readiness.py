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
                "adaptive split/merge/promote/demote remains a fixture-scale reference path",
                "native carrier behavior is not validated on full real capture datasets",
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
                "schema compatibility is still limited to the current scaffold version policy",
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
                "torch optimization scaffold can consume capture tensor batches when PyTorch is installed",
            ),
            gaps=(
                "backend is reference/scaffold quality rather than a carrier-complete production renderer",
                "training loop is fixture-scale and not a full production optimizer",
                "full-resolution tiled or GPU-native data loading is not implemented",
            ),
            next_steps=(
                "replace reference tensor paths with a carrier-complete differentiable renderer",
                "run torch optimization on real captures with memory-bounded batching",
            ),
        ),
        ReadinessPillar(
            id="cuda_backend",
            title="CUDA production backend",
            implemented=bool(cuda_sources.get("availableSourceCount", 0)),
            production_ready=False,
            evidence=(
                "CUDA source stubs are packaged and discoverable",
                "legacy cuda_kernels cuda-renderer-report remains a metadata-only launch contract",
                "callable aura.cuda_renderer boundary validates launch shape and returns an explicit CPU fallback batch",
                "aura cuda-kernel-build-report can probe extension build/load status",
                f"backend readiness reports {backend_readiness['sceneCarrierCudaCoverageRate']:.0%} scene-carrier CUDA production coverage",
            ),
            gaps=(
                "CUDA extension build is not attempted by this readiness report",
                "callable cuda_renderer fallback is not CUDA acceleration",
                "torch_carrier_kernel_report marks CUDA carrier kernels as not production ready",
                "GPU traversal and production kernel dispatch are missing",
            ),
            next_steps=(
                "implement and validate CUDA kernels for every native carrier",
                "make torch-kernel-report productionReady true before claiming CUDA readiness",
            ),
        ),
        ReadinessPillar(
            id="renderer_trainer",
            title="Renderer and trainer production path",
            implemented=True,
            production_ready=False,
            evidence=(
                "CPU reference rendering, ray query, reconstruction, and torch optimization scaffolds exist",
                "capture manifest conversion and tensor target planning share deterministic contracts",
                "capture proposal weights can be trained from labeled feature examples and reused in native region generation",
            ),
            gaps=(
                "renderer is not production real-time",
                "trainer is not a full reconstruction system for large real datasets",
                "proposal model is a lightweight logistic contract, not a full neural region proposal network",
                "secondary rays, relighting, streaming, and GPU BVH traversal are not production paths",
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
                "reference benchmark emits deterministic scaffold metrics",
                "readiness-report includes the CPU-only backend readiness contract used by reference benchmarks",
            ),
            gaps=(
                "no production benchmark results against COLMAP, NeRF/nerfstudio, 3DGS, 2DGS, or ray-traced GS baselines",
                "LPIPS is currently a deterministic proxy rather than a learned LPIPS backend",
                "paper claims must not include real-time performance, robustness, or better PSNR without new evidence",
            ),
            next_steps=(
                "run reproducible real-dataset baselines and publish PSNR/SSIM/LPIPS/FPS plus scene-behavior metrics",
                "keep current claims limited to a native adaptive radiance asset scaffold",
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
        f"{implemented}/{len(pillars)} readiness pillars have an implemented scaffold; "
        f"{ready}/{len(pillars)} are production ready. "
        "AURA is not production ready until CUDA kernels, renderer/trainer, and real baseline benchmarks are complete."
    )
