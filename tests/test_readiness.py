import json
import subprocess
import sys

from aura import ProductionReadinessReport, ReadinessPillar, production_readiness_report


def test_production_readiness_report_lists_implemented_and_missing_pillars():
    report = production_readiness_report()
    payload = report.to_dict()
    by_id = {pillar["id"]: pillar for pillar in payload["pillars"]}

    assert isinstance(report, ProductionReadinessReport)
    assert payload["format"] == "AURA_PRODUCTION_READINESS_REPORT"
    assert payload["productionReady"] is False
    assert payload["pillarCount"] == 6
    assert payload["implementedPillarCount"] >= 5
    assert payload["productionReadyPillarCount"] < payload["pillarCount"]
    assert set(by_id) == {
        "native_carriers",
        "package_validation",
        "torch_backend",
        "cuda_backend",
        "renderer_trainer",
        "benchmarks",
    }
    assert by_id["package_validation"]["implemented"] is True
    assert by_id["package_validation"]["productionReady"] is True
    assert by_id["cuda_backend"]["productionReady"] is False
    assert "torch_carrier_kernel_report marks CUDA carrier kernels as not production ready" in by_id["cuda_backend"]["gaps"]
    assert "callable cuda_renderer fallback is not CUDA acceleration" in by_id["cuda_backend"]["gaps"]
    assert by_id["renderer_trainer"]["productionReady"] is False
    assert "renderer real-time performance is not yet benchmarked at production resolution" in by_id["renderer_trainer"]["gaps"]
    assert any("secondary-ray/reflection integration remains future work" in gap for gap in by_id["renderer_trainer"]["gaps"])
    assert by_id["benchmarks"]["productionReady"] is False
    assert any("official full-split baseline" in step for step in by_id["benchmarks"]["nextSteps"])
    assert payload["torchCarrierKernels"]["productionReady"] is False
    assert payload["cudaKernelSources"]["format"] == "AURA_CUDA_KERNEL_SOURCE_REPORT"
    assert payload["legacyCudaRenderer"]["format"] == "AURA_CUDA_RENDERER_LAUNCH_REPORT"
    assert payload["legacyCudaRenderer"]["available"] is False
    assert payload["legacyCudaRenderer"]["productionReady"] is False
    assert payload["cudaRendererCallableBoundary"]["format"] == "AURA_CUDA_RENDERER_CALLABLE_BOUNDARY"
    assert payload["cudaRendererCallableBoundary"]["reportKind"] == "callable_cuda_renderer_fallback_boundary"
    assert payload["cudaRendererCallableBoundary"]["callableBoundaryReady"] is True
    assert payload["cudaRendererCallableBoundary"]["fallbackContractReady"] is True
    boundary = payload["cudaRendererCallableBoundary"]
    assert (
        boundary["fallbackAvailable"] is True
        or boundary["compiledCudaAvailable"] is True
        or boundary.get("compiledExecutionAvailable") is True
    )
    assert payload["cudaRendererCallableBoundary"]["fallbackBackend"] in {"cpu", "cuda"}
    assert isinstance(boundary["compiledCudaAvailable"], bool)
    assert payload["cudaRendererCallableBoundary"]["productionReady"] is False
    assert payload["backendReadiness"]["format"] == "AURA_BACKEND_READINESS_EVALUATION"
    assert payload["backendReadiness"]["sceneCarrierAutogradCoverageRate"] == 1.0
    assert payload["backendReadiness"]["productionCudaReady"] is False
    assert "carrier_cuda_kernels_not_production_ready" in payload["backendReadiness"]["productionBlockers"]
    assert "multi-scene Beta-vs-Gaussian evidence" in payload["summary"]
    assert "same-split external-method smoke/protocol baselines" in payload["summary"]


def test_readiness_pillar_serializes_json_safe_fields():
    pillar = ReadinessPillar(
        id="example",
        title="Example",
        implemented=True,
        production_ready=False,
        evidence=("implemented contract",),
        gaps=("missing production backend",),
        next_steps=("finish backend",),
    )

    assert pillar.to_dict() == {
        "id": "example",
        "title": "Example",
        "implemented": True,
        "productionReady": False,
        "evidence": ["implemented contract"],
        "gaps": ["missing production backend"],
        "nextSteps": ["finish backend"],
    }


def test_readiness_report_cli_prints_json():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "readiness-report"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_PRODUCTION_READINESS_REPORT"
    assert payload["productionReady"] is False
    assert payload["backendReadiness"]["requiresTorchImport"] is False
    assert payload["cudaRendererCallableBoundary"]["fallbackAvailable"] is True
    assert payload["cudaRendererCallableBoundary"]["productionReady"] is False
    assert {pillar["id"] for pillar in payload["missingOrIncomplete"]}.issuperset(
        {"native_carriers", "cuda_backend", "renderer_trainer", "benchmarks"}
    )
