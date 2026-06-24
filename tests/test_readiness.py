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
    assert isinstance(payload["productionReady"], bool)
    assert payload["pillarCount"] == 6
    assert payload["implementedPillarCount"] >= 5
    assert payload["productionReadyPillarCount"] <= payload["pillarCount"]
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
    if by_id["cuda_backend"]["productionReady"]:
        assert by_id["cuda_backend"]["gaps"] == []
        assert any("compiled CUDA dispatch artifact passed without fallback" in item for item in by_id["cuda_backend"]["evidence"])
    else:
        assert "torch_carrier_kernel_report marks CUDA carrier kernels as not production ready" in by_id["cuda_backend"]["gaps"]
        assert "callable cuda_renderer fallback is not CUDA acceleration" in by_id["cuda_backend"]["gaps"]
    if by_id["renderer_trainer"]["productionReady"]:
        assert any("PRISM additive extension contract" in item for item in by_id["renderer_trainer"]["evidence"])
    else:
        assert "renderer real-time performance is not yet benchmarked at production resolution" in by_id["renderer_trainer"]["gaps"]
        assert any("secondary-ray/reflection integration remains future work" in gap for gap in by_id["renderer_trainer"]["gaps"])
    if by_id["benchmarks"]["productionReady"]:
        assert any("publication validation report passed" in item for item in by_id["benchmarks"]["evidence"])
    else:
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
    assert "artifact-backed production-ready" in payload["summary"]
    assert "same-split publication baselines" in payload["summary"]


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
    assert isinstance(payload["productionReady"], bool)
    assert payload["backendReadiness"]["requiresTorchImport"] is False
    assert payload["cudaRendererCallableBoundary"]["fallbackAvailable"] is True
    assert payload["cudaRendererCallableBoundary"]["productionReady"] is False
    missing = {pillar["id"] for pillar in payload["missingOrIncomplete"]}
    if "renderer_trainer" not in missing:
        renderer = next(pillar for pillar in payload["pillars"] if pillar["id"] == "renderer_trainer")
        assert any("PRISM additive extension contract" in item for item in renderer["evidence"])
    if "benchmarks" not in missing:
        benchmarks = next(pillar for pillar in payload["pillars"] if pillar["id"] == "benchmarks")
        assert any("publication validation report passed" in item for item in benchmarks["evidence"])
    if "native_carriers" not in missing:
        native = next(pillar for pillar in payload["pillars"] if pillar["id"] == "native_carriers")
        assert any("local real captures" in item for item in native["evidence"])
    if "cuda_backend" not in missing:
        cuda = next(pillar for pillar in payload["pillars"] if pillar["id"] == "cuda_backend")
        assert any("compiled CUDA dispatch artifact passed without fallback" in item for item in cuda["evidence"])


def test_cuda_production_gate_requires_compiled_nonfallback_artifact(tmp_path, monkeypatch):
    from aura import readiness

    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    cuda = next(pillar for pillar in report["pillars"] if pillar["id"] == "cuda_backend")

    assert cuda["productionReady"] is False
    assert "production CUDA validation artifact is missing" in cuda["gaps"]


def test_cuda_production_gate_rejects_fallback_artifact(tmp_path, monkeypatch):
    from aura import readiness

    (tmp_path / "cuda_production_backend_2026-06-24.json").write_text(json.dumps({
        "format": "AURA_CUDA_PRODUCTION_BACKEND_REPORT",
        "passed": True,
        "compiledCudaDispatch": False,
        "fallbackUsed": True,
        "device": "cpu",
        "parity": {"maxAbsError": 0.0, "threshold": 0.001},
        "throughput": {"raysPerSecond": 1000.0, "minRaysPerSecond": 1.0},
    }))
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    cuda = next(pillar for pillar in report["pillars"] if pillar["id"] == "cuda_backend")

    assert cuda["productionReady"] is False
    assert "compiled CUDA dispatch artifact did not pass" in cuda["gaps"]


def test_cuda_production_gate_accepts_compiled_cuda_artifact(tmp_path, monkeypatch):
    from aura import readiness

    (tmp_path / "cuda_production_backend_2026-06-24.json").write_text(json.dumps({
        "format": "AURA_CUDA_PRODUCTION_BACKEND_REPORT",
        "passed": True,
        "compiledCudaDispatch": True,
        "fallbackUsed": False,
        "device": "cuda",
        "parity": {"maxAbsError": 0.0001, "threshold": 0.001},
        "throughput": {"raysPerSecond": 1000.0, "minRaysPerSecond": 1.0},
    }))
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    cuda = next(pillar for pillar in report["pillars"] if pillar["id"] == "cuda_backend")

    assert cuda["productionReady"] is True
    assert any("compiled CUDA dispatch" in item for item in cuda["evidence"])


def test_native_carrier_gate_requires_real_capture_artifact(tmp_path, monkeypatch):
    from aura import readiness

    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    native = next(pillar for pillar in report["pillars"] if pillar["id"] == "native_carriers")

    assert native["productionReady"] is False
    assert "native real-capture validation artifact is missing" in native["gaps"]


def test_native_carrier_gate_accepts_complete_local_real_capture_artifact(tmp_path, monkeypatch):
    from aura import readiness

    (tmp_path / "native_real_capture_validation_2026-06-24.json").write_text(json.dumps({
        "format": "AURA_NATIVE_REAL_CAPTURE_VALIDATION",
        "passed": True,
        "allLocalScenesComplete": True,
        "sceneCount": 8,
        "requiredSceneCount": 8,
        "meanDeltaPsnr": 0.8,
        "validatedCarrierFamilies": ["beta", "gaussian"],
        "missing": [],
    }))
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    native = next(pillar for pillar in report["pillars"] if pillar["id"] == "native_carriers")

    assert native["productionReady"] is True
    assert any("8 local real captures" in item for item in native["evidence"])


def test_torch_backend_gate_requires_cuda_real_capture_artifact(tmp_path, monkeypatch):
    from aura import readiness

    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    torch_backend = next(pillar for pillar in report["pillars"] if pillar["id"] == "torch_backend")

    assert torch_backend["productionReady"] is False
    assert "torch backend real-capture CUDA validation artifact is missing" in torch_backend["gaps"]


def test_torch_backend_gate_accepts_cuda_real_capture_artifact(tmp_path, monkeypatch):
    from aura import readiness

    (tmp_path / "torch_backend_validation_2026-06-24.json").write_text(json.dumps({
        "format": "AURA_TORCH_BACKEND_VALIDATION",
        "passed": True,
        "device": "cuda",
        "manifestFrameCount": 251,
        "manifestRegionCount": 129531,
        "loadedFrameCount": 1,
        "sceneElementCount": 2048,
        "packedBatchCount": 1,
        "packedTargetCount": 64,
        "maxBatchTargetCount": 64,
        "maxAllowedBatchTargets": 256,
        "finiteLosses": True,
        "renderSeconds": 0.1,
        "failures": [],
    }))
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    torch_backend = next(pillar for pillar in report["pillars"] if pillar["id"] == "torch_backend")

    assert torch_backend["productionReady"] is True
    assert any("real-capture packed CUDA render" in item for item in torch_backend["evidence"])


def test_renderer_trainer_gate_accepts_publication_validation_artifact(tmp_path, monkeypatch):
    from aura import readiness

    (tmp_path / "publication_validation_2026-06-24.json").write_text(json.dumps({
        "format": "AURA_PUBLICATION_VALIDATION_REPORT",
        "publicationReady": True,
        "passedGateCount": 4,
        "gateCount": 4,
        "gates": [
            {"id": "prism_additive_contract", "passed": True},
            {"id": "prism_cuda_fps", "passed": True},
            {"id": "secondary_ray_reflection", "passed": True},
            {"id": "inverse_materials", "passed": True},
        ],
        "claimBoundary": {"cannotClaim": ["full production-resolution FPS across all publication scenes"]},
    }))
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    renderer = next(pillar for pillar in report["pillars"] if pillar["id"] == "renderer_trainer")

    assert renderer["productionReady"] is True
    assert any("PRISM additive extension contract" in item for item in renderer["evidence"])


def test_benchmark_gate_accepts_publication_validation_artifact(tmp_path, monkeypatch):
    from aura import readiness

    (tmp_path / "publication_validation_2026-06-24.json").write_text(json.dumps({
        "format": "AURA_PUBLICATION_VALIDATION_REPORT",
        "publicationReady": True,
        "passedGateCount": 8,
        "gateCount": 8,
        "remainingGateIds": [],
        "gates": [
            {"id": "local_multiscene_quality", "passed": True},
            {"id": "dataset_audit", "passed": True},
            {"id": "external_method_baselines", "passed": True},
        ],
        "claimBoundary": {
            "canClaim": ["AURA has same-split external baseline metrics"],
            "cannotClaim": ["official external-repo leaderboard superiority"],
        },
    }))
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    benchmarks = next(pillar for pillar in report["pillars"] if pillar["id"] == "benchmarks")

    assert benchmarks["productionReady"] is True
    assert any("publication validation report passed" in item for item in benchmarks["evidence"])
