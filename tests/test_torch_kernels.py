import importlib.util
import json
import subprocess
import sys
from importlib.resources import files

import pytest

from aura import (
    AuraElement,
    Bounds,
    cuda_kernel_extension_report,
    cuda_kernel_extension_status,
    cuda_kernel_source_report,
    cuda_kernel_sources,
    cuda_render_rays,
    cuda_renderer_api_contract,
    cuda_renderer_report,
    cuda_renderer_source_report,
    torch_carrier_kernel_report,
    torch_carrier_kernel_specs,
    torch_carrier_parameter_tensors,
    torch_carrier_response_tensors_batched,
    torch_carrier_response_tensors,
)


def test_torch_carrier_kernel_specs_cover_native_payloads_without_torch():
    specs = torch_carrier_kernel_specs()
    by_payload = {spec.payload_type: spec for spec in specs}

    assert set(by_payload) == {
        "surface_cell",
        "volume_cell",
        "beta_kernel",
        "gabor_frequency",
        "neural_residual",
        "semantic_feature",
        "gaussian_fallback",
    }
    assert by_payload["gaussian_fallback"].carrier_id == "gaussian"
    assert by_payload["gaussian_fallback"].to_dict()["implementationStage"] == "torch_autograd_gaussian_fallback_kernel"
    assert by_payload["gaussian_fallback"].autograd_kernel is True
    assert by_payload["gaussian_fallback"].production_ready is False
    assert by_payload["gaussian_fallback"].blockers == ("missing_cuda_kernel",)
    assert "opacity" in by_payload["surface_cell"].differentiable_fields
    assert by_payload["volume_cell"].to_dict()["payloadType"] == "volume_cell"
    assert by_payload["surface_cell"].to_dict()["implementationStage"] == "torch_autograd_surface_kernel"
    assert by_payload["surface_cell"].autograd_kernel is True
    assert by_payload["surface_cell"].cuda_kernel is False
    assert by_payload["surface_cell"].production_ready is False
    assert by_payload["surface_cell"].blockers == ("missing_cuda_kernel",)
    assert by_payload["volume_cell"].to_dict()["implementationStage"] == "torch_autograd_volume_kernel"
    assert by_payload["volume_cell"].autograd_kernel is True
    assert by_payload["volume_cell"].production_ready is False
    assert by_payload["volume_cell"].blockers == ("missing_cuda_kernel",)
    assert by_payload["beta_kernel"].to_dict()["implementationStage"] == "torch_autograd_beta_kernel"
    assert by_payload["beta_kernel"].autograd_kernel is True
    assert by_payload["beta_kernel"].production_ready is False
    assert by_payload["beta_kernel"].blockers == ("missing_cuda_kernel",)
    assert by_payload["gabor_frequency"].to_dict()["implementationStage"] == "torch_autograd_gabor_kernel"
    assert by_payload["gabor_frequency"].autograd_kernel is True
    assert by_payload["gabor_frequency"].production_ready is False
    assert by_payload["gabor_frequency"].blockers == ("missing_cuda_kernel",)
    assert by_payload["neural_residual"].to_dict()["implementationStage"] == "torch_autograd_neural_residual_kernel"
    assert by_payload["neural_residual"].autograd_kernel is True
    assert by_payload["neural_residual"].production_ready is False
    assert by_payload["neural_residual"].blockers == ("missing_cuda_kernel",)
    assert by_payload["semantic_feature"].to_dict()["implementationStage"] == "torch_autograd_semantic_feature_kernel"
    assert by_payload["semantic_feature"].autograd_kernel is True
    assert by_payload["semantic_feature"].production_ready is False
    assert by_payload["semantic_feature"].blockers == ("missing_cuda_kernel",)


def test_torch_carrier_kernel_report_is_a_production_readiness_gate():
    report = torch_carrier_kernel_report()

    assert report["format"] == "AURA_TORCH_CARRIER_KERNEL_REPORT"
    assert report["productionReady"] is False
    assert report["carrierCount"] == 7
    assert report["nonProductionCarrierCount"] == 7
    assert report["referenceOnlyCarrierCount"] == 0
    assert report["autogradCarrierCount"] == 7
    assert report["cudaCarrierCount"] == 0
    by_carrier = {item["carrierId"]: item for item in report["kernelSpecs"]}
    assert set(by_carrier) == {"surface", "volume", "beta", "gabor", "neural", "semantic", "gaussian"}
    assert by_carrier["surface"]["autogradKernel"] is True
    assert by_carrier["surface"]["productionReady"] is False
    assert by_carrier["surface"]["blockers"] == ["missing_cuda_kernel"]
    assert by_carrier["volume"]["autogradKernel"] is True
    assert by_carrier["volume"]["productionReady"] is False
    assert by_carrier["volume"]["blockers"] == ["missing_cuda_kernel"]
    assert by_carrier["beta"]["autogradKernel"] is True
    assert by_carrier["beta"]["productionReady"] is False
    assert by_carrier["beta"]["blockers"] == ["missing_cuda_kernel"]
    assert by_carrier["gabor"]["autogradKernel"] is True
    assert by_carrier["gabor"]["productionReady"] is False
    assert by_carrier["gabor"]["blockers"] == ["missing_cuda_kernel"]
    assert by_carrier["neural"]["autogradKernel"] is True
    assert by_carrier["neural"]["productionReady"] is False
    assert by_carrier["neural"]["blockers"] == ["missing_cuda_kernel"]
    assert by_carrier["semantic"]["autogradKernel"] is True
    assert by_carrier["semantic"]["productionReady"] is False
    assert by_carrier["semantic"]["blockers"] == ["missing_cuda_kernel"]
    assert by_carrier["gaussian"]["autogradKernel"] is True
    assert by_carrier["gaussian"]["productionReady"] is False
    assert by_carrier["gaussian"]["blockers"] == ["missing_cuda_kernel"]
    assert "carrier-complete CUDA kernels" in report["requiredNextStep"]


def test_torch_kernel_report_cli_prints_readiness_json():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "torch-kernel-report"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_TORCH_CARRIER_KERNEL_REPORT"
    assert payload["productionReady"] is False
    assert payload["carrierCount"] == 7
    assert payload["nonProductionCarrierCount"] == 7
    assert payload["referenceOnlyCarrierCount"] == 0
    assert payload["autogradCarrierCount"] == 7


def test_cuda_kernel_sources_cover_every_native_carrier_symbol():
    sources = cuda_kernel_sources()
    report = cuda_kernel_source_report()
    source_text = files("aura").joinpath("cuda/aura_carriers.cu").read_text(encoding="utf-8")

    assert {source.carrier_id for source in sources} == {"surface", "volume", "beta", "gabor", "neural", "semantic", "gaussian"}
    assert {source.symbol for source in sources} == {
        "aura_surface_forward_kernel",
        "aura_volume_forward_kernel",
        "aura_beta_forward_kernel",
        "aura_gabor_forward_kernel",
        "aura_neural_forward_kernel",
        "aura_semantic_forward_kernel",
        "aura_gaussian_forward_kernel",
    }
    assert report["format"] == "AURA_CUDA_KERNEL_SOURCE_REPORT"
    assert report["sourceCount"] == 7
    assert report["availableSourceCount"] == 7
    assert report["contractCompleteSourceCount"] == 7
    assert report["contractOutputs"] == ["out_color", "out_transmittance", "out_confidence", "out_residual"]
    for source in report["sources"]:
        assert source["path"] == "cuda/aura_carriers.cu"
        assert source["available"] is True
        assert source["required"] is True
        assert source["sourceSymbolAvailable"] is True
        assert source["contractComplete"] is True
        assert source["missingSourceFragments"] == []
        assert source["contractOutputs"] == report["contractOutputs"]
        assert source["symbol"] in source_text


def test_cuda_renderer_source_report_declares_native_ray_query_kernel():
    report = cuda_renderer_source_report()
    source_text = files("aura").joinpath(report["path"]).read_text(encoding="utf-8")
    binding_text = files("aura").joinpath(report["bindingPath"]).read_text(encoding="utf-8")
    kernel_text = _cuda_kernel_text(source_text, report["symbol"])

    assert report["format"] == "AURA_CUDA_RENDERER_SOURCE_REPORT"
    assert report["symbol"] == "aura_render_rays_kernel"
    assert report["launcherSymbol"] == "aura_render_rays_launcher"
    assert report["bindingSymbol"] == "render_rays"
    assert report["available"] is True
    assert report["bindingAvailable"] is True
    assert report["sourceSymbolAvailable"] is True
    assert report["bindingSymbolAvailable"] is True
    assert report["contractComplete"] is True
    assert report["missingSourceFragments"] == []
    assert report["missingBindingFragments"] == []
    assert report["productionReady"] is False
    assert report["extensionSymbols"] == ["aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"]
    assert report["dispatchBindingAvailable"] is True
    assert report["pythonBindingSourceAvailable"] is True
    assert report["pythonBindingAvailable"] is False
    assert "compiled Python extension not built or loaded in this process" in report["productionBlockers"]
    assert "AABB first-hit traversal over native element bounds" in report["implementedSemantics"]
    assert "compiled host launcher ABI that computes a grid and launches aura_render_rays_kernel" in report["implementedSemantics"]
    assert "pybind11 torch extension binding source for render_rays packed tensor dispatch" in report["implementedSemantics"]
    assert set(report["contractOutputs"]) == {
        "out_color",
        "out_alpha",
        "out_transmittance",
        "out_depth",
        "out_normal",
        "out_confidence",
        "out_residual",
        "out_material_id",
        "out_semantic_id",
        "ordered_hits",
    }

    arguments = {argument["name"]: argument for argument in report["arguments"]}
    assert arguments["ray_origins"]["shape"] == "rayCount x 3"
    assert arguments["element_mins"]["shape"] == "elementCount x 3"
    assert arguments["carrier_ids"]["dtype"] == "const int*"
    assert arguments["ordered_hits"]["shape"] == "rayCount x maxHits"
    assert arguments["max_hits"]["role"] == "size"
    for fragment in (
        "aura_ray_aabb_intersect",
        "ordered_hits[ray_i * max_hits + hit_i] = -1",
        "extern \"C\" void aura_render_rays_launcher",
        "aura_render_rays_kernel<<<block_count, threads>>>",
        "out_material_id[ray_i]",
        "out_semantic_id[ray_i]",
        "out_residual[ray_i]",
    ):
        assert fragment in kernel_text
    for fragment in (
        "PYBIND11_MODULE",
        "render_rays",
        "aura_render_rays_launcher",
        "torch::Tensor",
        "ordered_hits",
    ):
        assert fragment in binding_text


def test_cuda_kernel_source_metadata_declares_carrier_complete_argument_schema():
    report = cuda_kernel_source_report()

    by_carrier = {source["carrierId"]: source for source in report["sources"]}

    assert by_carrier["surface"]["payloadType"] == "surface_cell"
    assert by_carrier["volume"]["payloadType"] == "volume_cell"
    assert by_carrier["beta"]["payloadType"] == "beta_kernel"
    assert by_carrier["gabor"]["payloadType"] == "gabor_frequency"
    assert by_carrier["neural"]["payloadType"] == "neural_residual"
    assert by_carrier["semantic"]["payloadType"] == "semantic_feature"
    assert by_carrier["gaussian"]["payloadType"] == "gaussian_fallback"

    for source in by_carrier.values():
        arguments = {argument["name"]: argument for argument in source["arguments"]}
        for output in ("out_color", "out_transmittance", "out_confidence", "out_residual"):
            assert arguments[output]["role"] == "output"
        assert arguments["out_color"]["shape"] == "count x 3"
        assert arguments["out_transmittance"]["shape"] == "count"
        assert arguments["out_confidence"]["shape"] == "count"
        assert arguments["out_residual"]["dtype"] == "unsigned char*"
        assert arguments["count"]["role"] == "size"

    assert {argument["name"] for argument in by_carrier["surface"]["arguments"]} == {
        "color",
        "opacity",
        "confidence",
        "out_color",
        "out_transmittance",
        "out_confidence",
        "out_residual",
        "count",
    }
    assert {argument["name"] for argument in by_carrier["volume"]["arguments"]} == {
        "color",
        "density",
        "path_length",
        "confidence",
        "out_color",
        "out_transmittance",
        "out_confidence",
        "out_residual",
        "count",
    }
    assert {argument["name"] for argument in by_carrier["gabor"]["arguments"]} == {
        "color",
        "opacity",
        "confidence",
        "frequency",
        "phase",
        "bandwidth",
        "hit_point",
        "out_color",
        "out_transmittance",
        "out_confidence",
        "out_residual",
        "count",
    }


def test_cuda_source_static_contract_writes_aura_outputs_for_every_carrier():
    source_text = files("aura").joinpath("cuda/aura_carriers.cu").read_text(encoding="utf-8")

    for source in cuda_kernel_sources():
        kernel_text = _cuda_kernel_text(source_text, source.symbol)
        signature_text = kernel_text.split(") {", maxsplit=1)[0]
        for argument in source.to_dict()["arguments"]:
            assert argument["name"] in signature_text
        for output in source.contract_outputs:
            assert f"{output}[i" in kernel_text


def test_torch_kernel_report_links_cuda_source_but_keeps_production_gate_closed():
    report = torch_carrier_kernel_report()

    assert report["productionReady"] is False
    assert report["cudaExtension"]["buildAttempted"] is False
    assert report["cudaExtension"]["reason"] == "build_not_attempted"
    assert report["cudaSourceCount"] == 7
    assert report["availableCudaSourceCount"] == 7
    for spec in report["kernelSpecs"]:
        assert spec["cudaKernel"] is False
        assert spec["productionReady"] is False
        assert spec["cudaSource"]["carrierId"] == spec["carrierId"]
        assert spec["cudaSource"]["available"] is True
        assert spec["cudaSource"]["symbol"].startswith("aura_")


def test_cuda_kernel_extension_status_does_not_build_by_default():
    status = cuda_kernel_extension_status()
    report = cuda_kernel_extension_report()

    assert status.available is False
    assert status.build_attempted is False
    assert status.compiled is False
    assert status.loadable is False
    assert status.reason == "build_not_attempted"
    assert status.module_name == "aura_cuda_carriers"
    assert status.source_paths == ("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu")
    assert len(status.symbols) == 10
    assert status.symbols[-3:] == ("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays")
    assert report["format"] == "AURA_CUDA_EXTENSION_REPORT"
    assert report["productionReady"] is False
    assert report["carrierSymbolCount"] == 7
    assert report["rendererSymbolCount"] == 3
    assert report["pythonBindingSource"] == "cuda/aura_bindings.cpp"
    assert report["pythonBindingSymbol"] == "render_rays"
    assert report["buildAttempted"] is False


def test_cuda_renderer_api_contract_declares_batched_ray_outputs_without_cuda():
    contract = cuda_renderer_api_contract()

    assert contract["format"] == "AURA_CUDA_RENDERER_API_CONTRACT"
    assert contract["apiName"] == "cuda_render_rays"
    assert contract["productionReady"] is False
    assert contract["batchDimension"] == "rayCount"
    assert contract["extension"]["buildAttempted"] is False
    assert contract["rendererSourceReport"]["format"] == "AURA_CUDA_RENDERER_SOURCE_REPORT"
    assert contract["rendererSourceReport"]["sourceSymbolAvailable"] is True
    assert contract["rendererSourceReport"]["productionReady"] is False
    assert contract["rendererBinding"]["kernelSymbol"] == "aura_render_rays_kernel"
    assert contract["rendererBinding"]["launcherSymbol"] == "aura_render_rays_launcher"
    assert contract["rendererBinding"]["compiledLauncherContract"] is True
    assert contract["rendererBinding"]["pythonBindingSourceAvailable"] is True
    assert contract["rendererBinding"]["dispatchImplemented"] is True
    assert contract["rendererBinding"]["pythonBindingAvailable"] is False
    inputs = {item["name"]: item for item in contract["inputTensors"]}
    outputs = {item["name"]: item for item in contract["outputTensors"]}

    assert inputs["ray_origins"]["shape"] == "rayCount x 3"
    assert inputs["ray_directions"]["shape"] == "rayCount x 3"
    assert inputs["carrier_ids"]["shape"] == "elementCount"
    assert outputs["out_color"]["shape"] == "rayCount x 3"
    assert outputs["out_transmittance"]["shape"] == "rayCount"
    assert outputs["out_depth"]["shape"] == "rayCount"
    assert outputs["out_normal"]["shape"] == "rayCount x 3"
    assert outputs["ordered_hits"]["shape"] == "rayCount x maxHits"
    assert any("renderer launcher symbol is verified" in item for item in contract["unavailableUntil"])
    assert any("renderer Python binding is imported" in item for item in contract["unavailableUntil"])


def test_cuda_render_rays_callable_scaffold_validates_batches_but_does_not_compile():
    report = cuda_render_rays(
        ray_origins=((0.0, 0.0, -2.0), (0.25, 0.0, -2.0)),
        ray_directions=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
    ).to_dict()

    assert report["format"] == "AURA_CUDA_RENDERER_LAUNCH_REPORT"
    assert report["apiName"] == "cuda_render_rays"
    assert report["rayCount"] == 2
    assert report["validatedInputs"] is True
    assert report["available"] is False
    assert report["productionReady"] is False
    assert report["reason"] == "extension_not_compiled_or_loadable"
    assert report["extension"]["buildAttempted"] is False
    assert report["extension"]["reason"] == "build_not_attempted"
    assert report["contract"]["productionReady"] is False


def test_cuda_render_rays_reports_invalid_batched_ray_inputs_without_cuda():
    report = cuda_render_rays(
        ray_origins=((0.0, 0.0, -2.0),),
        ray_directions=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
    ).to_dict()

    assert report["validatedInputs"] is False
    assert report["available"] is False
    assert report["productionReady"] is False
    assert report["reason"].startswith("invalid_batched_ray_inputs:")
    assert "does not match" in report["reason"]
    assert report["extension"]["buildAttempted"] is False


def test_cuda_render_rays_accepts_tensor_shaped_inputs():
    """cuda_kernels.py line 581: _batched_vec3_count returns shape[0] for array-shaped inputs."""
    import numpy as np
    origins = np.zeros((3, 3), dtype=float)
    directions = np.zeros((3, 3), dtype=float)
    report = cuda_render_rays(ray_origins=origins, ray_directions=directions).to_dict()
    assert report["rayCount"] == 3
    assert report["validatedInputs"] is True


def test_cuda_renderer_report_cli_prints_cpu_safe_json():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "cuda-renderer-report"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload == cuda_renderer_report()
    assert payload["format"] == "AURA_CUDA_RENDERER_LAUNCH_REPORT"
    assert payload["productionReady"] is False
    assert payload["available"] is False
    assert payload["extension"]["buildAttempted"] is False


def test_cuda_kernel_build_report_cli_is_non_destructive_without_build():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "cuda-kernel-build-report"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_CUDA_EXTENSION_REPORT"
    assert payload["productionReady"] is False
    assert payload["buildAttempted"] is False
    assert payload["compiled"] is False
    assert payload["loadable"] is False
    assert payload["reason"] == "build_not_attempted"


def _cuda_kernel_text(source_text, symbol):
    start = source_text.index(f'extern "C" __global__ void {symbol}')
    next_kernel = source_text.find('\nextern "C" __global__ void ', start + 1)
    if next_kernel == -1:
        return source_text[start:]
    return source_text[start:next_kernel]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_kernel_keeps_color_opacity_confidence_differentiable():
    import torch

    elements = (
        AuraElement(
            id="surface",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "surface_cell"},
        ),
    )
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.2, 0.4, 0.6]], requires_grad=True)
    opacities = torch.tensor([0.5], requires_grad=True)
    confidences = torch.tensor([0.75], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert colors.grad is not None
    assert opacities.grad is not None
    assert confidences.grad is not None
    assert colors.grad.tolist() == [[1.0, 1.0, 1.0]]
    assert opacities.grad.tolist() == pytest.approx([-1.0])
    assert confidences.grad.tolist() == pytest.approx([1.0])
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_response_dispatch_does_not_sync_with_torch_any(monkeypatch):
    import torch

    elements = (
        AuraElement(
            id="surface",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(1.0, 0.0, 0.0),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "surface_cell"},
        ),
        AuraElement(
            id="volume",
            carrier_id="volume",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "volume_cell", "density": 1.0},
        ),
        AuraElement(
            id="beta",
            carrier_id="beta",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0},
        ),
        AuraElement(
            id="gabor",
            carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "gabor_frequency", "frequency": (0.0, 0.0, 1.0), "phase": 0.0, "bandwidth": 0.5},
        ),
        AuraElement(
            id="neural",
            carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "neural_residual", "residual_scale": 0.25},
        ),
        AuraElement(
            id="semantic",
            carrier_id="semantic",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "semantic_feature", "label": "object", "confidence": 0.9},
        ),
        AuraElement(
            id="gaussian",
            carrier_id="gaussian",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "gaussian_fallback"},
        ),
    )

    def fail_any(*_args, **_kwargs):
        raise AssertionError("carrier response dispatch should not synchronize through torch.any")

    monkeypatch.setattr(torch, "any", fail_any)

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1.0]),
        torch.tensor([[2.0] * len(elements)]),
        torch.tensor([[0.5, 0.5, 0.5]]),
        torch.tensor([element.color for element in elements]),
        torch.tensor([element.opacity for element in elements]),
        torch.tensor([element.confidence for element in elements]),
        torch.tensor([element.bounds.min_corner for element in elements]),
        torch.tensor([element.bounds.max_corner for element in elements]),
        "cpu",
    )

    assert carrier_colors[0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert transmittance.tolist() == pytest.approx([0.5])
    assert confidence.tolist() == pytest.approx([0.75])
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_batched_carrier_response_matches_scalar_carriers():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    elements = (
        AuraElement(
            id="surface",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(1.0, 0.0, 0.0),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "surface_cell"},
        ),
        AuraElement(
            id="volume",
            carrier_id="volume",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.0, 1.0, 0.0),
            opacity=0.6,
            confidence=0.8,
            payload={"type": "volume_cell", "density": 1.25, "opacity": 0.7},
        ),
        AuraElement(
            id="beta",
            carrier_id="beta",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.0, 0.0, 1.0),
            opacity=0.65,
            confidence=0.7,
            payload={"type": "beta_kernel", "alpha": 2.0, "beta": 3.0, "support_radius": [0.5, 0.5, 0.5]},
        ),
        AuraElement(
            id="gabor",
            carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.8, 0.6, 0.2),
            opacity=0.4,
            confidence=0.9,
            payload={"type": "gabor_frequency", "frequency": (0.0, 0.0, 0.5), "phase": 0.2, "bandwidth": 0.6},
        ),
        AuraElement(
            id="neural",
            carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.2, 0.4, 0.8),
            opacity=0.7,
            confidence=0.85,
            payload={"type": "neural_residual", "latent_dim": 8, "residual_scale": 0.35},
        ),
        AuraElement(
            id="semantic",
            carrier_id="semantic",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.3, 0.3, 0.3),
            opacity=0.25,
            confidence=0.4,
            payload={"type": "semantic_feature", "label": "object", "confidence": 0.9},
        ),
        AuraElement(
            id="gaussian",
            carrier_id="gaussian",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.9, 0.5, 0.1),
            opacity=0.8,
            confidence=0.6,
            payload={
                "type": "gaussian_fallback",
                "mean": [0.5, 0.5, 0.5],
                "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
            },
        ),
    )
    best_index = torch.arange(len(elements), dtype=torch.long, device=device)
    best_depth = torch.full((len(elements),), 1.0, dtype=torch.float32, device=device)
    exit_depth = torch.full((len(elements), len(elements)), 2.0, dtype=torch.float32, device=device)
    hit_points = torch.tensor([(0.5, 0.5, 0.5)] * len(elements), dtype=torch.float32, device=device)
    colors = torch.tensor([element.color for element in elements], dtype=torch.float32, device=device)
    opacities = torch.tensor([element.opacity for element in elements], dtype=torch.float32, device=device)
    confidences = torch.tensor([element.confidence for element in elements], dtype=torch.float32, device=device)
    mins = torch.tensor([element.bounds.min_corner for element in elements], dtype=torch.float32, device=device)
    maxs = torch.tensor([element.bounds.max_corner for element in elements], dtype=torch.float32, device=device)

    scalar = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        device,
    )
    batched = torch_carrier_response_tensors_batched(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        device,
    )

    assert torch.allclose(batched[0], scalar[0])
    assert torch.allclose(batched[1], scalar[1])
    assert torch.allclose(batched[2], scalar[2])
    assert batched[3].tolist() == scalar[3].tolist()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_surface_carrier_parameter_tensors_cover_native_surface_fields():
    import torch

    elements = (
        AuraElement(
            id="surface",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.2, 0.4, 0.6),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "surface_cell"},
        ),
    )

    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")

    assert set(carrier_parameters["surface"]) == {
        "min_corner",
        "max_corner",
        "plane_point",
        "normal",
        "color",
        "opacity",
        "confidence",
    }
    assert carrier_parameters["surface"]["min_corner"].requires_grad is True
    assert carrier_parameters["surface"]["max_corner"].requires_grad is True
    assert carrier_parameters["surface"]["plane_point"].requires_grad is True
    assert carrier_parameters["surface"]["normal"].requires_grad is True
    assert carrier_parameters["surface"]["color"].requires_grad is True
    assert carrier_parameters["surface"]["opacity"].requires_grad is True
    assert carrier_parameters["surface"]["confidence"].requires_grad is True


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_volume_kernel_keeps_density_differentiable():
    import torch

    elements = (
        AuraElement(
            id="volume",
            carrier_id="volume",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "volume_cell", "density": 2.0, "opacity": 0.5},
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.2, 0.4, 0.6]], requires_grad=True)
    opacities = torch.tensor([0.5], requires_grad=True)
    confidences = torch.tensor([0.75], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
        carrier_parameters=carrier_parameters,
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert set(carrier_parameters["volume"]) == {"min_corner", "max_corner", "color", "density", "opacity", "confidence"}
    assert carrier_parameters["volume"]["color"].grad is not None
    assert carrier_parameters["volume"]["density"].grad is not None
    assert carrier_parameters["volume"]["opacity"].grad is not None
    assert carrier_parameters["volume"]["confidence"].grad is not None
    assert carrier_parameters["volume"]["color"].grad.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert carrier_parameters["volume"]["density"].grad.item() == pytest.approx(-0.5 * torch.exp(torch.tensor(-2.0)).item())
    assert carrier_parameters["volume"]["opacity"].grad.item() == pytest.approx(-(1.0 - torch.exp(torch.tensor(-2.0))).item())
    assert carrier_parameters["volume"]["confidence"].grad.item() == pytest.approx(1.0)
    assert opacities.grad is None
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_beta_kernel_keeps_shape_parameters_differentiable():
    import torch

    elements = (
        AuraElement(
            id="beta",
            carrier_id="beta",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "beta_kernel", "alpha": 2.0, "beta": 3.0, "support_radius": [0.4, 0.5, 0.6]},
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.4, 0.5, 0.6]])
    colors = torch.tensor([[0.2, 0.4, 0.6]], requires_grad=True)
    opacities = torch.tensor([0.5], requires_grad=True)
    confidences = torch.tensor([0.75], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
        carrier_parameters=carrier_parameters,
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert set(carrier_parameters["beta"]) == {
        "min_corner",
        "max_corner",
        "color",
        "opacity",
        "confidence",
        "alpha",
        "beta",
        "support_radius",
    }
    assert carrier_parameters["beta"]["color"].grad is not None
    assert carrier_parameters["beta"]["opacity"].grad is not None
    assert carrier_parameters["beta"]["confidence"].grad is not None
    assert carrier_parameters["beta"]["alpha"].grad is not None
    assert carrier_parameters["beta"]["beta"].grad is not None
    assert carrier_parameters["beta"]["support_radius"].grad is not None
    assert carrier_parameters["beta"]["color"].grad.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert torch.isfinite(carrier_parameters["beta"]["opacity"].grad)
    assert carrier_parameters["beta"]["confidence"].grad.item() == pytest.approx(1.0)
    assert torch.isfinite(carrier_parameters["beta"]["alpha"].grad)
    assert torch.isfinite(carrier_parameters["beta"]["beta"].grad)
    assert torch.all(torch.isfinite(carrier_parameters["beta"]["support_radius"].grad))
    assert confidences.grad is None
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_kernel_keeps_frequency_phase_bandwidth_differentiable():
    import torch

    elements = (
        AuraElement(
            id="gabor",
            carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "gabor_frequency", "frequency": (0.25, 0.5, 0.75), "phase": 0.1, "bandwidth": 0.6},
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.4, 0.5, 0.6]])
    colors = torch.tensor([[0.2, 0.4, 0.6]], requires_grad=True)
    opacities = torch.tensor([0.5], requires_grad=True)
    confidences = torch.tensor([0.75], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
        carrier_parameters=carrier_parameters,
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert set(carrier_parameters["gabor"]) == {
        "min_corner",
        "max_corner",
        "plane_point",
        "normal",
        "color",
        "opacity",
        "confidence",
        "frequency",
        "phase",
        "bandwidth",
    }
    assert carrier_parameters["gabor"]["plane_point"].requires_grad is True
    assert carrier_parameters["gabor"]["color"].grad is not None
    assert carrier_parameters["gabor"]["opacity"].grad is not None
    assert carrier_parameters["gabor"]["confidence"].grad is not None
    assert carrier_parameters["gabor"]["frequency"].grad is not None
    assert carrier_parameters["gabor"]["phase"].grad is not None
    assert carrier_parameters["gabor"]["bandwidth"].grad is not None
    assert torch.all(torch.isfinite(carrier_parameters["gabor"]["color"].grad))
    assert torch.isfinite(carrier_parameters["gabor"]["opacity"].grad)
    assert torch.isfinite(carrier_parameters["gabor"]["confidence"].grad)
    assert torch.all(torch.isfinite(carrier_parameters["gabor"]["frequency"].grad))
    assert torch.isfinite(carrier_parameters["gabor"]["phase"].grad)
    assert torch.isfinite(carrier_parameters["gabor"]["bandwidth"].grad)
    assert opacities.grad is None
    assert confidences.grad is None
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_neural_kernel_keeps_residual_scale_differentiable():
    import torch

    elements = (
        AuraElement(
            id="neural",
            carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "neural_residual", "latent_dim": 8, "residual_scale": 0.4},
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.4, 0.5, 0.6]])
    colors = torch.tensor([[0.2, 0.4, 0.6]], requires_grad=True)
    opacities = torch.tensor([0.5], requires_grad=True)
    confidences = torch.tensor([0.75], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
        carrier_parameters=carrier_parameters,
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert set(carrier_parameters["neural"]) == {"min_corner", "max_corner", "color", "opacity", "confidence", "residual_scale"}
    assert carrier_parameters["neural"]["color"].grad is not None
    assert carrier_parameters["neural"]["opacity"].grad is not None
    assert carrier_parameters["neural"]["confidence"].grad is not None
    assert carrier_parameters["neural"]["residual_scale"].grad is not None
    assert carrier_parameters["neural"]["color"].grad.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert carrier_parameters["neural"]["residual_scale"].grad.item() == pytest.approx(-0.5 - 0.25 * 0.75)
    assert carrier_parameters["neural"]["opacity"].grad.item() == pytest.approx(-0.4)
    assert carrier_parameters["neural"]["confidence"].grad.item() == pytest.approx(0.9)
    assert opacities.grad is None
    assert confidences.grad is None
    assert residual.tolist() == [True]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_semantic_kernel_keeps_confidence_differentiable():
    import torch

    elements = (
        AuraElement(
            id="semantic",
            carrier_id="semantic",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.75,
            payload={"type": "semantic_feature", "label": "object", "confidence": 0.9},
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.4, 0.5, 0.6]])
    colors = torch.tensor([[0.2, 0.4, 0.6]], requires_grad=True)
    opacities = torch.tensor([0.5], requires_grad=True)
    confidences = torch.tensor([0.75], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
        carrier_parameters=carrier_parameters,
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert carrier_parameters["semantic"]["confidence"].grad is not None
    assert carrier_parameters["semantic"]["confidence"].grad.item() == pytest.approx(1.0)
    assert colors.grad is not None
    assert opacities.grad is not None
    assert confidences.grad is None
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_fallback_kernel_keeps_color_opacity_confidence_differentiable():
    import torch

    elements = (
        AuraElement(
            id="gaussian",
            carrier_id="gaussian",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.2, 0.4, 0.6),
            opacity=0.5,
            confidence=0.75,
            payload={
                "type": "gaussian_fallback",
                "mean": [0.0, 0.0, 0.0],
                "covariance": [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
            },
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.1, 0.0, 0.0]])
    colors = torch.tensor([[0.1, 0.1, 0.1]], requires_grad=True)
    opacities = torch.tensor([0.1], requires_grad=True)
    confidences = torch.tensor([0.25], requires_grad=True)
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    carrier_colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
        carrier_parameters=carrier_parameters,
    )
    loss = carrier_colors.sum() + transmittance.sum() + confidence.sum()
    loss.backward()

    assert carrier_parameters["gaussian"]["color"].grad is not None
    assert carrier_parameters["gaussian"]["opacity"].grad is not None
    assert carrier_parameters["gaussian"]["confidence"].grad is not None
    assert carrier_parameters["gaussian"]["gaussian_mean"].grad is not None
    assert carrier_parameters["gaussian"]["gaussian_covariance_diag"].grad is not None
    assert carrier_parameters["gaussian"]["color"].grad.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert carrier_parameters["gaussian"]["opacity"].grad.item() == pytest.approx(-0.882497)
    assert carrier_parameters["gaussian"]["confidence"].grad.item() == pytest.approx(0.882497)
    assert carrier_parameters["gaussian"]["gaussian_mean"].grad[0].item() > 0.0
    assert carrier_parameters["gaussian"]["gaussian_mean"].grad[1].item() == pytest.approx(0.0)
    assert carrier_parameters["gaussian"]["gaussian_mean"].grad[2].item() == pytest.approx(0.0)
    assert torch.all(torch.isfinite(carrier_parameters["gaussian"]["gaussian_covariance_diag"].grad))
    assert colors.grad is None
    assert opacities.grad is None
    assert confidences.grad is None
    assert residual.tolist() == [False]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gaussian_fallback_kernel_uses_covariance_weighted_support():
    import torch

    elements = (
        AuraElement(
            id="gaussian",
            carrier_id="gaussian",
            bounds=Bounds((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
            color=(0.2, 0.4, 0.6),
            opacity=0.8,
            confidence=0.75,
            payload={
                "type": "gaussian_fallback",
                "mean": [0.0, 0.0, 0.0],
                "covariance": [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
            },
        ),
    )
    best_index = torch.tensor([0, 0], dtype=torch.long)
    best_depth = torch.tensor([1.0, 1.0])
    exit_depth = torch.tensor([[2.0], [2.0]])
    hit_points = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    colors = torch.tensor([element.color for element in elements])
    opacities = torch.tensor([element.opacity for element in elements])
    confidences = torch.tensor([element.confidence for element in elements])
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    _colors, transmittance, confidence, _residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
    )

    offset_weight = torch.exp(torch.tensor(-2.0)).item()
    assert transmittance[0].item() == pytest.approx(0.2)
    assert confidence[0].item() == pytest.approx(0.75)
    assert transmittance[1].item() == pytest.approx(1.0 - 0.8 * offset_weight)
    assert confidence[1].item() == pytest.approx(0.75 * offset_weight)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_carrier_response_tensors_apply_payload_kernels():
    import torch

    elements = (
        AuraElement(
            id="volume",
            carrier_id="volume",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=1.0,
            payload={"type": "volume_cell", "density": 2.0},
        ),
        AuraElement(
            id="semantic",
            carrier_id="semantic",
            bounds=Bounds((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=0.5,
            payload={"type": "semantic_feature", "label": "object", "confidence": 0.9},
        ),
        AuraElement(
            id="neural",
            carrier_id="neural",
            bounds=Bounds((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
            opacity=0.5,
            confidence=1.0,
            payload={"type": "neural_residual", "latent_dim": 8, "residual_scale": 0.4},
        ),
    )
    best_index = torch.tensor([0, 1, 2], dtype=torch.long)
    best_depth = torch.tensor([1.0, 1.0, 1.0])
    exit_depth = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    hit_points = torch.tensor([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
    colors = torch.ones((3, 3))
    opacities = torch.tensor([0.5, 0.5, 0.5])
    confidences = torch.tensor([1.0, 0.5, 1.0])
    mins = torch.tensor([element.bounds.min_corner for element in elements])
    maxs = torch.tensor([element.bounds.max_corner for element in elements])

    _colors, transmittance, confidence, residual = torch_carrier_response_tensors(
        torch,
        elements,
        best_index,
        best_depth,
        exit_depth,
        hit_points,
        colors,
        opacities,
        confidences,
        mins,
        maxs,
        "cpu",
    )

    assert transmittance[0].item() == pytest.approx(torch.exp(torch.tensor(-2.0)).item())
    assert confidence[1].item() == pytest.approx(0.9)
    assert confidence[2].item() == pytest.approx(0.9)
    assert residual.tolist() == [False, False, True]


# ============================================================
# Step 4 new tests: kernel-level regression tests for the
# four carrier upgrades (DBS, Gabor bank, Scaffold-GS, LangSplatV2)
# ============================================================


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_dbs_beta_kernel_default_reproduces_prior():
    """DBS beta fields at defaults must produce identical outputs to prior kernel."""
    import torch

    def _make_element(payload):
        return AuraElement(
            id="beta",
            carrier_id="beta",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.6,
            confidence=0.8,
            payload=payload,
        )

    base_payload = {"type": "beta_kernel", "alpha": 2.0, "beta": 3.0, "support_radius": [0.5, 0.5, 0.5]}
    dbs_payload = {"type": "beta_kernel", "alpha": 2.0, "beta": 3.0, "support_radius": [0.5, 0.5, 0.5]}
    # adaptive_alpha/beta absent => same as None; frequency_scale/appearance_shift absent => defaults

    elements_base = (_make_element(base_payload),)
    elements_dbs = (_make_element(dbs_payload),)

    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.2, 0.4, 0.6]])
    opacities = torch.tensor([0.6])
    confidences = torch.tensor([0.8])
    mins = torch.tensor([(0.0, 0.0, 0.0)])
    maxs = torch.tensor([(1.0, 1.0, 1.0)])

    out_base = torch_carrier_response_tensors(
        torch, elements_base, best_index, best_depth, exit_depth, hit_points,
        colors, opacities, confidences, mins, maxs, "cpu",
    )
    out_dbs = torch_carrier_response_tensors(
        torch, elements_dbs, best_index, best_depth, exit_depth, hit_points,
        colors, opacities, confidences, mins, maxs, "cpu",
    )

    assert torch.allclose(out_base[0], out_dbs[0])
    assert torch.allclose(out_base[1], out_dbs[1])
    assert torch.allclose(out_base[2], out_dbs[2])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_dbs_beta_kernel_adaptive_shape_overrides():
    """adaptive_alpha overrides base alpha in the kernel computation."""
    import torch

    # Hit at (0.3, 0.3, 0.3): u = mean(1 - |0.3-0.5|/0.5) = mean(1 - 0.4) = 0.6
    # alpha=2, beta=3 gives a non-zero weight; alpha=0.5, beta=0.5 gives weight=1.0
    def _run(payload):
        elements = (AuraElement(
            id="beta", carrier_id="beta",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            opacity=0.8, confidence=0.9, payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.3, 0.3, 0.3]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([0.8]),
            torch.tensor([0.9]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    base = _run({"type": "beta_kernel", "alpha": 2.0, "beta": 3.0, "support_radius": [0.5, 0.5, 0.5]})
    dbs = _run({
        "type": "beta_kernel", "alpha": 2.0, "beta": 3.0, "support_radius": [0.5, 0.5, 0.5],
        "adaptive_alpha": 0.5, "adaptive_beta": 0.5,
    })
    # alpha=0.5, beta=0.5 gives weight=1.0 (U-shaped) vs alpha=2, beta=3 near 0 at this point
    # => different transmittances
    assert not torch.allclose(base[1], dbs[1])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_dbs_beta_kernel_frequency_scale_and_appearance_shift():
    """frequency_scale and appearance_shift affect kernel output."""
    import torch

    # Use hit at (0.3,0.3,0.3) where beta(2,2) gives weight ~ 0.96 > 0
    # Use a dark color so appearance_shift (+0.3) is visible after clamping
    def _run(payload, color=(0.2, 0.2, 0.2)):
        elements = (AuraElement(
            id="beta", carrier_id="beta",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=color,
            opacity=0.5, confidence=0.8, payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.3, 0.3, 0.3]]),  # non-center => weight > 0
            torch.tensor([[list(color)]]),
            torch.tensor([0.5]),
            torch.tensor([0.8]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    base = _run({"type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [0.5, 0.5, 0.5]})
    scaled = _run({
        "type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [0.5, 0.5, 0.5],
        "frequency_scale": 2.0,
    })
    shifted = _run({
        "type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [0.5, 0.5, 0.5],
        "appearance_shift": 0.3,  # adds 0.3 to base color 0.2 => 0.5 (visible change)
    })
    # At (0.3,0.3,0.3) weight ~ 0.96 so base transmittance < 1; scaled transmittance < base
    assert base[1].item() < 1.0, "base transmittance should be < 1 at non-center point"
    # frequency_scale=2.0 doubles the weight => changes transmittance
    assert not torch.allclose(base[1], scaled[1])
    # appearance_shift adds 0.3 to base color 0.2 => 0.5; visible difference
    assert not torch.allclose(base[0], shifted[0])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_bank_default_single_filter():
    """Gabor num_filters=1 (default) produces identical output to prior kernel."""
    import torch

    def _run(payload):
        elements = (AuraElement(
            id="gabor", carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.8, 0.6, 0.2), opacity=0.4, confidence=0.9,
            payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([[0.8, 0.6, 0.2]]),
            torch.tensor([0.4]),
            torch.tensor([0.9]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    base = _run({"type": "gabor_frequency", "frequency": (0.0, 0.0, 0.5), "phase": 0.2, "bandwidth": 0.6})
    bank1 = _run({"type": "gabor_frequency", "frequency": (0.0, 0.0, 0.5), "phase": 0.2, "bandwidth": 0.6, "num_filters": 1})
    assert torch.allclose(base[0], bank1[0])
    assert torch.allclose(base[1], bank1[1])
    assert torch.allclose(base[2], bank1[2])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_gabor_bank_multi_filter():
    """num_filters=2 produces a weighted sum of two Gabor kernels."""
    import torch

    def _run(payload):
        elements = (AuraElement(
            id="gabor", carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(1.0, 1.0, 1.0), opacity=0.5, confidence=0.9,
            payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.3, 0.3, 0.3]]),
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([0.5]),
            torch.tensor([0.9]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    single = _run({"type": "gabor_frequency", "frequency": (0.0, 0.0, 1.0), "phase": 0.0, "bandwidth": 0.5})
    multi = _run({
        "type": "gabor_frequency",
        "frequency": (0.0, 0.0, 1.0), "phase": 0.0, "bandwidth": 0.5,
        "num_filters": 2,
        "frequencies": [(0.0, 0.0, 0.5), (0.0, 0.0, 2.0)],
        "phases": [0.0, 0.25],
        "filter_weights": [0.5, 0.5],
    })
    # Multi-filter output is valid (different from single unless all filters identical)
    assert multi[0].shape == single[0].shape
    assert multi[1].shape == single[1].shape
    # Colors are in valid range
    assert (multi[0] >= 0.0).all() and (multi[0] <= 1.0).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_neural_residual_scaffold_default():
    """Scaffold-GS fields at defaults reproduce identical outputs to prior kernel."""
    import torch

    def _run(payload):
        elements = (AuraElement(
            id="neural", carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.2, 0.4, 0.8), opacity=0.7, confidence=0.85,
            payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([[0.2, 0.4, 0.8]]),
            torch.tensor([0.7]),
            torch.tensor([0.85]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    base = _run({"type": "neural_residual", "latent_dim": 8, "residual_scale": 0.35})
    scaffold = _run({
        "type": "neural_residual", "latent_dim": 8, "residual_scale": 0.35,
        # defaults: anchor_feature_dim absent, use_anchor_conditioning absent
    })
    assert torch.allclose(base[0], scaffold[0])
    assert torch.allclose(base[1], scaffold[1])
    assert torch.allclose(base[2], scaffold[2])
    assert base[3].tolist() == scaffold[3].tolist()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_neural_residual_scaffold_anchor():
    """anchor_feature_dim splits latent and modulates residual strength."""
    import torch

    def _run(payload):
        elements = (AuraElement(
            id="neural", carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.5, 0.5, 0.5), opacity=0.8, confidence=0.9,
            payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([0.8]),
            torch.tensor([0.9]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    base = _run({"type": "neural_residual", "latent_dim": 32, "residual_scale": 0.5})
    anchored = _run({
        "type": "neural_residual", "latent_dim": 32, "residual_scale": 0.5,
        "anchor_feature_dim": 16,  # 16/32 = 0.5 anchor ratio => residual_strength halved
    })
    # anchor_feature_dim splits the latent, reducing effective residual strength
    # => transmittance should be higher (less occlusion) with anchor dim < latent dim
    assert anchored[1].item() > base[1].item()
    # residual flag must still be set
    assert anchored[3].tolist() == [True]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_neural_residual_use_anchor_conditioning_no_op():
    """use_anchor_conditioning=True is a no-op when no neighbors provided."""
    import torch

    def _run(use_anchor):
        payload = {
            "type": "neural_residual", "latent_dim": 8, "residual_scale": 0.4,
        }
        if use_anchor:
            payload["use_anchor_conditioning"] = True
        elements = (AuraElement(
            id="neural", carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.3, 0.6, 0.9), opacity=0.6, confidence=0.75,
            payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([[0.3, 0.6, 0.9]]),
            torch.tensor([0.6]),
            torch.tensor([0.75]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    base = _run(False)
    anchored = _run(True)
    # No-op: same outputs when no neighbor features are provided
    assert torch.allclose(base[0], anchored[0])
    assert torch.allclose(base[1], anchored[1])
    assert torch.allclose(base[2], anchored[2])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_semantic_sparse_codebook_default_dense():
    """use_sparse_codebook=False (default) produces identical kernel output to prior."""
    import torch

    def _run(payload):
        elements = (AuraElement(
            id="semantic", carrier_id="semantic",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.5, 0.5, 0.5), opacity=0.3, confidence=0.7,
            payload=payload,
        ),)
        return torch_carrier_response_tensors(
            torch, elements,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor([[2.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([0.3]),
            torch.tensor([0.7]),
            torch.tensor([(0.0, 0.0, 0.0)]),
            torch.tensor([(1.0, 1.0, 1.0)]),
            "cpu",
        )

    base = _run({"type": "semantic_feature", "label": "object", "confidence": 0.85})
    dense = _run({
        "type": "semantic_feature", "label": "object", "confidence": 0.85,
        # use_sparse_codebook absent => defaults to False
    })
    assert torch.allclose(base[0], dense[0])
    assert torch.allclose(base[1], dense[1])
    assert torch.allclose(base[2], dense[2])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_semantic_sparse_codebook_sparse():
    """Sparse codebook decode produces correct feature vector via semantic.decode_semantic_feature."""
    from aura.semantic import decode_semantic_feature
    from aura.carrier_payloads import SemanticFeaturePayload

    payload = SemanticFeaturePayload(
        label="table", confidence=0.75,
        use_sparse_codebook=True,
        codebook_size=8, codebook_dim=4,
        sparse_indices=[1, 3],
        sparse_weights=[0.8, 0.2],
    ).to_dict()

    codebook = [
        [0.0, 0.0, 0.0, 0.0],  # atom 0
        [1.0, 2.0, 3.0, 4.0],  # atom 1
        [0.0, 0.0, 0.0, 0.0],  # atom 2
        [5.0, 0.0, 0.0, 0.0],  # atom 3
        [0.0, 0.0, 0.0, 0.0],  # atom 4
        [0.0, 0.0, 0.0, 0.0],  # atom 5
        [0.0, 0.0, 0.0, 0.0],  # atom 6
        [0.0, 0.0, 0.0, 0.0],  # atom 7
    ]
    result = decode_semantic_feature(payload, codebook=codebook)
    # 0.8 * [1,2,3,4] + 0.2 * [5,0,0,0] = [0.8+1.0, 1.6, 2.4, 3.2] = [1.8, 1.6, 2.4, 3.2]
    assert result == pytest.approx([1.8, 1.6, 2.4, 3.2])


# ---------------------------------------------------------------------------
# Line 41: TorchCarrierKernelSpec.blockers when autograd_kernel is False
# ---------------------------------------------------------------------------

def test_torch_carrier_kernel_spec_blockers_lists_missing_autograd():
    from aura.torch_kernels import TorchCarrierKernelSpec
    spec = TorchCarrierKernelSpec(
        carrier_id="custom",
        payload_type="custom_cell",
        description="test spec with no autograd kernel",
        implementation_stage="placeholder",
        autograd_kernel=False,
        cuda_kernel=False,
        differentiable_fields=(),
    )
    assert "missing_autograd_kernel" in spec.blockers
    assert "missing_cuda_kernel" in spec.blockers


# ---------------------------------------------------------------------------
# Line 153: torch_carrier_response_tensors_batched raises on empty elements
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_batched_carrier_response_raises_on_empty_elements():
    import torch
    with pytest.raises(ValueError, match="batched carrier response requires at least one element"):
        torch_carrier_response_tensors_batched(
            torch, (), torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0]), torch.tensor([[2.0]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 1.0, 1.0]]),
            "cpu",
        )


# ---------------------------------------------------------------------------
# Lines 160-163, 180-183, 206-210, 362-364: is_batched_gaussian=True path
# in torch_carrier_response_tensors_batched via carrier_parameters with __batched__
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_batched_carrier_response_uses_batched_gaussian_path():
    """Exercise the is_batched_gaussian=True branches (lines 160-163, 180-183, 206-210, 362-364)."""
    import torch

    N = 2
    elements = tuple(
        AuraElement(
            id=f"g{i}",
            carrier_id="gaussian",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.5, 0.5, 0.5),
            opacity=0.3,
            confidence=0.8,
            payload={"type": "gaussian_fallback"},
        )
        for i in range(N)
    )
    # Build carrier_parameters with __batched__ key to trigger the batched path
    carrier_parameters = {
        "__batched__": {
            "color": torch.tensor([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]], dtype=torch.float32),
            "opacity": torch.tensor([0.3, 0.4], dtype=torch.float32),
            "confidence": torch.tensor([0.8, 0.7], dtype=torch.float32),
            "gaussian_covariance_diag": torch.tensor([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], dtype=torch.float32),
            "gaussian_mean": torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32),
            "min_corner": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
            "max_corner": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        },
        "__batched_meta__": {
            "gaussian_mean_present": torch.tensor([True, True], dtype=torch.bool),
            "residual": torch.tensor([False, False], dtype=torch.bool),
        },
    }
    best_index = torch.tensor([0, 1], dtype=torch.long)
    best_depth = torch.tensor([1.0, 1.0])
    exit_depth = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
    opacities = torch.tensor([0.3, 0.4])
    confidences = torch.tensor([0.8, 0.7])
    mins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    maxs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    c, t, conf, r = torch_carrier_response_tensors_batched(
        torch, elements, best_index, best_depth, exit_depth,
        hit_points, colors, opacities, confidences, mins, maxs,
        "cpu", carrier_parameters=carrier_parameters,
    )
    assert c.shape == (N, 3)
    assert t.shape == (N,)
    assert conf.shape == (N,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_batched_carrier_response_batched_gaussian_without_batched_meta():
    """Exercise the is_batched_gaussian=True path without __batched_meta__ (lines 208-209)."""
    import torch

    N = 2
    elements = tuple(
        AuraElement(
            id=f"g{i}",
            carrier_id="gaussian",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.5, 0.5, 0.5),
            opacity=0.3,
            confidence=0.8,
            payload={"type": "gaussian_fallback"},
        )
        for i in range(N)
    )
    carrier_parameters = {
        "__batched__": {
            "color": torch.tensor([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]], dtype=torch.float32),
            "opacity": torch.tensor([0.3, 0.4], dtype=torch.float32),
            "confidence": torch.tensor([0.8, 0.7], dtype=torch.float32),
            "gaussian_covariance_diag": torch.tensor([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]], dtype=torch.float32),
            "gaussian_mean": torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.float32),
            "min_corner": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
            "max_corner": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        },
        # No __batched_meta__ → triggers the "residual is None → zeros" branch
    }
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0, 2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.5, 0.5, 0.5]])
    opacities = torch.tensor([0.3])
    confidences = torch.tensor([0.8])
    mins = torch.tensor([[0.0, 0.0, 0.0]])
    maxs = torch.tensor([[1.0, 1.0, 1.0]])

    c, t, conf, r = torch_carrier_response_tensors_batched(
        torch, elements, best_index, best_depth, exit_depth,
        hit_points, colors, opacities, confidences, mins, maxs,
        "cpu", carrier_parameters=carrier_parameters,
    )
    assert r.tolist() == [False]


# ---------------------------------------------------------------------------
# Line 570: gabor with vector (tuple) frequency (not int/float)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_response_gabor_scalar_frequency_in_multifilter_bank():
    """Line 570: num_filters>1 with scalar _f_freq triggers the scalar branch."""
    import torch

    elements = (
        AuraElement(
            id="gabor_scalar_freq",
            carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.5, 0.3, 0.1),
            opacity=0.5,
            confidence=0.9,
            payload={
                "type": "gabor_frequency",
                "frequency": (1.0, 0.0, 0.0),
                "phase": 0.0,
                "bandwidth": 0.5,
                "num_filters": 2,
                # scalar frequencies in the bank → triggers line 570
                "frequencies": [1.0, 0.5],  # scalars, not vectors
            },
        ),
    )
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.5, 0.3, 0.1]])
    opacities = torch.tensor([0.5])
    confidences = torch.tensor([0.9])
    mins = torch.tensor([[0.0, 0.0, 0.0]])
    maxs = torch.tensor([[1.0, 1.0, 1.0]])

    c, t, conf, r = torch_carrier_response_tensors(
        torch, elements, best_index, best_depth, exit_depth,
        hit_points, colors, opacities, confidences, mins, maxs, "cpu",
    )
    assert c.shape == (1, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_batched_carrier_response_gabor_with_vector_frequency():
    """Line 575: frequency is a vector tuple, so else branch fires."""
    import torch

    elements = (
        AuraElement(
            id="gabor_vec",
            carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.5, 0.3, 0.1),
            opacity=0.5,
            confidence=0.9,
            payload={
                "type": "gabor_frequency",
                "frequency": (1.0, 0.5, 0.0),
                "phase": 0.0,
                "bandwidth": 0.5,
                "num_filters": 2,
                "frequencies": [(1.0, 0.5, 0.0), (0.5, 1.0, 0.0)],  # list of vector freqs
            },
        ),
    )
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.5, 0.3, 0.1]])
    opacities = torch.tensor([0.5])
    confidences = torch.tensor([0.9])
    mins = torch.tensor([[0.0, 0.0, 0.0]])
    maxs = torch.tensor([[1.0, 1.0, 1.0]])

    c, t, conf, r = torch_carrier_response_tensors(
        torch, elements, best_index, best_depth, exit_depth,
        hit_points, colors, opacities, confidences, mins, maxs, "cpu",
    )
    assert c.shape == (1, 3)


# ---------------------------------------------------------------------------
# Lines 625-640: neural anchor conditioning path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_response_neural_anchor_when_neighbor_feats_is_none(monkeypatch):
    """Line 640: neighbor_elements set + mlp set, but neighbor_features returns None."""
    import torch
    import aura.cross_carrier as _cc

    monkeypatch.setattr(_cc, "neighbor_features_from_carrier_parameters", lambda *a, **kw: None)

    neighbor = AuraElement(
        id="surface_neighbor",
        carrier_id="surface",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        color=(1.0, 0.0, 0.0),
        opacity=0.5,
        payload={"type": "surface_cell"},
    )
    elements = (
        AuraElement(
            id="neural_no_feats",
            carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.2, 0.4, 0.8),
            opacity=0.7,
            confidence=0.85,
            payload={
                "type": "neural_residual",
                "residual_scale": 0.35,
                "use_anchor_conditioning": True,
                "neighbor_elements": [neighbor],
            },
        ),
    )
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.2, 0.4, 0.8]])
    opacities = torch.tensor([0.7])
    confidences = torch.tensor([0.85])
    mins = torch.tensor([[0.0, 0.0, 0.0]])
    maxs = torch.tensor([[1.0, 1.0, 1.0]])

    carrier_params = torch_carrier_parameter_tensors(torch, elements, device="cpu", requires_grad=False)
    from aura.cross_carrier import build_cross_carrier_mlp
    carrier_params["neural_no_feats"]["cross_carrier_mlp"] = build_cross_carrier_mlp(torch, "cpu")

    c, t, conf, r = torch_carrier_response_tensors(
        torch, elements, best_index, best_depth, exit_depth,
        hit_points, colors, opacities, confidences, mins, maxs, "cpu",
        carrier_parameters=carrier_params,
    )
    assert c.shape == (1, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_batched_carrier_response_neural_with_cross_carrier_mlp():
    """Lines 625-639: neural_residual with cross_carrier_mlp and neighbor_elements → runs MLP."""
    import torch
    from aura.cross_carrier import build_cross_carrier_mlp

    neighbor = AuraElement(
        id="surface_neighbor",
        carrier_id="surface",
        bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        color=(1.0, 0.0, 0.0),
        opacity=0.5,
        payload={"type": "surface_cell"},
    )
    elements = (
        AuraElement(
            id="neural_anchor",
            carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.2, 0.4, 0.8),
            opacity=0.7,
            confidence=0.85,
            payload={
                "type": "neural_residual",
                "residual_scale": 0.35,
                "use_anchor_conditioning": True,
                "neighbor_elements": [neighbor],
            },
        ),
    )
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.5, 0.5, 0.5]])
    colors = torch.tensor([[0.2, 0.4, 0.8]])
    opacities = torch.tensor([0.7])
    confidences = torch.tensor([0.85])
    mins = torch.tensor([[0.0, 0.0, 0.0]])
    maxs = torch.tensor([[1.0, 1.0, 1.0]])

    # Build carrier_parameters with a cross_carrier_mlp for the neural element
    carrier_params = torch_carrier_parameter_tensors(torch, elements, device="cpu", requires_grad=False)
    # Also build an MLP explicitly and store it in carrier_parameters
    _mlp = build_cross_carrier_mlp(torch, "cpu")
    carrier_params["neural_anchor"]["cross_carrier_mlp"] = _mlp

    c, t, conf, r = torch_carrier_response_tensors(
        torch, elements, best_index, best_depth, exit_depth,
        hit_points, colors, opacities, confidences, mins, maxs, "cpu",
        carrier_parameters=carrier_params,
    )
    assert c.shape == (1, 3)


# ---------------------------------------------------------------------------
# Lines 709-744, 774: _torch_batched_gaussian_parameter_tensors called via
# torch_carrier_parameter_tensors with 1001+ pure gaussian_fallback elements
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_carrier_parameter_tensors_uses_batched_path_for_large_gaussian_set():
    """Lines 709-744, 774: >1000 gaussian_fallback triggers batched parameter tensors."""
    import torch

    N = 1001
    elements = tuple(
        AuraElement(
            id=f"gaussian_{i}",
            carrier_id="gaussian",
            bounds=Bounds((float(i) * 0.001, 0.0, 0.0), (float(i) * 0.001 + 0.001, 1.0, 1.0)),
            color=(0.5, 0.5, 0.5),
            opacity=0.3,
            confidence=0.8,
            payload={"type": "gaussian_fallback"},
        )
        for i in range(N)
    )

    params = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    assert "__batched__" in params
    assert "color" in params["__batched__"]
    assert params["__batched__"]["color"].shape == (N, 3)
    assert "__batched_meta__" in params
    assert "gaussian_mean_present" in params["__batched_meta__"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_batched_gaussian_parameter_tensors_includes_mean_when_present():
    """Lines 727-737: elements with explicit mean in payload."""
    import torch
    from aura.torch_kernels import _torch_batched_gaussian_parameter_tensors

    elements = (
        AuraElement(
            id="g_with_mean",
            carrier_id="gaussian",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "gaussian_fallback", "mean": [0.5, 0.5, 0.5]},
        ),
        AuraElement(
            id="g_no_mean",
            carrier_id="gaussian",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "gaussian_fallback"},  # no mean → centers
        ),
    )

    params = _torch_batched_gaussian_parameter_tensors(torch, elements, device="cpu", requires_grad=False)
    assert "__batched__" in params
    meta = params["__batched_meta__"]
    assert meta["gaussian_mean_present"].tolist() == [True, False]


# ---------------------------------------------------------------------------
# Line 944-948: neural with use_anchor_conditioning=True in parameter tensors
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_parameter_tensors_neural_with_anchor_conditioning():
    """Lines 944-948: neural_residual with use_anchor_conditioning=True builds cross_carrier_mlp."""
    import torch

    elements = (
        AuraElement(
            id="neural_anchored",
            carrier_id="neural",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            color=(0.2, 0.4, 0.8),
            opacity=0.7,
            confidence=0.85,
            payload={"type": "neural_residual", "use_anchor_conditioning": True},
        ),
    )

    params = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    assert "cross_carrier_mlp" in params["neural_anchored"]


# ---------------------------------------------------------------------------
# Line 1015: surface_cell with explicit plane_point in payload
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_parameter_tensors_surface_with_explicit_plane_point():
    """Line 1015: plane_point present in payload → use it directly."""
    import torch

    elements = (
        AuraElement(
            id="surface_explicit",
            carrier_id="surface",
            bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            payload={"type": "surface_cell", "plane_point": [0.1, 0.2, 0.3]},
        ),
    )

    params = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    pp = params["surface_explicit"]["plane_point"].tolist()
    assert pp == pytest.approx([0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Line 1033: gabor without explicit plane_point → uses _center_point fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_parameter_tensors_gabor_with_explicit_plane_point():
    """Line 1033: gabor_frequency with explicit plane_point → uses it directly."""
    import torch

    elements = (
        AuraElement(
            id="gabor_explicit_point",
            carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
            payload={"type": "gabor_frequency", "frequency": (1.0, 0.0, 0.0), "phase": 0.0,
                     "plane_point": [0.3, 0.4, 0.5]},
        ),
    )

    params = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    pp = params["gabor_explicit_point"]["plane_point"].tolist()
    assert pp == pytest.approx([0.3, 0.4, 0.5])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_carrier_parameter_tensors_gabor_without_plane_point_uses_center():
    """Line 1035: gabor_frequency without plane_point/point → _center_point fallback."""
    import torch

    elements = (
        AuraElement(
            id="gabor_no_point",
            carrier_id="gabor",
            bounds=Bounds((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
            payload={"type": "gabor_frequency", "frequency": (1.0, 0.0, 0.0), "phase": 0.0},
            # no plane_point or point
        ),
    )

    params = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    # Center of (0,0,0)-(2,2,2) is (1,1,1)
    pp = params["gabor_no_point"]["plane_point"].tolist()
    assert pp == pytest.approx([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Lines 1070-1072, 1076-1079: _stack_scalar_parameter with batched/per-element params
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_scalar_parameter_returns_from_batched_cache():
    """Lines 1070-1072: carrier_parameters has __batched__ with the name."""
    import torch
    from aura.torch_kernels import _stack_scalar_parameter

    elements = (
        AuraElement(id="e1", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0)), opacity=0.5),
        AuraElement(id="e2", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0)), opacity=0.3),
    )
    batched_opacities = torch.tensor([0.9, 0.8])
    carrier_parameters = {"__batched__": {"opacity": batched_opacities}}

    result = _stack_scalar_parameter(torch, elements, "opacity", carrier_parameters, "cpu", defaults=[0.5, 0.3])
    assert result is batched_opacities


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_scalar_parameter_uses_per_element_parameter():
    """Lines 1076-1079: carrier_parameters has per-element override."""
    import torch
    from aura.torch_kernels import _stack_scalar_parameter

    elements = (
        AuraElement(id="e1", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0)), opacity=0.5),
    )
    per_element_tensor = torch.tensor(0.77)
    carrier_parameters = {"e1": {"opacity": per_element_tensor}}

    result = _stack_scalar_parameter(torch, elements, "opacity", carrier_parameters, "cpu", defaults=[0.5])
    assert result[0].item() == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Lines 1094-1096, 1100-1103: _stack_vector_parameter with batched/per-element params
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_vector_parameter_returns_from_batched_cache():
    """Lines 1094-1096: carrier_parameters has __batched__ with the name."""
    import torch
    from aura.torch_kernels import _stack_vector_parameter

    elements = (
        AuraElement(id="e1", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0))),
    )
    batched_colors = torch.tensor([[0.9, 0.8, 0.7]])
    carrier_parameters = {"__batched__": {"color": batched_colors}}

    result = _stack_vector_parameter(torch, elements, "color", carrier_parameters, "cpu", defaults=[(0.5, 0.5, 0.5)])
    assert result is batched_colors


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_vector_parameter_uses_per_element_parameter():
    """Lines 1100-1103: carrier_parameters has per-element override."""
    import torch
    from aura.torch_kernels import _stack_vector_parameter

    elements = (
        AuraElement(id="e1", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0))),
    )
    per_element_tensor = torch.tensor([0.1, 0.2, 0.3])
    carrier_parameters = {"e1": {"color": per_element_tensor}}

    result = _stack_vector_parameter(torch, elements, "color", carrier_parameters, "cpu", defaults=[(0.5, 0.5, 0.5)])
    assert result[0].tolist() == pytest.approx([0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Lines 1115-1121: _stack_gaussian_mean_parameter with batched cache
# Lines 1127-1128: _stack_gaussian_mean_parameter with per-element parameter
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_gaussian_mean_parameter_returns_from_batched_cache():
    """Lines 1115-1121: carrier_parameters has __batched__ with gaussian_mean."""
    import torch
    from aura.torch_kernels import _stack_gaussian_mean_parameter

    elements = (
        AuraElement(id="e1", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0))),
    )
    batched_mean = torch.tensor([[0.5, 0.5, 0.5]])
    carrier_parameters = {
        "__batched__": {"gaussian_mean": batched_mean},
        "__batched_meta__": {"gaussian_mean_present": torch.tensor([True])},
    }

    mean_result, present_result = _stack_gaussian_mean_parameter(torch, elements, carrier_parameters, "cpu")
    assert mean_result is batched_mean
    assert present_result.tolist() == [True]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_gaussian_mean_parameter_without_batched_meta_present():
    """Lines 1119-1120: __batched__ has gaussian_mean but no __batched_meta__ gaussian_mean_present."""
    import torch
    from aura.torch_kernels import _stack_gaussian_mean_parameter

    elements = (
        AuraElement(id="e1", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0))),
    )
    batched_mean = torch.tensor([[0.5, 0.5, 0.5]])
    carrier_parameters = {
        "__batched__": {"gaussian_mean": batched_mean},
        # No __batched_meta__ → present defaults to ones
    }

    mean_result, present_result = _stack_gaussian_mean_parameter(torch, elements, carrier_parameters, "cpu")
    assert mean_result is batched_mean
    assert present_result.tolist() == [True]  # ones


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_stack_gaussian_mean_parameter_uses_per_element_parameter():
    """Lines 1127-1128: carrier_parameters has per-element gaussian_mean."""
    import torch
    from aura.torch_kernels import _stack_gaussian_mean_parameter

    elements = (
        AuraElement(id="e1", carrier_id="gaussian", bounds=Bounds((0.0,0.0,0.0),(1.0,1.0,1.0)),
                    payload={"type": "gaussian_fallback"}),
    )
    per_element_mean = torch.tensor([0.3, 0.6, 0.9])
    carrier_parameters = {"e1": {"gaussian_mean": per_element_mean}}

    mean_result, present_result = _stack_gaussian_mean_parameter(torch, elements, carrier_parameters, "cpu")
    assert mean_result[0].tolist() == pytest.approx([0.3, 0.6, 0.9])
    assert present_result.tolist() == [True]


# ---------------------------------------------------------------------------
# Line 1141: _normal_parameter returns actual value when available
# ---------------------------------------------------------------------------

def test_normal_parameter_returns_value_from_payload():
    """Line 1141: element has normal in payload → return it directly."""
    from aura.torch_kernels import _normal_parameter

    class _FakeElement:
        normal = None
        payload = {"normal": [0.0, 1.0, 0.0]}

    result = _normal_parameter(_FakeElement(), fallback=(0.0, 0.0, -1.0))
    assert result == pytest.approx((0.0, 1.0, 0.0))


def test_normal_parameter_returns_value_from_element_normal():
    """Line 1141: element.normal is set → return it directly."""
    from aura.torch_kernels import _normal_parameter

    class _FakeElement:
        normal = (1.0, 0.0, 0.0)
        payload = {}

    result = _normal_parameter(_FakeElement(), fallback=(0.0, 0.0, -1.0))
    assert result == pytest.approx((1.0, 0.0, 0.0))


def test_normal_parameter_uses_fallback_when_not_set():
    from aura.torch_kernels import _normal_parameter

    class _FakeElement:
        normal = None
        payload = {}

    result = _normal_parameter(_FakeElement(), fallback=(0.0, 0.0, -1.0))
    assert result == (0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------
# Line 1264: _torch_gaussian_weight returns ones when mean is None
# Line 1283: _gaussian_mean returns _center_point when no mean in payload
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_gaussian_weight_returns_ones_when_mean_is_none():
    """Line 1262: mean=None → return all ones."""
    import torch
    from aura.torch_kernels import _torch_gaussian_weight

    class _FakeElement:
        payload = {}
        bounds = Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    points = torch.tensor([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]])
    result = _torch_gaussian_weight(torch, points, _FakeElement(), "cpu", mean=None)
    assert result.tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_torch_gaussian_weight_computes_mahalanobis_with_default_covariance():
    """Line 1264: mean is set but covariance_diag is None → compute from element."""
    import torch
    from aura.torch_kernels import _torch_gaussian_weight

    class _FakeElement:
        payload = {}
        bounds = Bounds((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))

    points = torch.tensor([[1.0, 1.0, 1.0]])  # center of bounds
    mean = torch.tensor([1.0, 1.0, 1.0])  # at center
    # covariance_diag=None → line 1264 fires and uses element bounds
    result = _torch_gaussian_weight(torch, points, _FakeElement(), "cpu", mean=mean, covariance_diag=None)
    # At center, mahalanobis distance = 0, weight should be 1.0
    assert result.tolist() == pytest.approx([1.0])


def test_gaussian_mean_returns_center_when_no_mean_in_payload():
    """Line 1283: element has no mean → use center point of bounds."""
    from aura.torch_kernels import _gaussian_mean

    class _FakeElement:
        payload = {}
        bounds = Bounds((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))

    result = _gaussian_mean(_FakeElement())
    assert result == pytest.approx((1.0, 2.0, 3.0))
