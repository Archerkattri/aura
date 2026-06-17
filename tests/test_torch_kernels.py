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
    kernel_text = _cuda_kernel_text(source_text, report["symbol"])

    assert report["format"] == "AURA_CUDA_RENDERER_SOURCE_REPORT"
    assert report["symbol"] == "aura_render_rays_kernel"
    assert report["available"] is True
    assert report["sourceSymbolAvailable"] is True
    assert report["contractComplete"] is True
    assert report["missingSourceFragments"] == []
    assert report["productionReady"] is False
    assert "python binding dispatch missing" in report["productionBlockers"]
    assert "AABB first-hit traversal over native element bounds" in report["implementedSemantics"]
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
        "out_material_id[ray_i]",
        "out_semantic_id[ray_i]",
        "out_residual[ray_i]",
    ):
        assert fragment in kernel_text


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
    assert status.source_paths == ("cuda/aura_carriers.cu",)
    assert len(status.symbols) == 7
    assert report["format"] == "AURA_CUDA_EXTENSION_REPORT"
    assert report["productionReady"] is False
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
    assert any("renderer binding dispatches cuda_render_rays" in item for item in contract["unavailableUntil"])


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

    assert set(carrier_parameters["surface"]) == {"color", "opacity", "confidence"}
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
            payload={"type": "volume_cell", "density": 2.0},
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

    assert set(carrier_parameters["volume"]) == {"color", "density", "confidence"}
    assert carrier_parameters["volume"]["color"].grad is not None
    assert carrier_parameters["volume"]["density"].grad is not None
    assert carrier_parameters["volume"]["confidence"].grad is not None
    assert carrier_parameters["volume"]["color"].grad.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert carrier_parameters["volume"]["density"].grad.item() == pytest.approx(-torch.exp(torch.tensor(-2.0)).item())
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
            payload={"type": "beta_kernel", "alpha": 2.0, "beta": 3.0},
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

    assert set(carrier_parameters["beta"]) == {"color", "opacity", "alpha", "beta"}
    assert carrier_parameters["beta"]["color"].grad is not None
    assert carrier_parameters["beta"]["opacity"].grad is not None
    assert carrier_parameters["beta"]["alpha"].grad is not None
    assert carrier_parameters["beta"]["beta"].grad is not None
    assert carrier_parameters["beta"]["color"].grad.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert torch.isfinite(carrier_parameters["beta"]["opacity"].grad)
    assert torch.isfinite(carrier_parameters["beta"]["alpha"].grad)
    assert torch.isfinite(carrier_parameters["beta"]["beta"].grad)
    assert confidences.grad.tolist() == pytest.approx([1.0])
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

    assert set(carrier_parameters["gabor"]) == {"color", "frequency", "phase", "bandwidth"}
    assert carrier_parameters["gabor"]["color"].grad is not None
    assert carrier_parameters["gabor"]["frequency"].grad is not None
    assert carrier_parameters["gabor"]["phase"].grad is not None
    assert carrier_parameters["gabor"]["bandwidth"].grad is not None
    assert torch.all(torch.isfinite(carrier_parameters["gabor"]["color"].grad))
    assert torch.all(torch.isfinite(carrier_parameters["gabor"]["frequency"].grad))
    assert torch.isfinite(carrier_parameters["gabor"]["phase"].grad)
    assert torch.isfinite(carrier_parameters["gabor"]["bandwidth"].grad)
    assert opacities.grad is not None
    assert confidences.grad is not None
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

    assert set(carrier_parameters["neural"]) == {"color", "residual_scale"}
    assert carrier_parameters["neural"]["color"].grad is not None
    assert carrier_parameters["neural"]["residual_scale"].grad is not None
    assert carrier_parameters["neural"]["color"].grad.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert carrier_parameters["neural"]["residual_scale"].grad.item() == pytest.approx(-0.25 * 0.75)
    assert opacities.grad is not None
    assert confidences.grad is not None
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
            payload={"type": "gaussian_fallback"},
        ),
    )
    carrier_parameters = torch_carrier_parameter_tensors(torch, elements, device="cpu")
    best_index = torch.tensor([0], dtype=torch.long)
    best_depth = torch.tensor([1.0])
    exit_depth = torch.tensor([[2.0]])
    hit_points = torch.tensor([[0.4, 0.5, 0.6]])
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
    assert carrier_parameters["gaussian"]["color"].grad.tolist() == [1.0, 1.0, 1.0]
    assert carrier_parameters["gaussian"]["opacity"].grad.item() == pytest.approx(-1.0)
    assert carrier_parameters["gaussian"]["confidence"].grad.item() == pytest.approx(1.0)
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
