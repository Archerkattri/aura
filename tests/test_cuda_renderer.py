import importlib.util

import pytest

from aura import AuraElement, AuraScene, Bounds, Ray
from aura.cuda_kernels import CudaExtensionStatus
from aura.cuda_renderer import (
    cuda_render_rays,
    cuda_renderer_dispatch_contract,
    cuda_renderer_boundary_report,
    cuda_renderer_kernel_inputs,
    cuda_renderer_launch_config,
    cuda_renderer_reference_first_hit_indices,
    cuda_renderer_scene_buffers,
    cuda_renderer_symbol_probe,
    simulate_cuda_renderer_kernel,
)


def test_cuda_renderer_launch_config_validates_and_computes_grid():
    config = cuda_renderer_launch_config(
        257,
        threads_per_block=128,
        max_hits=4,
        fallback_backend="cpu",
        device="cuda:0",
    )

    assert config.block_count == 3
    assert config.to_dict() == {
        "rayCount": 257,
        "threadsPerBlock": 128,
        "blockCount": 3,
        "maxHits": 4,
        "fallbackBackend": "cpu",
        "device": "cuda:0",
        "requireCuda": False,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"ray_count": 0}, "ray_count must be positive"),
        ({"ray_count": 1, "threads_per_block": 0}, "threads_per_block must be positive"),
        ({"ray_count": 1, "threads_per_block": 2048}, "threads_per_block must be <= 1024"),
        ({"ray_count": 1, "max_hits": 0}, "max_hits must be positive"),
        ({"ray_count": 1, "fallback_backend": "fake"}, "fallback_backend must be one of"),
    ),
)
def test_cuda_renderer_launch_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cuda_renderer_launch_config(**kwargs)


def test_cuda_renderer_scene_buffers_match_renderer_kernel_abi():
    scene = AuraScene(
        name="cuda_scene_buffer_test",
        elements=(
            AuraElement(
                id="panel",
                carrier_id="surface",
                bounds=Bounds((-1.0, -0.5, 0.0), (0.0, 0.5, 0.2)),
                color=(0.8, 0.2, 0.1),
                opacity=0.75,
                confidence=0.9,
                material_id="enamel",
                semantic_id="tooth",
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="residual",
                carrier_id="neural",
                bounds=Bounds((0.0, -0.5, 0.1), (1.0, 0.5, 0.3)),
                color=(0.1, 0.2, 0.9),
                opacity=0.4,
                confidence=0.55,
                material_id="enamel",
                semantic_id="highlight",
                payload={"type": "neural_residual", "residual_scale": 0.2},
            ),
        ),
    )

    buffers = cuda_renderer_scene_buffers(scene)
    payload = buffers.to_dict()

    assert payload["format"] == "AURA_CUDA_RENDERER_SCENE_BUFFERS"
    assert buffers.element_ids == ("panel", "residual")
    assert buffers.carrier_ids == ("surface", "neural")
    assert buffers.carrier_kernel_ids == (0, 4)
    assert buffers.material_id_table == ("enamel",)
    assert buffers.semantic_id_table == ("tooth", "highlight")
    assert buffers.material_ids == (0, 0)
    assert buffers.semantic_ids == (0, 1)
    assert buffers.element_mins == pytest.approx((-1.0, -0.5, 0.0, 0.0, -0.5, 0.1))
    assert buffers.element_maxs == pytest.approx((0.0, 0.5, 0.2, 1.0, 0.5, 0.3))
    assert buffers.payload_params == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0))
    assert payload["colors"]["shape"] == [2, 3]
    assert payload["opacities"]["dtype"] == "float32"
    assert payload["payloadParams"]["shape"] == [2, 4]


def test_cuda_renderer_kernel_inputs_pack_rays_and_match_cpu_first_hits():
    scene = AuraScene(
        name="cuda_kernel_input_test",
        elements=(
            AuraElement(
                id="left",
                carrier_id="surface",
                bounds=Bounds((-1.0, -0.5, 0.0), (-0.1, 0.5, 0.2)),
                color=(1.0, 0.0, 0.0),
                opacity=0.8,
                confidence=0.9,
                material_id="matte",
                semantic_id="left_object",
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="right",
                carrier_id="semantic",
                bounds=Bounds((0.1, -0.5, 0.0), (1.0, 0.5, 0.2)),
                color=(0.0, 0.0, 1.0),
                opacity=0.6,
                confidence=0.7,
                semantic_id="right_object",
                payload={"type": "semantic_feature", "label": "right_object"},
            ),
        ),
    )
    ray_origins = ((-0.5, 0.0, -1.0), (0.5, 0.0, -1.0), (2.0, 0.0, -1.0))
    ray_directions = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))

    inputs = cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=3)
    payload = inputs.to_dict()
    kernel_args = inputs.to_kernel_args()
    rays = tuple(Ray(origin=origin, direction=direction) for origin, direction in zip(ray_origins, ray_directions))

    assert payload["format"] == "AURA_CUDA_RENDERER_KERNEL_INPUT_BUFFERS"
    assert payload["kernelSymbol"] == "aura_render_rays_kernel"
    assert inputs.ray_count == 3
    assert inputs.element_count == 2
    assert inputs.output_buffer_shapes()["ordered_hits"] == (3, 3)
    assert kernel_args["ray_count"] == 3
    assert kernel_args["element_count"] == 2
    assert kernel_args["max_hits"] == 3
    assert kernel_args["ray_origins"] == pytest.approx((-0.5, 0.0, -1.0, 0.5, 0.0, -1.0, 2.0, 0.0, -1.0))
    assert kernel_args["carrier_ids"] == (0, 5)
    assert kernel_args["payload_params"] == pytest.approx((0.0,) * 8)
    assert kernel_args["material_ids"] == (0, -1)
    assert kernel_args["semantic_ids"] == (0, 1)
    assert cuda_renderer_reference_first_hit_indices(scene, rays) == (0, 1, -1)


def test_cuda_renderer_kernel_simulation_matches_flat_abi_ordered_compositing_outputs():
    scene = AuraScene(
        name="cuda_kernel_simulation_test",
        elements=(
            AuraElement(
                id="front_surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                color=(0.9, 0.2, 0.1),
                opacity=0.75,
                confidence=0.8,
                material_id="paint",
                semantic_id="front",
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="back_residual",
                carrier_id="neural",
                bounds=Bounds((-0.5, -0.5, 0.3), (0.5, 0.5, 0.5)),
                color=(0.1, 0.2, 0.9),
                opacity=0.5,
                confidence=0.6,
                semantic_id="back",
                payload={"type": "neural_residual", "residual_scale": 0.2},
            ),
        ),
    )
    ray_origins = ((0.0, 0.0, -1.0), (0.0, 0.0, 0.25), (2.0, 0.0, -1.0))
    ray_directions = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    inputs = cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=2)

    simulation = simulate_cuda_renderer_kernel(inputs)
    payload = simulation.to_dict()

    assert payload["format"] == "AURA_CUDA_RENDERER_KERNEL_SIMULATION"
    assert payload["productionReady"] is False
    assert simulation.first_hit_indices == (0, 1, -1)
    assert simulation.ordered_hits == (0, 1, 1, -1, -1, -1)
    assert simulation.out_color == pytest.approx((0.6875, 0.175, 0.1875, 0.05, 0.1, 0.45, 0.0, 0.0, 0.0))
    assert simulation.out_alpha == pytest.approx((0.875, 0.5, 0.0))
    assert simulation.out_transmittance == pytest.approx((0.125, 0.5, 1.0))
    assert simulation.out_depth[0] == pytest.approx(1.0)
    assert simulation.out_depth[1] == pytest.approx(0.05)
    assert simulation.out_depth[2] > 1.0e30
    assert simulation.out_normal == pytest.approx((0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
    assert simulation.out_confidence == pytest.approx((0.7671428571, 0.57, 0.0))
    assert simulation.out_residual == (1, 1, 0)
    assert simulation.out_material_id == (0, -1, -1)
    assert simulation.out_semantic_id == (0, 1, -1)
    assert payload["orderedHits"]["shape"] == [3, 2]


def test_cuda_renderer_kernel_simulation_uses_payload_specific_carrier_responses():
    scene = AuraScene(
        name="cuda_payload_simulation_test",
        elements=(
            AuraElement(
                id="fog",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5)),
                color=(0.2, 0.4, 0.8),
                opacity=0.9,
                confidence=0.7,
                payload={"type": "volume_cell", "density": 2.0},
            ),
            AuraElement(
                id="detail",
                carrier_id="beta",
                bounds=Bounds((-0.5, -0.5, 0.6), (0.5, 0.5, 1.6)),
                color=(0.8, 0.1, 0.1),
                opacity=0.5,
                confidence=0.6,
                payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0},
            ),
        ),
    )
    inputs = cuda_renderer_kernel_inputs(
        scene,
        ((0.0, 0.0, -1.0),),
        ((0.0, 0.0, 1.0),),
        max_hits=2,
    )

    simulation = simulate_cuda_renderer_kernel(inputs)
    volume_transmittance = __import__("math").exp(-2.0 * 0.5)
    beta_transmittance = 1.0 - 0.5 * (8.0 / 9.0)

    assert inputs.scene.payload_params == pytest.approx((2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0))
    assert simulation.ordered_hits == (0, 1)
    assert simulation.out_transmittance[0] == pytest.approx(volume_transmittance * beta_transmittance)
    assert simulation.out_alpha[0] == pytest.approx(1.0 - volume_transmittance * beta_transmittance)
    assert simulation.out_color[0] > 0.0


def test_cuda_renderer_kernel_simulation_uses_beta_support_ellipsoid():
    scene = AuraScene(
        name="cuda_beta_support_simulation_test",
        elements=(
            AuraElement(
                id="detail",
                carrier_id="beta",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "beta_kernel",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "support_radius": [0.5, 0.5, 0.25],
                },
            ),
        ),
    )
    inputs = cuda_renderer_kernel_inputs(
        scene,
        ((0.0, 0.0, -1.0), (0.75, 0.0, -1.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        max_hits=1,
    )

    simulation = simulate_cuda_renderer_kernel(inputs)

    assert inputs.scene.beta_support_radii == pytest.approx((0.5, 0.5, 0.25))
    assert simulation.first_hit_indices == (0, -1)
    assert simulation.out_depth[0] == pytest.approx(1.75)
    assert simulation.out_depth[1] > 1.0e30


def test_cuda_renderer_kernel_simulation_uses_gaussian_ellipsoid_support():
    scene = AuraScene(
        name="cuda_gaussian_support_simulation_test",
        elements=(
            AuraElement(
                id="fallback",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 1.0],
                    "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                    "support_sigma": 1.0,
                },
            ),
        ),
    )
    inputs = cuda_renderer_kernel_inputs(
        scene,
        ((0.0, 0.0, -1.0), (0.75, 0.0, -1.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        max_hits=1,
    )

    simulation = simulate_cuda_renderer_kernel(inputs)

    assert inputs.scene.gaussian_means == pytest.approx((0.0, 0.0, 1.0))
    assert inputs.scene.gaussian_support_radius_sq == pytest.approx((1.0,))
    assert simulation.first_hit_indices == (0, -1)
    assert simulation.out_depth[0] == pytest.approx(1.5)
    assert simulation.out_depth[1] > 1.0e30


def test_cuda_renderer_kernel_simulation_uses_surface_plane_geometry():
    scene = AuraScene(
        name="cuda_surface_plane_simulation_test",
        elements=(
            AuraElement(
                id="panel",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.4)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                normal=(0.0, 0.0, 1.0),
                payload={"type": "surface_cell", "plane_point": [0.0, 0.0, 0.25]},
            ),
        ),
    )
    inputs = cuda_renderer_kernel_inputs(
        scene,
        ((0.0, 0.0, -1.0),),
        ((0.0, 0.0, 1.0),),
        max_hits=1,
    )

    simulation = simulate_cuda_renderer_kernel(inputs)

    assert inputs.scene.plane_points == pytest.approx((0.0, 0.0, 0.25))
    assert inputs.scene.plane_normals == pytest.approx((0.0, 0.0, 1.0))
    assert simulation.first_hit_indices == (0,)
    assert simulation.out_depth == pytest.approx((1.25,))
    assert simulation.out_normal == pytest.approx((0.0, 0.0, 1.0))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_compiled_extension_matches_kernel_simulation_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    scene = AuraScene(
        name="cuda_compiled_parity_scene",
        elements=(
            AuraElement(
                id="front_surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                color=(0.9, 0.2, 0.1),
                opacity=0.75,
                confidence=0.8,
                material_id="paint",
                semantic_id="front",
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="back_residual",
                carrier_id="neural",
                bounds=Bounds((-0.5, -0.5, 0.3), (0.5, 0.5, 0.5)),
                color=(0.1, 0.2, 0.9),
                opacity=0.5,
                confidence=0.6,
                semantic_id="back",
                payload={"type": "neural_residual", "residual_scale": 0.2},
            ),
        ),
    )
    ray_origins = ((0.0, 0.0, -1.0), (0.0, 0.0, 0.25), (2.0, 0.0, -1.0))
    ray_directions = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    inputs = cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=2)
    simulation = simulate_cuda_renderer_kernel(inputs)

    batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=2,
    )

    assert batch.backend == "cuda"
    assert batch.reason == "compiled_cuda_renderer_python_binding"
    assert batch.to_dict()["available"] is True
    assert _flatten_nested(batch.color) == pytest.approx(simulation.out_color)
    assert batch.opacity == pytest.approx(simulation.out_alpha)
    assert batch.transmittance == pytest.approx(simulation.out_transmittance)
    assert tuple(3.402823466e38 if depth is None else depth for depth in batch.depth) == pytest.approx(simulation.out_depth)
    assert _flatten_nested(tuple((0.0, 0.0, 0.0) if normal is None else normal for normal in batch.normal)) == pytest.approx(
        simulation.out_normal
    )
    assert batch.confidence == pytest.approx(simulation.out_confidence)
    assert tuple(int(value) for value in batch.residual) == simulation.out_residual
    assert tuple(hit[0]["kernelElementIndex"] if hit else -1 for hit in batch.ordered_hits) == simulation.first_hit_indices


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_compiled_extension_uses_beta_support_ellipsoid_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    scene = AuraScene(
        name="cuda_compiled_beta_support_scene",
        elements=(
            AuraElement(
                id="detail",
                carrier_id="beta",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "beta_kernel",
                    "alpha": 2.0,
                    "beta": 2.0,
                    "support_radius": [0.5, 0.5, 0.25],
                },
            ),
        ),
    )
    ray_origins = ((0.0, 0.0, -1.0), (0.75, 0.0, -1.0))
    ray_directions = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    simulation = simulate_cuda_renderer_kernel(cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=1))

    batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=1,
    )

    assert batch.backend == "cuda"
    assert batch.depth[0] == pytest.approx(1.75)
    assert batch.depth[1] is None
    assert tuple(3.402823466e38 if depth is None else depth for depth in batch.depth) == pytest.approx(simulation.out_depth)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_compiled_extension_uses_gaussian_ellipsoid_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    scene = AuraScene(
        name="cuda_compiled_gaussian_support_scene",
        elements=(
            AuraElement(
                id="fallback",
                carrier_id="gaussian",
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.0, 0.0, 1.0],
                    "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                    "support_sigma": 1.0,
                },
            ),
        ),
    )
    ray_origins = ((0.0, 0.0, -1.0), (0.75, 0.0, -1.0))
    ray_directions = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    simulation = simulate_cuda_renderer_kernel(cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=1))

    batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=1,
    )

    assert batch.backend == "cuda"
    assert batch.depth[0] == pytest.approx(1.5)
    assert batch.depth[1] is None
    assert tuple(3.402823466e38 if depth is None else depth for depth in batch.depth) == pytest.approx(simulation.out_depth)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_compiled_extension_uses_surface_plane_geometry_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    scene = AuraScene(
        name="cuda_compiled_surface_plane_scene",
        elements=(
            AuraElement(
                id="panel",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.4)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                normal=(0.0, 0.0, 1.0),
                payload={"type": "surface_cell", "plane_point": [0.0, 0.0, 0.25]},
            ),
        ),
    )
    ray_origins = ((0.0, 0.0, -1.0),)
    ray_directions = ((0.0, 0.0, 1.0),)
    simulation = simulate_cuda_renderer_kernel(cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=1))

    batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=1,
    )

    assert batch.backend == "cuda"
    assert batch.depth == pytest.approx((1.25,))
    assert batch.depth == pytest.approx(simulation.out_depth)
    assert _flatten_nested(batch.normal) == pytest.approx(simulation.out_normal)


def test_cuda_renderer_dispatch_contract_tracks_compiled_launcher_boundary():
    scene = AuraScene(
        name="cuda_dispatch_contract_test",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                color=(0.9, 0.2, 0.1),
                opacity=0.75,
                confidence=0.8,
                material_id="paint",
                semantic_id="front",
                payload={"type": "surface_cell"},
            ),
        ),
    )

    contract = cuda_renderer_dispatch_contract(
        scene,
        ray_origins=((0.0, 0.0, -1.0),),
        ray_directions=((0.0, 0.0, 1.0),),
        threads_per_block=64,
        max_hits=2,
        fallback_backend="cpu",
        extension=_unavailable_extension_status(),
    )
    payload = contract.to_dict()

    assert payload["format"] == "AURA_CUDA_RENDERER_DISPATCH_CONTRACT"
    assert payload["kernelSymbol"] == "aura_render_rays_kernel"
    assert payload["launcherSymbol"] == "aura_render_rays_launcher"
    assert payload["productionReady"] is False
    assert payload["dispatchReady"] is False
    assert payload["compiledExtensionAvailable"] is False
    assert payload["rendererSymbolsReady"] is False
    assert payload["pythonBindingAvailable"] is False
    assert payload["symbolProbe"]["format"] == "AURA_CUDA_RENDERER_SYMBOL_PROBE"
    assert payload["symbolProbe"]["dispatchSymbolsReady"] is False
    assert payload["symbolProbe"]["reason"] == "extension_unavailable: build_not_attempted"
    assert payload["reason"] == "compiled_cuda_renderer_extension_unavailable: build_not_attempted"
    assert payload["launchConfig"]["threadsPerBlock"] == 64
    assert payload["launchConfig"]["blockCount"] == 1
    assert payload["kernelArgs"]["ray_count"] == 1
    assert payload["kernelArgs"]["element_count"] == 1
    assert payload["outputBufferShapes"]["ordered_hits"] == [1, 2]
    assert "validate render_rays Python tensor dispatch on CUDA hardware" in payload["missingDispatchWork"]


def test_cuda_renderer_symbol_probe_distinguishes_loaded_symbol_states():
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )

    unavailable_module = cuda_renderer_symbol_probe(extension)
    missing_launcher = cuda_renderer_symbol_probe(
        extension,
        extension_module=type("FakeCudaModule", (), {"aura_render_rays_kernel": object(), "render_rays": object()})(),
    )
    ready_symbols = cuda_renderer_symbol_probe(
        extension,
        extension_module=type(
            "FakeCudaModule",
            (),
            {
                "aura_render_rays_kernel": object(),
                "aura_render_rays_launcher": object(),
                "render_rays": object(),
            },
        )(),
    )

    assert unavailable_module.dispatch_symbols_ready is False
    assert unavailable_module.to_dict()["reason"] == "extension_module_object_unavailable"
    assert missing_launcher.dispatch_symbols_ready is False
    assert missing_launcher.to_dict()["kernelSymbolAvailable"] is True
    assert missing_launcher.to_dict()["launcherSymbolAvailable"] is False
    assert missing_launcher.to_dict()["bindingSymbolAvailable"] is True
    assert missing_launcher.to_dict()["reason"] == "missing_symbols: aura_render_rays_launcher"
    assert ready_symbols.dispatch_symbols_ready is True
    assert ready_symbols.to_dict()["dispatchSymbolsReady"] is True
    assert ready_symbols.to_dict()["reason"] is None


def test_cuda_renderer_dispatch_contract_keeps_gate_closed_after_symbol_verification():
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    extension_module = type(
        "FakeCudaModule",
        (),
        {
            "aura_render_rays_kernel": object(),
            "aura_render_rays_launcher": object(),
            "render_rays": object(),
        },
    )()
    scene = AuraScene(
        name="cuda_symbol_verified_contract",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    contract = cuda_renderer_dispatch_contract(
        scene,
        ray_origins=((0.0, 0.0, -1.0),),
        ray_directions=((0.0, 0.0, 1.0),),
        extension=extension,
        extension_module=extension_module,
    )
    payload = contract.to_dict()

    assert payload["compiledExtensionAvailable"] is True
    assert payload["rendererSymbolsReady"] is True
    assert payload["pythonBindingAvailable"] is False
    assert payload["dispatchReady"] is False
    assert payload["productionReady"] is False
    assert payload["reason"] == "python_cuda_renderer_binding_missing"
    assert "validate render_rays Python tensor dispatch on CUDA hardware" in payload["missingDispatchWork"]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_uses_verified_python_binding_module():
    import torch

    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )

    class FakeCudaModule:
        aura_render_rays_kernel = object()
        aura_render_rays_launcher = object()

        @staticmethod
        def render_rays(
            ray_origins,
            ray_directions,
            element_mins,
            element_maxs,
            plane_points,
            plane_normals,
            beta_support_radii,
            gaussian_means,
            gaussian_inverse_covariances,
            gaussian_support_radius_sq,
            carrier_ids,
            colors,
            opacities,
            confidences,
            payload_params,
            material_ids,
            semantic_ids,
            max_hits,
            threads_per_block,
        ):
            del (
                ray_directions,
                element_mins,
                element_maxs,
                plane_points,
                plane_normals,
                beta_support_radii,
                gaussian_means,
                gaussian_inverse_covariances,
                gaussian_support_radius_sq,
                carrier_ids,
                payload_params,
                threads_per_block,
            )
            ray_count = int(ray_origins.shape[0])
            ordered_hits = torch.full((ray_count, max_hits), -1, dtype=torch.int32)
            ordered_hits[0, 0] = 0
            return {
                "out_color": colors[:1].repeat(ray_count, 1),
                "out_alpha": opacities[:1].repeat(ray_count),
                "out_transmittance": 1.0 - opacities[:1].repeat(ray_count),
                "out_depth": torch.tensor([1.0, 3.402823466e38], dtype=torch.float32),
                "out_normal": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
                "out_confidence": confidences[:1].repeat(ray_count),
                "out_residual": torch.tensor([0, 0], dtype=torch.uint8),
                "out_material_id": torch.tensor([int(material_ids[0].item()), -1], dtype=torch.int32),
                "out_semantic_id": torch.tensor([int(semantic_ids[0].item()), -1], dtype=torch.int32),
                "ordered_hits": ordered_hits,
            }

    scene = AuraScene(
        name="cuda_fake_binding_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                color=(0.9, 0.2, 0.1),
                opacity=0.75,
                confidence=0.8,
                material_id="paint",
                semantic_id="front",
                payload={"type": "surface_cell"},
            ),
        ),
    )

    batch = cuda_render_rays(
        scene,
        ray_origins=((0.0, 0.0, -1.0), (2.0, 0.0, -1.0)),
        ray_directions=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        extension=extension,
        extension_module=FakeCudaModule(),
        device="cuda",
        max_hits=2,
    )

    assert batch.backend == "cuda"
    assert batch.reason == "compiled_cuda_renderer_python_binding"
    assert batch.element_ids == ("surface", None)
    assert batch.carrier_ids == ("surface", None)
    assert batch.depth == (1.0, None)
    assert batch.material_ids == ("paint", None)
    assert batch.semantic_ids == ("front", None)
    assert batch.ordered_hits[0][0]["elementId"] == "surface"
    assert batch.to_dict()["available"] is True
    assert batch.to_dict()["productionReady"] is False


def _flatten_nested(values):
    return tuple(item for row in values for item in row)


def _unavailable_extension_status():
    return CudaExtensionStatus(
        available=False,
        build_attempted=False,
        compiled=False,
        loadable=False,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
        reason="build_not_attempted",
    )


def test_cuda_renderer_scene_buffers_reject_unknown_carrier_for_kernel_abi():
    scene = AuraScene(
        name="unsupported_cuda_carrier",
        elements=(
            AuraElement(
                id="custom",
                carrier_id="custom",
                bounds=Bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                payload={"type": "custom"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="unsupported CUDA renderer carrier id"):
        cuda_renderer_scene_buffers(scene)


def test_cuda_render_rays_cpu_fallback_matches_aura_ray_query_contract():
    scene = AuraScene(
        name="cuda_cpu_fallback_scene",
        elements=(
            AuraElement(
                id="fog",
                carrier_id="volume",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.4)),
                color=(0.1, 0.3, 0.8),
                opacity=0.5,
                confidence=0.6,
                payload={"type": "volume_cell", "density": 0.25},
            ),
            AuraElement(
                id="panel",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.2), (0.5, 0.5, 0.3)),
                color=(0.9, 0.2, 0.1),
                opacity=0.75,
                confidence=0.95,
                normal=(0.0, 0.0, -1.0),
                material_id="enamel",
                payload={"type": "surface_cell"},
            ),
        ),
    )

    batch = cuda_render_rays(
        scene,
        ray_origins=((0.0, 0.0, -1.0), (2.0, 0.0, -1.0)),
        ray_directions=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        fallback_backend="cpu",
        threads_per_block=64,
        max_hits=1,
        extension=_unavailable_extension_status(),
    )
    payload = batch.to_dict()
    expected_hit = scene.traverse_ray(Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))
    expected_miss = scene.traverse_ray(Ray(origin=(2.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)))

    assert payload["format"] == "AURA_CUDA_RENDERER_BATCH"
    assert payload["productionReady"] is False
    assert payload["available"] is False
    assert payload["backend"] == "cpu"
    assert payload["reason"] == "cuda_extension_unavailable_cpu_fallback"
    assert payload["launchConfig"]["rayCount"] == 2
    assert payload["launchConfig"]["threadsPerBlock"] == 64
    assert payload["launchConfig"]["blockCount"] == 1
    assert payload["extension"]["buildAttempted"] is False
    assert payload["extension"]["reason"] == "build_not_attempted"

    assert batch.color[0] == pytest.approx(expected_hit.result.color)
    assert batch.transmittance[0] == pytest.approx(expected_hit.result.transmittance)
    assert batch.opacity[0] == pytest.approx(expected_hit.result.opacity)
    assert batch.depth[0] == pytest.approx(expected_hit.result.depth)
    assert batch.normal[0] == expected_hit.result.normal
    assert batch.confidence[0] == pytest.approx(expected_hit.result.confidence)
    assert batch.material_ids[0] == expected_hit.result.material_id
    assert batch.semantic_ids[0] == expected_hit.result.semantic_id
    assert batch.residual[0] is expected_hit.result.residual
    assert batch.provenance[0] == expected_hit.result.provenance
    assert batch.element_ids[0] == expected_hit.ordered_hits[0].element_id
    assert batch.carrier_ids[0] == expected_hit.ordered_hits[0].carrier_id
    assert batch.ordered_hits[0][0]["elementId"] == expected_hit.ordered_hits[0].element_id
    assert batch.ordered_hit_overflow[0] is True

    assert batch.color[1] == pytest.approx(expected_miss.result.color)
    assert batch.transmittance[1] == pytest.approx(1.0)
    assert batch.depth[1] is None
    assert batch.element_ids[1] is None
    assert batch.carrier_ids[1] is None
    assert batch.provenance[1] == "miss"
    assert batch.ordered_hits[1] == ()
    assert batch.ordered_hit_overflow[1] is False


def test_cuda_renderer_boundary_report_distinguishes_callable_fallback_from_production_cuda():
    scene = AuraScene(
        name="cuda_boundary_report_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.8, 0.1, 0.2),
                opacity=0.7,
                confidence=0.9,
                payload={"type": "surface_cell"},
            ),
        ),
    )

    report = cuda_renderer_boundary_report(scene, fallback_backend="cpu", max_hits=4)

    assert report["format"] == "AURA_CUDA_RENDERER_BOUNDARY_REPORT"
    assert report["apiName"] == "aura.cuda_renderer.cuda_render_rays"
    assert report["callableBoundaryAvailable"] is True
    assert report["available"] is False
    assert report["productionReady"] is False
    assert report["rendererSource"]["format"] == "AURA_CUDA_RENDERER_SOURCE_REPORT"
    assert report["rendererSource"]["symbol"] == "aura_render_rays_kernel"
    assert report["rendererSource"]["sourceSymbolAvailable"] is True
    assert report["rendererSource"]["productionReady"] is False
    assert report["symbolProbe"]["format"] == "AURA_CUDA_RENDERER_SYMBOL_PROBE"
    assert report["symbolProbe"]["dispatchSymbolsReady"] is False
    assert report["fallbackProbe"]["executed"] is True
    assert report["fallbackProbe"]["backend"] == "cpu"
    assert report["fallbackProbe"]["rayCount"] == 1
    assert report["fallbackProbe"]["maxHits"] == 4
    assert report["kernelInputProbe"]["kernelSymbol"] == "aura_render_rays_kernel"
    assert report["kernelInputProbe"]["rayCount"] == 1
    assert report["kernelInputProbe"]["elementCount"] == 1
    assert report["kernelInputProbe"]["outputBufferShapes"]["ordered_hits"] == [1, 4]
    assert report["dispatchContractProbe"]["format"] == "AURA_CUDA_RENDERER_DISPATCH_CONTRACT"
    assert report["dispatchContractProbe"]["launcherSymbol"] == "aura_render_rays_launcher"
    assert report["dispatchContractProbe"]["dispatchReady"] is False
    assert report["dispatchContractProbe"]["rendererSymbolsReady"] is False
    assert report["dispatchContractProbe"]["pythonBindingAvailable"] is False
    assert report["dispatchContractProbe"]["symbolProbe"]["reason"] == "extension_unavailable: build_not_attempted"
    assert set(report["fallbackProbe"]["outputFields"]).issuperset(
        {"color", "transmittance", "depth", "normal", "confidence", "orderedHits"}
    )
    assert "compiled_cuda_renderer_dispatch_missing" in report["productionBlockers"]
    assert "not production CUDA acceleration" in report["notes"]


def test_cuda_renderer_boundary_report_without_scene_is_metadata_only():
    report = cuda_renderer_boundary_report()

    assert report["callableBoundaryAvailable"] is True
    assert report["productionReady"] is False
    assert report["fallbackProbe"] is None
    assert report["kernelInputProbe"] is None
    assert report["dispatchContractProbe"] is None


def test_cuda_render_rays_rejects_invalid_ray_batches_before_fallback():
    scene = AuraScene(name="invalid_cuda_batch_scene", elements=())

    with pytest.raises(ValueError, match="does not match"):
        cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        )

    with pytest.raises(ValueError, match="ray_directions must contain 3D ray vectors"):
        cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0),),
        )


def test_cuda_render_rays_refuses_to_fallback_when_cuda_is_required():
    scene = AuraScene(name="require_cuda_scene", elements=())

    try:
        batch = cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0, 1.0),),
            require_cuda=True,
        )
    except RuntimeError as exc:
        assert "CUDA renderer extension is unavailable" in str(exc) or "CUDA renderer Python dispatch is unavailable" in str(exc)
        return

    assert batch.backend == "cuda"
    assert batch.device == "cuda"
    assert batch.reason == "compiled_cuda_renderer_python_binding"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_torch_fallback_matches_aura_ray_query_contract():
    scene = AuraScene(
        name="cuda_torch_fallback_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(0.8, 0.1, 0.2),
                opacity=0.7,
                confidence=0.9,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    ray = Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0))

    batch = cuda_render_rays(
        scene,
        ray_origins=(ray.origin,),
        ray_directions=(ray.direction,),
        fallback_backend="torch",
        device="cpu",
    )
    expected = scene.traverse_ray(ray)

    assert batch.backend == "torch"
    assert batch.device == "cpu"
    assert batch.reason in {"cuda_extension_unavailable_torch_fallback", "explicit_torch_fallback"}
    assert batch.color[0] == pytest.approx(expected.result.color)
    assert batch.transmittance[0] == pytest.approx(expected.result.transmittance)
    assert batch.depth[0] == pytest.approx(expected.result.depth)
    assert batch.normal[0] == expected.result.normal
    assert batch.element_ids[0] == expected.ordered_hits[0].element_id
    assert batch.ordered_hits[0][0]["elementId"] == expected.ordered_hits[0].element_id
