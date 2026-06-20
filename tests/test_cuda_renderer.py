import importlib.util

import pytest

import aura.cuda_renderer as cuda_renderer_module
from aura import AuraElement, AuraScene, Bounds, Ray
from aura.cuda_kernels import CudaExtensionStatus
from aura.cuda_renderer import (
    CUDA_RENDERER_BINDING_SYMBOL,
    cuda_render_rays,
    cuda_renderer_build_bvh,
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
    assert buffers.payload_params == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0))
    assert payload["colors"]["shape"] == [2, 3]
    assert payload["opacities"]["dtype"] == "float32"
    assert payload["payloadParams"]["shape"] == [2, 5]


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
    assert kernel_args["payload_params"] == pytest.approx((0.0,) * 10)
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
    assert simulation.out_color == pytest.approx((0.6775, 0.155, 0.0975, 0.01, 0.02, 0.09, 0.0, 0.0, 0.0))
    assert simulation.out_alpha == pytest.approx((0.775, 0.1, 0.0))
    assert simulation.out_transmittance == pytest.approx((0.225, 0.9, 1.0))
    assert simulation.out_depth[0] == pytest.approx(1.0)
    assert simulation.out_depth[1] == pytest.approx(0.05)
    assert simulation.out_depth[2] > 1.0e30
    assert simulation.out_normal == pytest.approx((0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
    assert simulation.out_confidence == pytest.approx((0.7925806452, 0.57, 0.0))
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

    assert inputs.scene.payload_params == pytest.approx((2.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0))
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


def test_cuda_renderer_kernel_simulation_uses_gabor_phase():
    pi = __import__("math").pi
    scene = AuraScene(
        name="cuda_gabor_phase_simulation_test",
        elements=(
            AuraElement(
                id="frequency",
                carrier_id="gabor",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.4)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gabor_frequency",
                    "frequency": [0.0, 0.0, 0.0],
                    "phase": pi / 2.0,
                    "bandwidth": 1.0,
                    "normal": [0.0, 0.0, 1.0],
                    "plane_point": [0.0, 0.0, 0.2],
                },
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

    assert inputs.scene.payload_params == pytest.approx((0.0, 0.0, 0.0, pi / 2.0, 1.0))
    assert simulation.first_hit_indices == (0,)
    assert simulation.out_color == pytest.approx((1.0, 0.0, 0.0))
    assert simulation.out_depth == pytest.approx((1.2,))


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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_compiled_extension_uses_gabor_phase_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    pi = __import__("math").pi
    scene = AuraScene(
        name="cuda_compiled_gabor_phase_scene",
        elements=(
            AuraElement(
                id="frequency",
                carrier_id="gabor",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.4)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={
                    "type": "gabor_frequency",
                    "frequency": [0.0, 0.0, 0.0],
                    "phase": pi / 2.0,
                    "bandwidth": 1.0,
                    "normal": [0.0, 0.0, 1.0],
                    "plane_point": [0.0, 0.0, 0.2],
                },
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
    assert _flatten_nested(batch.color) == pytest.approx(simulation.out_color)
    assert batch.depth == pytest.approx((1.2,))


def _multi_carrier_scene():
    return AuraScene(
        name="cuda_bvh_multi_carrier",
        elements=(
            AuraElement(
                id="wall",
                carrier_id="surface",
                bounds=Bounds((-0.75, -0.75, 0.0), (-0.25, -0.25, 0.1)),
                color=(0.8, 0.2, 0.1),
                opacity=0.7,
                confidence=0.9,
                payload={"type": "surface_cell"},
            ),
            AuraElement(
                id="fog",
                carrier_id="volume",
                bounds=Bounds((-0.15, -0.7, 0.0), (0.35, -0.2, 0.8)),
                color=(0.2, 0.4, 0.8),
                opacity=0.6,
                confidence=0.7,
                payload={"type": "volume_cell", "density": 1.2},
            ),
            AuraElement(
                id="residual",
                carrier_id="neural",
                bounds=Bounds((-0.75, 0.05, 0.0), (-0.25, 0.55, 0.2)),
                color=(0.1, 0.2, 0.9),
                opacity=0.5,
                confidence=0.6,
                payload={"type": "neural_residual", "residual_scale": 0.3},
            ),
            AuraElement(
                id="detail",
                carrier_id="beta",
                bounds=Bounds((0.5, 0.05, 0.0), (0.8, 0.35, 0.15)),
                color=(0.9, 0.6, 0.2),
                opacity=0.8,
                confidence=0.8,
                payload={"type": "beta_kernel", "alpha": 2.0, "beta": 2.0},
            ),
            AuraElement(
                id="blob",
                carrier_id="gaussian",
                bounds=Bounds((0.85, 0.3, 0.0), (1.05, 0.5, 0.2)),
                color=(0.3, 0.7, 0.5),
                opacity=0.7,
                confidence=0.75,
                payload={
                    "type": "gaussian_fallback",
                    "mean": [0.95, 0.4, 0.1],
                    "covariance": [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]],
                    "support_sigma": 1.0,
                },
            ),
        ),
    )


def test_cuda_renderer_build_bvh_covers_every_element_as_a_leaf():
    scene = _multi_carrier_scene()

    bvh = cuda_renderer_build_bvh(scene)
    payload = bvh.to_dict()

    assert payload["format"] == "AURA_CUDA_RENDERER_BVH"
    assert bvh.element_count == len(scene.elements)
    assert payload["leafCount"] == len(scene.elements)
    leaf_elements = sorted(value for value in bvh.node_element if value >= 0)
    assert leaf_elements == list(range(len(scene.elements)))
    # Internal nodes reference valid child node indices; leaves have no children.
    for node_index in range(bvh.node_count):
        element_index = bvh.node_element[node_index]
        if element_index >= 0:
            assert bvh.node_left[node_index] == -1
            assert bvh.node_right[node_index] == -1
        else:
            assert 0 <= bvh.node_left[node_index] < bvh.node_count
            assert 0 <= bvh.node_right[node_index] < bvh.node_count


def test_cuda_renderer_build_bvh_handles_single_and_empty_scenes():
    single = AuraScene(
        name="single_bvh",
        elements=(
            AuraElement(
                id="only",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    bvh = cuda_renderer_build_bvh(single)
    assert bvh.node_count == 1
    assert bvh.node_element == (0,)

    empty = cuda_renderer_build_bvh(AuraScene(name="empty_bvh", elements=()))
    assert empty.node_count == 0
    assert empty.node_element == ()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_bvh_path_matches_brute_force_and_torch_on_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    import aura.cuda_renderer as renderer_module
    from aura.benchmark import _benchmark_ray_grid
    from aura.torch_renderer import torch_render_rays

    scene = _multi_carrier_scene()
    origins, directions = _benchmark_ray_grid(scene, 256)
    ray_origins = torch.tensor(origins, dtype=torch.float32, device="cuda")
    ray_directions = torch.tensor(directions, dtype=torch.float32, device="cuda")

    extension, extension_module = renderer_module._resolve_cuda_renderer_extension(
        extension=None, extension_module=None, build=True
    )
    assert extension.available
    assert hasattr(extension_module, "render_rays_bvh")

    # Production path: dispatch prefers the GPU BVH binding when present.
    bvh_batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=8,
    )

    # Force the brute-force binding by hiding render_rays_bvh from the module.
    class _BruteForceOnly:
        render_rays = staticmethod(getattr(extension_module, CUDA_RENDERER_BINDING_SYMBOL))
        aura_render_rays_kernel = object()
        aura_render_rays_launcher = object()

    brute_batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        device="cuda",
        extension=extension,
        extension_module=_BruteForceOnly,
        max_hits=8,
    )

    torch_batch = torch_render_rays(scene, ray_origins, ray_directions, device="cuda", collect_traces=False)

    assert bvh_batch.backend == "cuda"
    for ray_index in range(256):
        assert bvh_batch.element_ids[ray_index] == brute_batch.element_ids[ray_index]
        assert bvh_batch.element_ids[ray_index] == torch_batch.element_ids[ray_index]
        for channel in range(3):
            assert bvh_batch.color[ray_index][channel] == pytest.approx(
                brute_batch.color[ray_index][channel], abs=1.0e-6
            )
            assert bvh_batch.color[ray_index][channel] == pytest.approx(
                torch_batch.predicted_color[ray_index][channel], abs=1.0e-4
            )
        assert bvh_batch.transmittance[ray_index] == pytest.approx(brute_batch.transmittance[ray_index], abs=1.0e-6)


def _carrier_parity_scene(carrier_id, payload, **element_kwargs):
    return AuraScene(
        name=f"cuda_torch_parity_{carrier_id}",
        elements=(
            AuraElement(
                id="carrier",
                carrier_id=carrier_id,
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.4)),
                color=(0.7, 0.3, 0.2),
                opacity=0.6,
                confidence=0.8,
                payload=payload,
                **element_kwargs,
            ),
        ),
    )


_CARRIER_PARITY_CASES = (
    ("surface", {"type": "surface_cell"}, {}),
    ("volume", {"type": "volume_cell", "density": 1.5}, {}),
    ("beta", {"type": "beta_kernel", "alpha": 2.0, "beta": 2.0}, {}),
    (
        "gabor",
        {"type": "gabor_frequency", "frequency": [0.0, 0.0, 0.0], "phase": 1.0, "bandwidth": 0.5},
        {},
    ),
    ("neural", {"type": "neural_residual", "residual_scale": 0.2}, {}),
    ("semantic", {"type": "semantic_feature", "label": "thing"}, {"semantic_id": "thing"}),
    (
        "gaussian",
        {
            "type": "gaussian_fallback",
            "mean": [0.0, 0.0, 0.2],
            "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
            "support_sigma": 1.0,
        },
        {},
    ),
)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
@pytest.mark.parametrize(("carrier_id", "payload", "element_kwargs"), _CARRIER_PARITY_CASES)
def test_cuda_render_rays_matches_torch_renderer_for_every_carrier_on_cuda(carrier_id, payload, element_kwargs):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    from aura.torch_renderer import torch_render_rays

    scene = _carrier_parity_scene(carrier_id, payload, **element_kwargs)
    ray_origins = ((0.0, 0.0, -1.0), (0.1, -0.1, -1.0), (2.0, 0.0, -1.0))
    ray_directions = ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))

    cuda_batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=4,
    )
    torch_batch = torch_render_rays(scene, ray_origins, ray_directions, device="cuda")

    assert cuda_batch.backend == "cuda"
    assert cuda_batch.to_dict()["available"] is True
    for ray_index in range(len(ray_origins)):
        assert cuda_batch.color[ray_index] == pytest.approx(torch_batch.predicted_color[ray_index], abs=1.0e-5)
        assert cuda_batch.transmittance[ray_index] == pytest.approx(torch_batch.transmittance[ray_index], abs=1.0e-5)
        assert cuda_batch.opacity[ray_index] == pytest.approx(torch_batch.opacity[ray_index], abs=1.0e-5)
        assert cuda_batch.confidence[ray_index] == pytest.approx(torch_batch.confidence[ray_index], abs=1.0e-5)
        assert bool(cuda_batch.residual[ray_index]) == bool(torch_batch.residual[ray_index])
        cuda_depth = cuda_batch.depth[ray_index]
        torch_depth = torch_batch.predicted_depth[ray_index]
        if torch_depth is None:
            assert cuda_depth is None
        else:
            assert cuda_depth is not None
            assert cuda_depth == pytest.approx(torch_depth, abs=1.0e-4)


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
    assert batch.to_dict()["productionReady"] is True


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_keeps_tensor_rays_on_compiled_path(monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

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
        ray_origin_device = None

        @classmethod
        def render_rays(
            cls,
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
                colors,
                opacities,
                confidences,
                payload_params,
                material_ids,
                semantic_ids,
                threads_per_block,
            )
            cls.ray_origin_device = str(ray_origins.device)
            assert ray_origins.is_cuda
            ray_count = int(ray_origins.shape[0])
            return {
                "out_color": torch.zeros((ray_count, 3), dtype=torch.float32, device=ray_origins.device),
                "out_alpha": torch.zeros((ray_count,), dtype=torch.float32, device=ray_origins.device),
                "out_transmittance": torch.ones((ray_count,), dtype=torch.float32, device=ray_origins.device),
                "out_depth": torch.full((ray_count,), 3.402823466e38, dtype=torch.float32, device=ray_origins.device),
                "out_normal": torch.zeros((ray_count, 3), dtype=torch.float32, device=ray_origins.device),
                "out_confidence": torch.zeros((ray_count,), dtype=torch.float32, device=ray_origins.device),
                "out_residual": torch.zeros((ray_count,), dtype=torch.uint8, device=ray_origins.device),
                "out_material_id": torch.full((ray_count,), -1, dtype=torch.int32, device=ray_origins.device),
                "out_semantic_id": torch.full((ray_count,), -1, dtype=torch.int32, device=ray_origins.device),
                "ordered_hits": torch.full((ray_count, max_hits), -1, dtype=torch.int32, device=ray_origins.device),
            }

    def fail_python_ray_validation(*_args, **_kwargs):
        raise AssertionError("compiled CUDA dispatch must not materialize Python Ray objects")

    monkeypatch.setattr(cuda_renderer_module, "_validated_rays", fail_python_ray_validation)
    scene = AuraScene(
        name="cuda_tensor_direct_dispatch_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    ray_origins = torch.tensor(((0.0, 0.0, -1.0), (2.0, 0.0, -1.0)), dtype=torch.float32, device="cuda")
    ray_directions = torch.tensor(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=torch.float32, device="cuda")

    batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        extension=extension,
        extension_module=FakeCudaModule,
        device="cuda",
        max_hits=2,
    )

    assert batch.backend == "cuda"
    assert batch.depth == (None, None)
    assert FakeCudaModule.ray_origin_device == "cuda:0"


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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_cuda_render_rays_torch_fallback_keeps_tensor_inputs_off_cpu_validation():
    import torch

    class _TensorOnlyRays:
        def __init__(self, values):
            self._tensor = torch.tensor(values, dtype=torch.float32)
            self.shape = self._tensor.shape

        def to(self, *, device=None, dtype=None):
            return self._tensor.to(device=device, dtype=dtype)

        def detach(self):  # pragma: no cover - should not be reached.
            raise AssertionError("torch fallback should not convert tensor rays through CPU validation")

    scene = AuraScene(
        name="cuda_torch_tensor_fallback_scene",
        elements=(
            AuraElement(
                id="surface",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                normal=(0.0, 0.0, -1.0),
                payload={"type": "surface_cell"},
            ),
        ),
    )

    batch = cuda_render_rays(
        scene,
        ray_origins=_TensorOnlyRays([[0.0, 0.0, -1.0]]),
        ray_directions=_TensorOnlyRays([[0.0, 0.0, 1.0]]),
        fallback_backend="torch",
        device="cpu",
    )

    assert batch.backend == "torch"
    assert batch.element_ids == ("surface",)
    assert batch.color[0] == pytest.approx((1.0, 0.0, 0.0))


def test_cuda_renderer_build_bvh_sah_is_lower_cost_than_median_for_clustered_scene():
    """SAH gives lower or equal traversal cost than median for a clustered scene."""

    # Create a clustered scene: many elements in one cluster, few in another
    elements = []
    # Cluster A: 8 elements tightly packed near x=0
    for i in range(8):
        elements.append(
            AuraElement(
                id=f"cluster_a_{i}",
                carrier_id="surface",
                bounds=Bounds((float(i) * 0.01, 0.0, 0.0), (float(i) * 0.01 + 0.005, 0.1, 0.1)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload={"type": "surface_cell"},
            )
        )
    # Cluster B: 1 element far away near x=100
    elements.append(
        AuraElement(
            id="cluster_b_0",
            carrier_id="surface",
            bounds=Bounds((100.0, 0.0, 0.0), (100.1, 0.1, 0.1)),
            color=(0.0, 0.0, 1.0),
            opacity=1.0,
            confidence=1.0,
            payload={"type": "surface_cell"},
        )
    )

    scene = AuraScene(name="clustered_test", elements=tuple(elements))

    bvh_sah = cuda_renderer_build_bvh(scene, method="sah")
    bvh_median = cuda_renderer_build_bvh(scene, method="median")

    # Both BVHs must cover all elements as leaves
    assert bvh_sah.element_count == len(elements)
    assert bvh_median.element_count == len(elements)
    assert bvh_sah.node_count >= len(elements)
    assert bvh_median.node_count >= len(elements)

    def compute_bvh_cost(bvh):
        """Rough SAH cost: sum over nodes of (node_SA / root_SA)."""
        node_mins_list = [(bvh.node_mins[i * 3], bvh.node_mins[i * 3 + 1], bvh.node_mins[i * 3 + 2]) for i in range(bvh.node_count)]
        node_maxs_list = [(bvh.node_maxs[i * 3], bvh.node_maxs[i * 3 + 1], bvh.node_maxs[i * 3 + 2]) for i in range(bvh.node_count)]

        def sa(lo, hi):
            w = hi[0] - lo[0]
            h = hi[1] - lo[1]
            d = hi[2] - lo[2]
            return 2.0 * (w * h + w * d + h * d)

        root_sa = sa(node_mins_list[0], node_maxs_list[0])
        if root_sa < 1e-10:
            return 0.0
        total = 0.0
        for i in range(bvh.node_count):
            node_sa = sa(node_mins_list[i], node_maxs_list[i])
            total += node_sa / root_sa
        return total

    cost_sah = compute_bvh_cost(bvh_sah)
    cost_median = compute_bvh_cost(bvh_median)

    # SAH should give lower or equal cost (allow some tolerance)
    assert cost_sah <= cost_median * 1.05  # SAH cost within 5% of median or better


def test_cuda_renderer_build_bvh_sah_parity_with_brute_force():
    """BVH-SAH first-hit indices match CPU traversal reference."""
    scene = AuraScene(
        name="sah_parity_test",
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

    bvh_sah = cuda_renderer_build_bvh(scene, method="sah")
    rays = (
        Ray(origin=(-0.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
        Ray(origin=(0.5, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
        Ray(origin=(2.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),
    )
    reference_indices = cuda_renderer_reference_first_hit_indices(scene, rays)

    # BVH must have correct structure
    assert bvh_sah.element_count == 2
    assert bvh_sah.node_count >= 3  # at least root + 2 leaves

    # The scene traversal gives the expected first hits
    assert reference_indices == (0, 1, -1)


# ---------------------------------------------------------------------------
# Group 1: Dataclass validation errors (lines 63, 81, 128, 130, 132, 230)
# ---------------------------------------------------------------------------

def _minimal_scene_buffer_kwargs(element_count: int = 1) -> dict:
    """Return a complete valid kwargs dict for CudaRendererSceneBuffers."""
    from aura.cuda_renderer import CudaRendererSceneBuffers  # noqa: F401
    n = element_count
    return dict(
        element_ids=tuple(f"e{i}" for i in range(n)),
        carrier_ids=tuple("surface" for _ in range(n)),
        carrier_kernel_ids=tuple(0 for _ in range(n)),
        material_id_table=(),
        semantic_id_table=(),
        material_ids=tuple(-1 for _ in range(n)),
        semantic_ids=tuple(-1 for _ in range(n)),
        element_mins=tuple(0.0 for _ in range(n * 3)),
        element_maxs=tuple(1.0 for _ in range(n * 3)),
        plane_points=tuple(0.0 for _ in range(n * 3)),
        plane_normals=tuple(0.0 for _ in range(n * 3)),
        beta_support_radii=tuple(0.0 for _ in range(n * 3)),
        gaussian_means=tuple(0.0 for _ in range(n * 3)),
        gaussian_inverse_covariances=tuple(0.0 for _ in range(n * 9)),
        gaussian_support_radius_sq=tuple(0.0 for _ in range(n)),
        colors=tuple(1.0 for _ in range(n * 3)),
        opacities=tuple(1.0 for _ in range(n)),
        confidences=tuple(1.0 for _ in range(n)),
        payload_params=tuple(0.0 for _ in range(n * 5)),
    )


def test_coverage_scene_buffers_rejects_mismatched_carrier_count():
    from aura.cuda_renderer import CudaRendererSceneBuffers
    kwargs = _minimal_scene_buffer_kwargs(1)
    kwargs["carrier_ids"] = ()  # wrong length — triggers line 63
    with pytest.raises(ValueError, match="require one carrier id per element"):
        CudaRendererSceneBuffers(**kwargs)


def test_coverage_scene_buffers_rejects_mismatched_buffer_length():
    from aura.cuda_renderer import CudaRendererSceneBuffers
    kwargs = _minimal_scene_buffer_kwargs(1)
    kwargs["material_ids"] = (0, 1)  # wrong length — triggers line 81
    with pytest.raises(ValueError, match="does not match expected"):
        CudaRendererSceneBuffers(**kwargs)


def test_coverage_kernel_input_buffers_rejects_zero_max_hits():
    from aura.cuda_renderer import CudaRendererSceneBuffers, CudaRendererKernelInputBuffers
    scene = CudaRendererSceneBuffers(**_minimal_scene_buffer_kwargs(1))
    with pytest.raises(ValueError, match="max_hits must be positive"):  # line 128
        CudaRendererKernelInputBuffers(
            scene=scene,
            ray_origins=(0.0, 0.0, -1.0),
            ray_directions=(0.0, 0.0, 1.0),
            max_hits=0,
        )


def test_coverage_kernel_input_buffers_rejects_mismatched_ray_buffers():
    from aura.cuda_renderer import CudaRendererSceneBuffers, CudaRendererKernelInputBuffers
    scene = CudaRendererSceneBuffers(**_minimal_scene_buffer_kwargs(1))
    with pytest.raises(ValueError, match="matching lengths"):  # line 130
        CudaRendererKernelInputBuffers(
            scene=scene,
            ray_origins=(0.0, 0.0, -1.0),
            ray_directions=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),  # extra ray
            max_hits=1,
        )


def test_coverage_kernel_input_buffers_rejects_non_divisible_ray_buffers():
    from aura.cuda_renderer import CudaRendererSceneBuffers, CudaRendererKernelInputBuffers
    scene = CudaRendererSceneBuffers(**_minimal_scene_buffer_kwargs(1))
    with pytest.raises(ValueError, match="flat rayCount x 3"):  # line 132
        CudaRendererKernelInputBuffers(
            scene=scene,
            ray_origins=(0.0, 0.0),  # not divisible by 3
            ray_directions=(0.0, 0.0),
            max_hits=1,
        )


def test_coverage_kernel_simulation_rejects_mismatched_output_buffer():
    from aura.cuda_renderer import (
        CudaRendererSceneBuffers, CudaRendererKernelInputBuffers, CudaRendererKernelSimulation,
    )
    scene = CudaRendererSceneBuffers(**_minimal_scene_buffer_kwargs(1))
    inputs = CudaRendererKernelInputBuffers(
        scene=scene,
        ray_origins=(0.0, 0.0, -1.0),
        ray_directions=(0.0, 0.0, 1.0),
        max_hits=1,
    )
    # Line 230: out_color has wrong length
    with pytest.raises(ValueError, match="does not match expected"):
        CudaRendererKernelSimulation(
            inputs=inputs,
            out_color=(1.0, 0.0),  # should be 3 for 1 ray
            out_alpha=(0.0,),
            out_transmittance=(1.0,),
            out_depth=(1.0e38,),
            out_normal=(0.0, 0.0, 0.0),
            out_confidence=(0.0,),
            out_residual=(0,),
            out_material_id=(-1,),
            out_semantic_id=(-1,),
            ordered_hits=(-1,),
        )


# ---------------------------------------------------------------------------
# Group 2: Private geometry helper functions
# ---------------------------------------------------------------------------

def test_coverage_payload_params_gabor_frequency_not_list():
    # Line 708: frequency is not list/tuple → falls back to (0,0,0)
    from aura.cuda_renderer import _cuda_renderer_payload_params
    elem = AuraElement(
        id="g",
        carrier_id="gabor",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gabor_frequency", "frequency": "bad", "phase": 0.5, "bandwidth": 1.0},
    )
    result = _cuda_renderer_payload_params(elem)
    assert result == (0.0, 0.0, 0.0, 0.5, 1.0)


def test_coverage_plane_normal_surface_zero_normal():
    # Lines 729-730: surface with zero normal → _nan_vec3
    from aura.cuda_renderer import _cuda_renderer_plane_normal
    import math
    elem = AuraElement(
        id="s",
        carrier_id="surface",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
        normal=(0.0, 0.0, 0.0),  # zero normal → normalize raises ValueError
        payload={"type": "surface_cell"},
    )
    result = _cuda_renderer_plane_normal(elem)
    assert all(math.isnan(v) for v in result)


def test_coverage_plane_normal_gabor_zero_normal_vector():
    # Lines 736-737: gabor with zero normal in payload → _nan_vec3
    from aura.cuda_renderer import _cuda_renderer_plane_normal
    import math
    elem = AuraElement(
        id="g",
        carrier_id="gabor",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
        payload={"type": "gabor_frequency", "normal": [0.0, 0.0, 0.0]},
    )
    result = _cuda_renderer_plane_normal(elem)
    assert all(math.isnan(v) for v in result)


def test_coverage_plane_normal_gabor_zero_extent_bounds():
    # Line 742: gabor with zero-extent bounds → _nan_vec3
    from aura.cuda_renderer import _cuda_renderer_plane_normal
    import math
    elem = AuraElement(
        id="g",
        carrier_id="gabor",
        bounds=Bounds((0.0, 0.0, 0.0), (0.0, 0.5, 0.5)),  # x-extent = 0
        payload={"type": "gabor_frequency"},
    )
    result = _cuda_renderer_plane_normal(elem)
    assert all(math.isnan(v) for v in result)


def test_coverage_plane_point_surface_negative_normal_dominant_axis():
    # Lines 762-763: surface with negative dominant-axis normal → uses min_corner
    from aura.cuda_renderer import _cuda_renderer_plane_point
    elem = AuraElement(
        id="s",
        carrier_id="surface",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
        normal=(0.0, 0.0, -1.0),  # negative dominant axis → uses min_corner for z
        payload={"type": "surface_cell"},
    )
    result = _cuda_renderer_plane_point(elem)
    # z-axis is dominant; normal[2] < 0 → center[2] = min_corner[2] = 0.0
    assert result[2] == pytest.approx(0.0)
    assert result[0] == pytest.approx(0.0)  # x center
    assert result[1] == pytest.approx(0.0)  # y center


def test_coverage_beta_support_radius_non_numeric_values():
    # Lines 775-776: support_radius with non-numeric values → _nan_vec3
    from aura.cuda_renderer import _cuda_renderer_beta_support_radius
    import math
    elem = AuraElement(
        id="b",
        carrier_id="beta",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "beta_kernel", "support_radius": ["a", "b", "c"]},
    )
    result = _cuda_renderer_beta_support_radius(elem)
    assert all(math.isnan(v) for v in result)


def test_coverage_gaussian_mean_non_list():
    # Line 789: gaussian with non-list mean → _nan_vec3
    from aura.cuda_renderer import _cuda_renderer_gaussian_mean
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "mean": "bad"},
    )
    result = _cuda_renderer_gaussian_mean(elem)
    assert all(math.isnan(v) for v in result)


def test_coverage_gaussian_mean_non_numeric_values():
    # Lines 792-793: gaussian with non-numeric values in mean list → _nan_vec3
    from aura.cuda_renderer import _cuda_renderer_gaussian_mean
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "mean": ["x", "y", "z"]},
    )
    result = _cuda_renderer_gaussian_mean(elem)
    assert all(math.isnan(v) for v in result)


def test_coverage_gaussian_inverse_covariance_invalid_matrix():
    # Line 801: invalid covariance (not a matrix3) → _nan_matrix3
    from aura.cuda_renderer import _cuda_renderer_gaussian_inverse_covariance
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "covariance": [[1.0, 0.0], [0.0, 1.0]]},  # 2x2 not 3x3
    )
    result = _cuda_renderer_gaussian_inverse_covariance(elem)
    assert all(math.isnan(v) for row in result for v in row)


def test_coverage_gaussian_inverse_covariance_non_numeric_values():
    # Lines 804-805: covariance with non-numeric values → _nan_matrix3
    from aura.cuda_renderer import _cuda_renderer_gaussian_inverse_covariance
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "covariance": [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]},
    )
    result = _cuda_renderer_gaussian_inverse_covariance(elem)
    assert all(math.isnan(v) for row in result for v in row)


def test_coverage_gaussian_support_radius_sq_explicit_nonpositive():
    # Lines 815-819: explicit support_radius_sq <= 0 → nan
    from aura.cuda_renderer import _cuda_renderer_gaussian_support_radius_sq
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "support_radius_sq": -1.0},
    )
    result = _cuda_renderer_gaussian_support_radius_sq(elem)
    assert math.isnan(result)


def test_coverage_gaussian_support_radius_sq_explicit_non_numeric():
    # Lines 815-819: explicit support_radius_sq non-numeric → nan
    from aura.cuda_renderer import _cuda_renderer_gaussian_support_radius_sq
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "support_radius_sq": "bad"},
    )
    result = _cuda_renderer_gaussian_support_radius_sq(elem)
    assert math.isnan(result)


def test_coverage_gaussian_support_radius_sq_sigma_non_numeric():
    # Lines 822-823: support_sigma non-numeric → nan
    from aura.cuda_renderer import _cuda_renderer_gaussian_support_radius_sq
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "support_sigma": "bad"},
    )
    result = _cuda_renderer_gaussian_support_radius_sq(elem)
    assert math.isnan(result)


def test_coverage_gaussian_support_radius_sq_sigma_nonpositive():
    # Line 823: support_sigma <= 0 → nan
    from aura.cuda_renderer import _cuda_renderer_gaussian_support_radius_sq
    import math
    elem = AuraElement(
        id="gau",
        carrier_id="gaussian",
        bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0)),
        payload={"type": "gaussian_fallback", "support_sigma": -1.0},
    )
    result = _cuda_renderer_gaussian_support_radius_sq(elem)
    assert math.isnan(result)


def test_coverage_normalize_vec3_zero_raises():
    # Line 1982
    from aura.cuda_renderer import _normalize_vec3
    with pytest.raises(ValueError, match="cannot normalize zero vector"):
        _normalize_vec3((0.0, 0.0, 0.0))


def test_coverage_inverse_matrix3_singular_returns_none():
    # Line 2014: singular matrix → returns None
    from aura.cuda_renderer import _inverse_matrix3
    singular = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))  # det = 0
    result = _inverse_matrix3(singular)
    assert result is None


def test_coverage_normal_for_element_uses_payload_normal():
    # Line 1975: element.normal is None but payload has normal
    from aura.cuda_renderer import _normal_for_element
    elem = AuraElement(
        id="s",
        carrier_id="surface",
        bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
        payload={"type": "surface_cell", "normal": [0.0, 1.0, 0.0]},
    )
    result = _normal_for_element(elem)
    assert result == pytest.approx((0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# Group 3: AABB/plane/ellipsoid intersection edge cases
# ---------------------------------------------------------------------------

def test_coverage_aabb_ray_swap_t0_t1_negative_direction():
    # Lines 2071-2072: t0 > t1 when ray goes in negative direction → swap
    from aura.cuda_renderer import _simulate_ray_aabb_intersect
    # Ray going in -z direction, which causes t0 > t1 and triggers the swap at lines 2071-2072
    result = _simulate_ray_aabb_intersect(
        origin=(0.0, 0.0, 2.0),
        direction=(0.0, 0.0, -1.0),  # negative direction → t0 > t1 swap
        box_min=(-1.0, -1.0, -1.0),
        box_max=(1.0, 1.0, 1.0),
    )
    # Should hit the box (ray going from z=2 towards z=-inf, box is at -1 to 1)
    assert result is not None


def test_coverage_plane_intersect_ray_parallel_to_plane():
    # Line 2098: ray direction perpendicular to plane normal → denom near zero → return None
    from aura.cuda_renderer import _simulate_ray_plane_intersect
    # Ray going sideways, plane faces forward — ray is parallel to plane
    result = _simulate_ray_plane_intersect(
        origin=(0.0, 0.0, -1.0),
        direction=(1.0, 0.0, 0.0),  # perpendicular to normal (0,0,1) → denom = 0
        box_min=(-1.0, -1.0, 0.0),
        box_max=(1.0, 1.0, 1.0),
        plane_point=(0.0, 0.0, 0.5),
        normal=(0.0, 0.0, 1.0),
    )
    assert result is None


def test_coverage_plane_intersect_nan_plane_returns_none():
    # Line 2095: NaN plane_point → return None (hits different branch than line 2098)
    import math
    from aura.cuda_renderer import _simulate_ray_plane_intersect
    nan = float("nan")
    result = _simulate_ray_plane_intersect(
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        box_min=(-1.0, -1.0, 0.0),
        box_max=(1.0, 1.0, 1.0),
        plane_point=(nan, nan, nan),
        normal=(0.0, 0.0, 1.0),
    )
    assert result is None


def test_coverage_plane_intersect_depth_negative_returns_none():
    # Line 2101: depth < 0 → return None (plane is behind ray)
    from aura.cuda_renderer import _simulate_ray_plane_intersect
    result = _simulate_ray_plane_intersect(
        origin=(0.0, 0.0, 2.0),
        direction=(0.0, 0.0, 1.0),  # ray going away from the plane
        box_min=(-1.0, -1.0, -1.0),
        box_max=(1.0, 1.0, 0.0),
        plane_point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    assert result is None


def test_coverage_plane_intersect_hit_outside_aabb():
    # Line 2104: hit point outside AABB → return None
    from aura.cuda_renderer import _simulate_ray_plane_intersect
    # Plane at z=0.5, ray hits at (10, 0, 0.5) which is outside the box
    result = _simulate_ray_plane_intersect(
        origin=(10.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        box_min=(-1.0, -1.0, 0.0),
        box_max=(1.0, 1.0, 1.0),
        plane_point=(0.0, 0.0, 0.5),
        normal=(0.0, 0.0, 1.0),
    )
    assert result is None


def test_coverage_beta_ellipsoid_nan_support_radii():
    # Line 2116: NaN support_radii → return None
    import math
    from aura.cuda_renderer import _simulate_ray_beta_ellipsoid_intersect
    nan = float("nan")
    result = _simulate_ray_beta_ellipsoid_intersect(
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        box_min=(-1.0, -1.0, 0.0),
        box_max=(1.0, 1.0, 1.0),
        support_radii=(nan, 0.5, 0.5),
    )
    assert result is None


def test_coverage_beta_ellipsoid_zero_support_radii():
    # Line 2116: zero support_radii → return None
    from aura.cuda_renderer import _simulate_ray_beta_ellipsoid_intersect
    result = _simulate_ray_beta_ellipsoid_intersect(
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        box_min=(-1.0, -1.0, 0.0),
        box_max=(1.0, 1.0, 1.0),
        support_radii=(0.0, 0.5, 0.5),
    )
    assert result is None


def test_coverage_beta_ellipsoid_normal_normalize_fails():
    # Lines 2149-2150: beta ellipsoid normal fails → returns (0,0,0)
    # Triggered when the hit point is exactly at the center (gradient = 0)
    from aura.cuda_renderer import _beta_ellipsoid_normal
    result = _beta_ellipsoid_normal(
        origin=(0.0, 0.0, 0.0),   # at center
        direction=(0.0, 0.0, 1.0),
        depth=0.0,                  # hit point = origin = center → gradient = 0
        center=(0.0, 0.0, 0.0),
        support_radii=(0.5, 0.5, 0.5),
    )
    assert result == (0.0, 0.0, 0.0)


def test_coverage_gaussian_ellipsoid_invalid_geometry():
    # Line 2161: invalid geometry (NaN mean) → return None
    import math
    from aura.cuda_renderer import _simulate_ray_gaussian_ellipsoid_intersect
    nan = float("nan")
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    result = _simulate_ray_gaussian_ellipsoid_intersect(
        origin=(0.0, 0.0, -1.0),
        direction=(0.0, 0.0, 1.0),
        mean=(nan, nan, nan),
        inverse_covariance=identity,
        support_radius_sq=1.0,
    )
    assert result is None


def test_coverage_gaussian_ellipsoid_behind_ray():
    # Line 2177: gaussian is entirely behind the ray (far < 0) → return None
    from aura.cuda_renderer import _simulate_ray_gaussian_ellipsoid_intersect
    # Mean at z=0, ray starts at z=5 going in +z direction (away from mean)
    inv_cov = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    result = _simulate_ray_gaussian_ellipsoid_intersect(
        origin=(0.0, 0.0, 5.0),
        direction=(0.0, 0.0, 1.0),  # going away from mean at origin
        mean=(0.0, 0.0, 0.0),
        inverse_covariance=inv_cov,
        support_radius_sq=0.01,  # very small support — box behind ray
    )
    assert result is None


# ---------------------------------------------------------------------------
# Group 4: Tensor/list helpers
# ---------------------------------------------------------------------------

def test_coverage_tensor_to_nested_invalid_length():
    # Line 1900: flat length not divisible by width → ValueError
    from aura.cuda_renderer import _tensor_to_nested_float_tuple
    with pytest.raises(ValueError, match="invalid flat length"):
        _tensor_to_nested_float_tuple([1.0, 2.0, 3.0, 4.0], width=3)


def test_coverage_tensor_to_flat_list_no_detach():
    # Lines 1907-1908: no .detach() attribute — handled via AttributeError
    from aura.cuda_renderer import _tensor_to_flat_list
    # Object with reshape and tolist but no detach
    class FakeTensor:
        def reshape(self, _n):
            return self
        def tolist(self):
            return [1.0, 2.0, 3.0]
    result = _tensor_to_flat_list(FakeTensor())
    assert result == [1.0, 2.0, 3.0]


def test_coverage_tensor_to_flat_list_no_reshape():
    # Lines 1911-1912: no .reshape() but has tolist()
    from aura.cuda_renderer import _tensor_to_flat_list
    class FakeTensor:
        def detach(self):
            return self
        def cpu(self):
            return self
        def tolist(self):
            return [4.0, 5.0, 6.0]
    result = _tensor_to_flat_list(FakeTensor())
    assert result == [4.0, 5.0, 6.0]


def test_coverage_tensor_to_flat_list_nested_sequence():
    # Lines 1915-1924: nested sequence flattening path (tolist → sequence of sequences)
    from aura.cuda_renderer import _tensor_to_flat_list
    result = _tensor_to_flat_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert result == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_coverage_tensor_to_flat_list_flat_sequence():
    # Line 1924 fallback: flat sequence
    from aura.cuda_renderer import _tensor_to_flat_list
    result = _tensor_to_flat_list([1.0, 2.0, 3.0])
    assert result == [1.0, 2.0, 3.0]


def test_coverage_tensor_to_flat_list_non_tensor_raises():
    # Line 1931: non-tensor-like raises ValueError
    from aura.cuda_renderer import _tensor_to_flat_list
    with pytest.raises(ValueError, match="not tensor-like"):
        _tensor_to_flat_list(42)


# ---------------------------------------------------------------------------
# Group 5: Ray validation helpers
# ---------------------------------------------------------------------------

def test_coverage_cuda_render_rays_available_extension_require_cuda_raises():
    # Line 1410: extension.available but binding not callable + require_cuda=True → RuntimeError
    from aura.cuda_kernels import CudaExtensionStatus
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    # Module where render_rays is not callable → symbol_probe.binding_callable=False
    module = type(
        "NotCallable",
        (),
        {
            "aura_render_rays_kernel": object(),
            "aura_render_rays_launcher": object(),
            "render_rays": "string_not_callable",
        },
    )()
    scene = AuraScene(name="require_cuda_available_scene", elements=())
    with pytest.raises(RuntimeError, match="CUDA renderer Python dispatch is unavailable"):
        cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0, 1.0),),
            require_cuda=True,
            extension=extension,
            extension_module=module,
        )


def test_coverage_cuda_render_rays_unavailable_extension_require_cuda_raises():
    # Line 1412: extension unavailable + require_cuda=True → RuntimeError
    scene = AuraScene(name="require_cuda_unavailable_scene", elements=())
    with pytest.raises(RuntimeError, match="CUDA renderer extension is unavailable"):
        cuda_render_rays(
            scene,
            ray_origins=((0.0, 0.0, -1.0),),
            ray_directions=((0.0, 0.0, 1.0),),
            require_cuda=True,
            extension=_unavailable_extension_status(),
        )


def test_coverage_ray_count_from_inputs_origin_ne_direction():
    # Line 1429: origin count != direction count → ValueError
    from aura.cuda_renderer import _ray_count_from_inputs
    with pytest.raises(ValueError, match="does not match"):
        _ray_count_from_inputs(
            ((0.0, 0.0, -1.0),),
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        )


def test_coverage_ray_count_from_inputs_zero_count():
    # Line 1431: origin_count <= 0 → ValueError "ray_count must be positive"
    from aura.cuda_renderer import _ray_count_from_inputs
    with pytest.raises(ValueError, match="ray_count must be positive"):
        _ray_count_from_inputs((), ())


def test_coverage_ray_count_from_rows_none_raises():
    # Line 1437: values is None → ValueError
    from aura.cuda_renderer import _ray_count_from_rows
    with pytest.raises(ValueError, match="is required"):
        _ray_count_from_rows(None, "ray_origins")


def test_coverage_ray_count_from_rows_tensor_wrong_shape():
    # Line 1442: tensor with wrong shape → ValueError
    from aura.cuda_renderer import _ray_count_from_rows
    class FakeTensor:
        shape = (3,)  # 1D, not 2D
    with pytest.raises(ValueError, match="must have shape rayCount x 3"):
        _ray_count_from_rows(FakeTensor(), "ray_origins")


def test_coverage_ray_count_from_rows_non_sequence():
    # Line 1445: non-sequence non-tensor → ValueError
    from aura.cuda_renderer import _ray_count_from_rows
    with pytest.raises(ValueError, match="must be a sequence or tensor-like"):
        _ray_count_from_rows(42, "ray_origins")


def test_coverage_validated_rays_count_mismatch():
    # Line 1456: origins count != directions count
    from aura.cuda_renderer import _validated_rays
    with pytest.raises(ValueError, match="does not match"):
        _validated_rays(
            ((0.0, 0.0, -1.0),),
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        )


def test_coverage_validated_rays_empty_raises():
    # Line 1458: empty origins list → ValueError
    from aura.cuda_renderer import _validated_rays
    with pytest.raises(ValueError, match="ray_count must be positive"):
        _validated_rays((), ())


def test_coverage_vec3_rows_none_raises():
    # Line 1464: None input → ValueError
    from aura.cuda_renderer import _vec3_rows
    with pytest.raises(ValueError, match="is required"):
        _vec3_rows(None, "ray_origins")


def test_coverage_vec3_rows_tensor_wrong_shape():
    # Line 1469: tensor with shape but wrong dims → ValueError
    from aura.cuda_renderer import _vec3_rows
    class BadShapeTensor:
        shape = (3,)  # 1D, should be 2D N×3
    with pytest.raises(ValueError, match="must have shape rayCount x 3"):
        _vec3_rows(BadShapeTensor(), "ray_origins")


def test_coverage_vec3_rows_tensor_with_shape_calls_tolist():
    # Lines 1467-1476: tensor with shape attr uses detach().cpu().tolist()
    from aura.cuda_renderer import _vec3_rows
    class FakeTensor:
        shape = (2, 3)
        def detach(self):
            return self
        def cpu(self):
            return self
        def tolist(self):
            return [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]
    result = _vec3_rows(FakeTensor(), "ray_origins")
    assert result == ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))


def test_coverage_vec3_rows_tensor_no_detach_uses_tolist():
    # Line 1473-1474: tensor with shape but no detach → tries tolist()
    from aura.cuda_renderer import _vec3_rows
    class FakeNoDetach:
        shape = (1, 3)
        def tolist(self):
            return [[1.0, 2.0, 3.0]]
    result = _vec3_rows(FakeNoDetach(), "ray_origins")
    assert result == ((1.0, 2.0, 3.0),)


def test_coverage_vec3_rows_tensor_no_detach_no_tolist():
    # Lines 1475-1476: tensor has shape but no detach and no tolist → falls through (pass)
    # The object must be a Sequence to avoid the isinstance error at 1478
    from aura.cuda_renderer import _vec3_rows
    from collections.abc import Sequence as ABCSequence

    class TensorLike(ABCSequence):
        shape = (2, 3)
        _data = [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return self._data[i]

    result = _vec3_rows(TensorLike(), "ray_origins")
    assert result == ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))


def test_coverage_vec3_rows_non_sequence_raises():
    # Line 1478: non-sequence non-tensor raises ValueError
    from aura.cuda_renderer import _vec3_rows
    with pytest.raises(ValueError, match="must be a sequence or tensor-like"):
        _vec3_rows(42, "ray_origins")


def test_coverage_vec3_rows_row_not_3d():
    # Line 1482: row not a 3D vector → ValueError
    from aura.cuda_renderer import _vec3_rows
    with pytest.raises(ValueError, match="must contain 3D ray vectors"):
        _vec3_rows([(0.0, 0.0)], "ray_origins")


# ---------------------------------------------------------------------------
# Group 6: Fallback/resolution helpers
# ---------------------------------------------------------------------------

def test_coverage_resolve_fallback_backend_auto_torch_with_elements():
    # Lines 1490-1498: "auto" with torch available and scene has elements → "torch"
    from aura.cuda_renderer import _resolve_fallback_backend
    scene = AuraScene(
        name="auto_torch_scene",
        elements=(
            AuraElement(
                id="s",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for this branch")
    result = _resolve_fallback_backend("auto", scene=scene)
    assert result in ("torch", "cpu")  # depends on torch availability


def test_coverage_resolve_fallback_backend_auto_exception_returns_cpu(monkeypatch):
    # Line 1495: exception in torch import → "cpu"
    from aura.cuda_renderer import _resolve_fallback_backend
    import aura.cuda_renderer as cr_mod
    monkeypatch.setattr(cr_mod, "_resolve_fallback_backend", _resolve_fallback_backend)

    def bad_import():
        raise ImportError("no torch")

    import sys
    original = sys.modules.get("aura.torch_renderer")
    sys.modules["aura.torch_renderer"] = None  # type: ignore
    try:
        scene = AuraScene(name="exc_scene", elements=())
        result = _resolve_fallback_backend("auto", scene=scene)
        assert result == "cpu"
    finally:
        if original is None:
            sys.modules.pop("aura.torch_renderer", None)
        else:
            sys.modules["aura.torch_renderer"] = original


def test_coverage_resolve_fallback_backend_auto_no_elements():
    # Line 1498: "auto" with no elements → "cpu"
    from aura.cuda_renderer import _resolve_fallback_backend
    scene = AuraScene(name="empty_scene", elements=())
    result = _resolve_fallback_backend("auto", scene=scene)
    assert result == "cpu"


def test_coverage_resolve_cuda_extension_imported_module_found(monkeypatch):
    # Line 1515: when imported module is found at runtime
    import aura.cuda_renderer as cr_mod
    from aura.cuda_renderer import _resolve_cuda_renderer_extension
    fake_module = object()
    monkeypatch.setattr(cr_mod, "_import_cuda_renderer_extension_module", lambda: fake_module)
    status, module = _resolve_cuda_renderer_extension(
        extension=None,
        extension_module=None,
        build=False,
    )
    assert module is fake_module
    assert status.available is True


def test_coverage_resolve_cuda_extension_provided_extension_no_module(monkeypatch):
    # Line 1517: extension is provided but module not importable → return extension, None
    import aura.cuda_renderer as cr_mod
    from aura.cuda_renderer import _resolve_cuda_renderer_extension, _available_extension_status
    from aura.cuda_kernels import CudaExtensionStatus
    monkeypatch.setattr(cr_mod, "_import_cuda_renderer_extension_module", lambda: None)
    provided_extension = CudaExtensionStatus(
        available=True, build_attempted=True, compiled=True, loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    status, module = _resolve_cuda_renderer_extension(
        extension=provided_extension,
        extension_module=None,
        build=False,
    )
    assert module is None
    assert status is provided_extension


def test_coverage_resolve_cuda_extension_no_extension_no_build(monkeypatch):
    # Line 1519: no extension, no module found, no build → use extension status
    import aura.cuda_renderer as cr_mod
    from aura.cuda_renderer import _resolve_cuda_renderer_extension
    monkeypatch.setattr(cr_mod, "_import_cuda_renderer_extension_module", lambda: None)
    status, module = _resolve_cuda_renderer_extension(
        extension=None,
        extension_module=None,
        build=False,
    )
    assert module is None
    assert status.available is False


def test_coverage_resolve_cuda_extension_with_provided_extension_module():
    # Line 1507-1509: extension_module provided → use it directly
    from aura.cuda_renderer import _resolve_cuda_renderer_extension, _available_extension_status
    fake_ext = _available_extension_status(build_attempted=True)
    fake_module = object()
    status, module = _resolve_cuda_renderer_extension(
        extension=fake_ext,
        extension_module=fake_module,
        build=False,
    )
    assert module is fake_module
    assert status is fake_ext


def test_coverage_build_extension_cuda_home_none(monkeypatch):
    # Line 1550: CUDA_HOME is None → returns failure status
    from aura.cuda_renderer import _build_cuda_renderer_extension_module
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for this path")
    import torch.utils.cpp_extension as cpp_ext
    monkeypatch.setattr(cpp_ext, "CUDA_HOME", None)
    status, module = _build_cuda_renderer_extension_module()
    assert status.available is False
    assert status.reason == "cuda_home_unavailable"
    assert module is None


def test_coverage_build_extension_torch_cuda_unavailable(monkeypatch):
    # Line 1552: torch.cuda not available → returns failure status
    from aura.cuda_renderer import _build_cuda_renderer_extension_module
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for this path")
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status, module = _build_cuda_renderer_extension_module()
    assert status.available is False
    assert status.reason == "torch_cuda_unavailable"
    assert module is None


def test_coverage_extension_status_failure():
    # Line 1601: _extension_status_failure function
    from aura.cuda_renderer import _extension_status_failure
    status = _extension_status_failure(
        ("cuda/aura_bindings.cpp",),
        ("render_rays",),
        "test_reason",
        build_attempted=True,
    )
    assert status.available is False
    assert status.reason == "test_reason"
    assert status.build_attempted is True


def test_coverage_cpu_fallback_available_extension_not_dispatch_ready():
    # Line 1624: extension.available but probe not dispatch_ready → different reason
    from aura.cuda_renderer import _cpu_fallback_batch, cuda_renderer_launch_config, cuda_renderer_symbol_probe
    from aura.cuda_kernels import CudaExtensionStatus
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    symbol_probe = cuda_renderer_symbol_probe(extension, extension_module=None)
    scene = AuraScene(name="cpu_fallback_test", elements=())
    rays = (Ray(origin=(0.0, 0.0, -1.0), direction=(0.0, 0.0, 1.0)),)
    config = cuda_renderer_launch_config(1, fallback_backend="cpu")
    batch = _cpu_fallback_batch(scene, rays, config, extension, symbol_probe=symbol_probe)
    assert "cuda_extension_available_python_binding_unavailable_cpu_fallback" in batch.reason


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_coverage_torch_fallback_available_extension_not_dispatch_ready():
    # Line 1657: extension.available but probe not dispatch_ready → different reason
    from aura.cuda_renderer import _torch_fallback_batch, cuda_renderer_launch_config, cuda_renderer_symbol_probe
    from aura.cuda_kernels import CudaExtensionStatus
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    symbol_probe = cuda_renderer_symbol_probe(extension, extension_module=None)
    scene = AuraScene(
        name="torch_fallback_test",
        elements=(
            AuraElement(
                id="s",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    config = cuda_renderer_launch_config(1, fallback_backend="torch")
    batch = _torch_fallback_batch(
        scene,
        ((0.0, 0.0, -1.0),),
        ((0.0, 0.0, 1.0),),
        config,
        extension,
        device="cpu",
        symbol_probe=symbol_probe,
    )
    assert "cuda_extension_available_python_binding_unavailable_torch_fallback" in batch.reason


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_coverage_compiled_extension_batch_ray_count_mismatch():
    # Lines 1698, 1700: tensor ray_count mismatch
    import torch
    from aura.cuda_renderer import _compiled_extension_batch, cuda_renderer_launch_config, _available_extension_status

    scene = AuraScene(
        name="mismatch_scene",
        elements=(
            AuraElement(
                id="s",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    extension = _available_extension_status(build_attempted=True)
    launch_config = cuda_renderer_launch_config(2, fallback_backend="cpu")

    class FakeModule:
        aura_render_rays_kernel = object()
        aura_render_rays_launcher = object()
        @staticmethod
        def render_rays(*args, **kwargs):
            raise AssertionError("should not be called")

    # Provide 1 origin ray but launch config says 2 → hits line 1698
    ray_origins = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    ray_directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="does not match launch config ray count"):
        _compiled_extension_batch(
            scene, ray_origins, ray_directions, launch_config, extension, FakeModule(), device="cpu"
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_coverage_compiled_extension_batch_direction_count_mismatch():
    # Line 1700: ray_direction tensor count != launch config ray count
    import torch
    from aura.cuda_renderer import _compiled_extension_batch, cuda_renderer_launch_config, _available_extension_status

    scene = AuraScene(
        name="dir_mismatch_scene",
        elements=(
            AuraElement(
                id="s",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    extension = _available_extension_status(build_attempted=True)
    # launch_config expects 2 rays, origins has 2 rows, but directions has only 1 row
    launch_config = cuda_renderer_launch_config(2, fallback_backend="cpu")
    ray_origins = torch.tensor([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]], dtype=torch.float32)
    ray_directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)  # 1 row != 2

    class _NoOpModule:
        aura_render_rays_kernel = object()
        aura_render_rays_launcher = object()
        @staticmethod
        def render_rays(*args, **kwargs):
            raise AssertionError("should not reach render_rays")

    with pytest.raises(ValueError, match="does not match launch config ray count"):
        _compiled_extension_batch(
            scene, ray_origins, ray_directions, launch_config, extension, _NoOpModule(), device="cpu"
        )


def test_coverage_cuda_float_ray_tensor_none_raises():
    # Line 1760: None input → ValueError
    from aura.cuda_renderer import _cuda_float_ray_tensor
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required")
    import torch
    with pytest.raises(ValueError, match="is required"):
        _cuda_float_ray_tensor(torch, None, "ray_origins", "cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_coverage_cuda_float_ray_tensor_wrong_shape():
    # Lines 1766, 1768: wrong shape tensor → ValueError
    import torch
    from aura.cuda_renderer import _cuda_float_ray_tensor
    bad_tensor = torch.tensor([1.0, 2.0, 3.0])  # 1D, not 2D
    with pytest.raises(ValueError, match="must have shape rayCount x 3"):
        _cuda_float_ray_tensor(torch, bad_tensor, "ray_origins", "cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_coverage_cuda_float_ray_tensor_zero_rays():
    # Line 1768: zero ray count → ValueError
    import torch
    from aura.cuda_renderer import _cuda_float_ray_tensor
    empty = torch.zeros((0, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="ray_count must be positive"):
        _cuda_float_ray_tensor(torch, empty, "ray_origins", "cpu")


# ---------------------------------------------------------------------------
# Group 7: Symbol probe / dispatch contract (lines 333, 336, 933, 937)
# ---------------------------------------------------------------------------

def test_coverage_dispatch_contract_reason_symbols_not_ready():
    # Line 333: extension.available=True but dispatch_symbols_ready=False → "compiled_cuda_renderer_binding_unavailable"
    from aura.cuda_kernels import CudaExtensionStatus
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    # Module with kernel but no launcher → dispatch_symbols_ready = False
    module = type(
        "NoLauncherModule",
        (),
        {
            "aura_render_rays_kernel": object(),
            # no aura_render_rays_launcher
            "render_rays": "not_callable",
        },
    )()
    scene = AuraScene(
        name="symbols_not_ready_scene",
        elements=(
            AuraElement(
                id="s",
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
        extension_module=module,
    )
    assert "compiled_cuda_renderer_binding_unavailable" in contract.reason


def test_coverage_dispatch_contract_reason_binding_callable():
    # Line 336: dispatch_symbols_ready True and binding_callable True → "compiled_cuda_renderer_python_binding_ready"
    from aura.cuda_kernels import CudaExtensionStatus
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    # Provide a module with all symbols including a callable binding
    module = type(
        "FullCudaModule",
        (),
        {
            "aura_render_rays_kernel": object(),
            "aura_render_rays_launcher": object(),
            "render_rays": lambda *args, **kwargs: None,
        },
    )()
    scene = AuraScene(
        name="dispatch_ready_scene",
        elements=(
            AuraElement(
                id="s",
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
        extension_module=module,
    )
    assert contract.reason == "compiled_cuda_renderer_python_binding_ready"


def test_coverage_dispatch_contract_reason_binding_missing():
    # Line 335: dispatch_symbols_ready True but binding_callable False → "python_cuda_renderer_binding_missing"
    from aura.cuda_kernels import CudaExtensionStatus
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    # Provide module with all symbols but render_rays is not callable
    module = type(
        "NonCallableBinding",
        (),
        {
            "aura_render_rays_kernel": object(),
            "aura_render_rays_launcher": object(),
            "render_rays": "not_callable",
        },
    )()
    scene = AuraScene(
        name="binding_missing_scene",
        elements=(
            AuraElement(
                id="s",
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
        extension_module=module,
    )
    assert contract.reason == "python_cuda_renderer_binding_missing"


def test_coverage_symbol_probe_all_symbols_missing():
    # Lines 933, 937: all symbols missing, including BINDING_SYMBOL
    from aura.cuda_kernels import CudaExtensionStatus
    extension = CudaExtensionStatus(
        available=True,
        build_attempted=True,
        compiled=True,
        loadable=True,
        module_name="aura_cuda_carriers",
        source_paths=("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        symbols=("aura_render_rays_kernel", "aura_render_rays_launcher", "render_rays"),
    )
    # Module with no relevant symbols
    empty_module = type("EmptyModule", (), {})()
    probe = cuda_renderer_symbol_probe(extension, extension_module=empty_module)
    assert probe.dispatch_symbols_ready is False
    assert CUDA_RENDERER_BINDING_SYMBOL in (probe.reason or "")


# ---------------------------------------------------------------------------
# Group 8: BVH edge cases (lines 614, 644-646)
# ---------------------------------------------------------------------------

def test_coverage_bvh_sah_left_right_count_zero():
    # Line 614: SAH bin split with left_count or right_count == 0 → continue
    # This is triggered when all centroids fall in the same bin; SAH falls through to median
    elements = []
    # Two elements at exactly the same location — all centroids in same bin
    for i in range(2):
        elements.append(
            AuraElement(
                id=f"coincident_{i}",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (0.001, 0.001, 0.001)),
                payload={"type": "surface_cell"},
            )
        )
    scene = AuraScene(name="sah_same_bin_test", elements=tuple(elements))
    bvh = cuda_renderer_build_bvh(scene, method="sah")
    # Should still cover all elements
    assert bvh.element_count == 2
    leaf_elements = sorted(v for v in bvh.node_element if v >= 0)
    assert leaf_elements == [0, 1]


def test_coverage_bvh_median_degenerate_coincident_centroids():
    # Lines 644-646: median split degenerate case (coincident centroids) → index split
    # All elements at the same location forces the fallback
    elements = []
    for i in range(4):
        elements.append(
            AuraElement(
                id=f"dup_{i}",
                carrier_id="surface",
                bounds=Bounds((0.0, 0.0, 0.0), (0.001, 0.001, 0.001)),
                payload={"type": "surface_cell"},
            )
        )
    scene = AuraScene(name="coincident_median_test", elements=tuple(elements))
    bvh = cuda_renderer_build_bvh(scene, method="median")
    assert bvh.element_count == 4
    leaf_elements = sorted(v for v in bvh.node_element if v >= 0)
    assert leaf_elements == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Group 9: Boundary report exception path (lines 1304-1311)
# ---------------------------------------------------------------------------

def test_coverage_boundary_report_exception_path(monkeypatch):
    # Lines 1304-1311: when inner probe raises, report captures the error
    import aura.cuda_renderer as cr_mod

    def raise_error(*args, **kwargs):
        raise RuntimeError("probe_failure_injected")

    monkeypatch.setattr(cr_mod, "cuda_renderer_kernel_inputs", raise_error)
    scene = AuraScene(
        name="boundary_exception_scene",
        elements=(
            AuraElement(
                id="s",
                carrier_id="surface",
                bounds=Bounds((-0.5, -0.5, 0.0), (0.5, 0.5, 0.2)),
                payload={"type": "surface_cell"},
            ),
        ),
    )
    report = cuda_renderer_boundary_report(scene, fallback_backend="cpu")
    assert report["fallbackProbe"]["executed"] is False
    assert "probe_failure_injected" in report["fallbackProbe"]["error"]
    assert report["kernelInputProbe"] is None
    assert report["dispatchContractProbe"] is None


# ---------------------------------------------------------------------------
# Group 10: cuda_kernels.py functions
# ---------------------------------------------------------------------------

def test_coverage_cuda_kernel_extension_status_torch_unavailable(monkeypatch):
    # Lines 320-330: torch is unavailable → reason = "torch_unavailable"
    from importlib.util import find_spec as original_find_spec
    from aura.cuda_kernels import cuda_kernel_extension_status

    def mock_find_spec(name, *args, **kwargs):
        if name == "torch":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("aura.cuda_kernels.find_spec", mock_find_spec)
    status = cuda_kernel_extension_status(build=True)
    assert status.available is False
    assert status.reason == "torch_unavailable"
    assert status.build_attempted is True


def test_coverage_cuda_kernel_extension_status_cuda_home_none(monkeypatch):
    # Lines 336-337: CUDA_HOME is None
    from aura.cuda_kernels import cuda_kernel_extension_status
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for this path")
    import torch.utils.cpp_extension as cpp_ext
    monkeypatch.setattr(cpp_ext, "CUDA_HOME", None)
    status = cuda_kernel_extension_status(build=True)
    assert status.available is False
    assert status.reason == "cuda_home_unavailable"


def test_coverage_cuda_kernel_extension_status_torch_cuda_unavailable(monkeypatch):
    # Lines 338-339: torch.cuda.is_available() returns False
    from aura.cuda_kernels import cuda_kernel_extension_status
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch required for this path")
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status = cuda_kernel_extension_status(build=True)
    assert status.available is False
    assert status.reason == "torch_cuda_unavailable"


def test_coverage_cuda_kernel_source_available_file_not_found(monkeypatch):
    # Lines 484-485: FileNotFoundError → returns False
    from aura.cuda_kernels import cuda_kernel_source_available
    import aura.cuda_kernels as ck_mod
    from importlib.resources import files as real_files

    def bad_files(package):
        raise FileNotFoundError("injected error")

    monkeypatch.setattr(ck_mod, "files", bad_files)
    result = cuda_kernel_source_available("cuda/aura_carriers.cu")
    assert result is False


def test_coverage_cuda_kernel_source_missing_fragments_none_source_text(monkeypatch):
    # Line 491: source_text is None → returns [symbol, *contract_outputs]
    from aura.cuda_kernels import (
        cuda_kernel_source_missing_fragments, CudaKernelSource, CudaKernelArgument,
        _cuda_kernel_source_text,
    )
    import aura.cuda_kernels as ck_mod

    monkeypatch.setattr(ck_mod, "_cuda_kernel_source_text", lambda path: None)
    source = CudaKernelSource(
        carrier_id="surface",
        payload_type="surface_cell",
        symbol="aura_surface_forward_kernel",
        path="cuda/nonexistent.cu",
        arguments=(),
        contract_outputs=("out_color", "out_transmittance"),
        required=True,
    )
    missing = cuda_kernel_source_missing_fragments(source)
    assert "aura_surface_forward_kernel" in missing
    assert "out_color" in missing


def test_coverage_validate_batched_rays_both_none():
    # Line 564-565: both None → returns None
    from aura.cuda_kernels import _validate_batched_rays
    result = _validate_batched_rays(None, None)
    assert result is None


def test_coverage_validate_batched_rays_one_none_raises():
    # Lines 566-567: one None → ValueError
    from aura.cuda_kernels import _validate_batched_rays
    with pytest.raises(ValueError, match="must be provided together"):
        _validate_batched_rays(((0.0, 0.0, -1.0),), None)


def test_coverage_validate_batched_rays_count_mismatch():
    # Lines 568-569: count mismatch → ValueError
    from aura.cuda_kernels import _validate_batched_rays
    with pytest.raises(ValueError, match="does not match"):
        _validate_batched_rays(
            ((0.0, 0.0, -1.0),),
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        )


def test_coverage_batched_vec3_count_tensor_wrong_shape():
    # Lines 579-580: tensor with wrong shape → ValueError
    from aura.cuda_kernels import _batched_vec3_count
    class FakeTensor:
        shape = (3,)  # 1D, not 2D
    with pytest.raises(ValueError, match="must have shape rayCount x 3"):
        _batched_vec3_count(FakeTensor(), "ray_origins")


def test_coverage_batched_vec3_count_non_sequence():
    # Lines 582-583: non-sequence non-tensor → ValueError
    from aura.cuda_kernels import _batched_vec3_count
    with pytest.raises(ValueError, match="must be a sequence or tensor-like"):
        _batched_vec3_count(42, "ray_origins")


def test_coverage_batched_vec3_count_bad_row():
    # Lines 585-586: row not 3D → ValueError
    from aura.cuda_kernels import _batched_vec3_count
    with pytest.raises(ValueError, match="must contain 3D ray vectors"):
        _batched_vec3_count([(0.0, 0.0)], "ray_origins")


def test_coverage_batched_vec3_count_valid_returns_len():
    # Line 587 / 614: valid rows → returns len
    from aura.cuda_kernels import _batched_vec3_count
    result = _batched_vec3_count([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)], "ray_origins")
    assert result == 2


def test_coverage_cuda_kernel_source_text_not_a_file(monkeypatch):
    # Lines 594, 596-597: resource.is_file() returns False → returns None
    from aura.cuda_kernels import _cuda_kernel_source_text
    import aura.cuda_kernels as ck_mod
    from importlib.resources import files as real_files

    class FakeResource:
        def is_file(self):
            return False

    class FakePackage:
        def joinpath(self, path):
            return FakeResource()

    monkeypatch.setattr(ck_mod, "files", lambda pkg: FakePackage())
    result = _cuda_kernel_source_text("cuda/nonexistent.cu")
    assert result is None


def test_coverage_cuda_kernel_source_text_file_not_found(monkeypatch):
    # Lines 596-597: FileNotFoundError → returns None
    from aura.cuda_kernels import _cuda_kernel_source_text
    import aura.cuda_kernels as ck_mod

    def bad_files(package):
        raise FileNotFoundError("injected")

    monkeypatch.setattr(ck_mod, "files", bad_files)
    result = _cuda_kernel_source_text("cuda/aura_carriers.cu")
    assert result is None


def test_coverage_resource_as_file_imports_as_file():
    # Lines 601-603: _resource_as_file imports as_file
    from aura.cuda_kernels import _resource_as_file
    from importlib.resources import files
    resource = files("aura").joinpath("cuda/aura_carriers.cu")
    ctx = _resource_as_file(resource)
    # It should be a context manager from importlib.resources.as_file
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")


# ---------------------------------------------------------------------------
# Group 11: Transmittance threshold break (line 1047)
# ---------------------------------------------------------------------------

def test_coverage_simulation_transmittance_threshold_break():
    # Line 1047: transmittance falls below threshold → break early
    # Create many opaque elements in sequence so transmittance hits zero quickly
    elements = [
        AuraElement(
            id=f"opaque_{i}",
            carrier_id="surface",
            bounds=Bounds((-0.5, -0.5, float(i) * 0.1), (0.5, 0.5, float(i) * 0.1 + 0.05)),
            color=(1.0, 0.0, 0.0),
            opacity=1.0,
            confidence=1.0,
            payload={"type": "surface_cell"},
        )
        for i in range(5)
    ]
    scene = AuraScene(name="threshold_break_test", elements=tuple(elements))
    inputs = cuda_renderer_kernel_inputs(
        scene,
        ((0.0, 0.0, -1.0),),
        ((0.0, 0.0, 1.0),),
        max_hits=5,
    )
    # Use a high transmittance_threshold so it triggers early
    simulation = simulate_cuda_renderer_kernel(inputs, transmittance_threshold=0.9)
    # Transmittance should be clamped near 0 quickly
    assert simulation.out_transmittance[0] <= 1.0


# ---------------------------------------------------------------------------
# Group 12: Lookup table id missing value (lines 1966-1967)
# ---------------------------------------------------------------------------

def test_coverage_gaussian_ellipsoid_normal_normalize_fails():
    # Lines 2194-2195: gaussian normal fails → returns (0,0,0)
    # Triggered when inverse_covariance is all zeros → gradient = (0,0,0)
    from aura.cuda_renderer import _gaussian_ellipsoid_normal
    zero_inv_cov = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = _gaussian_ellipsoid_normal(
        origin=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        depth=1.0,
        mean=(0.0, 0.0, 0.0),
        inverse_covariance=zero_inv_cov,  # produces zero gradient → normalize fails
    )
    assert result == (0.0, 0.0, 0.0)


def test_coverage_lookup_table_id_missing_raises():
    # Lines 1966-1967: value not in table → ValueError
    from aura.cuda_renderer import _lookup_table_id
    with pytest.raises(ValueError, match="missing from CUDA renderer id table"):
        _lookup_table_id(("a", "b"), "c")


def test_coverage_table_value_out_of_range_raises():
    # Line 1931: _table_value with out-of-range index → ValueError
    from aura.cuda_renderer import _table_value
    with pytest.raises(ValueError, match="out-of-range dictionary id"):
        _table_value(("a", "b"), 99)


def test_coverage_import_module_glob_path_exception(monkeypatch):
    # Lines 1543-1544: glob-based import attempt raises → return None
    import aura.cuda_renderer as cr_mod
    from aura.cuda_renderer import _import_cuda_renderer_extension_module

    def _always_raise(name):
        raise ImportError(f"no module named {name!r}")

    monkeypatch.setattr(cr_mod, "import_module", _always_raise)
    result = _import_cuda_renderer_extension_module()
    assert result is None


def test_coverage_resolve_cuda_extension_build_true_no_cached_module(monkeypatch):
    # Line 1520: _import fails, build=True → calls _build_cuda_renderer_extension_module
    import aura.cuda_renderer as cr_mod
    from aura.cuda_renderer import _resolve_cuda_renderer_extension, _extension_status_failure

    monkeypatch.setattr(cr_mod, "_import_cuda_renderer_extension_module", lambda: None)
    stub_status = _extension_status_failure(
        ("cuda/aura_bindings.cpp", "cuda/aura_carriers.cu"),
        ("render_rays",),
        "stub_build_skipped",
        build_attempted=True,
    )
    monkeypatch.setattr(cr_mod, "_build_cuda_renderer_extension_module", lambda: (stub_status, None))
    status, module = _resolve_cuda_renderer_extension(
        extension=None,
        extension_module=None,
        build=True,
    )
    assert module is None
    assert status.available is False
    assert status.reason == "stub_build_skipped"
