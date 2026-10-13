import math

import pytest

from aura import AuraElement, AuraScene, Bounds
from aura.cuda_renderer import (
    cuda_render_rays,
    cuda_renderer_kernel_inputs,
    simulate_cuda_renderer_kernel,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def cuda_torch():
    torch = pytest.importorskip("torch", reason="torch is required for CUDA integration checks")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    return torch


def _flatten_nested(values):
    return tuple(item for row in values for item in row)


def _assert_compiled_cuda(batch):
    assert batch.backend == "cuda"
    assert batch.reason == "compiled_cuda_renderer_python_binding"
    assert batch.to_dict()["available"] is True


def _assert_kernel_parity(batch, simulation):
    _assert_compiled_cuda(batch)
    assert _flatten_nested(batch.color) == pytest.approx(simulation.out_color)
    assert batch.opacity == pytest.approx(simulation.out_alpha)
    assert batch.transmittance == pytest.approx(simulation.out_transmittance)
    actual_depth = tuple(3.402823466e38 if depth is None else depth for depth in batch.depth)
    assert actual_depth == pytest.approx(simulation.out_depth)
    actual_normals = tuple((0.0, 0.0, 0.0) if normal is None else normal for normal in batch.normal)
    assert _flatten_nested(actual_normals) == pytest.approx(simulation.out_normal)
    assert batch.confidence == pytest.approx(simulation.out_confidence)
    assert tuple(int(value) for value in batch.residual) == simulation.out_residual
    actual_hits = tuple(hit[0]["kernelElementIndex"] if hit else -1 for hit in batch.ordered_hits)
    assert actual_hits == simulation.first_hit_indices


def test_compiled_cuda_renderer_matches_kernel_simulation(cuda_torch):
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
    ray_directions = ((0.0, 0.0, 1.0),) * len(ray_origins)
    inputs = cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=2)
    simulation = simulate_cuda_renderer_kernel(inputs)

    batch = cuda_render_rays(
        scene,
        ray_origins,
        ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=2,
    )

    _assert_kernel_parity(batch, simulation)


@pytest.mark.parametrize(
    ("carrier_id", "payload", "element_kwargs", "ray_origins", "expected_depths"),
    (
        pytest.param(
            "beta",
            {"type": "beta_kernel", "alpha": 2.0, "beta": 2.0, "support_radius": [0.5, 0.5, 0.25]},
            {},
            ((0.0, 0.0, -1.0), (0.75, 0.0, -1.0)),
            (1.75, None),
            id="beta-support-ellipsoid",
        ),
        pytest.param(
            "gaussian",
            {
                "type": "gaussian_fallback",
                "mean": [0.0, 0.0, 1.0],
                "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                "support_sigma": 1.0,
            },
            {},
            ((0.0, 0.0, -1.0), (0.75, 0.0, -1.0)),
            (1.5, None),
            id="gaussian-support-ellipsoid",
        ),
        pytest.param(
            "surface",
            {"type": "surface_cell", "plane_point": [0.0, 0.0, 0.25]},
            {"normal": (0.0, 0.0, 1.0)},
            ((0.0, 0.0, -1.0),),
            (1.25,),
            id="surface-plane",
        ),
        pytest.param(
            "gabor",
            {
                "type": "gabor_frequency",
                "frequency": [0.0, 0.0, 0.0],
                "phase": math.pi / 2.0,
                "bandwidth": 1.0,
                "normal": [0.0, 0.0, 1.0],
                "plane_point": [0.0, 0.0, 0.2],
            },
            {},
            ((0.0, 0.0, -1.0),),
            (1.2,),
            id="gabor-phase",
        ),
    ),
)
def test_compiled_cuda_carrier_geometry_matches_kernel_simulation(
    cuda_torch, carrier_id, payload, element_kwargs, ray_origins, expected_depths
):
    scene = AuraScene(
        name=f"cuda_{carrier_id}_geometry",
        elements=(
            AuraElement(
                id="carrier",
                carrier_id=carrier_id,
                bounds=Bounds((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0)),
                color=(1.0, 0.0, 0.0),
                opacity=1.0,
                confidence=1.0,
                payload=payload,
                **element_kwargs,
            ),
        ),
    )
    ray_directions = ((0.0, 0.0, 1.0),) * len(ray_origins)
    inputs = cuda_renderer_kernel_inputs(scene, ray_origins, ray_directions, max_hits=1)
    simulation = simulate_cuda_renderer_kernel(inputs)

    batch = cuda_render_rays(
        scene,
        ray_origins,
        ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=1,
    )

    _assert_kernel_parity(batch, simulation)
    assert len(batch.depth) == len(expected_depths)
    for actual, expected in zip(batch.depth, expected_depths):
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected)


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


@pytest.mark.parametrize(("carrier_id", "payload", "element_kwargs"), _CARRIER_PARITY_CASES)
def test_compiled_cuda_matches_torch_renderer_for_each_carrier(
    cuda_torch, carrier_id, payload, element_kwargs
):
    from aura.torch_renderer import torch_render_rays

    scene = _carrier_parity_scene(carrier_id, payload, **element_kwargs)
    ray_origins = cuda_torch.tensor(
        ((0.0, 0.0, -1.0), (0.1, -0.1, -1.0), (2.0, 0.0, -1.0)),
        dtype=cuda_torch.float32,
        device="cuda",
    )
    ray_directions = cuda_torch.tensor(
        ((0.0, 0.0, 1.0),) * 3,
        dtype=cuda_torch.float32,
        device="cuda",
    )

    cuda_batch = cuda_render_rays(
        scene,
        ray_origins,
        ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=4,
    )
    torch_batch = torch_render_rays(scene, ray_origins, ray_directions, device="cuda")

    _assert_compiled_cuda(cuda_batch)
    for ray_index in range(int(ray_origins.shape[0])):
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


def test_compiled_cuda_bvh_matches_brute_force_and_torch(cuda_torch):
    from aura.torch_renderer import torch_render_rays

    scene = _multi_carrier_scene()
    ray_origins = cuda_torch.tensor(
        (
            (-0.5, -0.5, -1.0),
            (0.1, -0.45, -1.0),
            (-0.5, 0.3, -1.0),
            (0.65, 0.2, -1.0),
            (0.95, 0.4, -1.0),
            (2.0, 0.0, -1.0),
        ),
        dtype=cuda_torch.float32,
        device="cuda",
    )
    ray_directions = cuda_torch.tensor(
        ((0.0, 0.0, 1.0),) * int(ray_origins.shape[0]),
        dtype=cuda_torch.float32,
        device="cuda",
    )

    bvh_batch = cuda_render_rays(
        scene,
        ray_origins,
        ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=8,
        use_bvh=True,
    )
    brute_batch = cuda_render_rays(
        scene,
        ray_origins,
        ray_directions,
        device="cuda",
        require_cuda=True,
        fallback_backend="none",
        max_hits=8,
        use_bvh=False,
    )
    torch_batch = torch_render_rays(scene, ray_origins, ray_directions, device="cuda", collect_traces=False)

    _assert_compiled_cuda(bvh_batch)
    _assert_compiled_cuda(brute_batch)
    for ray_index in range(int(ray_origins.shape[0])):
        assert bvh_batch.element_ids[ray_index] == brute_batch.element_ids[ray_index]
        assert bvh_batch.element_ids[ray_index] == torch_batch.element_ids[ray_index]
        assert bvh_batch.color[ray_index] == pytest.approx(brute_batch.color[ray_index], abs=1.0e-6)
        assert bvh_batch.color[ray_index] == pytest.approx(torch_batch.predicted_color[ray_index], abs=1.0e-4)
        assert bvh_batch.transmittance[ray_index] == pytest.approx(brute_batch.transmittance[ray_index], abs=1.0e-6)
