"""Tests for the gsplat differentiable-CUDA-rasterizer training backend.

The conversion-math tests need torch (for eigendecomposition / quaternion ops)
and are skipped where the GPU stack is absent; they run on CPU tensors so they
do NOT require a GPU. The camera-conversion test is pure Python.
"""

import importlib.util

import pytest

import aura.gsplat_renderer as gr

_HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def test_manifest_frame_to_camera_is_world_to_camera():
    frame = {
        "intrinsics": {"width": 800, "height": 600, "fx": 500.0, "fy": 500.0,
                        "cx": 400.0, "cy": 300.0},
        "camera_origin": [0.0, 0.0, 5.0],
        "look_at": [0.0, 0.0, 0.0],
    }
    view, k, w, h = gr.manifest_frame_to_camera(frame, scale=0.5)
    assert (w, h) == (400, 300)
    # intrinsics scale with resolution
    assert k[0][0] == pytest.approx(250.0)
    assert k[0][2] == pytest.approx(200.0)
    # camera at +5z looking toward origin: forward = -z, so the view matrix maps
    # the world origin to +5 along the camera forward axis (row 2).
    import_origin = [0.0, 0.0, 0.0, 1.0]
    z_cam = sum(view[2][i] * import_origin[i] for i in range(4))
    assert z_cam == pytest.approx(5.0, abs=1e-5)


def _spd_scene():
    """A small AuraScene of Gaussian carriers with random SPD covariances."""
    import torch
    from aura.elements import AuraElement, Bounds
    from aura.carrier_payloads import GaussianFallbackPayload
    from aura.scene import AuraScene

    torch.manual_seed(0)
    elems, covs = [], []
    for i in range(5):
        a = torch.randn(3, 3)
        cov = a @ a.T + 0.3 * torch.eye(3)
        covs.append(cov)
        payload = GaussianFallbackPayload(
            mean=(float(i), 0.5, -1.0),
            covariance=tuple(tuple(float(x) for x in row) for row in cov.tolist()),
            source="test",
        ).to_dict()
        elems.append(AuraElement(
            id=f"g{i}", carrier_id="gaussian", bounds=Bounds((0, 0, 0), (1, 1, 1)),
            color=(0.1 * i, 0.4, 0.6), opacity=0.3 + 0.1 * i, confidence=1.0,
            payload=payload,
        ))
    scene = AuraScene(name="t", elements=tuple(elems), chunks=(), semantic_graph=None)
    return scene, covs


@requires_torch
def test_covariance_color_opacity_round_trip():
    """scene -> trainable params -> scene reconstructs covariance/color/opacity."""
    import torch

    scene, covs = _spd_scene()
    params, ctx = gr.scene_to_gaussian_params(scene, device="cpu")
    assert set(params) == {"means", "log_scales", "quats", "logit_opacities", "colors"}
    assert all(params[k].requires_grad for k in params)

    back = gr.gaussian_params_to_scene(params, ctx)
    gaussians = [e for e in back.elements if e.carrier_id == "gaussian"]
    assert len(gaussians) == len(covs)
    for i, e in enumerate(gaussians):
        cov_out = torch.tensor(e.payload["covariance"])
        assert (cov_out - covs[i]).abs().max().item() < 1e-4
        assert e.opacity == pytest.approx(0.3 + 0.1 * i, abs=1e-4)
        assert e.payload["source"] == "gsplat-trained"
        assert e.payload["mean"] == [float(i), 0.5, -1.0]


@requires_torch
def test_quat_rotation_matrix_inverse():
    """quat -> R -> quat is identity (up to sign) and R is orthonormal."""
    import torch

    torch.manual_seed(1)
    q = torch.randn(8, 4)
    q = q / q.norm(dim=1, keepdim=True)
    R = gr._quat_wxyz_to_rotation_matrix(q, torch)
    # orthonormal rotations: R R^T = I, det = +1
    eye = torch.eye(3).unsqueeze(0).expand(8, 3, 3)
    assert (R @ R.transpose(1, 2) - eye).abs().max().item() < 1e-5
    assert (torch.linalg.det(R) - 1.0).abs().max().item() < 1e-4
    q2 = gr._rotation_matrix_to_quat_wxyz(R, torch)
    # quaternion double cover: q and -q are the same rotation; compare |dot|.
    dots = (q * q2).sum(dim=1).abs()
    assert (dots - 1.0).abs().max().item() < 1e-4
