"""Differentiable CUDA rasterization backend for AURA-Core training.

The native torch training renderer (``torch_render_capture_training_objective``)
is a dense ``O(rays x carriers)`` ray-Gaussian evaluator: correct but far too
slow to converge a real scene (one full-coverage epoch over 129k carriers ~25
min, see ``docs/CONVERGENCE_TODO.md``), and AURA's CUDA renderer is forward-only
(no backward). This module gives AURA-Core a *fast differentiable CUDA
rasterizer* for the training step by using ``gsplat`` (a tiled, sorted,
``O(pixels)`` differentiable Gaussian rasterizer with analytic gradients).

It is deliberately structured as a renderer backend, not a fork of the training
pipeline: it reads the Gaussian carriers OUT of an :class:`AuraScene`, optimises
them against the manifest's posed images, and writes the trained Gaussians BACK
into a new :class:`AuraScene` so the result is an ordinary ``.aura`` package that
AURA's own forward renderer (``eval_psnr.py``) evaluates unchanged. A
hand-written native AURA backward kernel can later replace ``gsplat`` behind the
same two boundary functions (:func:`scene_to_gaussian_params` /
:func:`gaussian_params_to_scene`).

Convention bridge (verified against ``carrier_payloads`` + ``torch_renderer``):

* AURA Gaussian carrier: ``carrier_id == "gaussian"``,
  ``payload["type"] == "gaussian_fallback"`` with ``payload["mean"]`` (Vec3) and
  ``payload["covariance"]`` (full 3x3, linear). ``element.color`` is linear RGB
  in ``[0, 1]`` and ``element.opacity`` in ``[0, 1]``; the forward renderer uses
  both *directly* (no activation).
* gsplat trains ``means``, ``log_scales`` (scale = exp), ``quats`` (wxyz,
  normalised), ``logit_opacities`` (opacity = sigmoid) and per-Gaussian linear
  ``colors``. The covariance written back is ``R diag(exp(2*log_scale)) R^T``
  with ``R`` the rotation matrix of the same normalised wxyz quaternion gsplat
  rasterised with, so AURA sees the exact Gaussian gsplat optimised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Sequence

from .elements import AuraElement, Bounds
from .carrier_payloads import GaussianFallbackPayload
from .scene import AuraScene


def gsplat_available() -> bool:
    """True when torch + gsplat (the differentiable CUDA rasterizer) import."""

    try:  # pragma: no cover - exercised only where the GPU stack is installed
        import torch  # noqa: F401
        import gsplat  # noqa: F401
    except Exception:
        return False
    return True


def require_gsplat():
    """Import torch + gsplat, failing loudly with an actionable message."""

    try:
        import torch
        import gsplat
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "The gsplat training backend requires torch and gsplat (a "
            "differentiable CUDA rasterizer). Install them on the GPU machine: "
            "`pip install torch --index-url https://download.pytorch.org/whl/cu128` "
            "then `pip install gsplat`."
        ) from exc
    return torch, gsplat


# --------------------------------------------------------------------------- #
# Camera conversion (canonical home; scripts import from here).
# --------------------------------------------------------------------------- #


def _normalize3(v: Sequence[float]) -> list[float]:
    n = sqrt(sum(c * c for c in v))
    if n == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return [c / n for c in v]


def _cross3(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def manifest_frame_to_camera(frame: dict, scale: float):
    """Convert one AURA manifest frame to ``(viewmat 4x4, K 3x3, W, H)``.

    Exact inverse of the camera basis built in ``eval_psnr.render_frame_torch``
    and AURA's ray construction: forward = normalize(look_at - origin),
    right = normalize(forward x up), up_actual = right x forward. gsplat uses a
    world-to-camera view matrix with +Z forward, +X right, +Y down, so the
    camera-from-world rotation rows are ``[right, up_actual, forward]`` and the
    translation is ``-R @ origin``.
    """

    intr = frame["intrinsics"]
    full_w, full_h = int(intr["width"]), int(intr["height"])
    w = max(1, int(full_w * scale))
    h = max(1, int(full_h * scale))
    fx = float(intr["fx"]) * scale
    fy = float(intr["fy"]) * scale
    cx = float(intr["cx"]) * scale
    cy = float(intr["cy"]) * scale

    origin = [float(c) for c in frame["camera_origin"]]
    look_at = [float(c) for c in frame["look_at"]]
    up = [float(c) for c in frame.get("up", [0.0, -1.0, 0.0])]

    fwd = _normalize3([look_at[i] - origin[i] for i in range(3)])
    right = _normalize3(_cross3(fwd, up))
    up_actual = _cross3(right, fwd)

    rows = [right, up_actual, fwd]
    t = [-sum(rows[r][c] * origin[c] for c in range(3)) for r in range(3)]
    view = [
        [rows[0][0], rows[0][1], rows[0][2], t[0]],
        [rows[1][0], rows[1][1], rows[1][2], t[1]],
        [rows[2][0], rows[2][1], rows[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    k = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return view, k, w, h


# --------------------------------------------------------------------------- #
# Boundary: AuraScene <-> trainable gsplat parameters.
# --------------------------------------------------------------------------- #


def _is_gaussian(element: AuraElement) -> bool:
    payload = element.payload or {}
    return element.carrier_id == "gaussian" and payload.get("type") == "gaussian_fallback"


def scene_to_gaussian_params(scene: AuraScene, *, device: str):
    """Read Gaussian carriers into trainable gsplat leaf tensors.

    Returns ``(params, ctx)`` where ``params`` is a dict of ``requires_grad``
    leaf tensors (``means`` ``[N,3]``, ``log_scales`` ``[N,3]``, ``quats``
    ``[N,4]`` wxyz, ``logit_opacities`` ``[N]``, ``colors`` ``[N,3]``) and
    ``ctx`` carries the element ids and the untouched non-Gaussian elements so
    the trained scene can be reassembled.
    """

    torch, _ = require_gsplat()
    gaussians = [e for e in scene.elements if _is_gaussian(e)]
    if not gaussians:
        raise ValueError("scene has no gaussian_fallback carriers to train")

    means = torch.tensor(
        [[float(c) for c in e.payload["mean"]] for e in gaussians],
        dtype=torch.float32, device=device,
    )
    cov = torch.tensor(
        [[[float(x) for x in row] for row in e.payload["covariance"]] for e in gaussians],
        dtype=torch.float32, device=device,
    )  # [N, 3, 3]
    colors = torch.tensor(
        [[float(c) for c in e.color] for e in gaussians],
        dtype=torch.float32, device=device,
    ).clamp(0.0, 1.0)
    opac = torch.tensor([float(e.opacity) for e in gaussians], dtype=torch.float32, device=device)

    # Symmetrise then eigendecompose covariance into rotation + axis variances.
    # cuSOLVER's batched syevd rejects a single very large 3x3 batch
    # (CUSOLVER_STATUS_INVALID_VALUE on 129k+), so eigh is run in GPU-sized
    # chunks (fast) with a CPU fallback. A tiny diagonal regulariser keeps
    # degenerate (zero-extent) seeds well-posed.
    cov = 0.5 * (cov + cov.transpose(1, 2))
    cov = cov + 1e-9 * torch.eye(3, dtype=cov.dtype, device=device).unsqueeze(0)
    eigvals, eigvecs = _batched_eigh(cov, torch)
    variances = eigvals.clamp(min=1e-12)
    scales = torch.sqrt(variances)  # [N, 3]
    log_scales = torch.log(scales.clamp(min=1e-9))
    # eigvecs is a rotation (or reflection); make it a proper rotation (det +1).
    det = torch.linalg.det(eigvecs)
    eigvecs = eigvecs.clone()
    eigvecs[:, :, 0] = eigvecs[:, :, 0] * det.unsqueeze(1)  # flip one axis if reflection
    quats = _rotation_matrix_to_quat_wxyz(eigvecs, torch)  # [N, 4]
    logit_opac = torch.logit(opac.clamp(1e-4, 1 - 1e-4))

    params = {
        "means": means.clone().requires_grad_(True),
        "log_scales": log_scales.clone().requires_grad_(True),
        "quats": quats.clone().requires_grad_(True),
        "logit_opacities": logit_opac.clone().requires_grad_(True),
        "colors": colors.clone().requires_grad_(True),
    }
    ctx = {
        "gaussian_elements": tuple(gaussians),  # originals; geometry replaced on writeback
        "non_gaussian": tuple(e for e in scene.elements if not _is_gaussian(e)),
        "scene_name": scene.name,
        "chunks": getattr(scene, "chunks", ()),
        "semantic_graph": getattr(scene, "semantic_graph", None),
    }
    return params, ctx


def seed_gaussian_params_from_regions(regions, *, device: str):
    """Seed trainable gsplat leaf tensors DIRECTLY from manifest point regions.

    This is the fast GPU-first seed path: it skips both the heavy image-tensor
    load and the ``decompose_evidence`` scene/BVH build (single-threaded CPU,
    minutes for 129k points), so the rasterizer starts almost immediately. It
    replicates ``decomposition._payload_for``'s Gaussian seed exactly — mean is
    the region bounding-box centre, the covariance is diagonal with variance =
    half-extent² per axis (so scale = half-extent, identity rotation) — and
    builds everything as vectorised tensors on ``device``.
    """

    torch, _ = require_gsplat()
    if not regions:
        raise ValueError("no seed regions provided")
    mins = torch.tensor([list(r.bounds.min_corner) for r in regions],
                        dtype=torch.float32, device=device)
    maxs = torch.tensor([list(r.bounds.max_corner) for r in regions],
                        dtype=torch.float32, device=device)
    means = 0.5 * (mins + maxs)
    half = (0.5 * (maxs - mins)).clamp(min=1e-4)  # sigma per axis (variance = half²)
    log_scales = torch.log(half)
    quats = torch.zeros((len(regions), 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity rotation (wxyz)
    colors = torch.tensor([list(r.color) for r in regions],
                          dtype=torch.float32, device=device).clamp(0.0, 1.0)
    opac = torch.tensor([float(r.opacity) for r in regions],
                        dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)
    params = {
        "means": means.clone().requires_grad_(True),
        "log_scales": log_scales.clone().requires_grad_(True),
        "quats": quats.clone().requires_grad_(True),
        "logit_opacities": torch.logit(opac).clone().requires_grad_(True),
        "colors": colors.clone().requires_grad_(True),
    }
    ctx = {
        "element_ids": [r.id for r in regions],
        "gaussian_elements": (),
        "scene_name": "aura_gsplat_train",
        "chunks": (),
        "semantic_graph": None,
    }
    return params, ctx


def _batched_eigh(cov, torch, chunk: int = 16384):
    """Symmetric eigendecomposition over a large [N,3,3] batch.

    cuSOLVER rejects one huge batched syevd call, so decompose in GPU-sized
    chunks; if any chunk errors (driver quirk), fall back to CPU LAPACK for it.
    Returns (eigvals [N,3] ascending, eigvecs [N,3,3] columns).
    """

    n = cov.shape[0]
    if n <= chunk:
        try:
            return torch.linalg.eigh(cov)
        except Exception:
            vals, vecs = torch.linalg.eigh(cov.detach().cpu())
            return vals.to(cov.device), vecs.to(cov.device)
    vals_parts, vecs_parts = [], []
    for start in range(0, n, chunk):
        block = cov[start : start + chunk]
        try:
            v, q = torch.linalg.eigh(block)
        except Exception:
            v, q = torch.linalg.eigh(block.detach().cpu())
            v, q = v.to(cov.device), q.to(cov.device)
        vals_parts.append(v)
        vecs_parts.append(q)
    return torch.cat(vals_parts, dim=0), torch.cat(vecs_parts, dim=0)


def _rotation_matrix_to_quat_wxyz(R, torch):
    """Batched rotation matrix [N,3,3] -> normalised wxyz quaternion [N,4]."""

    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    trace = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-12)) / 2.0
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * w)
    quat = torch.stack([w, x, y, z], dim=1)
    return quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-12)


def _quat_wxyz_to_rotation_matrix(quat, torch):
    """Batched normalised wxyz quaternion [N,4] -> rotation matrix [N,3,3]."""

    q = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def gaussian_params_to_scene(params: dict, ctx: dict) -> AuraScene:
    """Write trained gsplat leaf tensors back into a new :class:`AuraScene`.

    Covariance is reconstructed as ``R diag(exp(2*log_scale)) R^T`` from the
    same normalised wxyz quaternion gsplat rasterised, so AURA's forward
    renderer sees the exact optimised Gaussian. Colors/opacities are written
    directly (linear RGB / sigmoid), matching the no-activation accessor chain.
    """

    torch, _ = require_gsplat()
    with torch.no_grad():
        means = params["means"].detach()
        scales = torch.exp(params["log_scales"].detach())
        quats = params["quats"].detach()
        opac = torch.sigmoid(params["logit_opacities"].detach())
        colors = params["colors"].detach().clamp(0.0, 1.0)

        R = _quat_wxyz_to_rotation_matrix(quats, torch)  # [N,3,3]
        S2 = torch.diag_embed(scales * scales)  # [N,3,3]
        cov = R @ S2 @ R.transpose(1, 2)  # [N,3,3]
        cov = 0.5 * (cov + cov.transpose(1, 2))  # numerically symmetric

        means_l = means.cpu().tolist()
        cov_l = cov.cpu().tolist()
        scales_l = scales.cpu().tolist()
        opac_l = opac.cpu().tolist()
        colors_l = colors.cpu().tolist()

    import dataclasses

    originals = ctx.get("gaussian_elements", ())
    element_ids = ctx.get("element_ids")
    n = len(means_l)
    elements: list[AuraElement] = []
    for i in range(n):
        mean = tuple(float(c) for c in means_l[i])
        cov_rows = tuple(tuple(float(x) for x in row) for row in cov_l[i])
        sigma = tuple(max(float(s), 1e-6) for s in scales_l[i])
        # 3-sigma AABB (axis-aligned bound large enough for the rotated ellipsoid).
        radius = 3.0 * max(sigma)
        bounds = Bounds(
            min_corner=tuple(mean[j] - radius for j in range(3)),
            max_corner=tuple(mean[j] + radius for j in range(3)),
        )
        payload = GaussianFallbackPayload(
            mean=mean, covariance=cov_rows, source="gsplat-trained"
        ).to_dict()
        new_fields = dict(
            bounds=bounds,
            color=tuple(float(c) for c in colors_l[i]),
            opacity=float(opac_l[i]),
            payload=payload,
        )
        if i < len(originals):
            # Preserve id / chunk_id / semantic_id / metadata etc. of the seed
            # carrier; only the trained geometry/appearance changes. (When
            # densification grows N beyond the seed count, extra Gaussians get a
            # fresh id and no chunk link.)
            elements.append(dataclasses.replace(originals[i], **new_fields))
        else:
            eid = (
                element_ids[i]
                if element_ids and i < len(element_ids)
                else f"gsplat_gaussian_{i:06d}"
            )
            elements.append(
                AuraElement(
                    id=eid,
                    carrier_id="gaussian",
                    confidence=1.0,
                    **new_fields,
                )
            )

    elements.extend(ctx.get("non_gaussian", ()))
    # Training moved geometry, so the seed-time LOD chunk partition no longer
    # encloses these carriers. Re-derive the carrier/LOD chunks (with bounds
    # that enclose the trained positions) via the same helper the seed path uses.
    from .decomposition import carrier_lod_elements_and_chunks

    chunked_elements, chunks = carrier_lod_elements_and_chunks(tuple(elements))
    from .semantic import SemanticGraph

    semantic_graph = ctx.get("semantic_graph") or SemanticGraph()
    return AuraScene(
        name=ctx.get("scene_name", "aura_gsplat_train"),
        elements=chunked_elements,
        chunks=chunks,
        semantic_graph=semantic_graph,
    )


# --------------------------------------------------------------------------- #
# Training.
# --------------------------------------------------------------------------- #


@dataclass
class GsplatTrainConfig:
    iterations: int = 7000
    scale: float = 0.25
    position_lr: float = 1.6e-4
    log_scale_lr: float = 5e-3
    quat_lr: float = 1e-3
    opacity_lr: float = 5e-2
    color_lr: float = 2.5e-3
    ssim_weight: float = 0.2
    densify: bool = False
    densify_grad2d: float = 2e-4
    refine_start_iter: int = 500
    refine_stop_iter: int = 5000
    refine_every: int = 100
    reset_every: int = 3000
    log_every: int = 100
    log: Callable[[str], None] | None = None


def _load_image_rgb(path: Path, torch, device, target_w: int, target_h: int):
    """Load an image as a [H,W,3] float tensor in [0,1], resized to target."""

    import imageio.v3 as imageio

    array = imageio.imread(path)  # [h,w,3] or [h,w,4] uint8
    img = torch.from_numpy(array[..., :3].copy()).to(device).float() / 255.0
    h, w = img.shape[0], img.shape[1]
    if (w, h) != (target_w, target_h):
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(
            img, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        img = img.squeeze(0).permute(1, 2, 0).contiguous()
    return img  # [H,W,3]


def train_scene_gsplat(
    seed_params: dict,
    ctx: dict,
    manifest: dict,
    *,
    config: GsplatTrainConfig,
    device: str = "cuda",
):
    """Optimise seed Gaussian params with the gsplat differentiable rasterizer
    and return ``(trained_scene, history)``.

    ``seed_params`` / ``ctx`` come from either :func:`scene_to_gaussian_params`
    (train an existing AuraScene) or :func:`seed_gaussian_params_from_regions`
    (fast GPU-first seed straight from manifest points). ``history`` carries the
    loss trace and final Gaussian count (which may exceed the seed count when
    ``config.densify`` is enabled).
    """

    torch, gsplat = require_gsplat()
    from gsplat import rasterization

    log = config.log or (lambda _msg: None)
    root = Path(manifest.get("root", "."))
    frames = [f for f in manifest["frames"] if (root / f["image_path"]).exists()]
    if not frames:
        raise ValueError("manifest has no readable training frames")

    # gsplat's DefaultStrategy mutates a canonically-named ParameterDict + the
    # matching per-key optimizers IN PLACE when it grows/prunes Gaussians, so we
    # adopt gsplat's exact key names: means / scales (log) / quats (wxyz) /
    # opacities (logit) / colors. This is the single source of truth for both
    # the fixed-count and densifying paths.
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(seed_params["means"].detach().clone()),
            "scales": torch.nn.Parameter(seed_params["log_scales"].detach().clone()),
            "quats": torch.nn.Parameter(seed_params["quats"].detach().clone()),
            "opacities": torch.nn.Parameter(seed_params["logit_opacities"].detach().clone()),
            "colors": torch.nn.Parameter(seed_params["colors"].detach().clone()),
        }
    ).to(device)
    lr = {
        "means": config.position_lr,
        "scales": config.log_scale_lr,
        "quats": config.quat_lr,
        "opacities": config.opacity_lr,
        "colors": config.color_lr,
    }
    optimizers = {
        key: torch.optim.Adam([splats[key]], lr=lr[key], eps=1e-15) for key in splats
    }

    strategy = None
    strategy_state = None
    if config.densify:
        # scene_scale calibrates DefaultStrategy's absolute size thresholds; use
        # the seed point cloud's spread about its centroid.
        with torch.no_grad():
            centroid = splats["means"].mean(dim=0, keepdim=True)
            scene_scale = float((splats["means"] - centroid).norm(dim=1).mean().clamp(min=1e-3))
        strategy = gsplat.DefaultStrategy(
            grow_grad2d=config.densify_grad2d,
            refine_start_iter=config.refine_start_iter,
            refine_stop_iter=config.refine_stop_iter,
            refine_every=config.refine_every,
            reset_every=config.reset_every,
            absgrad=True,
            key_for_gradient="means2d",
            verbose=False,
        )
        strategy.check_sanity(splats, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    def render(frame, scale):
        view, k, w, h = manifest_frame_to_camera(frame, scale)
        viewmat = torch.tensor(view, dtype=torch.float32, device=device).unsqueeze(0)
        kmat = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
        out, _alpha, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=splats["colors"],
            viewmats=viewmat,
            Ks=kmat,
            width=w,
            height=h,
            packed=False,
            absgrad=bool(config.densify),
        )
        return out[0], info, w, h  # out[0]: [H,W,3]

    def _l1_ssim(rendered, gt):
        l1 = torch.abs(rendered - gt).mean()
        if config.ssim_weight <= 0:
            return l1
        ssim = _ssim(rendered, gt, torch)
        return (1 - config.ssim_weight) * l1 + config.ssim_weight * (1 - ssim)

    history = {"loss": []}
    n_frames = len(frames)
    for it in range(config.iterations):
        frame = frames[it % n_frames]
        _v, _k, w, h = manifest_frame_to_camera(frame, config.scale)
        gt = _load_image_rgb(root / frame["image_path"], torch, device, w, h)
        rendered, info, _w, _h = render(frame, config.scale)
        loss = _l1_ssim(rendered, gt)

        if strategy is not None:
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers, state=strategy_state, step=it, info=info
            )
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        if strategy is not None:
            strategy.step_post_backward(
                params=splats, optimizers=optimizers, state=strategy_state,
                step=it, info=info, packed=False,
            )

        if it % config.log_every == 0 or it == config.iterations - 1:
            history["loss"].append((it, float(loss.detach())))
            log(f"  [gsplat] iter {it + 1}/{config.iterations}  loss={float(loss.detach()):.4f}  "
                f"N={splats['means'].shape[0]}")

    trained = {
        "means": splats["means"],
        "log_scales": splats["scales"],
        "quats": splats["quats"],
        "logit_opacities": splats["opacities"],
        "colors": splats["colors"],
    }
    trained_scene = gaussian_params_to_scene(trained, ctx)
    history["final_gaussian_count"] = int(splats["means"].shape[0])
    history["seed_gaussian_count"] = len(ctx["gaussian_elements"])
    return trained_scene, history


def _ssim(a, b, torch, window: int = 11, sigma: float = 1.5):
    """Single-scale SSIM between two [H,W,3] tensors in [0,1]."""

    # [3,1,H,W]
    x = a.permute(2, 0, 1).unsqueeze(1)
    y = b.permute(2, 0, 1).unsqueeze(1)
    coords = torch.arange(window, dtype=torch.float32, device=a.device) - (window - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :])[None, None]  # [1,1,w,w]
    pad = window // 2

    def filt(t):
        return torch.nn.functional.conv2d(t, kernel, padding=pad)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = filt(x * x) - mu_x2
    sigma_y = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


def render_scene_gsplat(scene: AuraScene, frame: dict, scale: float, *, device: str = "cuda"):
    """Render a (trained) scene's Gaussians with gsplat through one manifest
    frame. Returns ``(W, H, flat_rgb_list)`` in ``[0,1]`` matching the eval
    harness's expected layout, for a training-renderer-consistent PSNR."""

    torch, gsplat = require_gsplat()
    from gsplat import rasterization

    params, _ctx = scene_to_gaussian_params(scene, device=device)
    view, k, w, h = manifest_frame_to_camera(frame, scale)
    with torch.no_grad():
        out, _alpha, _info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["log_scales"]),
            opacities=torch.sigmoid(params["logit_opacities"]),
            colors=params["colors"],
            viewmats=torch.tensor(view, dtype=torch.float32, device=device).unsqueeze(0),
            Ks=torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0),
            width=w,
            height=h,
            packed=False,
        )
    flat = out[0].clamp(0.0, 1.0).reshape(-1).cpu().tolist()
    return w, h, flat
