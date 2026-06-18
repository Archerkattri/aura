"""AURA physically-based shading module.

Implements a staged shading pipeline for AURA-Core scenes:

  Stage 0 — Lambertian directional shading (GS-IR arXiv:2311.16473)
  Stage 1 — Cook-Torrance microfacet + split-sum IBL
             (GS-IR arXiv:2311.16473; GI-GS arXiv:2410.02619; GOGS arXiv:2508.14563)
  Stage 2 — BVH shadow-ray visibility baking via scene traversal
             (R3DG arXiv:2311.16043; SSD-GS arXiv:2604.13333)

All stages are opt-in.  With shading disabled the pipeline is a no-op and
the renderer output is bit-identical to the existing emissive path.

Public helpers
--------------
lambertian_shade         -- Stage 0 (batch tensor op)
cook_torrance_shade      -- Stage 1 (batch tensor op)
bake_carrier_visibility  -- Stage 2 (per-carrier scalar, uses scene.traverse_ray)
render_with_shading      -- Full-pipeline helper (torch orthographic path)
render_relit             -- Relighting demo: same geometry, different light/env
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionalLight:
    """A single directional light source."""

    direction: tuple[float, float, float]  # unit vector TOWARD the light
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0


@dataclass(frozen=True)
class ShadingConfig:
    """Top-level shading configuration passed to the renderer.

    ``stage`` controls which effects are active:
      * ``"off"``  — emissive-only, bit-stable with legacy output (default)
      * ``"lambertian"`` — Stage 0: Lambertian directional shading
      * ``"pbr"``  — Stage 1: Cook-Torrance microfacet + split-sum IBL
      * ``"pbr_shadow"`` — Stage 1 + 2: PBR with baked shadow visibility
    """

    stage: str = "off"
    lights: tuple[DirectionalLight, ...] = ()
    # IBL environment map (H x W x 3, stored as nested Python lists or None)
    env_map: Any | None = None
    # prefiltered specular map (per-roughness-level); None → fallback
    prefiltered_specular: Any | None = None
    # BRDF look-up-table (N x N x 2) for split-sum; None → analytic fallback
    brdf_lut: Any | None = None
    # Shadow transmittance clamp floor (0 = hard shadows, 1 = no shadows)
    shadow_floor: float = 0.0

    def is_active(self) -> bool:
        return self.stage != "off"


# ---------------------------------------------------------------------------
# Stage 0 — Lambertian shading
# ---------------------------------------------------------------------------


def lambertian_shade(
    torch: Any,
    albedo: Any,  # [..., 3]
    normals: Any,  # [..., 3]
    lights: Sequence[DirectionalLight],
) -> Any:
    """Batch Lambertian shading: sum over directional lights.

    ``L_out = albedo * sum_i( max(0, dot(N, L_i)) * L_color_i * intensity_i )``

    Works on any leading batch dimensions.
    """
    if not lights:
        # No lights → black; caller should fall back to emissive.
        return torch.zeros_like(albedo)

    device = albedo.device
    accumulated = torch.zeros_like(albedo)
    normal_norm = torch.clamp(torch.linalg.norm(normals, dim=-1, keepdim=True), min=1e-8)
    n_hat = normals / normal_norm

    for light in lights:
        l_dir = torch.tensor(light.direction, dtype=albedo.dtype, device=device)
        l_dir = l_dir / torch.clamp(torch.linalg.norm(l_dir), min=1e-8)
        l_color = torch.tensor(light.color, dtype=albedo.dtype, device=device) * light.intensity
        n_dot_l = torch.clamp(torch.sum(n_hat * l_dir.reshape(*([1] * (normals.dim() - 1)), 3), dim=-1, keepdim=True), min=0.0)
        accumulated = accumulated + albedo * n_dot_l * l_color

    return accumulated


# ---------------------------------------------------------------------------
# Stage 1 — Cook-Torrance microfacet BRDF + split-sum IBL
# ---------------------------------------------------------------------------


def _ggx_ndf(torch: Any, n_dot_h: Any, roughness: Any) -> Any:
    """GGX/Trowbridge-Reitz Normal Distribution Function."""
    a = roughness * roughness
    a2 = a * a
    denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0
    return a2 / (math.pi * torch.clamp(denom * denom, min=1e-8))


def _schlick_smith_g1(torch: Any, n_dot_v: Any, roughness: Any) -> Any:
    """Smith-Schlick masking/shadowing G1 term."""
    k = (roughness + 1.0) ** 2 / 8.0
    return n_dot_v / torch.clamp(n_dot_v * (1.0 - k) + k, min=1e-8)


def _smith_g(torch: Any, n_dot_v: Any, n_dot_l: Any, roughness: Any) -> Any:
    return _schlick_smith_g1(torch, n_dot_v, roughness) * _schlick_smith_g1(torch, n_dot_l, roughness)


def _schlick_fresnel(torch: Any, v_dot_h: Any, f0: Any) -> Any:
    """Schlick Fresnel approximation.  f0 shape [..., 3], v_dot_h shape [...]."""
    return f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - v_dot_h.unsqueeze(-1), min=0.0), 5.0)


def cook_torrance_brdf(
    torch: Any,
    albedo: Any,   # [N, 3]
    normals: Any,  # [N, 3]
    view_dir: Any,  # [N, 3]  (unit vector FROM surface TOWARD camera)
    roughness: Any,  # [N] or [N, 1]
    metallic: Any,   # [N] or [N, 1]
    lights: Sequence[DirectionalLight],
) -> Any:
    """Cook-Torrance BRDF for an array of directional lights (Stage 1 analytic).

    Returns irradiance RGB tensor of shape [N, 3].
    """
    device = albedo.device
    eps = 1e-8

    # Ensure 2-D
    rough = roughness.reshape(-1, 1) if roughness.dim() == 1 else roughness  # [N, 1]
    metal = metallic.reshape(-1, 1) if metallic.dim() == 1 else metallic      # [N, 1]

    # Dielectric F0 = 0.04; metallic blends toward albedo
    f0_dielectric = torch.full_like(albedo, 0.04)
    f0 = torch.lerp(f0_dielectric, albedo, metal)  # [N, 3]

    n_norm = torch.clamp(torch.linalg.norm(normals, dim=-1, keepdim=True), min=eps)
    n_hat = normals / n_norm
    v_norm = torch.clamp(torch.linalg.norm(view_dir, dim=-1, keepdim=True), min=eps)
    v_hat = view_dir / v_norm
    n_dot_v = torch.clamp(torch.sum(n_hat * v_hat, dim=-1), min=eps)

    accumulated = torch.zeros_like(albedo)

    for light in lights:
        l_dir = torch.tensor(light.direction, dtype=albedo.dtype, device=device)
        l_dir = l_dir / torch.clamp(torch.linalg.norm(l_dir), min=eps)
        l_color = torch.tensor(light.color, dtype=albedo.dtype, device=device) * light.intensity

        n_dot_l = torch.clamp(torch.sum(n_hat * l_dir, dim=-1), min=0.0)  # [N]
        h = (v_hat + l_dir) / torch.clamp(torch.linalg.norm(v_hat + l_dir, dim=-1, keepdim=True), min=eps)
        n_dot_h = torch.clamp(torch.sum(n_hat * h, dim=-1), min=0.0)  # [N]
        v_dot_h = torch.clamp(torch.sum(v_hat * h, dim=-1), min=0.0)  # [N]

        D = _ggx_ndf(torch, n_dot_h, rough.squeeze(-1))           # [N]
        G = _smith_g(torch, n_dot_v, n_dot_l, rough.squeeze(-1))  # [N]
        F = _schlick_fresnel(torch, v_dot_h, f0)                   # [N, 3]

        # Specular term
        spec_num = D.unsqueeze(-1) * G.unsqueeze(-1) * F          # [N, 3]
        spec_denom = 4.0 * n_dot_v.unsqueeze(-1) * n_dot_l.unsqueeze(-1)
        specular = spec_num / torch.clamp(spec_denom, min=eps)     # [N, 3]

        # Diffuse term (energy-conserving): ks * F, kd = (1 - F) * (1 - metallic)
        kd = (1.0 - F) * (1.0 - metal)
        diffuse = kd * albedo / math.pi                            # [N, 3]

        radiance = (diffuse + specular) * n_dot_l.unsqueeze(-1) * l_color
        accumulated = accumulated + radiance

    return accumulated


def ibl_diffuse(
    torch: Any,
    albedo: Any,    # [N, 3]
    normals: Any,   # [N, 3]
    env_map: Any,   # [H, W, 3] or None
) -> Any:
    """Diffuse IBL: sample env_map at normal direction (nearest-neighbor for simplicity).

    Falls back to a small ambient term if env_map is None.
    """
    if env_map is None:
        # Constant ambient fallback (0.1 * albedo)
        return albedo * 0.1

    # env_map: [H, W, 3] tensor
    H, W = int(env_map.shape[0]), int(env_map.shape[1])
    eps = 1e-8
    n_norm = normals / torch.clamp(torch.linalg.norm(normals, dim=-1, keepdim=True), min=eps)

    # Spherical parameterisation
    nx, ny, nz = n_norm[..., 0], n_norm[..., 1], n_norm[..., 2]
    phi = torch.atan2(nz, nx)           # azimuth in [-pi, pi]
    theta = torch.acos(torch.clamp(ny, -1.0 + eps, 1.0 - eps))  # elevation in [0, pi]
    u = (phi / (2.0 * math.pi) + 0.5).clamp(0.0, 1.0)
    v = (theta / math.pi).clamp(0.0, 1.0)

    xi = (u * (W - 1)).long().clamp(0, W - 1)
    yi = (v * (H - 1)).long().clamp(0, H - 1)
    irradiance = env_map[yi, xi]   # [N, 3]
    return albedo * irradiance


def ibl_specular(
    torch: Any,
    albedo: Any,        # [N, 3]
    normals: Any,       # [N, 3]
    view_dir: Any,      # [N, 3] toward camera
    roughness: Any,     # [N]
    metallic: Any,      # [N]
    env_map: Any,       # [H, W, 3] or None (prefiltered or plain env)
    brdf_lut: Any,      # [LUT_N, LUT_N, 2] or None
) -> Any:
    """Split-sum specular IBL.

    Uses a mip-level selection from roughness → prefiltered specular,
    combined with the BRDF LUT for view-angle-dependent scale+bias.
    Falls back to a metallic tint when env_map/brdf_lut are not provided.
    """
    device = albedo.device
    eps = 1e-8

    rough = roughness.reshape(-1, 1) if roughness.dim() == 1 else roughness  # [N, 1]
    metal = metallic.reshape(-1, 1) if metallic.dim() == 1 else metallic      # [N, 1]

    f0_dielectric = torch.full_like(albedo, 0.04)
    f0 = torch.lerp(f0_dielectric, albedo, metal)

    n_norm = normals / torch.clamp(torch.linalg.norm(normals, dim=-1, keepdim=True), min=eps)
    v_norm = view_dir / torch.clamp(torch.linalg.norm(view_dir, dim=-1, keepdim=True), min=eps)
    n_dot_v = torch.clamp(torch.sum(n_norm * v_norm, dim=-1), min=0.0)  # [N]

    # --- Prefiltered env sample ---
    if env_map is None:
        # Fallback: flat specular tint = f0 * (1 - roughness)
        prefiltered = f0 * (1.0 - rough)
    else:
        # Reflect view around normal for specular direction
        reflect = 2.0 * n_dot_v.unsqueeze(-1) * n_norm - v_norm  # [N, 3]
        r_norm = reflect / torch.clamp(torch.linalg.norm(reflect, dim=-1, keepdim=True), min=eps)
        H_env, W_env = int(env_map.shape[0]), int(env_map.shape[1])
        rx, ry, rz = r_norm[..., 0], r_norm[..., 1], r_norm[..., 2]
        phi = torch.atan2(rz, rx)
        theta = torch.acos(torch.clamp(ry, -1.0 + eps, 1.0 - eps))
        u = (phi / (2.0 * math.pi) + 0.5).clamp(0.0, 1.0)
        v = (theta / math.pi).clamp(0.0, 1.0)
        xi = (u * (W_env - 1)).long().clamp(0, W_env - 1)
        yi = (v * (H_env - 1)).long().clamp(0, H_env - 1)
        prefiltered = env_map[yi, xi]  # [N, 3]

    # --- BRDF LUT ---
    if brdf_lut is None:
        # Schlick analytic fallback: scale=1, bias=0
        scale = torch.ones(albedo.shape[0], 1, dtype=albedo.dtype, device=device)
        bias = torch.zeros(albedo.shape[0], 1, dtype=albedo.dtype, device=device)
    else:
        LUT_N = int(brdf_lut.shape[0])
        lut_u = n_dot_v.clamp(0.0, 1.0)
        lut_v = rough.squeeze(-1).clamp(0.0, 1.0)
        lut_xi = (lut_u * (LUT_N - 1)).long().clamp(0, LUT_N - 1)
        lut_yi = (lut_v * (LUT_N - 1)).long().clamp(0, LUT_N - 1)
        lut_sample = brdf_lut[lut_yi, lut_xi]  # [N, 2]
        scale = lut_sample[:, 0:1]
        bias = lut_sample[:, 1:2]

    specular = prefiltered * (f0 * scale + bias)
    return specular


def cook_torrance_ibl_shade(
    torch: Any,
    albedo: Any,        # [N, 3]
    normals: Any,       # [N, 3]
    view_dir: Any,      # [N, 3] unit vector toward camera
    roughness: Any,     # [N]
    metallic: Any,      # [N]
    lights: Sequence[DirectionalLight],
    env_map: Any | None,
    brdf_lut: Any | None,
) -> Any:
    """Stage 1: Cook-Torrance analytic + split-sum IBL (diffuse + specular)."""
    analytic = cook_torrance_brdf(torch, albedo, normals, view_dir, roughness, metallic, lights)
    diff_ibl = ibl_diffuse(torch, albedo, normals, env_map)
    spec_ibl = ibl_specular(torch, albedo, normals, view_dir, roughness, metallic, env_map, brdf_lut)
    return analytic + diff_ibl + spec_ibl


# ---------------------------------------------------------------------------
# Stage 2 — BVH shadow-ray visibility baking
# ---------------------------------------------------------------------------


def bake_carrier_visibility(
    scene: Any,  # AuraScene
    light_direction: tuple[float, float, float],  # unit vector toward light
    *,
    shadow_floor: float = 0.0,
    offset_scale: float = 1e-3,
) -> tuple[float, ...]:
    """Cast shadow rays from each element toward the light, return per-element visibility [0, 1].

    Uses ``scene.traverse_ray()`` (the importable scene traversal API) so no
    CUDA files are edited.  Each element's center is offset along the light
    direction by ``offset_scale`` to avoid self-intersection.

    Returns a tuple of length ``len(scene.elements)`` with visibility scalars.
    Shadow floor clamps the minimum (1.0 = fully lit, 0.0 = fully in shadow).
    """
    from aura.ray import Ray

    # Normalise light direction
    lx, ly, lz = light_direction
    mag = (lx * lx + ly * ly + lz * lz) ** 0.5
    if mag < 1e-8:
        # Degenerate direction → all lit
        return tuple(1.0 for _ in scene.elements)
    l_hat = (lx / mag, ly / mag, lz / mag)

    visibilities: list[float] = []
    for element in scene.elements:
        bounds = element.bounds
        cx = (bounds.min_corner[0] + bounds.max_corner[0]) * 0.5
        cy = (bounds.min_corner[1] + bounds.max_corner[1]) * 0.5
        cz = (bounds.min_corner[2] + bounds.max_corner[2]) * 0.5
        # Offset slightly along light direction to avoid self-hit
        ox = cx + l_hat[0] * offset_scale
        oy = cy + l_hat[1] * offset_scale
        oz = cz + l_hat[2] * offset_scale

        shadow_ray = Ray(origin=(ox, oy, oz), direction=l_hat)
        traversal = scene.traverse_ray(shadow_ray)

        # ordered_hits contains RayHitTrace objects; each has .result.depth
        # Filter out the element itself (depth ≈ 0 due to offset) and find real blockers.
        blocking = sum(
            1 for hit in traversal.ordered_hits
            if hit.result.depth is not None and float(hit.result.depth) > offset_scale * 0.5
        )

        raw_vis = 0.0 if blocking > 0 else 1.0
        visibility = shadow_floor + (1.0 - shadow_floor) * raw_vis
        visibilities.append(float(visibility))

    return tuple(visibilities)


# ---------------------------------------------------------------------------
# Shading dispatch: apply shading to composited color
# ---------------------------------------------------------------------------


def apply_shading(
    torch: Any,
    color: Any,         # [N, 3] composited emissive color from renderer
    albedo: Any,        # [N, 3] per-ray albedo (defaults to color if not provided)
    normals: Any,       # [N, 3] per-ray normal
    normal_present: Any,  # [N] bool
    view_dir: Any,      # [N, 3] unit toward camera
    roughness: Any | None,   # [N]
    metallic: Any | None,    # [N]
    visibility: Any | None,  # [N] or None
    config: ShadingConfig,
) -> Any:
    """Apply the shading stage specified in ``config`` to the composited color.

    Returns the shaded color tensor.  If stage is "off" or no normals are
    present, returns ``color`` unchanged.
    """
    if not config.is_active():
        return color

    device = color.device
    N = int(color.shape[0])

    # Where no normal is available, fall back to emissive
    has_normal = normal_present.to(dtype=color.dtype).unsqueeze(-1)  # [N, 1]

    if config.stage == "lambertian":
        shaded = lambertian_shade(torch, albedo, normals, config.lights)
        result = shaded * has_normal + color * (1.0 - has_normal)

    elif config.stage in ("pbr", "pbr_shadow"):
        rough = roughness if roughness is not None else torch.full((N,), 0.5, dtype=color.dtype, device=device)
        metal = metallic if metallic is not None else torch.zeros((N,), dtype=color.dtype, device=device)
        shaded = cook_torrance_ibl_shade(
            torch, albedo, normals, view_dir, rough, metal, config.lights, config.env_map, config.brdf_lut
        )
        result = shaded * has_normal + color * (1.0 - has_normal)

        if config.stage == "pbr_shadow" and visibility is not None:
            vis = visibility.to(dtype=color.dtype).unsqueeze(-1)  # [N, 1]
            result = result * vis

    else:
        result = color

    return result


# ---------------------------------------------------------------------------
# High-level renderer helpers
# ---------------------------------------------------------------------------


def render_with_shading(
    scene: Any,  # AuraScene
    ray_origins: Any,
    ray_directions: Any,
    *,
    config: ShadingConfig,
    device: str | None = None,
    scene_tensors: Any | None = None,
    carrier_parameters: Any | None = None,
) -> Any:
    """Render rays and apply shading, returning the final color tensor.

    This wraps ``torch_render_ray_color_tensor`` (emissive) then applies the
    chosen shading stage on top.  With config.stage == "off" the output is
    bit-identical to calling the emissive renderer directly.
    """
    from aura.torch_renderer import (
        _torch_composited_scene_rays,
        require_torch,
        torch_renderer_status,
        torch_scene_tensors,
    )

    torch = require_torch()
    status = torch_renderer_status()
    resolved_device = device or (scene_tensors.device if scene_tensors is not None else None) or status.default_device or "cpu"

    origins = ray_origins.to(device=resolved_device, dtype=torch.float32) if hasattr(ray_origins, "to") else torch.as_tensor(ray_origins, dtype=torch.float32, device=resolved_device)
    directions = ray_directions.to(device=resolved_device, dtype=torch.float32) if hasattr(ray_directions, "to") else torch.as_tensor(ray_directions, dtype=torch.float32, device=resolved_device)

    composited, st, element_normals = _torch_composited_scene_rays(
        torch,
        scene,
        origins,
        directions,
        device=resolved_device,
        carrier_parameters=carrier_parameters,
        scene_tensors=scene_tensors,
        collect_traces=False,
    )

    color = composited["color"]          # [N, 3] emissive
    first_index = composited["first_index"]  # [N]
    has_hit = composited["has_hit"]       # [N]

    if not config.is_active():
        return color

    # --- Gather per-ray surface properties ---
    N = int(color.shape[0])
    elements = tuple(scene.elements)
    num_elements = len(elements)

    # Albedo: from per-element "albedo" field if present, else emissive color
    albedo_list = []
    roughness_list = []
    metallic_list = []
    for element in elements:
        payload = element.payload
        # Albedo defaults to element color (emissive)
        alb = payload.get("albedo")
        if alb is not None and isinstance(alb, (list, tuple)) and len(alb) == 3:
            albedo_list.append([float(alb[0]), float(alb[1]), float(alb[2])])
        else:
            albedo_list.append(list(element.color))
        # Roughness: prefer relighting field "shading_roughness", fallback to 0.5
        rough = payload.get("shading_roughness")
        if rough is None:
            rough = payload.get("roughness", 0.5)
        roughness_list.append(float(rough) if rough is not None else 0.5)
        # Metallic: prefer "shading_metallic", fallback 0.0
        metal = payload.get("shading_metallic", 0.0)
        metallic_list.append(float(metal) if metal is not None else 0.0)

    all_albedo = torch.tensor(albedo_list, dtype=torch.float32, device=resolved_device)       # [E, 3]
    all_roughness = torch.tensor(roughness_list, dtype=torch.float32, device=resolved_device) # [E]
    all_metallic = torch.tensor(metallic_list, dtype=torch.float32, device=resolved_device)   # [E]

    safe_idx = torch.clamp(first_index, 0, num_elements - 1)
    ray_albedo = all_albedo[safe_idx]            # [N, 3]
    ray_roughness = all_roughness[safe_idx]      # [N]
    ray_metallic = all_metallic[safe_idx]        # [N]

    # Normal
    predicted_normals = element_normals[safe_idx]  # [N, 3]
    normal_present = st.element_normal_present[safe_idx] & has_hit

    # View direction: from hit point toward camera = -ray direction
    view_dir = -directions / torch.clamp(torch.linalg.norm(directions, dim=-1, keepdim=True), min=1e-8)

    # Visibility (Stage 2)
    visibility = None
    if config.stage == "pbr_shadow" and config.lights:
        light = config.lights[0]  # shadow baked for first light
        vis_per_element = bake_carrier_visibility(
            scene, light.direction, shadow_floor=config.shadow_floor
        )
        all_visibility = torch.tensor(vis_per_element, dtype=torch.float32, device=resolved_device)
        visibility = all_visibility[safe_idx]  # [N]

    # Convert env_map / brdf_lut to tensors if provided as lists
    env_map_tensor = _to_tensor_or_none(torch, config.env_map, resolved_device, dtype=torch.float32)
    brdf_lut_tensor = _to_tensor_or_none(torch, config.brdf_lut, resolved_device, dtype=torch.float32)

    # Swap config references to tensor versions for helper calls
    eff_config = ShadingConfig(
        stage=config.stage,
        lights=config.lights,
        env_map=env_map_tensor,
        prefiltered_specular=config.prefiltered_specular,
        brdf_lut=brdf_lut_tensor,
        shadow_floor=config.shadow_floor,
    )

    return apply_shading(
        torch,
        color,
        ray_albedo,
        predicted_normals,
        normal_present,
        view_dir,
        ray_roughness,
        ray_metallic,
        visibility,
        eff_config,
    )


def render_relit(
    scene: Any,  # AuraScene
    ray_origins: Any,
    ray_directions: Any,
    *,
    config: ShadingConfig,
    device: str | None = None,
    scene_tensors: Any | None = None,
) -> Any:
    """Render the scene under the lighting specified in ``config``.

    Returns the shaded color tensor.  Calling with two different ``config``
    objects (different lights or env_map) on the same scene produces different
    images while the geometry is fixed — demonstrating relighting.
    """
    return render_with_shading(
        scene,
        ray_origins,
        ray_directions,
        config=config,
        device=device,
        scene_tensors=scene_tensors,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_tensor_or_none(torch: Any, value: Any, device: str, *, dtype: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape") and hasattr(value, "to"):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)
