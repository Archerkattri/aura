"""Cross-carrier neural-residual MLP for AURA-Core (Scaffold-GS, arXiv:2312.00109).

This module implements a small differentiable MLP that takes neighboring carriers'
features (color, opacity, residual_scale, geometry centroid) as input and produces
a non-trivial residual correction to the neural-residual carrier's response.

Architecture:
  - Input:  [color_r, color_g, color_b, opacity, residual_scale, cx, cy, cz]
             per neighbor, concatenated then mean-pooled -> 8-D conditioning vector.
  - Hidden: two Linear(8->16, bias=True) -> ReLU layers.
  - Output: Linear(16->1) -> tanh -> scalar correction in (-1, 1), scaled by 0.5.

The MLP weights are real torch.nn.Parameters stored in a CrossCarrierMLP object,
and surfaced into torch_carrier_parameter_tensors via the "mlp_*" key group so
the existing optimizer can train them.

Default behavior (use_anchor_conditioning=False / no neighbors):
  - The function returns 0.0 exactly, and the neural_residual path is unchanged.
  - Existing tests that don't set use_anchor_conditioning continue to pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# MLP definition (pure torch, lazy-imported so the module is importable without torch)
# ---------------------------------------------------------------------------

_NEIGHBOR_FEATURE_DIM = 8   # [r, g, b, opacity, residual_scale, cx, cy, cz]
_HIDDEN_DIM = 16
_OUTPUT_DIM = 1


def build_cross_carrier_mlp(torch: Any, device: str) -> Any:
    """Construct a small 2-layer MLP for cross-carrier residual correction.

    The network is an ``nn.Module`` so its parameters can be discovered by
    ``torch.nn.Module.parameters()`` / named_parameters().  However we also
    expose the individual weight / bias tensors as plain tensors in the
    carrier_parameters dict so the existing SGD / Adam paths in
    torch_optimizer.py can train them without knowing about nn.Module.

    Returns an ``nn.Sequential`` on the requested device.
    """
    import torch as _torch
    import torch.nn as _nn

    model = _nn.Sequential(
        _nn.Linear(_NEIGHBOR_FEATURE_DIM, _HIDDEN_DIM, bias=True),
        _nn.ReLU(),
        _nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM, bias=True),
        _nn.ReLU(),
        _nn.Linear(_HIDDEN_DIM, _OUTPUT_DIM, bias=True),
        _nn.Tanh(),
    )
    model = model.to(device)
    # Xavier init so the MLP is non-trivially different for different inputs
    for m in model.modules():
        if isinstance(m, _nn.Linear):
            _nn.init.xavier_uniform_(m.weight)
            _nn.init.zeros_(m.bias)
    return model


def mlp_parameter_tensors_from_module(
    torch: Any,
    mlp: Any,
    *,
    requires_grad: bool = True,
) -> Dict[str, Any]:
    """Extract named weight/bias tensors from the MLP module into a flat dict.

    Keys: ``mlp_w0``, ``mlp_b0``, ``mlp_w1``, ``mlp_b1``, ``mlp_w2``, ``mlp_b2``.
    These are the *same* tensor objects (not copies), so gradients computed
    through the module accumulate on them — and any update via the carrier_parameters
    dict is immediately visible in the module.
    """
    params: Dict[str, Any] = {}
    layer_idx = 0
    for m in mlp.modules():
        if type(m).__name__ == "Linear":
            t = m.weight
            if requires_grad and not t.requires_grad:
                t.requires_grad_(True)
            params[f"mlp_w{layer_idx}"] = t
            t = m.bias
            if requires_grad and not t.requires_grad:
                t.requires_grad_(True)
            params[f"mlp_b{layer_idx}"] = t
            layer_idx += 1
    return params


def build_neighbor_feature_vector(
    torch: Any,
    neighbor_colors: Any,      # (N, 3) float32
    neighbor_opacities: Any,   # (N,)   float32
    neighbor_residuals: Any,   # (N,)   float32
    neighbor_centroids: Any,   # (N, 3) float32
    device: str,
) -> Any:
    """Build the mean-pooled 8-D conditioning vector for the MLP.

    Concatenates [color(3), opacity(1), residual_scale(1), centroid(3)] = 8
    per neighbor then takes the mean across neighbors.  This is differentiable
    end-to-end with respect to all inputs.
    """
    # neighbor_colors: (N, 3), neighbor_opacities: (N, 1), etc.
    N = neighbor_colors.shape[0]
    op = neighbor_opacities.view(N, 1)
    rs = neighbor_residuals.view(N, 1)
    # Each row: [r, g, b, opacity, residual_scale, cx, cy, cz]
    feats = torch.cat([neighbor_colors, op, rs, neighbor_centroids], dim=1)  # (N, 8)
    return feats.mean(dim=0)  # (8,)


def cross_carrier_residual_correction(
    torch: Any,
    mlp: Any,
    neighbor_colors: Any,
    neighbor_opacities: Any,
    neighbor_residuals: Any,
    neighbor_centroids: Any,
    device: str,
) -> Any:
    """Run MLP forward pass and return a scalar correction in (-0.5, 0.5).

    This is the REAL cross-carrier computation: output depends on neighbor
    features and MLP weights, and is differentiable w.r.t. both.

    Returns a scalar tensor (shape []) with requires_grad=True (if the MLP
    params do).
    """
    feat = build_neighbor_feature_vector(
        torch, neighbor_colors, neighbor_opacities, neighbor_residuals,
        neighbor_centroids, device,
    )  # (8,)
    raw = mlp(feat.unsqueeze(0))  # (1, 1) — tanh output in (-1, 1)
    correction = raw.squeeze() * 0.5  # scale to (-0.5, 0.5)
    return correction


def neighbor_features_from_carrier_parameters(
    torch: Any,
    neighbor_elements: Sequence[Any],
    carrier_parameters: Optional[Dict[str, Dict[str, Any]]],
    device: str,
) -> Optional[Tuple[Any, Any, Any, Any]]:
    """Extract (colors, opacities, residuals, centroids) tensors from neighbors.

    Returns None when no neighbors are available (triggers no-op / default path).
    """
    if not neighbor_elements:
        return None

    colors_list: List[Any] = []
    opacities_list: List[Any] = []
    residuals_list: List[Any] = []
    centroids_list: List[Any] = []

    for elem in neighbor_elements:
        eparams = (carrier_parameters or {}).get(elem.id, {})

        # Color
        if "color" in eparams:
            c = eparams["color"].detach().float().to(device)
        else:
            c = torch.tensor(list(elem.color)[:3], dtype=torch.float32, device=device)
        if c.shape[0] < 3:
            c = torch.cat([c, torch.zeros(3 - c.shape[0], device=device)])
        colors_list.append(c[:3])

        # Opacity
        if "opacity" in eparams:
            o = eparams["opacity"].detach().float().to(device).view(())
        else:
            o = torch.tensor(float(elem.opacity), dtype=torch.float32, device=device)
        opacities_list.append(o.view(1).squeeze())

        # Residual scale (default 0 for non-neural carriers)
        if "residual_scale" in eparams:
            r = eparams["residual_scale"].detach().float().to(device).view(())
        else:
            r = torch.tensor(
                float(elem.payload.get("residual_scale", 0.0))
                if hasattr(elem, "payload") else 0.0,
                dtype=torch.float32, device=device,
            )
        residuals_list.append(r.view(1).squeeze())

        # Centroid
        bounds = elem.bounds
        cx = (float(bounds.min_corner[0]) + float(bounds.max_corner[0])) / 2.0
        cy = (float(bounds.min_corner[1]) + float(bounds.max_corner[1])) / 2.0
        cz = (float(bounds.min_corner[2]) + float(bounds.max_corner[2])) / 2.0
        centroids_list.append(torch.tensor([cx, cy, cz], dtype=torch.float32, device=device))

    colors = torch.stack(colors_list)        # (N, 3)
    opacities = torch.stack(opacities_list)  # (N,)
    residuals = torch.stack(residuals_list)  # (N,)
    centroids = torch.stack(centroids_list)  # (N, 3)
    return colors, opacities, residuals, centroids
