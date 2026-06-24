"""Binary carrier sidecar for `.aura` packages — fast tensor I/O.

The JSON `elements.json` stores one dict per carrier, which does not scale: a
3.4M-carrier scene takes ~22 min to load (pure-Python JSON + 3.4M AuraElement
objects). For the train -> render/eval loop the renderers only need the carrier
*tensors* (means/scales/quats/opacity/colour/SH + per-carrier PRISM footprint),
not the full asset object graph.

This module writes those tensors as a single compressed `carriers.npz` next to
the package (or anywhere) and loads them back in well under a second per million
carriers. The full `.aura` JSON remains the asset-contract format; this is the
fast path for rendering/eval/iteration.
"""

from __future__ import annotations

from pathlib import Path

CARRIERS_NPZ = "carriers.npz"


def save_carriers(
    path,
    *,
    means,            # [N,3]
    scales,           # [N,3] (linear, not log)
    quats,            # [N,4] wxyz (normalised)
    opacity,          # [N]   in [0,1]
    colors=None,      # [N,3] flat linear RGB (when sh_degree == 0)
    sh=None,          # [N,K,3] SH coefficients (when sh_degree > 0)
    sh_degree=0,
    ftypes=None,      # [N] int PRISM footprint codes (optional)
    freq=None,        # [N,2] gabor freq (optional)
    phase=None,       # [N]   gabor phase (optional)
    beta=None,        # [N]   Beta-kernel shape (Deformable Beta Splatting; optional)
    sb=None,          # [N,L,6] spherical-Beta view-dependent colour lobes (optional)
    confidence=None,  # [N]   per-carrier confidence in [0,1] (optional)
):
    """Write carrier tensors to ``<path>/carriers.npz`` (path may be a package
    dir or a file path). Accepts torch tensors or numpy arrays."""
    import numpy as np

    def _np(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype="float32")

    out = Path(path)
    target = out / CARRIERS_NPZ if out.is_dir() or not out.suffix else out
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "means": _np(means), "scales": _np(scales), "quats": _np(quats),
        "opacity": _np(opacity), "sh_degree": np.int64(sh_degree),
    }
    if sh is not None:
        data["sh"] = _np(sh)
    if colors is not None:
        data["colors"] = _np(colors)
    if ftypes is not None:
        data["ftypes"] = (ftypes.detach().cpu().numpy() if hasattr(ftypes, "detach") else np.asarray(ftypes)).astype("int64")
    if freq is not None:
        data["freq"] = _np(freq)
    if phase is not None:
        data["phase"] = _np(phase)
    if beta is not None:
        data["beta"] = _np(beta)
    if sb is not None:
        data["sb"] = _np(sb)
    if confidence is not None:
        data["confidence"] = _np(confidence)
    np.savez(target, **data)
    return target


def has_carriers(path) -> bool:
    p = Path(path)
    return (p / CARRIERS_NPZ).exists() if p.is_dir() else p.suffix == ".npz" and p.exists()


def load_carriers(path, *, device="cuda"):
    """Load carrier tensors from a ``carriers.npz`` (or package dir). Returns a
    dict with torch tensors: means/scales/quats/opacity (+ colors or sh +
    sh_degree, + ftypes/freq/phase if present). Returns None if absent."""
    import numpy as np
    import torch

    p = Path(path)
    f = p / CARRIERS_NPZ if p.is_dir() else p
    if not f.exists():
        return None
    z = np.load(f)
    t = lambda k: torch.from_numpy(z[k]).to(device)
    out = {
        "means": t("means"), "scales": t("scales"), "quats": t("quats"),
        "opacity": t("opacity"), "sh_degree": int(z["sh_degree"]),
    }
    if "sh" in z:
        out["sh"] = t("sh")
    if "colors" in z:
        out["colors"] = t("colors")
    if "ftypes" in z:
        out["ftypes"] = torch.from_numpy(z["ftypes"]).to(device).long()
    if "freq" in z:
        out["freq"] = t("freq")
    if "phase" in z:
        out["phase"] = t("phase")
    if "beta" in z:
        out["beta"] = t("beta")
    if "sb" in z:
        out["sb"] = t("sb")
    if "confidence" in z:
        out["confidence"] = t("confidence")
    return out


def render_carriers_gsplat(carriers, frame, scale, *, device="cuda"):
    """Render loaded carrier tensors with gsplat through one manifest frame.
    Returns (W, H, flat_rgb). The fast path for the train->eval loop — no
    package/scene round-trip."""
    import torch
    from gsplat import rasterization
    from .gsplat_renderer import manifest_frame_to_camera

    view, k, w, h = manifest_frame_to_camera(frame, scale)
    vm = torch.tensor(view, dtype=torch.float32, device=device).unsqueeze(0)
    K = torch.tensor(k, dtype=torch.float32, device=device).unsqueeze(0)
    shd = int(carriers.get("sh_degree", 0))
    colors = carriers["sh"] if "sh" in carriers else carriers["colors"]
    with torch.no_grad():
        out, _, _ = rasterization(
            means=carriers["means"], quats=carriers["quats"], scales=carriers["scales"],
            opacities=carriers["opacity"], colors=colors, viewmats=vm, Ks=K, width=w, height=h,
            sh_degree=(shd if shd and shd > 0 else None),
        )
    return w, h, out[0].clamp(0, 1).reshape(-1).cpu().tolist()


def carriers_from_params(params, *, sh_degree=0, ftypes=None, freq=None, phase=None):
    """Build the save_carriers kwargs from the canonical training param tensors
    (means, log_scales, quats, logit_opacities, colors-or-SH)."""
    import torch

    kw = dict(
        means=params["means"],
        scales=torch.exp(params["log_scales"]),
        quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True).clamp(min=1e-12),
        opacity=torch.sigmoid(params["logit_opacities"]),
        sh_degree=sh_degree,
        ftypes=ftypes, freq=freq, phase=phase,
    )
    c = params["colors"]
    if c.dim() == 3:
        kw["sh"] = c
    else:
        kw["colors"] = c.clamp(0, 1)
    return kw
