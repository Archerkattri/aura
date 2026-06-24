"""Per-carrier confidence — multi-view observation support.

The capability contract calls for a *confidence* per carrier (no mainstream 3DGS
method exposes one). A principled, training-data-grounded signal: **how many
posed views actually observe the carrier**. A carrier that projects inside many
camera frusta (in front of each) is supported by lots of evidence; one seen by a
single grazing view is speculative. We count in-frustum, in-front projections
across the manifest's cameras and squash the count into [0,1].

This is geometric observation support — orthogonal to (and combinable with)
opacity. It is cheap: O(N · V) projections, GPU-batched over carriers per view.
"""
from __future__ import annotations


def multiview_confidence(carriers, manifest, *, scale=1.0, device="cuda", saturate=12):
    """Return per-carrier confidence [N] in [0,1] = 1 - exp(-views_observing/saturate).

    `views_observing[i]` counts manifest frames whose camera sees carrier i
    (projects within the image and lies in front of the camera). `saturate` is the
    view count at which confidence reaches ~0.63 (1-1/e); ~3·saturate → ~0.95.
    """
    import torch
    from .gsplat_renderer import manifest_frame_to_camera

    means = carriers["means"].to(device)             # [N,3]
    n = means.shape[0]
    counts = torch.zeros(n, dtype=torch.float32, device=device)
    ones = torch.ones(n, 3, dtype=torch.float32, device=device)
    homog = torch.cat([means, ones[:, :1]], dim=1)    # [N,4]

    for fr in manifest["frames"]:
        view, k, w, h = manifest_frame_to_camera(fr, scale)
        vm = torch.tensor(view, dtype=torch.float32, device=device)   # [4,4] world->cam
        K = torch.tensor(k, dtype=torch.float32, device=device)       # [3,3]
        cam = (homog @ vm.T)[:, :3]                   # [N,3] camera-space
        z = cam[:, 2]
        infront = z > 1e-4
        zc = torch.clamp(z, min=1e-4)
        u = K[0, 0] * cam[:, 0] / zc + K[0, 2]
        v = K[1, 1] * cam[:, 1] / zc + K[1, 2]
        inview = infront & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        counts += inview.float()

    conf = 1.0 - torch.exp(-counts / float(saturate))
    return conf


def attach_confidence(carriers, manifest, *, scale=1.0, device="cuda", saturate=12):
    """Return a shallow copy of `carriers` with a `confidence` tensor added."""
    out = dict(carriers)
    out["confidence"] = multiview_confidence(
        carriers, manifest, scale=scale, device=device, saturate=saturate)
    return out
