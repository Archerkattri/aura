"""Render-loss per-carrier reliability label for the P2 head-to-head study.

P0's reliability label (``per_carrier_reliability.py``) is a **colour-agreement
proxy**: it compares each carrier's stored colour to the median GT colour where the
carrier *projects* into held-out views. It never renders, so it is occlusion-blind
(an interior carrier samples whatever occludes it) and it ignores how much the
carrier actually contributes to the image. P2 replaces the proxy with a label
measured from the **actual alpha-composited render** on held-out views.

Method (exact blend-weight attribution). The rendered image is *linear* in the
carrier colours: ``R_p = Σ_i w_{i,p} c_i`` where ``w_{i,p} = α_i T_i`` is carrier
``i``'s alpha-compositing blend weight at pixel ``p`` (``T_i`` = transmittance in
front — so occluded/low-opacity carriers get ``w ≈ 0`` for free). Because ``R`` is
linear in ``c``, ``∂(Σ_p R_p·f_p)/∂c_i = Σ_p w_{i,p} f_p`` is EXACT for any per-pixel
weight ``f`` that does not depend on ``c_i``. Three colour backward passes per
held-out view therefore recover, per carrier, exactly:

    W_i   = Σ_p w_{i,p}                (total blend weight / visibility)
    A_i   = Σ_p w_{i,p} GT_p           (blend-weighted GT colour it paints)
    B_i   = Σ_p w_{i,p} GT_p²          (blend-weighted GT colour²)

and the carrier's blend-weighted rendering **squared error** decomposes in closed
form (no first-order / ablation approximation):

    SE_i  = Σ_ch (c_{i,ch}² W_i − 2 c_{i,ch} A_{i,ch} + B_{i,ch})
    dist_i = sqrt(SE_i / W_i)          # blend-weighted RMS colour distance (L2)
    reliability_i = exp(−β · dist_i)   # same functional form as the colour label

This is a faithful per-carrier attribution of the rendered L1/L2 error, weighted by
each carrier's real contribution to held-out views — a carrier that paints the
wrong colour *where it is actually visible* scores low; an occluded or invisible
carrier (``W_i`` below a floor) is unlabelled. It is the P0 colour label upgraded to
use the true alpha composite instead of an occlusion-blind projection.

(An earlier first-order gate-gradient ``ΔL_i ≈ −∂L/∂g_i`` was tried and rejected: in
an over-complete splat scene individual carriers are highly redundant, so the
first-order leave-out delta is dominated by noise and does not track true finite
ablation — see the P2 doc. The blend-weight attribution above is exact, not
first-order, because the render is genuinely linear in colour.)

Features (export-time ``train_agree`` colour agreement, the view-count ``raw_conf``
heuristic, ``opacity``) and the ``labeled`` carrier set are copied verbatim from the
P0 colour npz for the SAME carriers, so ``calibrate_confidence.py`` runs identically
and the only thing that changes head-to-head is the label. Accuracy job, safe on
shared GPUs (gpu-usage-policy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def weighted_error_reliability(se, w, *, beta: float = 4.0, weight_floor: float = 1e-4):
    """Pure reliability rule (GPU-free, the reference the CUDA loop implements).

    ``se`` = per-carrier blend-weighted squared colour error ``Σ_ch(c²W − 2cA + B)``
    summed over held-out views; ``w`` = per-carrier total blend weight ``Σ W_i``.
    Returns ``(reliability[N], labeled[N])`` where
    ``reliability_i = exp(−β·sqrt(se_i/w_i))`` (the blend-weighted RMS L2 colour
    distance squashed exactly like the colour label) and ``labeled`` marks carriers
    whose accumulated visibility clears ``weight_floor`` (occluded / invisible
    carriers are unlabelled, not mislabelled)."""
    import numpy as np

    se = np.asarray(se, dtype="float64")
    w = np.asarray(w, dtype="float64")
    labeled = w > weight_floor
    dist = np.zeros_like(w)
    safe = labeled
    dist[safe] = np.sqrt(np.clip(se[safe], 0.0, None) / w[safe])
    reliability = np.where(labeled, np.exp(-beta * dist), 0.0)
    return reliability.astype("float32"), labeled


def build_render_loss_labels(
    carriers_path: str,
    manifest_path: str,
    color_npz: str,
    out: str,
    *,
    holdout: int = 8,
    beta: float = 4.0,
    weight_floor: float = 1e-4,
    device: str = "cuda",
    scale: float = 1.0,
    prune_check: bool = True,
    prune_fracs=(0.1, 0.2),
    seed: int = 0,
) -> dict:
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from aura.gsplat_renderer import manifest_frame_to_camera
    from gsplat import rasterization
    import imageio.v3 as imageio

    torch.set_num_threads(2)
    dev = device if torch.cuda.is_available() else "cpu"

    z = dict(np.load(Path(carriers_path)))
    means = torch.tensor(z["means"], dtype=torch.float32, device=dev)
    quats = torch.tensor(z["quats"], dtype=torch.float32, device=dev)
    scales = torch.tensor(z["scales"], dtype=torch.float32, device=dev)
    opacity = torch.tensor(np.clip(z["opacity"], 1e-6, 1.0), dtype=torch.float32, device=dev)
    sh_degree = int(z["sh_degree"]) if "sh_degree" in z else 0
    if "colors" in z:
        base_colors = torch.tensor(np.clip(z["colors"], 0, 1), dtype=torch.float32, device=dev)
    else:  # SH: use DC term as flat colour for the linear-in-colour attribution
        sh = z["sh"]
        base_colors = torch.tensor(np.clip(0.5 + 0.2820948 * sh[:, 0, :], 0, 1),
                                   dtype=torch.float32, device=dev)
    shd = sh_degree if sh_degree and sh_degree > 0 else None
    n = means.shape[0]

    manifest = json.load(open(manifest_path))
    root_raw = Path(manifest.get("root", "."))
    mparent = Path(manifest_path).resolve().parent
    _bases = [root_raw, Path.cwd() / root_raw, mparent / root_raw, mparent, Path.cwd()]
    frames = manifest["frames"]
    test_frames = [f for i, f in enumerate(frames) if i % holdout == 0]

    def _img_path(fr):
        p = Path(fr["image_path"])
        for base in _bases:
            if (base / p).exists():
                return base / p
        return root_raw / p

    def camera(fr):
        view, k, w, h = manifest_frame_to_camera(fr, scale)
        vm = torch.tensor(view, dtype=torch.float32, device=dev).unsqueeze(0)
        K = torch.tensor(k, dtype=torch.float32, device=dev).unsqueeze(0)
        return vm, K, w, h

    def load_gt(fr, w, h):
        img = imageio.imread(_img_path(fr))
        gt = torch.tensor(img[..., :3].copy(), dtype=torch.float32, device=dev) / 255.0
        if (gt.shape[1], gt.shape[0]) != (w, h):
            gt = torch.nn.functional.interpolate(
                gt.permute(2, 0, 1).unsqueeze(0), size=(h, w),
                mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).contiguous()
        return gt

    # --- exact blend-weight attribution: W, A, B accumulated over held-out views.
    # The render is linear in colour, so each of these is an EXACT gradient. We
    # recompute the (deterministic) forward per weighting rather than retain one big
    # graph, keeping peak memory at a single forward+backward (fits the shared GPUs).
    W = torch.zeros(n, device=dev)          # Σ_p w_{i,p}
    A = torch.zeros(n, 3, device=dev)       # Σ_p w_{i,p} GT_p
    Bacc = torch.zeros(n, 3, device=dev)    # Σ_p w_{i,p} GT_p^2
    per_view_l1 = []

    def _weighted_grad(vm, K, w, h, pixel_weight):
        """∂(Σ_p Σ_ch img[p,ch]·pixel_weight[p,ch])/∂colp  == Σ_p w_{i,p}·pixel_weight
        (exact, since img is linear in colp). pixel_weight is [H,W,3] or a scalar."""
        colp = base_colors.clone().requires_grad_(True)
        img, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacity,
            colors=colp, viewmats=vm, Ks=K, width=w, height=h, sh_degree=shd)
        (img[0] * pixel_weight).sum().backward()
        g = colp.grad.detach().clone()
        del img, colp
        return g

    for fr in test_frames:
        vm, K, w, h = camera(fr)
        gt = load_gt(fr, w, h)
        ones = torch.ones_like(gt)
        with torch.no_grad():
            img, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacity,
                colors=base_colors, viewmats=vm, Ks=K, width=w, height=h, sh_degree=shd)
            per_view_l1.append(float((img[0] - gt).abs().mean()))
            del img
        W += _weighted_grad(vm, K, w, h, ones)[:, 0]     # W_i (channel-independent)
        A += _weighted_grad(vm, K, w, h, gt)             # A_{i,ch}
        Bacc += _weighted_grad(vm, K, w, h, gt * gt)     # B_{i,ch}

    c = base_colors                                     # [N,3]
    se = (c * c * W[:, None] - 2.0 * c * A + Bacc).sum(dim=1)   # SE_i
    W_np = W.cpu().numpy().astype("float64")
    se_np = se.cpu().numpy().astype("float64")
    reliability, active = weighted_error_reliability(
        se_np, W_np, beta=beta, weight_floor=weight_floor)

    # --- copy P0 features + labeled set for the SAME carriers ---------------
    cd = dict(np.load(color_npz))
    if cd["raw_conf"].shape[0] != n:
        raise SystemExit(f"carrier count mismatch: color npz {cd['raw_conf'].shape[0]} vs carriers {n}")
    labeled = cd["labeled"] & active     # P0 colour label's >=min_obs AND render-visible

    out_d = dict(
        raw_conf=cd["raw_conf"].astype("float32"),
        train_agree=cd["train_agree"].astype("float32"),
        reliability=reliability,                 # render-loss (blend-weighted) reliability
        blend_weight=W_np.astype("float32"),
        weighted_sq_err=se_np.astype("float32"),
        opacity=z["opacity"].astype("float32"),
        heldout_obs=cd["heldout_obs"] if "heldout_obs" in cd else active.astype("int32"),
        train_obs=cd["train_obs"] if "train_obs" in cd else np.zeros(n, "int32"),
        labeled=labeled,
        label=np.array("render_loss"),
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **out_d)

    lm = labeled
    summary = {
        "carriers": int(n),
        "label": "render_loss",
        "method": "exact_blend_weighted_color_error",
        "test_views": len(test_frames),
        "beta": beta,
        "mean_per_view_L1": round(float(np.mean(per_view_l1)), 5),
        "labeled_fraction": round(float(lm.mean()), 4),
        "reliability_mean_labeled": round(float(reliability[lm].mean()), 4),
        "reliability_std_labeled": round(float(reliability[lm].std()), 4),
        "corr_trainagree_reliability": round(float(
            np.corrcoef(cd["train_agree"][lm], reliability[lm])[0, 1]), 4),
        "corr_opacity_reliability": round(float(
            np.corrcoef(z["opacity"][lm], reliability[lm])[0, 1]), 4),
        "corr_rawconf_reliability": round(float(
            np.corrcoef(cd["raw_conf"][lm], reliability[lm])[0, 1]), 4),
        "corr_colorlabel_reliability": round(float(
            np.corrcoef(cd["reliability"][lm], reliability[lm])[0, 1]), 4),
        "out": out,
    }

    if prune_check:
        summary["directional_prune"] = directional_prune_check(
            means, quats, scales, opacity, base_colors, shd, test_frames, camera,
            load_gt, reliability, labeled, z["opacity"], fracs=prune_fracs, seed=seed,
            dev=dev, rasterization=rasterization, torch=torch, np=np)

    print(json.dumps(summary, indent=2))
    return summary


def directional_prune_check(means, quats, scales, opacity, colors, shd, test_frames,
                            camera, load_gt, reliability, labeled, opacity_np, *,
                            fracs, seed, dev, rasterization, torch, np):
    """Utility validation with TRUE finite ablation on held-out views: prune the
    least-reliable carriers by the render-loss label and by opacity, and report the
    held-out render L1 at each budget. A good render-loss label should let you drop
    the least-reliable carriers with less render-quality loss than dropping the
    lowest-opacity carriers (and much less than dropping the MOST-reliable)."""
    views = test_frames
    cams, gts = [], []
    for fr in views:
        vm, K, w, h = camera(fr)
        cams.append((vm, K, w, h))
        gts.append(load_gt(fr, w, h))

    def l1(op):
        tot = 0.0
        with torch.no_grad():
            for (vm, K, w, h), gt in zip(cams, gts):
                img, _, _ = rasterization(
                    means=means, quats=quats, scales=scales, opacities=op,
                    colors=colors, viewmats=vm, Ks=K, width=w, height=h, sh_degree=shd)
                tot += float((img[0] - gt).abs().mean())
        return tot / len(views)

    n = means.shape[0]
    base = l1(opacity)
    rng = np.random.default_rng(seed)
    rel_order = np.argsort(reliability)           # least reliable first
    opa_order = np.argsort(opacity_np)            # lowest opacity first
    rows = []
    for f in fracs:
        k = int(f * n)
        drop_rel = torch.tensor(rel_order[:k], device=dev)
        drop_rel_top = torch.tensor(rel_order[-k:], device=dev)
        drop_opa = torch.tensor(opa_order[:k], device=dev)
        drop_rnd = torch.tensor(rng.permutation(n)[:k], device=dev)
        r = {}
        for name, idx in (("drop_least_reliable", drop_rel),
                          ("drop_most_reliable", drop_rel_top),
                          ("drop_lowest_opacity", drop_opa),
                          ("drop_random", drop_rnd)):
            op = opacity.clone()
            op[idx] = 1e-6
            r[name] = round(l1(op), 5)
        rows.append({"prune_frac": f, **r})
    return {"base_L1": round(base, 5), "budgets": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carriers", required=True, help=".aura dir or carriers.npz")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--color-npz", required=True,
                    help="P0 colour reliability npz (features + labeled set reused)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=int, default=8)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="render resolution for the label (1.0=native; use the "
                         "carrier training scale for memory-capped scenes, e.g. garden 0.5)")
    ap.add_argument("--no-prune-check", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cp = Path(a.carriers)
    cpath = cp / "carriers.npz" if cp.is_dir() else cp
    build_render_loss_labels(
        str(cpath), a.manifest, a.color_npz, a.out, holdout=a.holdout,
        beta=a.beta, scale=a.scale, device=a.device, prune_check=not a.no_prune_check)


if __name__ == "__main__":
    main()
