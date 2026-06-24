#!/usr/bin/env python3
"""Cross-family carrier routing — the open post-3DGS thesis, tested cleanly.

The truck ablation showed adaptive β-routing within the Beta family does NOT beat a
good global β (smooth scene). The real question is whether *different kernel
FAMILIES* matter for *different content*: Gabor (oscillatory) for high-frequency
texture, Gaussian/Beta (smooth bump) for smooth regions.

This isolates that question as a 2D fit (no multi-view / pose confound): fit real
image crops with PRISM at a MATCHED carrier budget under {gaussian, beta, gabor,
mix}, on HIGH-FREQUENCY crops vs SMOOTH crops. Hypothesis: gabor/mix beat gaussian
on high-freq content and tie on smooth content — i.e. carrier family matters when
(and only when) the content has structure a single family can't capture compactly.
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch

from aura.prism import (
    project_gaussians, quats_scales_to_cov3d, composite,
    gaussian_footprint, beta_footprint, gabor_footprint,
)


def dispatch_footprint(dx, dy, conic, torch_, *, ftype=0, freq=None, phase=0.0):
    if int(ftype) == 2:      # gabor
        return gabor_footprint(dx, dy, conic, torch_, freq=freq, phase=phase)
    if int(ftype) == 1:      # beta
        return beta_footprint(dx, dy, conic, torch_, beta=2.0)
    return gaussian_footprint(dx, dy, conic, torch_)


def load_crop(img_path, x0, y0, size):
    from eval_psnr import load_jpg_as_rgb
    w, h, flat = load_jpg_as_rgb(str(img_path))
    a = np.array(flat, "float32").reshape(h, w, 3)
    return a[y0:y0 + size, x0:x0 + size]


def highfreq_energy(crop):
    g = crop.mean(2)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return float(gx + gy)


def fit(target, mode, n_carriers, iters, device, seed):
    """Fit `target` [H,W,3] with n_carriers under a carrier mode. Returns PSNR."""
    torch.manual_seed(seed)
    H, W, _ = target.shape
    tgt = torch.tensor(target, dtype=torch.float32, device=device)
    g = int(math.sqrt(n_carriers))
    n = g * g
    # carriers on a frontal grid at z=2; camera at origin, +z forward
    lin = torch.linspace(-1, 1, g, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    means = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.full((n,), 2.0, device=device)], 1)
    means = means.clone().requires_grad_(True)
    logscale = torch.full((n, 3), math.log(0.06), device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1).clone().requires_grad_(True)
    logit_o = torch.full((n,), 0.5, device=device, requires_grad=True)
    colors = torch.rand(n, 3, device=device, requires_grad=True)
    focal = W / 2.0
    K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1.0]], device=device)
    vm = torch.eye(4, device=device)

    params = [means, logscale, quats, logit_o, colors]
    freq = phase = None
    if mode in ("gabor", "mix"):
        freq = (torch.rand(n, 2, device=device) * 0.4 + 0.1).requires_grad_(True)
        phase = (torch.rand(n, device=device) * 6.28).requires_grad_(True)
        params += [freq, phase]
    if mode == "gaussian":
        ftypes = torch.zeros(n, dtype=torch.long, device=device)
    elif mode == "beta":
        ftypes = torch.ones(n, dtype=torch.long, device=device)
    elif mode == "gabor":
        ftypes = torch.full((n,), 2, dtype=torch.long, device=device)
    else:  # mix: gabor on high-gradient grid cells, gaussian elsewhere
        gv = torch.tensor(np.abs(np.gradient(target.mean(2))[0]) +
                          np.abs(np.gradient(target.mean(2))[1]), device=device)
        # sample the local gradient at each carrier's grid centre
        cidx = (((means.detach()[:, :2] + 1) / 2) * torch.tensor([W - 1, H - 1.0], device=device)).long().clamp(0)
        local = gv[cidx[:, 1].clamp(0, H - 1), cidx[:, 0].clamp(0, W - 1)]
        thr = torch.quantile(local, 0.6)
        ftypes = torch.where(local >= thr, torch.full_like(local, 2).long(), torch.zeros_like(local).long())

    opt = torch.optim.Adam(params, lr=0.03)
    for it in range(iters):
        cov = quats_scales_to_cov3d(quats, torch.exp(logscale), torch)
        proj = project_gaussians(means, cov, vm, K, W, H, torch)
        extra = {"ftype": ftypes[proj.index]}
        if freq is not None:
            extra["freq"] = freq[proj.index]; extra["phase"] = phase[proj.index]
        img = composite(proj, colors.clamp(0, 1), torch.sigmoid(logit_o), W, H, torch,
                        footprint=dispatch_footprint, footprint_extra=extra)
        loss = ((img - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mse = float(((img.clamp(0, 1) - tgt) ** 2).mean())
    return 10 * math.log10(1.0 / mse) if mse > 0 else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "data/tanks/truck/images/000001.jpg"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    # a few crops; we label each by high-frequency energy
    crops = {
        "A(0,0)": (0, 0), "B(600,200)": (600, 200), "C(300,400)": (300, 400),
        "D(900,500)": (900, 500), "E(1200,300)": (1200, 300),
    }
    results = []
    for name, (x0, y0) in crops.items():
        crop = load_crop(a.image, x0, y0, a.size)
        if crop.shape[0] < a.size or crop.shape[1] < a.size:
            continue
        hf = highfreq_energy(crop)
        row = {"crop": name, "hf": round(hf, 4)}
        for mode in ("gaussian", "beta", "gabor", "mix"):
            row[mode] = round(fit(crop, mode, a.n, a.iters, a.device, seed=0), 2)
        row["gabor-gauss"] = round(row["gabor"] - row["gaussian"], 2)
        row["mix-gauss"] = round(row["mix"] - row["gaussian"], 2)
        results.append(row)
        print(row, flush=True)

    results.sort(key=lambda r: r["hf"])
    print("\n=== sorted by high-frequency energy (low → high) ===")
    print(f"{'crop':12} {'hf':>7} {'gauss':>7} {'beta':>7} {'gabor':>7} {'mix':>7} {'gab-gau':>8} {'mix-gau':>8}")
    for r in results:
        print(f"{r['crop']:12} {r['hf']:>7} {r['gaussian']:>7} {r['beta']:>7} {r['gabor']:>7} {r['mix']:>7} {r['gabor-gauss']:>8} {r['mix-gauss']:>8}")


if __name__ == "__main__":
    main()
