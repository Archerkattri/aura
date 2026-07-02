#!/usr/bin/env python3
"""P0 pruning-sweep visual: calibrated-confidence vs opacity carrier pruning.

Renders ONE held-out view under a keep-fraction sweep (100% -> 10%), side by
side: LEFT keeps the highest **calibrated confidence** carriers, RIGHT keeps the
highest **opacity** carriers (the engine pruning default). Each side is labeled
with the keep-%, the rendered PSNR vs the ground-truth image, and -- the P0
property -- the mean held-out **reliability** of the kept carriers.

HONEST FINDING (verified 2026-07-02, this is NOT the naive "opacity destroys the
scene" story; that story does not hold and cannot):

  * Rendered PSNR is dominated by OPACITY, because opacity is exactly the
    alpha-compositing blend weight -- keeping the highest-opacity carriers keeps
    the pixels you actually see, so the opacity render stays clean at every
    budget. (This is why opacity pruning is the 3DGS standard.)
  * The P0 killer property is a different axis: the *reliability* of the carriers
    you keep. Calibrated confidence keeps the carriers that agree with held-out
    observations -- retained reliability ~0.90 at a 10%-keep budget vs opacity's
    ~0.50 (at/below random). Opacity keeps good-looking-but-unreliable carriers
    and ships NO reliability guarantee; that is exactly the gap P0 fills.

So the figure tells the true, more useful story: opacity gives you a clean render
with no trust; calibrated confidence gives you a certified, budget-controllable
reliability -- a capability a bare 3DGS/DBS splat does not have.

The calibrator is fit EXACTLY as experiments/calibrate_confidence.py (isotonic
PAVA on the train-view colour-agreement feature, disjoint half of the labeled
carriers, seed 0), then applied to every carrier to rank the full set.

Usage (accuracy job -- safe on shared GPUs, see gpu-usage-policy):
    OMP_NUM_THREADS=2 .gpu_venv/bin/python experiments/make_pruning_sweep_gif.py \
        --scene room --frame 8
Outputs: assets/pruning_sweep.gif + assets/pruning_30pct.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# dataviz-validated P0 palette (consistent with assets/p0_selection_auc.png).
CAL = "#2a78d6"      # calibrated confidence (AURA)
OPAC = "#eb6834"     # opacity (engine default)
ORACLE = "#6f6d67"   # oracle ceiling reference (neutral)
INK = "#0b0b0b"
MUTED = "#6b6963"
SURFACE = "#fbfbf9"
METER_BG = "#e6e5df"

SCENES = {
    "room": ("room-gsplat", "room-manifest"),
    "truck": ("truck-sidecar", "truck-pts129k-manifest"),
    "garden": ("garden-gsplat", "garden-manifest"),
    "kitchen": ("kitchen-gsplat", "kitchen-manifest"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room", choices=list(SCENES))
    ap.add_argument("--frame", type=int, default=8, help="manifest frame index (held-out view)")
    ap.add_argument("--scale", type=float, default=0.5, help="render intrinsics scale")
    ap.add_argument("--steps", type=int, default=8, help="keep-fraction steps 100%->10%")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.25)
    ap.add_argument("--out-gif", default=str(REPO / "assets" / "pruning_sweep.gif"))
    ap.add_argument("--out-png", default=str(REPO / "assets" / "pruning_30pct.png"))
    a = ap.parse_args()

    import numpy as np
    import torch
    import imageio.v3 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    sys.path.insert(0, str(REPO / "src"))
    from aura.gsplat_renderer import manifest_frame_to_camera
    from aura.calibration import IsotonicConfidenceCalibrator
    from gsplat import rasterization

    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aura_name, manifest_name = SCENES[a.scene]

    # --- carriers (index-aligned with reliability_<scene>.npz) ---
    car = dict(np.load(REPO / f"outputs/{aura_name}.aura/carriers.npz"))
    means = torch.tensor(car["means"], dtype=torch.float32, device=dev)
    scales = torch.tensor(car["scales"], dtype=torch.float32, device=dev)
    quats = torch.tensor(car["quats"], dtype=torch.float32, device=dev)
    opacity_np = np.clip(car["opacity"], 0.0, 1.0)
    opac = torch.tensor(opacity_np, dtype=torch.float32, device=dev)
    colors = torch.tensor(np.clip(car["colors"], 0.0, 1.0), dtype=torch.float32, device=dev)
    N = means.shape[0]

    # --- reliability + calibrator (identical protocol to calibrate_confidence.py) ---
    rel = np.load(REPO / f"outputs/reliability_{a.scene}.npz")
    labeled = rel["labeled"]
    train_agree = rel["train_agree"]
    reliability = rel["reliability"]
    feat = train_agree[labeled]
    rl = reliability[labeled]
    m = feat.shape[0]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(m)
    half = m // 2
    cal_idx = perm[:half]
    calib = IsotonicConfidenceCalibrator().fit(feat[cal_idx], rl[cal_idx])
    conf_all = calib.predict(train_agree)  # calibrated confidence, every carrier

    conf_t = torch.tensor(conf_all, dtype=torch.float32, device=dev)
    order_conf = torch.argsort(conf_t, descending=True)
    order_opac = torch.argsort(opac, descending=True)

    # oracle reliability ceiling over the labeled set (frame-independent).
    def retained_reliability(order_np, k):
        kept = order_np[:k]
        lab = labeled[kept]
        return float(reliability[kept][lab].mean()) if lab.any() else float("nan")

    order_conf_np = order_conf.cpu().numpy()
    order_opac_np = order_opac.cpu().numpy()
    order_oracle_np = np.argsort(-reliability)  # keep true-highest-reliability first

    # --- the held-out view + GT ---
    manifest = json.load(open(REPO / f"outputs/{manifest_name}.json"))
    root = REPO / manifest["root"]
    frame = manifest["frames"][a.frame]
    view, k, W, H = manifest_frame_to_camera(frame, a.scale)
    vm = torch.tensor(view, dtype=torch.float32, device=dev).unsqueeze(0)
    K = torch.tensor(k, dtype=torch.float32, device=dev).unsqueeze(0)

    gt_img = iio.imread(root / frame["image_path"])[..., :3]
    gt = torch.from_numpy(gt_img.copy()).to(dev).float() / 255.0
    if (gt.shape[1], gt.shape[0]) != (W, H):
        gt = torch.nn.functional.interpolate(
            gt.permute(2, 0, 1).unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False)[0].permute(1, 2, 0).contiguous()

    def render(idx_tensor):
        with torch.no_grad():
            out, _, _ = rasterization(
                means=means[idx_tensor], quats=quats[idx_tensor], scales=scales[idx_tensor],
                opacities=opac[idx_tensor], colors=colors[idx_tensor],
                viewmats=vm, Ks=K, width=W, height=H)
        return out[0].clamp(0.0, 1.0)

    def psnr(img):
        mse = torch.mean((img - gt) ** 2).item()
        return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))

    def to_np(img):
        return (img.cpu().numpy() * 255.0).astype("uint8")

    fracs = np.linspace(1.0, 0.1, a.steps)

    # Precompute every frame's renders + metrics.
    frames_data = []
    print(f"scene={a.scene} frame={a.frame} carriers={N} view={W}x{H} labeled={int(labeled.sum())}")
    print(f"{'keep%':>6} {'conf_PSNR':>10} {'opac_PSNR':>10} {'conf_rel':>9} {'opac_rel':>9} {'oracle_rel':>10}")
    for f in fracs:
        kk = max(1, int(round(f * N)))
        rc = render(order_conf[:kk]); ro = render(order_opac[:kk])
        img_c, img_o = to_np(rc), to_np(ro)
        p_c, p_o = psnr(rc), psnr(ro)
        rel_c = retained_reliability(order_conf_np, kk)
        rel_o = retained_reliability(order_opac_np, kk)
        rel_oracle = retained_reliability(order_oracle_np, kk)
        frames_data.append(dict(frac=float(f), keep=kk, img_c=img_c, img_o=img_o,
                                psnr_c=p_c, psnr_o=p_o, rel_c=rel_c, rel_o=rel_o,
                                rel_oracle=rel_oracle))
        print(f"{int(round(f*100)):>5}% {p_c:>10.2f} {p_o:>10.2f} {rel_c:>9.3f} {rel_o:>9.3f} {rel_oracle:>10.3f}")

    # ---------- shared frame renderer (matplotlib) ----------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
    })

    def draw_panel(ax_img, ax_meter, img, title, color, psnr_v, rel_v, rel_oracle):
        ax_img.clear(); ax_meter.clear()
        ax_img.imshow(img); ax_img.set_xticks([]); ax_img.set_yticks([])
        for s in ax_img.spines.values():
            s.set_color(color); s.set_linewidth(2.0)
        ax_img.set_title(title, fontsize=10.5, color=INK, pad=5, fontweight="bold")
        # PSNR (honest secondary metric) top-left on the image.
        ax_img.text(0.015, 0.965, f"render PSNR  {psnr_v:.1f} dB", transform=ax_img.transAxes,
                    ha="left", va="top", fontsize=8.2, color="white",
                    bbox=dict(boxstyle="round,pad=0.22", fc=(0, 0, 0, 0.55), ec="none"))
        # reliability meter (the P0 property, hero metric).
        ax_meter.set_xlim(0, 1); ax_meter.set_ylim(0, 1)
        ax_meter.set_xticks([]); ax_meter.set_yticks([])
        for s in ax_meter.spines.values():
            s.set_visible(False)
        ax_meter.add_patch(plt.Rectangle((0, 0.30), 1.0, 0.40, fc=METER_BG, ec="none"))
        ax_meter.add_patch(plt.Rectangle((0, 0.30), max(0.0, rel_v), 0.40, fc=color, ec="none"))
        ax_meter.axvline(rel_oracle, 0.18, 0.82, color=ORACLE, lw=1.4, ls=(0, (3, 2)))
        ax_meter.text(0.0, 0.86, "kept-carrier reliability", ha="left", va="bottom",
                      fontsize=7.8, color=MUTED)
        ax_meter.text(1.0, 0.86, f"{rel_v:.2f}", ha="right", va="bottom",
                      fontsize=9.5, color=INK, fontweight="bold")

    fig = plt.figure(figsize=(8.0, 4.15), dpi=100)
    axL = fig.add_axes([0.020, 0.300, 0.470, 0.520])
    axR = fig.add_axes([0.510, 0.300, 0.470, 0.520])
    mL = fig.add_axes([0.035, 0.150, 0.440, 0.075])
    mR = fig.add_axes([0.525, 0.150, 0.440, 0.075])
    sup = fig.text(0.5, 0.95, "", ha="center", va="top", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.5, 0.058,
             "Opacity preserves the render (it is the blend weight) but keeps unreliable carriers;",
             ha="center", va="center", fontsize=7.6, color=MUTED)
    fig.text(0.5, 0.034,
             "calibrated confidence keeps the most reliable carriers (bar tracks the dashed oracle ceiling).",
             ha="center", va="center", fontsize=7.6, color=MUTED)
    fig.text(0.5, 0.011, "Held-out view;  render PSNR vs ground truth.",
             ha="center", va="center", fontsize=6.8, color=MUTED)

    def render_frame(fd):
        pct = int(round(fd["frac"] * 100))
        sup.set_text(f"Carrier pruning  —  keep {pct}%  ({fd['keep']:,} of {N:,} carriers)")
        draw_panel(axL, mL, fd["img_c"], "Calibrated confidence (AURA)", CAL,
                   fd["psnr_c"], fd["rel_c"], fd["rel_oracle"])
        draw_panel(axR, mR, fd["img_o"], "Opacity (engine default)", OPAC,
                   fd["psnr_o"], fd["rel_o"], fd["rel_oracle"])

    # ---------- GIF (PillowWriter) ----------
    out_gif = Path(a.out_gif); out_gif.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=a.fps)
    # hold the first and last frames a touch longer for readability.
    seq = [frames_data[0]] + frames_data + [frames_data[-1], frames_data[-1]]
    with writer.saving(fig, str(out_gif), dpi=100):
        for fd in seq:
            render_frame(fd)
            writer.grab_frame()
    size_mb = out_gif.stat().st_size / 1e6
    print(f"wrote {out_gif}  ({size_mb:.2f} MB, {len(seq)} frames, {W*2}x{H} render px)")

    # ---------- static 3-panel PNG (full / confidence@30% / opacity@30%) ----------
    k30 = max(1, int(round(0.30 * N)))
    full_img = to_np(render(torch.arange(N, device=dev)))
    full_p = psnr(render(torch.arange(N, device=dev)))
    c30 = to_np(render(order_conf[:k30])); c30_p = psnr(render(order_conf[:k30]))
    o30 = to_np(render(order_opac[:k30])); o30_p = psnr(render(order_opac[:k30]))
    rel_c30 = retained_reliability(order_conf_np, k30)
    rel_o30 = retained_reliability(order_opac_np, k30)

    fig2 = plt.figure(figsize=(9.6, 3.2), dpi=150)
    fig2.suptitle(f"AURA P0 — pruning to 30% of carriers ({a.scene}, held-out view)",
                  fontsize=12, color=INK, fontweight="bold", y=1.02)
    specs = [
        ("Full  (100%)", full_img, INK, f"PSNR {full_p:.1f} dB", None),
        ("Calibrated confidence  @30%", c30, CAL, f"PSNR {c30_p:.1f} dB", rel_c30),
        ("Opacity  @30%", o30, OPAC, f"PSNR {o30_p:.1f} dB", rel_o30),
    ]
    for i, (title, img, color, psnr_txt, rel_v) in enumerate(specs):
        ax = fig2.add_axes([0.006 + i * 0.333, 0.10, 0.323, 0.80])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(2.2)
        ax.set_title(title, fontsize=10, color=INK, pad=4, fontweight="bold")
        label = psnr_txt if rel_v is None else f"{psnr_txt}   •   kept reliability {rel_v:.2f}"
        ax.text(0.02, 0.03, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.0, color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.55), ec="none"))
    out_png = Path(a.out_png)
    fig2.savefig(out_png, bbox_inches="tight", facecolor=SURFACE, dpi=150)
    print(f"wrote {out_png}  (30%-keep: conf PSNR {c30_p:.1f}/rel {rel_c30:.2f}  "
          f"vs opac PSNR {o30_p:.1f}/rel {rel_o30:.2f})")


if __name__ == "__main__":
    main()
