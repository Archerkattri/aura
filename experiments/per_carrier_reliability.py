"""Produce a per-carrier held-out reliability signal for the P0 calibration study.

Reliability label = does a carrier's stored appearance agree with what INDEPENDENT
held-out views actually observe? For each carrier we project its centre into every
held-out (test-split) camera, sample the ground-truth pixel colour where it lands
in-front and in-frame, and compare the robust (median) observed colour to the
carrier's own colour. Carriers that sit on real, consistently-observed surface
agree; floaters / mis-placed / rarely-seen carriers disagree. This is the honest,
GPU-cheap reliability signal the P0 calibration + certificate consume
(docs/P0_CALIBRATED_CONFIDENCE.md).

Caveat (documented, not hidden): a single projected pixel carries the COMPOSITED
scene colour, so an occluded carrier can disagree even if well-placed. Averaging a
robust statistic over many held-out views suppresses this; we also require a
minimum number of held-out observations before a carrier is labelled.

Accuracy job, not a timing job — safe to run on shared GPUs (see gpu-usage-policy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aura", default="outputs/truck-sidecar.aura",
                    help=".aura package dir with carriers.npz")
    ap.add_argument("--manifest", default="outputs/truck-pts129k-manifest.json")
    ap.add_argument("--holdout", type=int, default=8,
                    help="every Nth frame is held out (llffhold convention)")
    ap.add_argument("--min-obs", type=int, default=3,
                    help="min held-out observations to label a carrier")
    ap.add_argument("--beta", type=float, default=4.0,
                    help="reliability = exp(-beta * colour L2 distance)")
    ap.add_argument("--conf-saturate", type=float, default=12.0)
    ap.add_argument("--out", default="outputs/reliability_truck.npz")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from aura.gsplat_renderer import manifest_frame_to_camera

    torch.set_num_threads(2)  # good neighbour on a shared box
    dev = a.device if torch.cuda.is_available() else "cpu"

    carriers = dict(np.load(Path(a.aura) / "carriers.npz"))
    means = torch.tensor(carriers["means"], dtype=torch.float32, device=dev)
    colors = torch.tensor(np.clip(carriers["colors"], 0, 1), dtype=torch.float32, device=dev)
    opacity = np.clip(carriers["opacity"], 0, 1)
    n = means.shape[0]
    homog = torch.cat([means, torch.ones(n, 1, device=dev)], dim=1)  # [N,4]

    manifest = json.load(open(a.manifest))
    root = Path(manifest.get("root", Path(a.manifest).parent))
    if not root.is_absolute():
        root = (Path(a.manifest).resolve().parent / root)
    # Resolve root robustly: image_path is relative to the scene root.
    frames = manifest["frames"]
    test_frames = [f for i, f in enumerate(frames) if i % a.holdout == 0]
    train_frames = [f for i, f in enumerate(frames) if i % a.holdout != 0]

    def _img_path(fr):
        p = Path(fr["image_path"])
        for base in (root, Path("data/tanks/truck"), Path(a.manifest).resolve().parent):
            if (base / p).exists():
                return base / p
        return root / p

    import imageio.v3 as imageio

    # --- held-out observed colours: accumulate sum + count + sum of squares ---
    obs_sum = torch.zeros(n, 3, device=dev)
    obs_cnt = torch.zeros(n, device=dev)
    # For a robust centre we keep a running set via mean; median would need all
    # obs stored ([N, V, 3]). With <~40 test views we can afford the full stack.
    stacks: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    def project(fr):
        view, k, w, h = manifest_frame_to_camera(fr, 1.0)
        vm = torch.tensor(view, dtype=torch.float32, device=dev)
        K = torch.tensor(k, dtype=torch.float32, device=dev)
        cam = (homog @ vm.T)[:, :3]
        z = cam[:, 2]
        infront = z > 1e-4
        zc = torch.clamp(z, min=1e-4)
        u = K[0, 0] * cam[:, 0] / zc + K[0, 2]
        v = K[1, 1] * cam[:, 1] / zc + K[1, 2]
        inview = infront & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return u, v, w, h, inview

    def agreement_over(fr_list):
        """Robust colour-agreement of every carrier over a view list: returns
        (agreement in [0,1], observation count). agreement = exp(-beta * L2 of the
        carrier colour vs the median observed GT colour) over in-frame views."""
        big = torch.tensor(1e6, device=dev)
        cols, msks, cnt = [], [], torch.zeros(n, device=dev)
        for fr in fr_list:
            u, v, w, h, inview = project(fr)
            img = imageio.imread(_img_path(fr))
            H, W = img.shape[0], img.shape[1]
            gt = torch.tensor(img, dtype=torch.float32, device=dev) / 255.0
            ui = torch.clamp(u.round().long(), 0, W - 1)
            vi = torch.clamp(v.round().long(), 0, H - 1)
            sampled = gt[vi, ui, :]
            cols.append(torch.where(inview[:, None], sampled, big))
            msks.append(inview.float())
            cnt += inview.float()
        stack = torch.stack(cols, dim=1)                 # [N,V,3]
        median_obs, _ = stack.median(dim=1)              # no-obs -> ~1e6
        dist = torch.linalg.norm(colors - median_obs, dim=1)
        agree = torch.exp(-a.beta * dist).clamp(0, 1)
        # carriers never observed -> agreement 0 (unsupported, honestly low)
        agree = torch.where(cnt > 0, agree, torch.zeros_like(agree))
        return agree, cnt

    # TARGET: held-out (test-split) colour agreement = the reliability label.
    reliability, obs_cnt = agreement_over(test_frames)
    # FEATURE (export-time, no held-out GT): TRAIN-split colour agreement -- a
    # floater/inconsistency detector computable at export. Disjoint views from
    # the target, so no leakage.
    train_agree, _ = agreement_over(train_frames)
    # Legacy heuristic feature (train view COUNT -> squash); kept to show it is
    # saturated / non-discriminative on a densely-captured scene ("before").
    train_cnt = torch.zeros(n, device=dev)
    for fr in train_frames:
        _, _, _, _, inview = project(fr)
        train_cnt += inview.float()
    raw_conf = 1.0 - torch.exp(-train_cnt / a.conf_saturate)

    obs_cnt_np = obs_cnt.cpu().numpy()
    labeled = obs_cnt_np >= a.min_obs

    out = dict(
        raw_conf=raw_conf.cpu().numpy().astype("float32"),       # count heuristic
        train_agree=train_agree.cpu().numpy().astype("float32"),  # informative feat
        reliability=reliability.cpu().numpy().astype("float32"),  # held-out target
        opacity=opacity.astype("float32"),
        heldout_obs=obs_cnt_np.astype("int32"),
        train_obs=train_cnt.cpu().numpy().astype("int32"),
        labeled=labeled,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    import numpy as _np
    rel_np = reliability.cpu().numpy()
    ta_np = train_agree.cpu().numpy()
    print(json.dumps({
        "carriers": int(n),
        "test_views": len(test_frames),
        "train_views": len(train_frames),
        "labeled_fraction": float(labeled.mean()),
        "reliability_mean_labeled": float(rel_np[labeled].mean()),
        "reliability_std_labeled": float(rel_np[labeled].std()),
        "raw_conf_mean": float(raw_conf.mean().item()),
        "raw_conf_std": float(raw_conf.std().item()),
        "train_agree_mean": float(ta_np[labeled].mean()),
        "train_agree_std": float(ta_np[labeled].std()),
        "corr_trainagree_reliability": float(
            _np.corrcoef(ta_np[labeled], rel_np[labeled])[0, 1]),
        "corr_opacity_reliability": float(
            _np.corrcoef(opacity[labeled], rel_np[labeled])[0, 1]),
        "corr_rawconf_reliability": float(
            _np.corrcoef(raw_conf.cpu().numpy()[labeled], rel_np[labeled])[0, 1]),
        "out": a.out,
    }, indent=2))


if __name__ == "__main__":
    main()
