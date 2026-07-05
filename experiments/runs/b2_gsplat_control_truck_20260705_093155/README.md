# B2 — true gsplat-3DGS matched-budget control (Truck)

Launched 2026-07-05 on the shared 5090 box while gaussianfeels (SNU lab) used the cards.
This is the **B2 headline control**: a genuine gsplat-3DGS (MCMC strategy) baseline at the
**same 1,000,000-primitive budget** every DBS beta/gauss arm used, so the AURA typed-carrier
win is controlled against *real* 3DGS instead of a frozen-β DBS ablation.

## What ran
- Backend: gsplat-main source tree (`/tmp/aura_sota_repos/gsplat-main`), JIT CUDA ops,
  `.gpu_venv` (torch 2.11+cu128). NOT pip gsplat 1.5.3, NOT the `.dbs_venv` fork.
- Scene: `data/tanks/truck`, `data_factor=1` (native 979×546 images/ — matches DBS truck arm).
- Strategy: `mcmc`, `--strategy.cap-max 1000000`, `max_steps=30000` (default).
- Split: `test_every=8` (== DBS `llffhold=8`). LPIPS net = `vgg` (matches DBS evaluator).
- GPU: `CUDA_VISIBLE_DEVICES=1` (GPU1). CPU threads capped to 4.
- Exact command: see `launch.sh` in this dir.

## Where results land
- Checkpoints + eval stats: `/tmp/gsplat_control/truck/` (`stats/val_step*.json`).
- Training log: `train.log` (this dir).
- Collector: `python experiments/collect_gsplat_control.py` reads the stats and writes
  `outputs/gsplat_control.json` with per-scene `gsplat final` (step-30k, standard) +
  `gsplat best` (best test PSNR, mirrors DBS `iteration_best` test-metric selection).

## Reference numbers to beat / compare (Truck, 1M budget)
- DBS-Beta (adaptive typed carrier): **26.394 dB** PSNR  (`/tmp/dbs_multiscene/truck_beta`)
  — the knowledge pack quotes 26.352 for the ablation-specific run; both are the "Beta" arm.
- DBS frozen-Gaussian control: **25.962 dB** PSNR  (`/tmp/dbs_multiscene/truck_gauss`)
  — pack quotes 26.017. The established matched-budget win is **+0.335 dB** (Beta − frozen-Gauss).

## What closes B2
A true gsplat-3DGS PSNR at matched 1M budget on Truck (both `final@30k` and `best`), placed
next to Beta 26.35–26.39 / frozen-Gauss 25.96–26.02. Interpretation:
- If gsplat-3DGS ≈ frozen-Gauss (~26.0) → the +0.335 dB Beta win holds against *real* 3DGS.
- If gsplat-3DGS lands between the two or above Beta → the win shrinks / flips; record honestly
  (project ethos: negatives are published, not hidden).
Follow-up (not done here): UBS-6D arm (arXiv 2510.03312) — only if wired; otherwise a later add.
