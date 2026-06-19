# AURA Benchmark Results — Truck Run 5 (Partial)

**Date:** 2026-06-20  
**Status:** Training in progress — partial results

---

## Training Status

| Parameter | Value |
|-----------|-------|
| Run | `truck-3k-run5` |
| Dataset | Tanks & Temples — truck scene |
| Carriers | 129,531 gaussian carriers |
| Current iteration | 600 / 3000 (20%) |
| Training loss at iter 1 | 0.3415 |
| Training loss at iter 600 | ~0.0200 |
| Loss trend | Rapid drop iter 1→10, stabilising ~0.022–0.035 |
| Target iterations | 3000 |
| Output path | `outputs/truck-3k-run5.aura` (not yet saved) |
| Command | `python3 -m aura.cli train ... --iterations 3000 --tile-size 256 --max-targets-per-batch 256 --max-targets-per-frame 16 --skip-validation --disable-evolution` |

### Loss Curve (run5, every ~50 iters)

```
iter   1 / 3000  loss = 0.3415   ← initial
iter  10 / 3000  loss = 0.0227   ← rapid convergence
iter 100 / 3000  loss = 0.0271
iter 200 / 3000  loss = 0.0320
iter 300 / 3000  loss = 0.0202
iter 400 / 3000  loss = 0.0243
iter 500 / 3000  loss = 0.0325
iter 600 / 3000  loss = 0.0200   ← latest
```

Loss is oscillating in the 0.017–0.038 range from iter 10 onward (stochastic batch sampling). No divergence, no OOM since the fix at iter 140.

---

## Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 |
| VRAM total | 8,188 MiB |
| VRAM used (at eval time) | ~3,897 MiB (training process active) |
| Training PID | 486634 |

---

## Evaluation: truck-pts129k-batched-smoke.aura

Because `truck-3k-run5.aura` checkpoint has not been saved yet (training saves on completion), evaluation was run against the closest available checkpoint:  
**`outputs/truck-pts129k-batched-smoke.aura`** — a smoke-test pass (iteration 0 only, 16 batches, 129,531 gaussian elements, same carrier configuration as run5).

This checkpoint is effectively untrained (only one forward pass), so its PSNR reflects the initialisation quality of the 129k gaussian scene, not a converged result.

### eval_psnr.py output (64×36 thumbnail, 10 frames)

```
Loading outputs/truck-pts129k-batched-smoke.aura...
Scene: 129531 elements
Evaluating 10 frames (stride=25)
  [1/10] 000001.jpg: PSNR= 8.31 dB  MSE=0.1477
  [2/10] 000026.jpg: PSNR=10.57 dB  MSE=0.0878
  [3/10] 000051.jpg: PSNR= 9.36 dB  MSE=0.1158
  [4/10] 000076.jpg: PSNR= 7.43 dB  MSE=0.1807
  [5/10] 000101.jpg: PSNR= 6.53 dB  MSE=0.2222
  [6/10] 000126.jpg: PSNR= 7.99 dB  MSE=0.1587
  [7/10] 000151.jpg: PSNR= 8.27 dB  MSE=0.1488
  [8/10] 000176.jpg: PSNR= 8.21 dB  MSE=0.1509
  [9/10] 000201.jpg: PSNR= 8.03 dB  MSE=0.1573
 [10/10] 000226.jpg: PSNR= 8.29 dB  MSE=0.1483
```

**Average PSNR: 8.30 dB** (64×36 thumbnail, iteration-0 checkpoint, 10 frames)

> **Note:** The low PSNR is expected. The smoke checkpoint has only completed one optimiser step. The 3DGS comparison numbers are measured after full 30k-iter training at full resolution; this eval uses a 64×36 thumbnail which can also suppress high-frequency detail and may bias PSNR downward slightly relative to full-res metrics.

---

## Baseline Reference

| Method | PSNR on Truck (T&T) | Notes |
|--------|---------------------|-------|
| 3DGS (Kerbl et al. 2023) | ~25.19 dB | Full-res, 30k iters |
| 3DGS literature range | ~26–27 dB | Various re-implementations |
| MP-GS / multi-primitive | ~25–27 dB | Depends on primitive budget |
| **AURA run5 (partial)** | **TBD** | Iter 600/3000, checkpoint not yet saved |
| AURA smoke checkpoint | 8.30 dB | Iter 0, 64×36 thumbnail |

---

## Notes and Next Steps

1. **Full eval pending:** `truck-3k-run5.aura` will be saved when training completes (est. iter 3000). Re-run `scripts/eval_psnr.py` against it at full resolution for a fair comparison.
2. **eval_psnr.py API note:** The script uses `result.samples` which is an outdated API (`TorchRenderBatch` uses `.predicted_color`). Use `_psnr2.py` pattern (`torch_render_ray_color_tensor` with 128-ray batches) for OOM-safe eval on 129k-carrier scenes.
3. **OOM fix:** Applied at prior session — `--max-targets-per-batch 256 --max-targets-per-frame 16` keeps GPU memory within 8 GB during training.
4. **Loss convergence:** The loss dropped ~15× from iter 1 to iter 10, then plateaus with stochastic noise. This is consistent with Adam on Gaussian splatting objectives. No pathological divergence seen.
5. **benchmark-real-scene CLI:** Available via `python3 -m aura.cli benchmark-real-scene <package_dir>`. Can be run post-training for structured comparison against 3DGS/NeRF fixtures.
