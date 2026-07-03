# P2 — full-resolution reproduction and a render-loss reliability label

P2 stress-tests the P0 killer property (calibrated, certified, exported per-carrier
confidence) along the two axes P0 left open:

- **P2a — full resolution.** P0 fit its carriers at *reduced* resolution. Does the
  calibrated-confidence story survive at full resolution, or was it a low-res
  artifact?
- **P2b — a render-loss label.** P0's reliability label is a **colour-agreement
  proxy** that never renders. Does the story survive when the label is measured from
  the **actual alpha-composited render** on held-out views?

Every number below traces to a committed JSON in `outputs/` (force-added; `outputs/`
is gitignored). Reproduce with `experiments/run_p2.sh <scene> <gpu>` then
`experiments/collect_p2.py` (→ `outputs/p2_summary.json`).

## What P0 reduced, and what "full resolution" means here

`aura train-gsplat --scale s` (and the P2 trainer) render and compute the training
loss at a linear fraction `s` of each camera's native resolution: the GT image is
bilinearly downsampled to `s×`, the intrinsics scale by `s`, and gsplat rasterizes
at `s·W × s·H` (`src/aura/gsplat_renderer.py:manifest_frame_to_camera`,
`_load_image_rgb`, `train_scene_gsplat`). **P0 used `--scale 0.25`** on the native
`images/` COLMAP manifest — so the P0 carriers were fit against **quarter-linear-
resolution** renders. (Note the reliability-*label* projection in
`per_carrier_reliability.py` already used `scale=1.0`, i.e. full-res GT — so what P0
reduced was the **trained carriers**, not the label evaluation.) For these scenes
`0.25×` native is `≈ images_4 / images_2`, i.e. the *standard* 3DGS Mip-360 training
resolution; **full resolution here is `--scale 1.0` = native `images/`**, 16× the
pixels P0 optimised against.

Native resolutions: **truck 979×546**, **room 3114×2075**, **kitchen 3115×2078**,
**garden 5187×3361**. **All four scenes are trained at native `1.0`** (garden's
17.4 MP fit during a transient GPU-memory window; the run took ~24 min on a shared
5090 under a concurrent ~27 GB external job). The one memory concession is garden's
**render-loss label**, whose 17.4 MP *rasterization* OOMs on the shared GPUs, so it is
rendered at `0.5` (the carriers are still native `1.0`); the colour/occlusion labels,
which only project points into full-res GT, run at native `1.0` for every scene.

P2 also **fixes a P0 protocol leak**: P0 trained on *all* frames, so the every-8th
views its reliability label calls "held-out" were actually seen in training. P2
**holds every-8th out of training** (`train_carriers_p2.py`, llffhold `holdout=8`),
so both labels are genuinely held-out. One consequence is that absolute reliability
(and hence absolute selection AUC) sits a little below P0's mildly-optimistic
figures; the conclusions are stated relative to each run's own oracle ceiling and to
opacity, which is what transfers.

## Reproduction checks against P0 (overlap points)

1. **Calibration + certificate math is unchanged.** Re-running
   `experiments/calibrate_confidence.py` on the committed P0 `reliability_*.npz`
   reproduces every committed `outputs/calib_<scene>{,_depth}.json` **byte-for-byte**
   (all 8, verified this session). So any P2 movement is from the carriers/label, not
   the calibration code.
2. **The label pipeline reproduces P0 at matched resolution.** The P2 **quarter
   control** (`--scale 0.25`, the same resolution as P0, but with the clean
   train-split) recovers P0's correlation/ECE story on every scene (e.g. room
   corr(feature, reliability) `0.961` vs P0 `0.96`, ECE `0.470→0.0013` vs P0
   `0.464→0.0016`). The small residual vs P0 is the leakage fix, not resolution.

## P2a — full resolution (colour-agreement label)

Per scene, colour-agreement label: `corr` = correlation of the export-time
train-view feature with held-out reliability; ECE raw→calibrated; selection AUC
(calibrated / oracle ceiling / opacity); certificate kept-fraction at ε=0.6, α=0.1.
Rows: **P0** = committed `calib_<scene>.json` (0.25, all-frames, the pre-P2 baseline);
**quarter** = P2 control (0.25, clean train-split); **FULL-RES** = P2 headline (1.0,
train-split); **depth** = full-res occlusion-aware label.

| scene | condition | corr | ECE raw→cal | AUC cal | oracle | opacity | cert kept |
|---|---|--:|--:|--:|--:|--:|--:|
| **truck** | P0 (0.25, all-frames) | — | 0.586→0.0014 | 0.581 | 0.601 | 0.367 | 1.00 |
| | quarter (0.25, split) | 0.93 | 0.671→0.0014 | 0.492 | 0.507 | 0.287 | 0.74 |
| | **FULL-RES (1.0, split)** | **0.93** | 0.661→0.0017 | **0.505** | 0.520 | 0.302 | 0.77 |
| | full-res depth | 0.86 | 0.651→0.0022 | 0.507 | 0.536 | 0.326 | 0.81 |
| **garden** | P0 (0.25, all-frames) | — | 0.551→0.0015 | 0.605 | 0.619 | 0.450 | 1.00 |
| | quarter (0.25, split) | 0.93 | 0.554→0.0017 | 0.602 | 0.617 | 0.450 | 1.00 |
| | **FULL-RES (1.0, split)** | **0.92** | 0.503→0.0010 | **0.642** | 0.659 | 0.482 | 1.00 |
| | full-res depth | 0.89 | 0.505→0.0017 | 0.640 | 0.666 | 0.480 | 1.00 |
| **kitchen** | P0 (0.25, all-frames) | — | 0.557→0.0006 | 0.629 | 0.633 | 0.458 | 1.00 |
| | quarter (0.25, split) | 0.98 | 0.567→0.0006 | 0.619 | 0.623 | 0.448 | 1.00 |
| | **FULL-RES (1.0, split)** | **0.98** | 0.537→0.0007 | **0.646** | 0.651 | 0.478 | 1.00 |
| | full-res depth | 0.97 | 0.529→0.0008 | 0.656 | 0.663 | 0.497 | 1.00 |
| **room** | P0 (0.25, all-frames) | — | 0.464→0.0016 | 0.720 | 0.729 | 0.534 | 1.00 |
| | quarter (0.25, split) | 0.96 | 0.470→0.0013 | 0.715 | 0.724 | 0.529 | 1.00 |
| | **FULL-RES (1.0, split)** | **0.96** | 0.448→0.0019 | **0.734** | 0.743 | 0.546 | 1.00 |
| | full-res depth | 0.94 | 0.467→0.0024 | 0.716 | 0.729 | 0.535 | 1.00 |

**Verdict — the P0 story holds at full resolution.** The export-time train-view
colour-agreement feature still predicts held-out reliability at **r ≈ 0.92–0.98**
(occlusion-aware depth label 0.86–0.97); the view-count and opacity heuristics stay
uninformative (|corr| ≤ 0.17). Isotonic calibration still drops ECE by **~240–770×**
to ~0.001–0.002. Calibrated confidence lands **within ~1–3% of the oracle ceiling**
(truck 2.9%, garden 2.6%, kitchen 0.8%, room 1.2%) and beats opacity, the raw
heuristic, and random on every scene. Crucially, **full-res is at or slightly above
the 0.25 quarter control on every scene** (AUC 0.505 vs 0.492, 0.642 vs 0.602, 0.646
vs 0.619, 0.734 vs 0.715) — the P0 conclusions are *not* a reduced-resolution
artifact; if anything full resolution helps a little. Absolute AUC sits a touch below
P0's committed values, which the quarter control localises to the train/test-leakage
fix (it moves the same way at 0.25 and 1.0), not to resolution.

## P2b — a render-loss reliability label

### Method: exact blend-weight attribution (not a first-order ablation)

A leave-one-out render-loss delta per carrier is what one wants, but a first-order
gate gradient `ΔL_i ≈ −∂L/∂g_i` (gate = per-carrier opacity multiplier) was tried and
**rejected**: in an over-complete splat scene individual carriers are highly
redundant, so the first-order delta is dominated by noise and does *not* track true
finite ablation (measured correlation to true group-ablation deltas ≈ 0.11; ranking
by it, removing the "most-harmful" carriers still *raised* held-out loss). It is not a
usable label.

Instead P2 uses an **exact** attribution. The rendered image is *linear* in the
carrier colours, `R_p = Σ_i w_{i,p} c_i`, where `w_{i,p} = α_i T_i` is carrier `i`'s
alpha-compositing blend weight at pixel `p` (`T_i` = transmittance in front, so
occluded / low-opacity carriers get `w ≈ 0` for free). Because `R` is linear in `c`,
`∂(Σ_p R_p f_p)/∂c_i = Σ_p w_{i,p} f_p` is **exact** for any pixel weighting `f`
independent of `c_i`. Three colour-backward passes per held-out view therefore recover
per carrier, exactly, `W_i = Σ_p w_{i,p}`, `A_i = Σ_p w_{i,p} GT_p`, and
`B_i = Σ_p w_{i,p} GT_p²`, and the carrier's blend-weighted rendering error closes in
form:

```
SE_i   = Σ_ch (c_{i,ch}² W_i − 2 c_{i,ch} A_{i,ch} + B_{i,ch})
dist_i = sqrt(SE_i / W_i)              # blend-weighted RMS colour distance
reliability_i = exp(−β · dist_i),  β=4 # same squash as the colour label
```

This is the P0 colour label upgraded to the **true alpha composite**: it scores each
carrier by whether it paints the right colour *where it is actually visible*, is
occlusion-exact (carriers with `W_i` below a floor are unlabelled, not mislabelled),
and is a faithful per-carrier attribution of the rendered L2 error. It is not a
first-order approximation — the render is genuinely linear in colour.
(`experiments/render_loss_reliability.py`; pure rule + tests in
`tests/test_render_loss_reliability.py`.) Features and the labelled carrier set are
copied from the P0 colour npz for the **same** carriers, so only the label changes
head-to-head.

### Head-to-head (colour proxy vs render-loss label, full-res carriers)

Same full-res (native-1.0) carriers and same labelled carrier set; only the label
changes. `corr(feat)` = export feature vs the label; `corr(c↔r)` = colour label vs
render-loss label; `lab` = labelled fraction; then ECE, AUC (cal/oracle/opacity),
cert kept@ε=0.6. (Garden's render-loss label is rendered at 0.5; see above.)

| scene | label | corr(feat) | corr(c↔r) | lab | ECE raw→cal | AUC cal | oracle | opacity | cert kept |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **truck** | colour (proxy) | 0.93 | — | 0.99 | 0.661→0.0017 | 0.505 | 0.520 | 0.302 | 0.77 |
| | **render-loss** | 0.66 | 0.64 | 0.71 | 0.730→0.0013 | 0.353 | 0.407 | 0.237 | 0.26 |
| **garden** | colour (proxy) | 0.92 | — | 0.94 | 0.503→0.0010 | 0.642 | 0.659 | 0.482 | 1.00 |
| | **render-loss** | 0.75 | 0.75 | 0.93 | 0.622→0.0009 | 0.444 | 0.481 | 0.343 | 0.79 |
| **kitchen** | colour (proxy) | 0.98 | — | 1.00 | 0.537→0.0007 | 0.646 | 0.651 | 0.478 | 1.00 |
| | **render-loss** | 0.81 | 0.82 | 0.99 | 0.646→0.0006 | 0.451 | 0.479 | 0.346 | 0.78 |
| **room** | colour (proxy) | 0.96 | — | 1.00 | 0.448→0.0019 | 0.734 | 0.743 | 0.546 | 1.00 |
| | **render-loss** | 0.78 | 0.78 | 0.96 | 0.537→0.0014 | 0.578 | 0.618 | 0.443 | 1.00 |

**Verdict — the killer property survives the stricter label, with honestly weaker
margins.** Under the render-grounded label: (1) calibration still crushes ECE by
2–3 orders of magnitude (to ~0.001); (2) the export-time colour feature still
*predicts* the render-loss label, but more weakly — **r ≈ 0.66–0.81** vs the proxy's
0.92–0.98 — because the render label penalises occlusion and visibility-weighted
colour error the occlusion-blind proxy misses (the two labels themselves correlate
only r ≈ 0.64–0.82); (3) calibrated confidence stays **above opacity/random and near
oracle, but the oracle gap widens to ~6–13%** (vs ~1–3% for colour); (4) the conformal
certificate stays valid but becomes **more selective** — at ε=0.6 the kept fraction
drops from ~1.0 (colour) toward the floater-heavy scenes' true reliability (truck
keeps 0.26, garden/kitchen ~0.78, room 1.0), i.e. the render label honestly refuses to
certify carriers the proxy over-credited.

**Honest caveat, reconfirmed.** A directional finite-ablation prune
(`directional_prune` in the render-loss summaries) reproduces P0's caveat under the
render label. **Opacity is the render-preserving prune on every scene**: dropping the
lowest-opacity carriers leaves held-out render L1 essentially unchanged (e.g. room
0.0408→0.0408, garden 0.0581→0.0587 at a 10%-keep-out), while dropping the
least-*reliable* carriers costs at least as much and usually more L1 (room
0.0408→0.0458, kitchen 0.0664→0.0722, garden 0.0581→0.0665; truck's least-reliable are
render-invisible so both are ~free). This is the same structural fact P0 flagged:
opacity *is* the alpha blend weight, so the least-reliable carriers are often the most
*visible* (load-bearing but slightly wrong-coloured) — reliability and
render-PSNR-preservation optimise different things. The P2 render-loss label makes this
explicit rather than resolving it.

## Reproduce

```bash
# full pipeline for one scene (train fr@1.0 + quarter@0.25 + 3 labels + 4 calibrations)
experiments/run_p2.sh room 0            # native full-res (also truck, kitchen, garden)
# garden's 17.4 MP render-loss RASTER OOMs on a shared GPU; render that one label at 0.5:
experiments/render_loss_reliability.py --carriers outputs/garden-fr.aura \
  --manifest outputs/garden-fr-manifest.json --color-npz outputs/reliability_garden_fr.npz \
  --out outputs/reliability_renderloss_garden_fr.npz --scale 0.5
experiments/collect_p2.py               # -> outputs/p2_summary.json
```
