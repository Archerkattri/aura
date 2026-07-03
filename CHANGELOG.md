# Changelog

All notable changes to AURA (Adaptive Unified Radiance Asset) are recorded here.
This is a research repository with its own git history. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Every claim below is backed by a
committed artifact — negatives are kept, not hidden.

## [Unreleased]

_Nothing yet._

## [0.2.0] — 2026-07-03

First tagged release. Consolidates the P0→P2 calibrated-confidence arc into a
citable version: the killer property (calibrated, certified, exported per-carrier
confidence), cross-scene calibrator transfer, full-resolution reproduction, and a
render-grounded reliability label — plus the honest P0 evaluation-leak correction
(the Truck colour pruning certificate drops from certifying 100% to certifying 77%
of carriers on the clean, genuinely held-out split, `1.00 → 0.77`). **The public
API, the `.aura` package format, and the KHR/USD exports are unchanged.** Two items
stay open: the garden native render-loss label needs an idle GPU (17.4 MP raster
OOMs under load, so P2 renders it at half resolution), and P3 independent
re-captures would harden the reliability story further. The four README result
figures regenerate from committed data via `experiments/make_hardening_figures.py`.
The change history that produced this release follows.

### P2 — full-resolution reproduction + render-loss reliability label (2026-07-03)

Stress-tests the P0 killer property along the two axes P0 left open: full resolution
(P0 fit carriers at `--scale 0.25`, quarter-linear) and a reliability label measured
from the actual alpha-composited render instead of P0's colour-agreement proxy. Also
fixes a P0 protocol leak (P0 trained on all frames, so its "held-out" reliability
views were seen in training; P2 holds every-8th out of training).

#### Added
- **Full-resolution + render-loss pipeline** (`experiments/train_carriers_p2.py`,
  `experiments/render_loss_reliability.py`, `experiments/run_p2.sh`,
  `experiments/collect_p2.py` → `outputs/p2_summary.json`). Per scene it trains
  train-split carriers at native `--scale 1.0` (all four scenes, garden's 17.4 MP
  included) plus a `0.25` quarter-resolution control, then runs the colour,
  occlusion-aware, and render-loss labels + four calibration/certificate reports. Only
  garden's render-loss *label* is rendered at `0.5` (its 17.4 MP rasterization OOMs on
  the shared GPUs; the carriers stay native). Reports
  `outputs/calib_<scene>_{fr,fr_depth,q,renderloss}.json`.
- **Render-loss label via exact blend-weight attribution.** The render is linear in
  carrier colour, so `∂⟨render,GT⟩/∂colour_i` recovers the *exact* alpha-compositing
  blend weights; from them each carrier's blend-weighted rendering error
  `exp(−β·sqrt(SE_i/W_i))` is an occlusion-exact, visibility-weighted per-carrier
  render-loss attribution (a first-order gate-gradient ablation was tried and
  rejected — noise in over-complete scenes). Pure rule + 5 tests in
  `tests/test_render_loss_reliability.py`.
- Write-up `docs/P2_FULLRES_RENDERLOSS.md` (every number traces to a JSON).

#### Validated
- **P0 story holds at full resolution.** The export-time colour feature still predicts
  held-out reliability at r ≈ 0.92–0.98, calibration still drops ECE ~240–770× to
  ~0.001–0.002, and calibrated confidence is within ~1–3% of its oracle and beats
  opacity/heuristic/random. Full-res is at or slightly above the 0.25 quarter control
  on every scene — the property is not a low-res artifact. Reproduction checks:
  re-running `calibrate_confidence.py` on the committed P0 npz reproduces
  `calib_<scene>{,_depth}.json` byte-for-byte, and the quarter control recovers P0's
  correlation/ECE at matched resolution (residual = the leakage fix).
- **Killer property survives the render-loss label, with honestly weaker margins.**
  Calibration still crushes ECE 2–3 orders; the colour feature still predicts the
  render label but more weakly (r ≈ 0.66–0.81, since it penalises occlusion /
  visibility-weighted error the proxy misses); calibrated confidence stays above
  opacity with the oracle gap widening to ~6–13%; the conformal certificate stays
  valid but becomes more selective (kept 0.26–1.0 at ε=0.6; truck 0.26). A directional
  prune reconfirms P0's honest caveat under the render label: reliability and
  render-PSNR-preservation optimise different things.

### P1 — cross-scene calibrator transfer + certificate operating study (2026-07-03)

Two measurements the P0 write-up explicitly declined, both pure CPU/numpy over the
committed P0 reliability data — no retraining, no GPU.

#### Added
- **P1a cross-scene calibrator transfer** (`experiments/cross_scene_transfer.py` →
  `outputs/cross_scene_transfer.json`, full 4×4×2 matrix). For every ordered scene
  pair over {truck, garden, kitchen, room} × label {color, depth}: fit the isotonic
  calibrator on all labeled carriers of the source, evaluate on the target's held-out
  eval half (the diagonal reproduces `calib_<scene>.json` exactly); report ECE (raw /
  transferred / in-scene), selection AUC (transferred / in-scene / opacity / oracle),
  and the ε=0.6, α=0.1 conformal certificate computed with the transferred confidence
  on the target's own local calibration split.
- **P1b certificate operating study** (`experiments/cert_sweep.py` →
  `outputs/cert_sweep.json`). Extends the P0 ε-sweep (Truck-only) to all four scenes ×
  both labels over ε 0.30→0.65 (α=0.1); records certified τ, kept fraction, empirical
  kept risk, and the selectivity onset ε per scene.
- Write-up `docs/P1_CROSS_SCENE.md`; regression test `tests/test_cross_scene_transfer.py`
  (2 tests, synthetic npz, self-contained).

#### Validated
- **Transfer holds / degrades gracefully.** Selection AUC transfers essentially for
  free — an isotonic calibrator is monotone and AUC is rank-based, so transferred vs
  in-scene AUC agree within ±0.0004 on all 24 off-diagonal pairs. Absolute calibration
  (ECE) degrades gracefully: transferred off-diagonal ECE mean 0.008 (color) / 0.026
  (depth), worst case 0.017 / 0.058 (both truck→room) — still 1–2 orders of magnitude
  below the uncalibrated raw heuristic (~0.54). Neighbouring indoor scenes
  (kitchen↔room) transfer best; object-centric Truck is the outlier source/target.
- **Certificate stays valid under transfer.** All 24 transferred certificates are
  certified at ε=0.6 (color kept 1.00; depth kept 0.914–1.00) because the conformal
  threshold is fit on the target's local split — the honest deployment recipe is "ship
  one calibrator + a small per-scene conformal set."
- **Certificate selectivity mapped on all scenes.** The onset ε* (largest ε with
  kept<1.0) is Truck 0.59/0.62, Garden 0.57/0.56, Kitchen 0.56/0.55, Room 0.47/0.48
  (color/depth); it tracks each scene's mean reliability, confirming the certificate
  is neither vacuous nor over-eager.

### P0 killer property — calibrated, certified, exported per-carrier confidence (2026-07-01 → 2026-07-02)

The successor axis a bare 3DGS/DBS splat lacks: a per-carrier confidence a
downstream engine can *trust and prune against with a guarantee*. Answers audit
blocker B1 (no demonstrated killer property) and major M3 (confidence was an
uncalibrated heuristic).

#### Added
- **Calibration module** `src/aura/calibration.py` (CPU / numpy, 10 tests):
  - `IsotonicConfidenceCalibrator` (PAVA) — monotone map from the raw multi-view
    heuristic to a calibrated reliability.
  - `conformal_prune_certificate` — distribution-free split-conformal risk control
    (Hoeffding UCB on retained-set unreliability): the most-inclusive threshold `τ`
    that keeps mean unreliability ≤ `ε` at confidence `1−α`.
  - `selection_quality_curve` — downstream demonstrator (retained reliability across
    pruning budgets).
  - `attach_calibrated_confidence` — replaces the heuristic value with the calibrated
    one and flags `confidence_calibrated=True`.
- **`aura calibrate-confidence <package> <reliability.npz>` CLI**, wiring the
  calibrated value into export so it ships as the `_AURA_CONFIDENCE` vendor
  attribute in the `KHR_gaussian_splatting` GLB.
- **Occlusion-aware reliability label** (`--label depth_aware`): a per-block
  front-surface z-buffer counts a carrier only in held-out views where it is the
  visible front surface, fixing the interior-occlusion false-low of the colour label.
- **OpenUSD 26.03 schema export**: `write_usd_gaussian_splat` / `aura export-usd
  --schema` emits the official `UsdVolParticleField3DGaussianSplat` schema
  (usd-core 26.05) with a confidence vendor channel; 3 round-trip tests. Closes
  audit item E6/P2 (USD preview previously predated the official schema).
- Experiment drivers `experiments/per_carrier_reliability.py` and
  `experiments/calibrate_confidence.py`; per-scene reports
  `outputs/calib_<scene>{,_depth}.json`.
- Authoritative write-up `docs/P0_CALIBRATED_CONFIDENCE.md` and the four-scene
  figure `assets/p0_selection_auc.png` (+ its generator
  `experiments/make_p0_selection_auc_figure.py`).

#### Changed
- **Corrected reliability estimator**: the robust observed-colour centre is now a
  masked `nanmedian` over each carrier's *observed* held-out views, replacing an
  earlier sentinel-median that poisoned carriers seen in fewer than half the views.
  Required for the occlusion-aware label; it supersedes the earlier (2026-07-01)
  Truck/Garden colour figures but leaves the headline conclusion unchanged.
- README gains a P0 "Calibrated Confidence (killer property)" section; the Current
  Status table now reads "validated on 4 real scenes".

#### Validated (four real scenes)
Truck (129k carriers), Garden (Mip-NeRF-360 outdoor, 120k), Kitchen (indoor, 120k),
Room (indoor, 107k):
- The export-time **train-view colour-agreement** feature predicts held-out
  reliability **r = 0.91–0.98** on all four; the shipped view-count heuristic is
  uninformative (|r| ≤ 0.13).
- Isotonic calibration drops **ECE from 0.46–0.59 to 0.0006–0.0016** (~300–900×).
- **Selection AUC** (mean retained reliability across pruning budgets): calibrated
  confidence **0.58–0.72, within 1–4% of the oracle ceiling**, beating opacity
  (0.37–0.53, at or below random) at every budget; at a 10%-keep budget calibrated
  retains 0.77–0.90 vs opacity's 0.31–0.49.
- The property **survives the occlusion-aware label** (within 1–9% of oracle;
  corr 0.75–0.97).

#### Honest notes
- The earlier "opacity is uniformly negatively correlated with reliability" reading
  did **not** survive the estimator fix — it was partly a sentinel-median artifact.
  The surviving, scene-independent claim: **opacity is a poor pruning signal
  everywhere** (at or below random); calibrated confidence beats it at every budget.
- The reliability label is still a colour-agreement proxy (not a photometric render
  loss) and the depth buffer is a coarse block z-buffer; both are conservative
  (they under-credit rather than over-credit a carrier).

### Typed-carrier asset foundation (2026-06-24 → 2026-06-25)

The DBS-Beta typed-carrier quality path, the asset/export layer, and the PRISM
additive extension — plus the local publication and SOTA-A/B evidence arc.

#### Added
- **DBS-Beta typed-carrier renderer** as the primary typed quality path (gsplat
  stays the Gaussian path); per-carrier confidence field with KHR export attribute;
  unified ray-query payload over trained carriers; a relighting layer over trained
  carriers; DINOv2 → CLIP open-vocabulary semantic query.
- **PRISM** (Pluggable Radiance-prImitive Splatting Module) as an *additive*
  extension over the gsplat/DBS-Beta quality path — pure-PyTorch renderer plus a
  custom CUDA path — routing Gabor/neural footprints only.
- **Export surfaces**: `KHR_gaussian_splatting` GLB (position/colour/opacity/
  rotation/scale/SH), USD ASCII preview, and a `.aura` package + `carriers.npz`
  sidecar for fast local rendering/eval.
- **Evidence & gates**: artifact-backed publication-validation report; local
  same-split external baselines (COLMAP, NeRF, 3DGS/gsplat-control, 2DGS-style,
  ray-traced-GS-style) and official 2DGS + 3DGUT 30k same-split rows on all 8
  audited scenes; SOTA A/B pass (DINOv3-small/timm, official 2DGS, 3DGUT);
  submission package, paper outline, and leaderboard-ablation schema.

#### Results
- Beta beats the fixed-Gaussian control on every audited scene, **mean +0.80 dB
  PSNR**, and reaches comparable quality at **~half the carriers** on Truck.
  *(Caveat: the control is a frozen-β DBS ablation, not real gsplat 3DGS, and
  Mip-360 eval used image downsamples — this reproduces DBS's published claim,
  arXiv 2501.18630, it is not our novelty.)*

#### Honest negatives (kept as publishable content)
- Adaptive per-carrier β does **not** beat a good global β (learned 26.352 <
  uniform β=2 26.421).
- Cross-family mix-routing **never** beats the best single family.
- The typed +dB win decomposes to a spherical-Beta colour model (~+0.4 dB), not to
  adaptivity (~0); an earlier +0.8 dB "typed win" was a camera-roll pose-bug
  artifact (fixed 2026-06-24).

#### Claim boundary
- Local artifact-backed A/B readiness only; **no** official-leaderboard SOTA claim,
  no production-FPS-everywhere claim, and third-party viewer compatibility is a
  structural check, not a runtime guarantee.
