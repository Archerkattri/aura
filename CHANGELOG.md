# Changelog

All notable changes to AURA (Adaptive Unified Radiance Asset) are recorded here.
This is a research repository with its own git history; it is pre-release and
tracks toward `v0.1.0-dev`, so everything currently lives under **Unreleased**.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Every
claim below is backed by a committed artifact — negatives are kept, not hidden.

## [Unreleased]

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
