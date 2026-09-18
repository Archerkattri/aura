# P7 — Relighting decision protocol (v0.8: real inverse rendering vs formal descope)

This note is the **pre-registered decision protocol** for AURA's relighting path
(audit item **M1**). It fixes, *before* the GPU attempt, exactly what will be
tried, the benchmarks and numeric bar it must clear, the descope alternative, and
the rule that chooses between them — so the v0.8 outcome is a measurement, not a
narrative. Scope discipline matches the rest of the repo: every target number below
is either a **CONFIRMED** figure quoted from a cited paper table, or explicitly
marked **UNCONFIRMED** (to be filled at eval time) rather than guessed.

Companion code: the CPU-buildable, GPU-runnable evaluation harness
`experiments/relight_benchmark_harness.py` (+ `tests/test_relight_harness.py`)
already implements the metrics, dataset loaders, and results tables used here.

## 1. Current state (the preview being judged)

`src/aura/relight.py` is a **preview**, not inverse rendering:

- **Albedo** = the baked SH DC term (`0.5 + C0·dc`) — the diffuse colour the scene
  trained with; it still contains residual baked-in shading.
- **Normals** = the covariance short axis; **unsigned** (both faces lit, brighter
  kept) and noisy for near-isotropic carriers.
- **No per-carrier material optimization** — nothing is fit to observations; the
  Lambertian / Cook-Torrance BRDFs in `shading.py` are applied *forward* to these
  guessed fields. The B3 publication gate honestly titles this "Relighting response
  on real asset" and only checks that the relit output *responds* to light/albedo.

## 2. What v0.8 will attempt

A genuine per-scene inverse-rendering pass on GPU:

1. **Per-carrier material optimization** — optimize an explicit per-carrier
   `(albedo, roughness, metallic)` (the `RelightingPayload` contract already exists)
   against the multi-view training images under the known/estimated illumination,
   so albedo is *de-lit* rather than the baked DC colour.
2. **Signed-normal recovery** — replace the unsigned covariance axis with a
   sign-disambiguated, optimized normal (view-consistency / photometric loss),
   which is what the TensoIR normal-MAE metric actually measures.
3. **Evaluation** on **TensoIR-Synthetic** (4 object scenes; GT albedo + GT normals
   + relighting under novel environment maps) and **Stanford-ORB** (real objects,
   ground-truth environment relighting), via the harness, against the published
   object-centric relighting SOTA.

## 3. The numeric bar

**Metrics** (all implemented in the harness): relit-render PSNR/SSIM/LPIPS under
novel illumination; **albedo PSNR** where GT albedo exists; **normal mean-angular-
error (deg)** where GT normals exist. Stanford-ORB additionally uses PSNR-H (HDR)
and PSNR-L (LDR after sRGB tone-map).

### 3a. Important correction to the M1 baseline list

The audit (`CLAUDE.md` M1) names **SSD-GS**, **R3GW**, and **GS³** as the relighting
SOTA "to meet on TensoIR / Stanford-ORB." Verified against the papers, **none of the
three actually reports TensoIR-Synthetic or Stanford-ORB numbers** — they evaluate
on different relighting regimes:

| Named baseline | TensoIR? | Stanford-ORB? | Datasets it *does* report | Own-dataset headline | Status |
|---|---|---|---|---|---|
| **SSD-GS** — Scattering & Shadow Decomposition for Relightable 3DGS (arXiv:2604.13333, Victoria Univ. Wellington) | No | No | NRHints OLAT (real), GS³-synthetic, SSS-GS-synthetic | OLAT relighting PSNR/SSIM/LPIPS (Tables 1, 3) | **UNCONFIRMED on TensoIR/ORB — not reported** |
| **R3GW** — Relightable 3D Gaussians for Outdoor Scenes in the Wild (arXiv:2603.02801) | No | No | NeRF-OSR (outdoor, in-the-wild) | NVS under varying illumination on NeRF-OSR | **UNCONFIRMED — different problem class (outdoor scenes, not object-centric)** |
| **GS³** — Efficient Relighting with Triple Gaussian Splatting (arXiv:2410.11419) | No | No | Lightstage, NRHints, synthetic point-lit | e.g. NRHints Pikachu PSNR 32.39, Cup-Fabric 37.09 (PSNR/SSIM/LPIPS only; **no albedo/normal metrics**) | **UNCONFIRMED on TensoIR/ORB — not reported** |

(Named-baseline own-dataset numbers above are CONFIRMED from each paper's HTML;
sources in §5.)

**Resolution (pre-registered):** keep TensoIR-Synthetic + Stanford-ORB as the eval
standard (they are the field's object-relighting benchmarks and the only ones with
GT albedo/normals), and anchor the numeric bar to the **object-centric Gaussian
relighting SOTA that *does* report there** (below). SSD-GS / GS³ / R3GW are retained
as qualitative cross-references and, if a head-to-head is wanted, would require
running AURA on *their* datasets (NRHints OLAT / lightstage / NeRF-OSR) — a separate,
different-capture-regime effort, not the v0.8 object-relighting bar.

### 3b. TensoIR-Synthetic bar — CONFIRMED published numbers

| Method | Albedo PSNR↑ | NVS PSNR↑ | Relight PSNR↑ | Relight SSIM↑ | Relight LPIPS↓ | Normal MAE°↓ | Source (CONFIRMED) |
|---|---:|---:|---:|---:|---:|---:|---|
| TensoIR (CVPR'23) | 29.275 | 35.088 | 28.580 | 0.944 | 0.081 | 4.100 | GS-IR Tab. [A] |
| GS-IR (CVPR'24) | 30.286 | 35.333 | 24.374 | 0.885 | 0.096 | 4.948 | GS-IR Tab. [A] |
| Relightable-3DGS / R3DG | 29.27 | 33.35 | 27.37 | — | — | 5.927 | PT-IR Tab. 1 [B] |
| SVG-IR (arXiv:2504.06815) | 30.48 | 36.71 | 31.10 | — | — | 4.358 | PT-IR Tab. 1 [B] |
| IRGS (CVPR'25, arXiv:2412.15867) | 33.796¹ | 35.43 | 29.907 | — | — | 4.112 | IRGS Tab. 2 [C] / PT-IR [B] |
| Path-traced IR (arXiv:2606.09606) | 32.12 | 36.17 | 31.84 | — | — | 4.028 | PT-IR Tab. 1 [B] |

¹ IRGS albedo PSNR is 33.796 in its own Table 2 [C] but re-tabulated as 30.62 in
PT-IR Table 1 [B]; the gap is an albedo scale-alignment convention difference. SSIM/
LPIPS are only CONFIRMED for the TensoIR and GS-IR rows (source [A]); "—" = not
captured, not zero.

**Reading of the field:** object-relighting SOTA sits at **relight PSNR ≈ 28–32 dB,
albedo PSNR ≈ 29–32 dB, normal MAE ≈ 4°**. Even GS-IR — a strong Gaussian method —
lands at relight PSNR 24.37 and normal MAE 4.95, i.e. *entering the arena credibly
is itself a high bar* for a method whose starting point is a baked-DC preview.

### 3c. Stanford-ORB bar — UNCONFIRMED (fill at eval time)

Stanford-ORB (Kuang et al., NeurIPS 2023; live leaderboard at
`https://stanfordorb.github.io/`) reports **PSNR-H / PSNR-L / SSIM / LPIPS** and
geometry (depth/normal) scores on 14 real objects. The Gaussian-method rows on ORB
were **not confirmable from the paper HTML fetched for this note** (IRGS and PT-IR
show ORB only qualitatively). The GPU eval session **must pull the current
published Gaussian-relighting PSNR-L / PSNR-H from the live leaderboard** and record
it here before applying the rule; treat every Stanford-ORB target as UNCONFIRMED
until then.

## 4. Pre-registered decision rule

Run the v0.8 attempt on **all four** TensoIR-Synthetic scenes (armadillo, ficus,
hotdog, lego) at the standard eval resolution, single run, plus the Stanford-ORB
eval set. Report the harness aggregate **before** any per-scene inspection (no
cherry-picking). Then:

**PROMOTE** to "real inverse-rendered relighting" (v0.8 = real) **iff all** hold:

- (i) mean TensoIR **relight PSNR ≥ 27.0 dB** — within ~2 dB of TensoIR (28.58) /
  IRGS (29.91), i.e. clears the R3DG (27.37) floor;
- (ii) mean TensoIR **albedo PSNR ≥ 27.0 dB** — within ~3 dB of GS-IR (30.29),
  demonstrating genuine de-lighting vs the baked DC;
- (iii) mean TensoIR **signed normal MAE ≤ 8.0°** — within ~2× of the ~4° SOTA,
  demonstrating real signed-normal recovery (the preview's unsigned axis cannot
  score this);
- (iv) Stanford-ORB **relight PSNR-L within 3 dB** of the current published Gaussian
  leaderboard entry (threshold value UNCONFIRMED — set from the live leaderboard at
  eval time).

**DESCOPE** (default if any criterion fails): formally rename the capability to a
**"confidence-weighted relighting preview"** everywhere it is user-facing —
`README.md`, `docs/`, the CLI (`relight-preview` stays; help/text updated),
`src/aura/relight.py`, and the B3 gate title — leaning into AURA's existing
calibrated per-carrier confidence (P0): the preview annotates/weights its relit
output by carrier confidence, which is honest and unique, rather than claiming
material recovery. Keep the honest B3 gate ("Relighting response on real asset")
and the preview scope note unchanged. Record the measured sub-bar numbers as a
published **honest negative** (the repo's ethos: negatives are kept, not hidden).

Rationale for the asymmetry: PROMOTE requires *all* criteria because "real inverse
rendering" is a conjunction of de-lit albedo + signed normals + relight fidelity;
missing any one means the preview framing is the honest one.

## 5. How the harness plugs in

`experiments/relight_benchmark_harness.py`:

- `--dataset {tensoir,stanford_orb}` resolves `AURA_TENSOIR_ROOT` /
  `AURA_STANFORD_ORB_ROOT`; **never downloads** — prints the source URL and exits
  `2` when absent (URLs in §3).
- Reads an `aura_relight_eval.json` manifest (schema in the module docstring)
  mapping GT frames to the GPU-rendered predictions, then computes relit
  PSNR/SSIM/LPIPS, albedo PSNR (where GT albedo exists), and signed **and** unsigned
  normal MAE (where GT normals exist), and writes a JSON + Markdown table.
- `--smoke` runs the full metric pipeline on synthetic fixtures with no dataset/GPU
  (CI-tested end to end). LPIPS is a real metric only when the `lpips` backbone
  loads; otherwise the step reports an honest backend label and `null`, never a
  fabricated score.

The GPU session's only new work is (a) the material/normal optimizer, (b) rendering
predictions, (c) writing the manifest, and (d) filling the Stanford-ORB UNCONFIRMED
threshold — then this rule decides v0.8.

## Sources (confirmed via arXiv HTML, 2026-07-03)

- **[A] GS-IR** — arXiv:2311.16473v3, TensoIR-Synthetic comparison table (TensoIR &
  GS-IR rows, incl. relight SSIM/LPIPS).
- **[B] Path-traced Inverse Rendering (PT-IR)** — arXiv:2606.09606, Table 1
  (TensoIR): TensoIR / GS-IR / R3DG / SVG-IR / IRGS / proposed rows.
- **[C] IRGS** — arXiv:2412.15867, Table 2 (TensoIR): relight PSNR 29.907, albedo
  PSNR 33.796, normal MAE 4.112.
- **SSD-GS** — arXiv:2604.13333 (datasets: NRHints OLAT, GS³-synthetic, SSS-GS);
  **R3GW** — arXiv:2603.02801 (dataset: NeRF-OSR); **GS³** — arXiv:2410.11419
  (datasets: lightstage / NRHints / synthetic point-lit). All three: **no TensoIR /
  Stanford-ORB numbers reported.**
- **Stanford-ORB** — Kuang et al., NeurIPS 2023, arXiv:2310.16044;
  leaderboard `https://stanfordorb.github.io/` (Gaussian ORB numbers UNCONFIRMED
  here — pull at eval time).
