# AURA roadmap — to a genuine post-3DGS engine

_Authoritative as of 2026-06-24 (supersedes the pre-pose-fix parts of
STATUS_AND_ROADMAP.md)._

## Where AURA actually is

- **Quality: competitive with 3DGS.** After fixing a camera-roll pose bug
  (`view_rotation`; +6.6 dB, `experiments/direct_pose_test.py`), AURA reaches
  **18.4 dB @0.25** and **19.4 dB full-res/30k/SH/densify** on T&T-Truck — on par
  with vanilla 3DGS (see `docs/lineage_truck.png`). **But this quality path is the
  gsplat backend** — i.e. AURA's competitive number *is* 3DGS. That is honest, and
  it means AURA is **not yet the post-3DGS vision** — it is 3DGS + a typed-carrier
  scaffold.
- **`.aura` format: fixed.** Binary `carriers.npz` sidecar (`aura/carrier_io.py`)
  loads in seconds instead of ~22 min for millions of carriers.
- **PRISM (AURA's own typed-carrier rasterizer): a quality dead-end as built.**
  PRISM-native quality *degrades* with carrier density (cap256 = 12.4 dB,
  cap2048 = 8.8 dB @0.25/5k; NaN at 4096 before grad clipping). Its from-scratch
  front-to-back compositor has training-dynamics problems with many overlapping
  carriers that gsplat solved through years of engineering (opacity reset, tuned
  densification, conditioned gradients, MCMC). **Fighting to make PRISM beat
  gsplat is the wrong battle.**

## What the research says is left (verified, cited 2025-26 survey)

There is **no consensus successor to 3DGS** — the field is single-axis
improvements on splat+rasterize. AURA's umbrella thesis (one typed-carrier asset
with unified ray-query/edit/relight/confidence/LOD/export) is a **real unfilled
gap**. Two make-or-break claims, both currently UNPROVEN in AURA:

1. **Adaptive mixed-type carriers winning on radiance QUALITY** (PSNR/LPIPS), not
   just geometry/compactness. Only the **Beta family** beats 3DGS on PSNR
   (Deformable Beta Splatting +0.45, Universal Beta +1.46 dB). Mixed-per-region
   *quality* win is open white space (only MP-GS gestures at it, surfaces only,
   no code).
2. **The unified capability contract as a WORKING integrated system** (not a file
   format). Verified that no method spans relight+query+edit+export+confidence+LOD.

## The plan

### Track 1 — typed-carrier quality win via DBS (the highest-certainty path)

Do **not** try to fix PRISM's compositor. Instead adopt **Deformable Beta
Splatting** (`github.com/RongLiu-Leo/beta-splatting`) — a *gsplat-derived* CUDA
rasterizer (installs as `beta_splatting`, no conflict with gsplat) with a
learnable per-primitive Beta shape. Steps:
1. **Build status (2026-06-24):** DBS's CUDA *compiles successfully on sm_120*
   (the `compute_sb` spherical-Beta fwd/bwd kernels built). **Caveat:** DBS's fork
   installs under the `gsplat` package name (namespace collision with the real
   gsplat 1.5.3 AURA uses), so it must live in an **isolated venv** (`.dbs_venv`)
   and be driven via a subprocess bridge. The `.aura` carrier sidecar
   (`carrier_io`) is format-agnostic, so DBS-trained carriers write back cleanly.
2. Add a `train-beta` backend mirroring `train-gsplat` but using DBS rasterization
   (means/scales/quats/opacity/SH + learnable Beta `sb` params). Seed from the
   same COLMAP carriers.
3. Reproduce DBS's claimed **+0.45–1.46 dB over 3DGS at ~45% params** on Truck —
   this is the first genuine "typed beats Gaussian on quality" result for AURA.
4. Then the *novel* step: **adaptive per-region routing** between Beta shapes
   (and Gaussian as the β→∞ limit), and prove the mix beats the best single
   setting at matched budget. This is the open contribution.

### Track 2 — the unified asset contract (the other make-or-break claim)

These differentiate AURA from "just 3DGS" and work on the good gsplat/DBS-trained
carriers (no need to beat gsplat on PSNR). Integrate, don't reinvent:
- **Export to the standard** `glTF KHR_gaussian_splatting` (ratifying Q2 2026) —
  replace the custom gltf_writer path so AURA assets are engine-ready.
- **Ray-query / secondary rays / relighting**: wire 3DGRT-style ray tracing
  (open, Apache) over the trained carriers; AURA already has `cuda_renderer`
  ray-query + `shading.py` relighting scaffolds to connect.
- **Semantics + confidence**: per-carrier confidence *coupled to* a semantic
  embedding (the one cross-axis gap nobody fills) — integrate LangSplat-style
  features.
- Expose all of the above through one `rayQuery(r) -> {color, depth, normal,
  material, semantic_id, confidence, ...}` interface (the carrier contract).

### Track 3 — fold PRISM down

Keep PRISM as the *research substrate* (it proves typed footprints train with
gradients, and is real-time for forward rendering), but stop presenting it as the
quality engine. The quality engine is gsplat now, DBS next.

## Honest differentiators to defend (vs the two traps)

- **Unified Gaussian Primitives** (2406.09733) already claims one-carrier +
  relight + edit → AURA must differentiate on *typed routing + semantics +
  confidence + real-time standard export*.
- **glTF KHR_gaussian_splatting** is a *storage* contract → AURA must be the
  *editable/relightable/queryable computational* asset layer above the file.

**Bottom line:** AURA's vision is a real, open gap. The make-or-break is (1) a
mixed-typed-carrier *quality* win (via DBS, not PRISM) and (2) the integrated
contract as a working system. Everything else is integration of public,
citable components.
