# AURA-Core — honest status & roadmap to a genuine post-3DGS step

_Last updated 2026-06-23 (first real GPU bring-up)._

AURA's thesis (see `../FINDINGS.md`, brain note "Photogrammetry → NeRF → 3DGS →
AURA"): keep 3DGS speed/quality but move from visual-only splats to **adaptive,
typed, ray-traceable radiance carriers** under one query / edit / confidence /
LOD / export contract. This file states honestly how far the code is.

## What is real today (verified on the Tanks&Temples truck scene, RTX 5090)

- **Differentiable training that converges.** `aura train-gsplat` seeds Gaussian
  carriers directly from COLMAP points (GPU, ~1s) and optimises them with the
  gsplat differentiable CUDA rasterizer (full-image L1+SSIM, adaptive
  densification 129k→404k). Trained Gaussians are written back into a normal
  `.aura` package.
- **Two renderer paths, by design:**
  - *Primary view* — the tiled rasterizer (`--renderer gsplat`): full
    front-to-back alpha compositing of all carriers. This is the high-fidelity
    path. The trained scene evaluates at **14.47 dB / SSIM 0.33 @0.25 scale**,
    matching the executed gsplat 3DGS baseline (14.04 dB) — i.e. AURA's pipeline
    reproduces 3DGS-class quality.
  - *Secondary ray / query* — the 3D ray renderer (`--renderer cuda|torch`):
    ray-Gaussian intersection for ray-query/secondary-ray semantics. It caps
    hits/ray and uses a support-radius cutoff, so it is **not** the primary-view
    quality path (it scored 8.71 dB on the same scene). Use `gsplat` for image
    quality; the ray path is for the asset-contract query features.
- **Carrier coverage fix, fast seeding, GPU SSIM/eigh** — see git log and
  `CONVERGENCE_TODO.md`.

## The honest gap: AURA is currently a 3DGS-class pipeline, not yet a new primitive

Every real training run optimises **only `gaussian_fallback` carriers**. The
seven carrier types in `carriers.py` (surface, volume, bounded kernel / beta,
gabor, neural residual, semantic, gaussian) are assigned by `assignment.py`'s
thresholds and stored as typed structs, but **no non-Gaussian carrier is trained
with gradients** — `torch_kernels.py` marks them all `missing_cuda_kernel`. So
functionally, the trained asset today is 3DGS with a typed-carrier/asset-contract
scaffold around it.

## What makes it a genuine post-3DGS step (the work that remains)

Ordered by leverage. Each plugs in behind the boundary already built in
`gsplat_renderer.py` (`scene_to_gaussian_params` / `gaussian_params_to_scene`),
so the `.aura` asset contract and eval harness stay unchanged.

1. **View-dependent appearance (spherical harmonics).** Biggest single quality
   lever toward true 3DGS (~25 dB): replace per-carrier flat RGB with SH
   coefficients (gsplat supports `colors=[N,K,3]`, `sh_degree=d` natively). Store
   SH on the carrier (payload/metadata), render with SH in both the rasterizer
   and ray paths. ~1–2 days, gsplat-native, low risk.
2. **Typed differentiable carriers** (the core differentiator) — **STARTED
   2026-06-23.** `src/aura/rasterizer_native.py` is an AURA-native differentiable
   rasterizer (pure-torch, autograd, GPU) with a *pluggable per-carrier 2D
   footprint*: `gaussian_footprint` (validated to match gsplat at ~31 dB),
   `beta_footprint` (bounded polynomial, Deformable-Beta-style), `gabor_footprint`
   (oscillatory). A non-Gaussian **Beta carrier now trains end-to-end with
   gradients** through it (test_rasterizer_native: L1 0.087→0.003) — the first
   time AURA optimises a non-Gaussian primitive. Remaining: wire it into the
   `train-gsplat`/scene boundary so mixed-type scenes train at scale, add a tiled
   / CUDA fast path (current dense compositor is O(M·H·W), fine for validation /
   modest scenes), and integrate the published kernels:
   - *Beta / Universal-Beta kernels* — Deformable Beta Splatting ships a CUDA
     rasterizer; integrate it as a second carrier backend (refs:
     `vault/01-raw/papers/aura/arxiv-2501.18630`, `2510.03312`).
   - *Gabor carriers* for high-frequency texture (`arxiv-2504.11003`).
   - *Splattable neural primitives* with analytic line integrals
     (`arxiv-2510.08491`) for the neural-residual carrier.
   - Requires a per-type differentiable rasterizer + a unified compositor; this
     is the real research effort and the thing that makes AURA ≠ 3DGS.
3. **Exact / ray-traceable volumetric rendering** (EVER `arxiv-2410.01804`,
   3DGRT `arxiv-2407.07090`): replace the billboard alpha approximation with
   exact volumetric ellipsoid / ray-traced compositing so the secondary-ray path
   is physically consistent and the primary path loses the popping artefacts.
4. **End-to-end carrier assignment learning.** `allocation.py`'s
   semantic-graph-governed soft assignment exists but is off by default
   (`use_graph_governed=False`) and not wired into the training objective; make
   the carrier-type choice differentiable/learned, not a fixed threshold.
5. **Native-primitive engine export.** `gltf_writer.py` / `usd_writer.py` export
   only Gaussians today; add export for the trained native carrier types so the
   asset is genuinely engine-ready beyond splats.

## Bottom line

The bring-up is done and honest: AURA trains, converges, and renders at
3DGS-class quality through its own pipeline, with the asset/query scaffold in
place. Item (2) — differentiable typed carriers — is the single thing that would
make AURA a genuinely new representation rather than a well-architected 3DGS
wrapper. It is scoped above and is the recommended next investment.
