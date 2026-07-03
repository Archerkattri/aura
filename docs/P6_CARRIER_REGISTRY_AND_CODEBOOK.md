# P6 — Carrier registry honesty + LangSplatV2-style codebook semantics

This note covers audit item **M4** (carrier registry honesty, CPU half) and the
**P4 codebook-semantics** infrastructure. It is the source of truth for what each
carrier type can back, and for what the semantic codebook layer does and does not
validate. Scope discipline: every claim below is either validated on CPU in CI, or
explicitly marked GPU-gated.

## 1. Carrier truth table (verified in code, 2026-07-03)

The registry (`src/aura/carriers.py:default_registry`) advertises **seven** carrier
types. They are not equally real. Verified state:

| Carrier | Registry kind | Real-scene training | Demo / 2D training | Render support (PRISM / hybrid) | Maturity |
|---|---|---|---|---|---|
| **gaussian** | `GAUSSIAN_FALLBACK` | **Yes** — gsplat quality backend; committed `outputs/calib_<scene>.json` for truck/garden/kitchen/room | — | primary quality backend (gsplat rasterization); `gaussian_footprint` in PRISM | **trained** |
| **beta** | `BETA_KERNEL` | **Yes** — DBS-Beta quality backend (+0.335 dB matched-budget); same calib corpus | — | primary quality backend (DBS-Beta); `beta_footprint` in PRISM; kept off the PRISM layer by default | **trained** |
| **gabor** | `GABOR_FREQUENCY` | No | Yes — `gabor_footprint` renders; assignable via `assign_footprints` / `train_carriers_prism`; validated on 2D crops only | real PRISM extension footprint (hybrid code 2 → `gabor_footprint`) | **demo** |
| **neural** | `NEURAL_RESIDUAL` | No | Partial — `make_neural_footprint` trains a 2D ring (test), but is **orphaned** (no caller) and **experimental** | **no PRISM kernel**; hybrid code 3 composites via an **explicit Gaussian fallback** | **demo** |
| **surface** | `SURFACE_CELL` | No | No | typed payload / contract only; no dedicated footprint | **metadata** |
| **volume** | `VOLUME_CELL` | No | No | typed contract only (a `volumetric` composite flag exists but no trained family) | **metadata** |
| **semantic** | `SEMANTIC_FEATURE` | No | No | not a render carrier (`primary_render=False`); label graph + per-carrier `semantic_id`; DINO features distilled only on GPU, uncommitted | **metadata** |

Two families train and render on real scenes with committed evidence; two have real
or partial footprints validated only in demo/2D use; three are typed-metadata
contracts with no trained render family behind them.

## 2. The maturity contract

Every `CarrierSpec` now carries an explicit `maturity` field
(`trained` | `demo` | `metadata`), surfaced through:

- `aura.carriers.CarrierSpec.maturity` and `aura.carriers.carrier_maturity_map()`;
- the runtime export report (`runtime_export._carrier_export_entry` → `"maturity"`);
- the README typed-carrier table.

Contract meaning:

- **trained** — trains and renders on real scenes with committed evidence
  (`calib_<scene>.json`). Only **gaussian** and **beta** today.
- **demo** — has a footprint that renders, but validated only in a demo / 2D / PRISM
  extension setting, not on real full scenes. **gabor**, **neural** today.
- **metadata** — a typed contract / payload / graph only; no trained render family.
  **surface**, **volume**, **semantic** today.

### Publication gate

`publication._carrier_registry_honesty_gate` (registered in
`publication_validation_report`) enforces the contract as a content-checked gate,
mirroring the existing real-scene gates. It **FAILS** when:

1. a registered carrier is **absent** from `CARRIER_MATURITY_CONTRACT` (an
   unlabelled type is an implicit over-claim), or advertises an unknown maturity;
2. the registry's advertised maturity **disagrees** with the contract (e.g. gabor
   flipped to `trained`);
3. a carrier claims **`trained`** without committed real-scene evidence — i.e. it is
   not in `TRAINED_CARRIER_EVIDENCE` (only gaussian/beta), or its
   `calib_<scene>.json` artifacts are missing.

Tests: `tests/test_publication_gates_real.py` (passes on committed artifacts; fails
on a demo-carrier-claims-trained tamper, an unlisted-type tamper, and a
missing-evidence tamper) and `tests/test_carrier_maturity.py`.

### The silent-fallback fix (hybrid.py)

Previously `hybrid._prism_layer` routed the unimplemented neural code (3) to the
Gaussian branch with no signal — the exact silent-substitution failure this repo
forbids. Now the fallback is **explicit**:

- `hybrid.footprint_routing(ftypes)` (pure/CPU-testable) reports, per carrier, the
  layer (`primary`/`prism`), the footprint actually used, and a `fallback` field
  that reads `"fallback:gaussian"` for neural;
- `render_hybrid` emits a `RuntimeWarning` when a neural carrier is composited via
  the Gaussian fallback, and populates an optional `provenance` dict with the
  fallback list;
- existing hybrid tests are unchanged (no neural carrier is rendered in them), so
  behavior for existing packages is identical — the fallback is now *visible*, not
  changed.

### The orphaned neural footprint (prism.py)

`prism.make_neural_footprint` is a Splat-the-Net-style neural primitive that is not
wired into any render or training path and has no real-scene evidence. It is now
**quarantined** behind `enable_experimental=True` (raises `NotImplementedError`
otherwise) with a docstring stating its unvalidated status. It is **not deleted** —
a later GPU session may revive it. The one exercise of it (a 2D-ring fit in
`tests/test_prism.py`) opts in explicitly and is labelled demo-stage.

## 3. Codebook design (`src/aura/codebook.py`)

LangSplatV2 (arXiv:2507.07136) trick: heavy per-carrier semantic features live once
in a shared **K-entry codebook**; each carrier stores only a small integer index.

- `fit_codebook(features, k, seed=0, iters=50)` — deterministic k-means (seeded
  k-means++ init, empty clusters re-seeded to the worst-fit point). `k ≤ N`.
- `assign_codes(features, codebook)` — nearest-centroid index per carrier, in the
  codebook's compact dtype (**uint8** when `k ≤ 256`, else **uint16**).
- `reconstruct(codes, codebook)` — gather features back from indices.
- `compression_report(...)` — original `N·d·4` bytes vs compressed
  `k·d·4 + N·code_bytes`; ratio + reconstruction RMSE / Frobenius-relative error.
- `query_codebook` / `open_vocab_query` — **open-vocab query in O(K·d + N)**: score
  the K codebook entries against the query embedding (`O(K·d)`), then fan out to
  carriers by index (`O(N)`), instead of the dense `O(N·d)` scan.
- `save_codebook` / `load_codebook` — `codebook.npz` + JSON metadata sidecar,
  aligned with the `carriers.npz` sidecar pattern so the codebook sits next to the
  carriers it indexes.

Storage layout mirrors carriers: the heavy `[k, d]` float32 matrix is written once;
carriers carry a `[N]` uint8/uint16 index array.

## 4. Honest scope — validated vs GPU-gated

**Validated on CPU (CI-safe, synthetic):** `tests/test_codebook.py` — deterministic
fit, uint8/uint16 index selection, reconstruction error on well-separated clusters,
compression ratio and its exact byte formula, the `O(K·d + N)` code-indexed query
fan-out (carrier scores equal the per-code score gathered by index), in-cluster
open-vocab retrieval, and `save`/`load` round-trip.

**Validated on a real distilled feature tensor (local_data):** the multi-view DINO
distillation itself is GPU-gated (`experiments/semantic_distill.py`, runs in
`.dbs_venv`, writes `carrier_features.npz`). When such a tensor is present the
`local_data` test in `tests/test_codebook.py` fits the codebook on it; otherwise it
skips (never faked). Measured on the committed local truck DINOv2 distillation:

| Metric | Value |
|---|---|
| Seen carriers × feature dim | 998,950 × 384 (float32) |
| Original feature bytes | 1,534,387,200 (~1.53 GB) |
| Codebook (k=64) compressed bytes | 1,097,254 (~1.05 MB), uint8 indices |
| **Compression ratio** | **≈ 1398×** |
| Reconstruction relative error (k=64) | 0.319 (decreases as k grows) |

The compression is dominated by N ≫ k: a million carriers each drop from a 1536-byte
feature to a single byte, and the 64×384 codebook is amortized once.

**GPU-gated next step (NOT done here):** real-scene *feature distillation* into the
committed asset — lifting DINO features onto trained carriers and shipping the
codebook + indices inside the `.aura` sidecar. Today the distilled features live in a
scratch `.npz`; the codebook layer above is ready to consume them, but wiring the
distillation output into the export path and validating open-vocab retrieval quality
against LangSplatV2 is a GPU session. Until then, semantic remains a **metadata**
carrier and the open-vocab reel is backed by the GPU distillation experiment, not by
committed per-carrier features.

## 5. What backs the open-vocab query demo today

The README "open-vocabulary query" reel (`docs/semantic_query_*.png`) is produced by
two GPU experiments, **not** by committed asset data:

1. `experiments/semantic_distill.py` — multi-view DINOv2/DINOv3 patch features are
   projected onto Beta carriers, visibility-aggregated, L2-normalized, then
   MiniBatchKMeans-clustered into groups. Writes `carrier_features.npz` to a scratch
   dir (uncommitted).
2. `experiments/semantic_query.py` — renders each group, CLIP-image-embeds it, and
   ranks groups against the CLIP text embedding.

The shipped library carries the *contract* (`SemanticGraph` labels,
`SemanticFeaturePayload` with sparse-codebook fields, `decode_semantic_feature`) and
now the *codebook layer* (`aura.codebook`), but no committed per-carrier feature
tensor. That is why `semantic` is a **metadata** carrier: the query capability is
demonstrated by the GPU experiments, not backed by data inside the asset.
