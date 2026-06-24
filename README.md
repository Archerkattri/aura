# AURA — Adaptive Unified Radiance Asset

> **Photogrammetry → NeRF → 3D Gaussian Splatting → AURA**

AURA turns posed photos into a **typed, queryable, relightable, engine-ready 3D
radiance asset**. It builds on a Gaussian rasterizer for speed and quality, and adds
the layer Gaussian splatting lacks: adaptive *typed* carriers, per-primitive
semantics and confidence, relighting, ray queries, and a standards-compliant export.

<p align="center">
  <img src="docs/truck_orbit.gif" width="70%" alt="AURA reconstruction of the Tanks & Temples Truck"><br>
  <em>AURA reconstruction of Tanks &amp; Temples — Truck (26.4 dB).</em>
</p>

![Ground truth · COLMAP · NeRF · 3DGS · AURA](docs/lineage_truck.png)

## Contents

- [What you get](#what-you-get)
- [Quality](#quality)
- [Capabilities](#capabilities)
- [How it's built](#how-its-built)
- [Install](#install)
- [Usage](#usage)
- [Reproduce](#reproduce)
- [Gallery](#gallery)
- [License](#license)

## What you get

- **Typed radiance carriers** — not one Gaussian everywhere. AURA reconstructs with
  adaptive primitives (Beta, Gabor, neural, Gaussian) so each region gets the
  carrier that fits it. Beta carriers are both **more accurate and ~2× more compact**
  than fixed Gaussians at matched budget.
- **A real asset, not a splat dump** — every carrier carries colour, geometry, a
  **confidence** value, and a **semantic** descriptor, and the whole scene answers a
  unified **ray query** `{color, depth, normal, confidence, semantic_id}`.
- **Relightable** — carriers re-shade under arbitrary lights (Lambertian /
  Cook-Torrance), unlike a baked splat cloud.
- **Open-vocabulary search** — type "a wheel" and AURA highlights it.
- **Engine-ready export** — standards-compliant `KHR_gaussian_splatting` glTF/GLB
  that loads in three.js / PlayCanvas / Babylon.

## Quality

Tanks & Temples — Truck, held-out views:

| Representation | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Carriers |
|---|---|---|---|---|
| fixed Gaussian | 26.02 | 0.890 | 0.128 | 1.0 M |
| **AURA (adaptive Beta)** | **26.35** | **0.896** | **0.122** | 1.0 M |
| **AURA (adaptive Beta)** | 26.07 | 0.890 | 0.139 | **0.5 M** |

AURA's typed carriers beat a fixed Gaussian of the same count — and reach the same
quality with **half the carriers**.

![GT vs fixed Gaussian vs adaptive Beta](docs/beta_vs_gauss_truck.png)

![Beta needs half the carriers for equal quality](docs/compactness_curve.png)

## Capabilities

### Engine-ready export — `KHR_gaussian_splatting`

Export trained carriers to the ratified Khronos
[`KHR_gaussian_splatting`](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting)
glTF/GLB extension — position, colour+opacity, rotation, scale, and higher-order SH —
so an AURA scene loads as real splats in any conformant engine.

```bash
aura export-splat scene/carriers.npz --output scene.glb
```

### Relighting

A carrier is treated as a surface element (normal + albedo) and re-shaded under
arbitrary lights, then rasterized — the same scene responds to a moving light:

![relighting under a moving light](docs/relight_sweep.gif)

### Per-carrier confidence

Every carrier carries a confidence from multi-view observation support — green is
well-observed, red flags speculative geometry — stored in the asset and exported as
a glTF attribute.

![confidence heatmap](docs/confidence_truck.png)

```bash
aura confidence scene/carriers.npz scene/manifest.json
```

### Semantics + open-vocabulary query

Multi-view vision features (DINOv2) are lifted onto carriers, giving a coherent
semantic segmentation; a CLIP text query then selects regions by name.

![semantic segmentation](docs/semantic_distill_truck.png)
![open-vocabulary query: "a wheel"](docs/semantic_query_truck.png)

### Unified ray query

One call answers a ray with the full payload — `{color, depth, normal, confidence,
semantic_id, transmittance}` — over the trained carriers:

```bash
aura ray-query scene/carriers.npz --origin 0 0 0 --direction 0 0 1
```

![expected-depth geometry](docs/truck_depth_orbit.gif)

## How it's built

AURA is an **engine + contract layer**:

- **Engine — gsplat.** Fast, high-quality Gaussian rasterization (and, increasingly,
  ray tracing) is a solved, well-optimised problem. AURA uses
  [gsplat](https://github.com/nerfstudio-project/gsplat) as its Gaussian backend
  rather than reinventing it.
- **Typed carriers — Beta / Gabor / neural.** Quality and compactness come from
  carriers gsplat doesn't have. AURA trains **Deformable Beta Splatting** carriers
  (a learnable bounded kernel + spherical-Beta colour) for the headline results.
- **PRISM — the typed-carrier rasterizer.** `aura.prism` is a differentiable, GPU,
  pluggable-footprint rasterizer that **extends** the Gaussian engine to splat
  non-Gaussian carriers (Beta, Gabor, neural) under one pipeline — use gsplat for
  Gaussian quality, PRISM where a region needs a carrier type gsplat can't express.

| Footprint | Kernel |
|---|---|
| gaussian | `exp(-½·conic)` (3DGS-style) |
| beta | bounded `(1-r/3)^β` (Deformable Beta) |
| gabor | oscillatory envelope (high-frequency texture) |
| neural | bounded MLP over Fourier features |

- **Asset contract.** Carriers live in a schema-validated `.aura` package (typed
  registry, chunks/LOD, semantic graph, confidence) plus a fast binary
  `carriers.npz` sidecar, and export to `KHR_gaussian_splatting`.

Built on a current stack: PyTorch 2.11 (CUDA 12.8), gsplat, Deformable Beta
Splatting, DINOv2 + OpenCLIP for semantics, and the ratified glTF
`KHR_gaussian_splatting` standard.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gpu,assets]"
```

The `gpu` extra adds PyTorch, `assets` adds `imageio` (EXR/video), `dev` adds pytest.
For the CUDA renderer: `aura cuda-kernel-build-report --build`.

## Usage

```bash
# 1. ingest posed captures (or a COLMAP sparse model)
aura colmap-to-capture-manifest <scene>/sparse/0 --root <scene> \
    --image-dir <scene>/images --output scene/manifest.json --point-seeded

# 2. reconstruct
aura train-gsplat scene/manifest.json --output scene.aura --scale 1.0   # Gaussian backend
aura train-prism  scene/manifest.json --output scene.aura --carrier beta --densify

# 3. use the asset
aura export-splat scene.aura --output scene.glb                          # engine export
aura confidence   scene.aura scene/manifest.json                         # confidence field
aura ray-query    scene.aura --origin 0 0 0 --direction 0 0 1            # query payload
aura render       scene.aura --backend torch --output view.ppm           # render
aura validate-package scene.aura && aura inspect-package scene.aura
```

## Reproduce

The headline Beta results use the Deformable Beta Splatting backend in an isolated
environment; figures and GIFs are produced by the scripts in `experiments/`:

```bash
bash scripts/fetch_scene.sh truck data/tanks/truck     # data
bash experiments/dbs_truck_ablation.sh                 # typed Beta vs fixed Gaussian
bash experiments/dbs_compactness_sweep.sh              # compactness (½ the carriers)
python experiments/render_turntable.py                 # reconstruction GIF
python experiments/relight_fork_gif.py                 # relighting GIF
python experiments/semantic_distill.py                 # semantic segmentation
python experiments/semantic_query.py                   # open-vocab query
```

## Gallery

All on Tanks & Temples — Truck, rendered through the trained carriers.

| | |
|---|---|
| **Reconstruction** (26.4 dB)<br>![orbit](docs/truck_orbit.gif) | **Expected depth**<br>![depth](docs/truck_depth_orbit.gif) |
| **Relighting**<br>![relight](docs/relight_sweep.gif) | **Confidence**<br>![confidence](docs/confidence_truck.png) |
| **Semantic segmentation**<br>![semantics](docs/semantic_distill_truck.png) | **Open-vocab query** ("a wheel")<br>![query](docs/semantic_query_truck.png) |
| **Typed vs fixed**<br>![beta vs gauss](docs/beta_vs_gauss_truck.png) | **Compactness**<br>![compactness](docs/compactness_curve.png) |

## Repository map

```text
src/aura/        the library — reconstruction, carriers, rasterizers, asset contract
  prism.py / prism_cuda.py   PRISM typed-carrier differentiable rasterizer
  gltf_splat.py              KHR_gaussian_splatting export
  relight.py confidence.py carrier_query.py   relight / confidence / ray-query
  carrier_io.py             fast binary carriers.npz sidecar
  schemas/                  JSON Schemas for the .aura package
scripts/         eval, dataset fetch, baselines, DBS↔AURA bridge
experiments/     reproduction scripts + figure/GIF renderers
tests/           deterministic contract, renderer, and CLI tests
docs/            figures & GIFs (this README is the single source of truth)
```

## License

MIT License. See [LICENSE](LICENSE).
