#!/usr/bin/env python3
"""AURA relighting benchmark harness — TensoIR-Synthetic and Stanford-ORB.

This is the **CPU-buildable, GPU-runnable-later** evaluation harness for AURA's
relighting path (audit item M1 / the v0.8 decision — see
``docs/P7_RELIGHT_DECISION.md``). It does three things and nothing more:

1. Locate a benchmark dataset via an environment variable, printing a clear
   instruction + URL and exiting ``2`` when it is absent. **It never downloads
   data.**
2. Load (ground-truth, prediction) pairs and compute the standard inverse-rendering
   metrics: relit-render PSNR / SSIM / LPIPS under novel illumination, albedo PSNR
   where GT albedo exists, and normal mean-angular-error where GT normals exist.
3. Write a results table (JSON + Markdown).

A ``--smoke`` mode runs the *full* metric pipeline on tiny in-memory synthetic
fixtures so the harness is CI-testable end to end without any dataset or GPU.

Scope honesty
-------------
Today AURA relighting is a **preview** (baked-SH albedo that still contains
shading; unsigned covariance normals; no per-carrier material optimization — see
``src/aura/relight.py``). This harness does NOT itself render AURA predictions: the
per-carrier albedo/roughness optimization + signed-normal recovery that would make
AURA competitive is the GPU attempt gated for v0.8. The harness scores whatever
predicted images you hand it via ``--predictions-dir`` / the eval manifest, so the
same code path can grade the preview today and the trained relighter later.

Environment variables (dataset roots — this harness does not fetch them)
------------------------------------------------------------------------
* ``AURA_TENSOIR_ROOT``       — root of the TensoIR-Synthetic dataset.
* ``AURA_STANFORD_ORB_ROOT``  — root of the Stanford-ORB dataset.

Eval-manifest layout (portable, what the loader reads)
------------------------------------------------------
Point the dataset root at a directory containing ``aura_relight_eval.json``::

    {
      "dataset": "tensoir",
      "samples": [
        {
          "scene": "lego", "view": "r_000",
          "relit_gt":   "lego/relit/r_000.png",     # required
          "relit_pred": "preds/lego/r_000.png",      # required (GPU render)
          "albedo_gt":  "lego/albedo/r_000.png",     # optional
          "albedo_pred":"preds/lego/r_000_albedo.png",
          "normal_gt":  "lego/normal/r_000.npy",     # optional
          "normal_pred":"preds/lego/r_000_normal.npy",
          "mask":       "lego/mask/r_000.png"        # optional (metrics restricted)
        }
      ]
    }

Paths are resolved relative to the dataset root (or ``--predictions-dir`` for the
``*_pred`` fields when that flag is given), or may be absolute. Colour/albedo images
are read as ``[0,1]`` (``uint8`` PNG ``/255``; ``.npy`` taken as-is). Normal maps:
``.npy`` is assumed to hold signed ``[-1,1]`` vectors; a PNG/JPG normal is decoded
as ``2*x-1``. This explicit manifest is what makes the harness portable and testable
— when you mount the real TensoIR / Stanford-ORB release, write one manifest that
maps its GT frames to your rendered predictions.

Exit codes
----------
* ``0`` success · ``1`` usage error · ``2`` dataset missing · ``3`` no eval samples
  (dataset present but no predictions / manifest to grade).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_DATASET_MISSING = 2
EXIT_NO_SAMPLES = 3

TENSOIR_ENV = "AURA_TENSOIR_ROOT"
STANFORD_ORB_ENV = "AURA_STANFORD_ORB_ROOT"

# Where to obtain each benchmark (the harness never downloads).
_DATASETS = {
    "tensoir": (
        TENSOIR_ENV,
        "TensoIR-Synthetic",
        "https://haian-jin.github.io/TensoIR/  (data: "
        "https://zenodo.org/records/7880113 — Zhang et al., CVPR 2023)",
    ),
    "stanford_orb": (
        STANFORD_ORB_ENV,
        "Stanford-ORB",
        "https://stanfordorb.github.io/  (code/data: "
        "https://github.com/StanfordORB/Stanford-ORB — Kuang et al., NeurIPS 2023)",
    ),
}

# LPIPS is a real perceptual metric only when the `lpips` package + its pretrained
# backbone are available. Set this env var (CI does) to skip the network-touching
# model load and keep the pipeline hermetic; the LPIPS step still runs and reports
# an honest backend label rather than a fabricated number.
DISABLE_LPIPS_ENV = "AURA_RELIGHT_DISABLE_LPIPS"


# --------------------------------------------------------------------------- #
# Metrics (pure numpy; CPU; no dataset/GPU needed)
# --------------------------------------------------------------------------- #
def psnr(pred, gt, *, max_val: float = 1.0) -> float:
    """PSNR in dB between two images/arrays in ``[0, max_val]``.

    Returns ``inf`` for identical inputs (MSE == 0), matching the standard
    convention used across AURA's eval scripts.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"psnr shape mismatch: {pred.shape} vs {gt.shape}")
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def _rgb_to_luma(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _gaussian_kernel(size: int, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2.0 * sigma * sigma))
    k = np.outer(g, g)
    return k / k.sum()


def _filter2d_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'valid'-region 2D correlation via shift-and-accumulate (memory-light)."""
    kh, kw = kernel.shape
    H, W = img.shape
    oh, ow = H - kh + 1, W - kw + 1
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * img[i:i + oh, j:j + ow]
    return out


def ssim(pred, gt, *, max_val: float = 1.0) -> float:
    """Luminance SSIM (Gaussian window, K1=0.01, K2=0.03), 'valid' region mean.

    Accepts ``HxWx3`` colour or ``HxW`` grayscale. Uses an 11x11 window on images
    at least 11px per side (the standard); on smaller inputs it shrinks the window
    to the largest odd size that fits so tiny fixtures still evaluate.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"ssim shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim == 3:
        pred, gt = _rgb_to_luma(pred), _rgb_to_luma(gt)
    H, W = pred.shape
    win = min(11, H, W)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        raise ValueError(f"image too small for SSIM ({H}x{W}); need >= 3px per side")
    k = _gaussian_kernel(win)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    mu_x = _filter2d_valid(pred, k)
    mu_y = _filter2d_valid(gt, k)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = _filter2d_valid(pred * pred, k) - mu_x2
    sig_y = _filter2d_valid(gt * gt, k) - mu_y2
    sig_xy = _filter2d_valid(pred * gt, k) - mu_xy
    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
    return float(np.mean(num / den))


def normal_mean_angular_error(
    pred, gt, *, mask=None, signed: bool = True, degrees: bool = True, eps: float = 1e-8
) -> float:
    """Mean angular error between predicted and GT normals.

    ``pred``/``gt`` are ``HxWx3`` or ``Nx3`` vector fields (need not be unit — they
    are normalized here). ``signed=True`` is the standard TensoIR metric; with
    ``signed=False`` the sign ambiguity is folded away (angle in ``[0, 90]``), which
    is the fair reading for AURA's *unsigned* covariance-axis preview normals.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"normal shape mismatch: {pred.shape} vs {gt.shape}")
    pn = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), eps, None)
    gn = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), eps, None)
    dot = np.sum(pn * gn, axis=-1)
    if not signed:
        dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)
    if mask is not None:
        ang = ang[np.asarray(mask, dtype=bool)]
    if ang.size == 0:
        return float("nan")
    mean = float(np.mean(ang))
    return math.degrees(mean) if degrees else mean


def lpips_distance(pred, gt, *, net: str = "alex", device: str = "cpu"):
    """Perceptual LPIPS distance. Returns ``(value_or_None, backend_label)``.

    Honest by construction: a real number only when the ``lpips`` package and its
    pretrained backbone load successfully (backend ``"lpips-<net>"``). If the
    package is absent, disabled via ``AURA_RELIGHT_DISABLE_LPIPS``, or the backbone
    cannot be fetched (offline CI), it returns ``(None, <reason>)`` — never a
    fabricated perceptual score.
    """
    if os.environ.get(DISABLE_LPIPS_ENV):
        return None, "disabled"
    try:
        import torch
        import lpips as lpips_lib
    except Exception:
        return None, "unavailable"
    try:
        fn = lpips_lib.LPIPS(net=net, verbose=False).to(device).eval()

        def _t(img):
            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = np.repeat(a[..., None], 3, axis=-1)
            t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
            return t * 2.0 - 1.0  # lpips expects [-1, 1]

        with torch.no_grad():
            d = fn(_t(pred), _t(gt))
        return float(d.reshape(-1)[0]), f"lpips-{net}"
    except Exception as exc:  # offline backbone download, etc. — stay honest.
        return None, f"error:{type(exc).__name__}"


# --------------------------------------------------------------------------- #
# Sample container + per-sample evaluation
# --------------------------------------------------------------------------- #
@dataclass
class RelightSample:
    scene: str
    view: str
    relit_pred: np.ndarray
    relit_gt: np.ndarray
    albedo_pred: Optional[np.ndarray] = None
    albedo_gt: Optional[np.ndarray] = None
    normal_pred: Optional[np.ndarray] = None
    normal_gt: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


def evaluate_sample(s: RelightSample) -> dict:
    """Compute every metric whose ground truth is present for one sample."""
    m: dict = {"scene": s.scene, "view": s.view}
    mask_flat = None if s.mask is None else np.asarray(s.mask, dtype=bool)
    m["relit_psnr"] = psnr(s.relit_pred, s.relit_gt)
    m["relit_ssim"] = ssim(s.relit_pred, s.relit_gt)
    val, backend = lpips_distance(s.relit_pred, s.relit_gt)
    m["relit_lpips"] = val
    m["relit_lpips_backend"] = backend
    if s.albedo_gt is not None and s.albedo_pred is not None:
        m["albedo_psnr"] = psnr(s.albedo_pred, s.albedo_gt)
    if s.normal_gt is not None and s.normal_pred is not None:
        m["normal_mae_deg"] = normal_mean_angular_error(
            s.normal_pred, s.normal_gt, mask=mask_flat, signed=True)
        m["normal_mae_deg_unsigned"] = normal_mean_angular_error(
            s.normal_pred, s.normal_gt, mask=mask_flat, signed=False)
    return m


def aggregate(rows: Sequence[dict]) -> dict:
    """Mean of each finite numeric metric across sample rows."""
    keys: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    agg: dict = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows
                if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                and math.isfinite(r[k])]
        if vals:
            agg[k] = sum(vals) / len(vals)
    return agg


# --------------------------------------------------------------------------- #
# Dataset resolution + loading
# --------------------------------------------------------------------------- #
def resolve_dataset_root(dataset: str, env: Optional[dict] = None) -> Path:
    """Resolve a dataset root from its env var, or exit ``2`` with instructions."""
    env = os.environ if env is None else env
    if dataset not in _DATASETS:
        raise SystemExit(EXIT_USAGE)
    var, name, source = _DATASETS[dataset]
    root = env.get(var)
    if not root or not Path(root).is_dir():
        sys.stderr.write(
            f"\n[relight-harness] {name} dataset not found.\n"
            f"  Set {var} to the dataset root (currently: {root!r}).\n"
            f"  This harness does NOT download data. Obtain {name} from:\n"
            f"    {source}\n"
            f"  Then provide an eval manifest (aura_relight_eval.json — see the\n"
            f"  module docstring) mapping GT frames to rendered predictions and re-run:\n"
            f"    {var}=/path/to/{dataset} PYTHONPATH=src \\\n"
            f"      python experiments/relight_benchmark_harness.py "
            f"--dataset {dataset} --predictions-dir <renders>\n\n"
        )
        raise SystemExit(EXIT_DATASET_MISSING)
    return Path(root)


def _load_array(path: Path, kind: str) -> np.ndarray:
    """Load a colour/albedo/normal/mask array as float64.

    ``.npy`` is taken as-is (colour clipped to [0,1]; normal assumed signed).
    Image files are read via imageio/PIL, scaled to [0,1]; a normal image is
    decoded from [0,1] to [-1,1].
    """
    if path.suffix.lower() == ".npy":
        a = np.load(path).astype(np.float64)
    else:
        try:
            import imageio.v2 as imageio
            raw = np.asarray(imageio.imread(path))
        except Exception:
            from PIL import Image
            raw = np.asarray(Image.open(path))
        a = raw.astype(np.float64)
        if np.issubdtype(raw.dtype, np.integer):
            a = a / float(np.iinfo(raw.dtype).max)
        if kind == "normal":
            a = a[..., :3] * 2.0 - 1.0
    if kind in ("color", "albedo"):
        a = np.clip(a[..., :3] if a.ndim == 3 else a, 0.0, 1.0)
    elif kind == "mask":
        a = (a > 0.5)
    return a


def load_manifest_samples(
    dataset: str,
    root: Path,
    predictions_dir: Optional[Path] = None,
    scenes: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> list[RelightSample]:
    """Load samples from ``<root>/aura_relight_eval.json``.

    Returns ``[]`` (caller exits ``3``) when the manifest is absent — that is the
    signal that the dataset is mounted but no predictions/manifest exist yet. The
    GPU relight session writes the manifest + prediction renders; this loader is
    the stable, dataset-layout-agnostic contract they target.
    """
    manifest_path = root / "aura_relight_eval.json"
    if not manifest_path.is_file():
        sys.stderr.write(
            f"[relight-harness] no eval manifest at {manifest_path}.\n"
            f"  The dataset root is present but has no aura_relight_eval.json.\n"
            f"  Write one (see module docstring) mapping GT frames to your rendered\n"
            f"  predictions, then re-run.\n"
        )
        return []
    manifest = json.loads(manifest_path.read_text())

    def _resolve(rel: Optional[str], is_pred: bool) -> Optional[Path]:
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            return p
        base = predictions_dir if (is_pred and predictions_dir is not None) else root
        return base / p

    samples: list[RelightSample] = []
    for entry in manifest.get("samples", []):
        if scenes and entry.get("scene") not in scenes:
            continue
        gt_p = _resolve(entry.get("relit_gt"), False)
        pred_p = _resolve(entry.get("relit_pred"), True)
        if gt_p is None or pred_p is None or not gt_p.is_file() or not pred_p.is_file():
            continue
        alb_gt = _resolve(entry.get("albedo_gt"), False)
        alb_pred = _resolve(entry.get("albedo_pred"), True)
        nrm_gt = _resolve(entry.get("normal_gt"), False)
        nrm_pred = _resolve(entry.get("normal_pred"), True)
        mask_p = _resolve(entry.get("mask"), False)
        samples.append(RelightSample(
            scene=entry.get("scene", "?"),
            view=entry.get("view", gt_p.stem),
            relit_gt=_load_array(gt_p, "color"),
            relit_pred=_load_array(pred_p, "color"),
            albedo_gt=_load_array(alb_gt, "albedo") if alb_gt and alb_gt.is_file() else None,
            albedo_pred=_load_array(alb_pred, "albedo") if alb_pred and alb_pred.is_file() else None,
            normal_gt=_load_array(nrm_gt, "normal") if nrm_gt and nrm_gt.is_file() else None,
            normal_pred=_load_array(nrm_pred, "normal") if nrm_pred and nrm_pred.is_file() else None,
            mask=_load_array(mask_p, "mask") if mask_p and mask_p.is_file() else None,
        ))
        if limit is not None and len(samples) >= limit:
            break
    return samples


# --------------------------------------------------------------------------- #
# Result assembly + writers
# --------------------------------------------------------------------------- #
def build_result(dataset: str, rows: Sequence[dict], *, scope: str) -> dict:
    lpips_backends = sorted({r.get("relit_lpips_backend", "n/a") for r in rows})
    return {
        "format": "AURA_RELIGHT_BENCHMARK_REPORT",
        "dataset": dataset,
        "scope": scope,
        "note": (
            "AURA relighting is a preview (baked-SH albedo, unsigned covariance "
            "normals, no per-carrier material optimization). Scores below grade the "
            "supplied predictions; see docs/P7_RELIGHT_DECISION.md for the v0.8 bar."
        ),
        "nSamples": len(rows),
        "lpipsBackends": lpips_backends,
        "aggregate": aggregate(rows),
        "perSample": list(rows),
    }


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.4f}"
    return str(v)


def results_markdown(result: dict) -> str:
    agg = result["aggregate"]
    lines = [
        f"# AURA relighting benchmark — {result['dataset']}",
        "",
        f"- scope: **{result['scope']}**",
        f"- samples: {result['nSamples']}",
        f"- LPIPS backend(s): {', '.join(result['lpipsBackends']) or 'n/a'}",
        "",
        f"> {result['note']}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    label = {
        "relit_psnr": "Relit PSNR (dB) ↑",
        "relit_ssim": "Relit SSIM ↑",
        "relit_lpips": "Relit LPIPS ↓",
        "albedo_psnr": "Albedo PSNR (dB) ↑",
        "normal_mae_deg": "Normal MAE, signed (deg) ↓",
        "normal_mae_deg_unsigned": "Normal MAE, unsigned (deg) ↓",
    }
    for key in ["relit_psnr", "relit_ssim", "relit_lpips", "albedo_psnr",
                "normal_mae_deg", "normal_mae_deg_unsigned"]:
        if key in agg:
            lines.append(f"| {label[key]} | {_fmt(agg[key])} |")
    lines.append("")
    return "\n".join(lines)


def write_results(result: dict, json_path: Optional[Path] = None,
                  md_path: Optional[Path] = None) -> None:
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2) + "\n")
    if md_path is not None:
        md_path = Path(md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(results_markdown(result) + "\n")


def print_summary(result: dict) -> None:
    print(f"\n[relight-harness] {result['dataset']}  ·  {result['nSamples']} samples "
          f"·  scope={result['scope']}")
    for k, v in result["aggregate"].items():
        print(f"    {k:28s} {_fmt(v)}")
    print(f"    lpips backend(s): {', '.join(result['lpipsBackends']) or 'n/a'}")


# --------------------------------------------------------------------------- #
# Smoke fixtures (in-memory; exercises the whole pipeline, no dataset/GPU)
# --------------------------------------------------------------------------- #
def make_smoke_samples(n: int = 3, size: int = 48, seed: int = 0) -> list[RelightSample]:
    """Tiny random fixtures with GT albedo + GT normals so every metric branch runs.

    Predictions are the GT plus small noise (finite PSNR) and slightly rotated
    normals (nonzero MAE) — enough to exercise, not benchmark.
    """
    rng = np.random.default_rng(seed)
    out: list[RelightSample] = []
    for i in range(n):
        gt = rng.random((size, size, 3))
        pred = np.clip(gt + rng.normal(0, 0.05, gt.shape), 0.0, 1.0)
        alb_gt = rng.random((size, size, 3))
        alb_pred = np.clip(alb_gt + rng.normal(0, 0.05, alb_gt.shape), 0.0, 1.0)
        nrm_gt = rng.normal(0, 1, (size, size, 3))
        nrm_gt /= np.clip(np.linalg.norm(nrm_gt, axis=-1, keepdims=True), 1e-8, None)
        nrm_pred = np.clip(nrm_gt + rng.normal(0, 0.1, nrm_gt.shape), -1, 1)
        mask = rng.random((size, size)) > 0.2
        out.append(RelightSample(
            scene="smoke", view=f"v{i:03d}",
            relit_pred=pred, relit_gt=gt,
            albedo_pred=alb_pred, albedo_gt=alb_gt,
            normal_pred=nrm_pred, normal_gt=nrm_gt, mask=mask,
        ))
    return out


def run_smoke(json_path: Optional[Path] = None, md_path: Optional[Path] = None) -> dict:
    samples = make_smoke_samples()
    rows = [evaluate_sample(s) for s in samples]
    result = build_result("smoke", rows, scope="smoke")
    write_results(result, json_path, md_path)
    print_summary(result)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AURA relighting benchmark harness (TensoIR-Synthetic / Stanford-ORB).")
    p.add_argument("--dataset", choices=sorted(_DATASETS.keys()),
                   help="benchmark to evaluate (required unless --smoke).")
    p.add_argument("--smoke", action="store_true",
                   help="run the full metric pipeline on synthetic fixtures (no dataset/GPU).")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="root for the manifest's *_pred paths (GPU-rendered relit frames).")
    p.add_argument("--scenes", nargs="*", default=None, help="restrict to these scene names.")
    p.add_argument("--limit", type=int, default=None, help="cap number of samples.")
    p.add_argument("--output-json", type=Path, default=None, help="write JSON report here.")
    p.add_argument("--output-md", type=Path, default=None, help="write Markdown report here.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        run_smoke(args.output_json, args.output_md)
        return EXIT_OK
    if not args.dataset:
        sys.stderr.write("[relight-harness] --dataset is required unless --smoke.\n")
        return EXIT_USAGE
    root = resolve_dataset_root(args.dataset)  # exits 2 if absent
    samples = load_manifest_samples(
        args.dataset, root, args.predictions_dir, args.scenes, args.limit)
    if not samples:
        sys.stderr.write(
            "[relight-harness] no eval samples to grade — provide predictions and an "
            "aura_relight_eval.json manifest under the dataset root.\n")
        return EXIT_NO_SAMPLES
    rows = [evaluate_sample(s) for s in samples]
    result = build_result(args.dataset, rows, scope="benchmark")
    write_results(result, args.output_json, args.output_md)
    print_summary(result)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
