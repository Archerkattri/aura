"""LangSplatV2-style semantic feature codebook for AURA carriers.

Per-carrier semantic features (e.g. distilled DINO descriptors, ~384-D float32)
are heavy: at N carriers they cost ``N * d * 4`` bytes and an open-vocabulary query
that scores every carrier against a text embedding costs ``O(N * d)``. LangSplatV2
(arXiv:2507.07136) removes both costs with a shared codebook: the heavy vectors live
*once* in a K-entry codebook and each carrier stores only a small integer index into
it. This module implements that layer, CPU-only and deterministic:

  * :func:`fit_codebook` — fit a K-entry codebook (deterministic k-means, seeded).
  * :func:`assign_codes` — map each carrier feature to its nearest codebook index
    (uint8 when ``K <= 256`` else uint16).
  * :func:`reconstruct` — rebuild per-carrier features from indices + codebook.
  * :func:`compression_report` — compression ratio + reconstruction error.
  * :func:`query_codebook` / :func:`open_vocab_query` — the open-vocab path: score
    the K codebook entries against a query embedding (``O(K*d)``), then fan out to
    carriers by index (``O(N)``) — ``O(K*d + N)`` instead of ``O(N*d)``.
  * :func:`save_codebook` / :func:`load_codebook` — ``.npz`` + JSON serialization
    aligned with the ``carriers.npz`` sidecar pattern.

Everything here is numpy-only (no torch, no GPU) so it runs in CI. Distilling the
per-carrier feature tensor from real scenes (multi-view DINO lifting) is the
GPU-gated step upstream of this module — see ``experiments/semantic_distill.py`` —
and is NOT performed here; this module operates on whatever feature matrix it is
handed (synthetic in CI, distilled features when available).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CODEBOOK_NPZ = "codebook.npz"


def _code_dtype(k: int) -> np.dtype:
    """uint8 when the codebook fits in 256 entries, else uint16 (LangSplatV2 trick:
    carriers carry the smallest index type that addresses the codebook)."""
    if k <= 0:
        raise ValueError("codebook size k must be positive")
    if k <= 256:
        return np.dtype(np.uint8)
    if k <= 65536:
        return np.dtype(np.uint16)
    raise ValueError("codebook size k > 65536 does not fit a uint16 index")


@dataclass(frozen=True)
class Codebook:
    """A fitted K-entry feature codebook.

    ``centroids`` is ``[k, dim]`` float32; carriers reference rows of it by index.
    ``seed``/``iters`` are recorded so a fit is reproducible and auditable.
    """

    centroids: np.ndarray  # [k, dim] float32
    seed: int
    iters: int

    @property
    def k(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def code_dtype(self) -> np.dtype:
        return _code_dtype(self.k)


def _as_features(features: Any) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2-D [N, dim] array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("features must contain at least one carrier")
    return arr


def _kmeans_plusplus_init(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic k-means++ seeding using the provided seeded RNG."""
    n = features.shape[0]
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    first = int(rng.integers(n))
    centers[0] = features[first]
    # Squared distance to the nearest chosen center, updated incrementally.
    closest = np.sum((features - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:  # all points already coincide with a center
            centers[c] = features[int(rng.integers(n))]
        else:
            probs = closest / total
            idx = int(rng.choice(n, p=probs))
            centers[c] = features[idx]
        dist = np.sum((features - centers[c]) ** 2, axis=1)
        closest = np.minimum(closest, dist)
    return centers


def fit_codebook(features: Any, k: int, *, seed: int = 0, iters: int = 50,
                 tol: float = 1e-5) -> Codebook:
    """Fit a ``k``-entry codebook over per-carrier ``features`` with deterministic
    (seeded) k-means. ``k`` must be <= the number of carriers. Empty clusters are
    re-seeded to the point farthest from its center, so every codebook row is used.
    """
    feats = _as_features(features)
    n = feats.shape[0]
    if k > n:
        raise ValueError(f"codebook size k={k} cannot exceed carrier count N={n}")
    _code_dtype(k)  # validate index width up front
    rng = np.random.default_rng(seed)
    centers = _kmeans_plusplus_init(feats, k, rng)
    prev_inertia = None
    for _ in range(max(1, iters)):
        # Assign: nearest center by squared Euclidean distance.
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; drop ||x||^2 (constant per row).
        cross = feats @ centers.T                      # [N, k]
        cnorm = np.sum(centers ** 2, axis=1)           # [k]
        d2 = cnorm[None, :] - 2.0 * cross              # argmin-equivalent
        labels = np.argmin(d2, axis=1)
        # Update: mean of assigned points; re-seed empties to the worst-fit point.
        new_centers = centers.copy()
        for c in range(k):
            members = feats[labels == c]
            if members.shape[0] > 0:
                new_centers[c] = members.mean(axis=0)
            else:
                full = np.sum((feats - centers[labels]) ** 2, axis=1)
                new_centers[c] = feats[int(np.argmax(full))]
        recon = new_centers[labels]
        inertia = float(np.sum((feats - recon) ** 2))
        centers = new_centers
        if prev_inertia is not None and abs(prev_inertia - inertia) <= tol * max(prev_inertia, 1e-12):
            break
        prev_inertia = inertia
    return Codebook(centroids=centers.astype(np.float32), seed=int(seed), iters=int(iters))


def assign_codes(features: Any, codebook: Codebook) -> np.ndarray:
    """Assign each carrier feature to its nearest codebook entry. Returns a
    ``[N]`` array of indices in the codebook's compact ``code_dtype``."""
    feats = _as_features(features)
    if feats.shape[1] != codebook.dim:
        raise ValueError(f"feature dim {feats.shape[1]} != codebook dim {codebook.dim}")
    cross = feats @ codebook.centroids.T
    cnorm = np.sum(codebook.centroids ** 2, axis=1)
    d2 = cnorm[None, :] - 2.0 * cross
    return np.argmin(d2, axis=1).astype(codebook.code_dtype)


def reconstruct(codes: Any, codebook: Codebook) -> np.ndarray:
    """Rebuild per-carrier features ``[N, dim]`` from integer ``codes`` and the
    shared codebook (a plain gather — this is the decode side of the trick)."""
    idx = np.asarray(codes)
    if idx.ndim != 1:
        raise ValueError(f"codes must be a 1-D [N] array, got shape {idx.shape}")
    if idx.size and (idx.min() < 0 or idx.max() >= codebook.k):
        raise ValueError("codes index outside the codebook range")
    return codebook.centroids[idx.astype(np.int64)]


def compression_report(features: Any, codebook: Codebook, codes: Any) -> dict[str, Any]:
    """Report codebook compression ratio and reconstruction error.

    ``original`` = ``N*dim*4`` float32 bytes. ``compressed`` = codebook matrix
    (``k*dim*4``) shared once + per-carrier index (``N * code_bytes``). Error is the
    per-carrier feature reconstruction error (RMSE + Frobenius-relative).
    """
    feats = _as_features(features)
    idx = np.asarray(codes)
    recon = reconstruct(idx, codebook)
    n, dim = feats.shape
    code_bytes = int(codebook.code_dtype.itemsize)
    original_bytes = int(n * dim * 4)
    codebook_bytes = int(codebook.k * dim * 4)
    index_bytes = int(n * code_bytes)
    compressed_bytes = codebook_bytes + index_bytes
    diff = feats - recon
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    fro = float(np.linalg.norm(feats))
    relative_error = float(np.linalg.norm(diff) / fro) if fro > 0 else 0.0
    return {
        "carriers": int(n),
        "feature_dim": int(dim),
        "codebook_size": int(codebook.k),
        "code_dtype": codebook.code_dtype.name,
        "original_bytes": original_bytes,
        "codebook_bytes": codebook_bytes,
        "index_bytes": index_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": float(original_bytes / compressed_bytes) if compressed_bytes else 0.0,
        "reconstruction_mse": mse,
        "reconstruction_rmse": rmse,
        "reconstruction_relative_error": relative_error,
    }


def query_codebook(query_embedding: Any, codebook: Codebook, codes: Any, *,
                   normalize: bool = False, top_k: int | None = None) -> dict[str, Any]:
    """Open-vocabulary query in ``O(K*d + N)``.

    Scores the ``K`` codebook entries against ``query_embedding`` (``O(K*d)``) then
    fans out to carriers by index (``O(N)``) — never the ``O(N*d)`` dense scan. With
    ``normalize=True`` the query and codebook rows are L2-normalized so the score is
    cosine similarity. Returns ``code_scores`` ``[K]``, ``carrier_scores`` ``[N]``,
    and (when ``top_k`` is set) the indices of the top carriers.
    """
    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if q.shape[0] != codebook.dim:
        raise ValueError(f"query dim {q.shape[0]} != codebook dim {codebook.dim}")
    centroids = codebook.centroids
    if normalize:
        q = q / (np.linalg.norm(q) + 1e-12)
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    code_scores = centroids @ q                       # [K]  — the only O(K*d) work
    idx = np.asarray(codes)
    carrier_scores = code_scores[idx.astype(np.int64)]  # [N]  — O(N) gather
    out: dict[str, Any] = {
        "code_scores": code_scores,
        "carrier_scores": carrier_scores,
    }
    if top_k is not None:
        k = int(min(max(top_k, 0), carrier_scores.shape[0]))
        top = np.argsort(-carrier_scores)[:k]
        out["top_carriers"] = top
        out["top_scores"] = carrier_scores[top]
    return out


def open_vocab_query(query_embedding: Any, codebook: Codebook, codes: Any, *,
                     normalize: bool = True, top_k: int = 10) -> np.ndarray:
    """Convenience wrapper: return the indices of the ``top_k`` carriers best
    matching ``query_embedding`` (cosine by default)."""
    return query_codebook(query_embedding, codebook, codes,
                          normalize=normalize, top_k=top_k)["top_carriers"]


def save_codebook(path: Any, codebook: Codebook, codes: Any, *,
                  report: dict[str, Any] | None = None) -> Path:
    """Serialize a codebook + carrier codes to ``<path>/codebook.npz`` (path may be a
    package dir or a file), plus a ``<stem>.json`` metadata sidecar, mirroring the
    ``carriers.npz`` pattern so the codebook lives next to the carriers it indexes."""
    out = Path(path)
    target = out / CODEBOOK_NPZ if out.is_dir() or not out.suffix else out
    target.parent.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(codes).astype(codebook.code_dtype)
    np.savez(
        target,
        codebook=codebook.centroids.astype(np.float32),
        codes=idx,
        k=np.int64(codebook.k),
        dim=np.int64(codebook.dim),
        seed=np.int64(codebook.seed),
        iters=np.int64(codebook.iters),
    )
    meta = {
        "format": "AURA_SEMANTIC_CODEBOOK",
        "k": codebook.k,
        "dim": codebook.dim,
        "seed": codebook.seed,
        "iters": codebook.iters,
        "code_dtype": codebook.code_dtype.name,
        "carriers": int(idx.shape[0]),
    }
    if report is not None:
        meta["compression"] = report
    target.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return target


def load_codebook(path: Any) -> tuple[Codebook, np.ndarray]:
    """Load a codebook + carrier codes written by :func:`save_codebook`. Returns
    ``(Codebook, codes)``."""
    p = Path(path)
    f = p / CODEBOOK_NPZ if p.is_dir() else p
    if not f.exists():
        raise FileNotFoundError(f"no codebook at {f}")
    z = np.load(f)
    codebook = Codebook(
        centroids=np.asarray(z["codebook"], dtype=np.float32),
        seed=int(z["seed"]),
        iters=int(z["iters"]),
    )
    codes = np.asarray(z["codes"]).astype(codebook.code_dtype)
    return codebook, codes
