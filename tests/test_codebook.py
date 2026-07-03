"""Tests for the LangSplatV2-style semantic feature codebook (aura.codebook).

CI-safe: pure numpy, synthetic clustered features. A ``local_data`` test validates
on a distilled real per-carrier feature tensor when one is present on disk (the
distillation itself is GPU-gated; see experiments/semantic_distill.py), else skips.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura import codebook as C  # noqa: E402


def _blobs(k=6, per=50, dim=32, spread=10.0, noise=0.1, seed=1):
    """k well-separated Gaussian blobs -> (features [N,dim], labels [N])."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= spread
    feats, labels = [], []
    for c in range(k):
        feats.append(centers[c] + rng.normal(scale=noise, size=(per, dim)).astype(np.float32))
        labels.append(np.full(per, c))
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels), centers


# --- fit / assign / reconstruct -------------------------------------------------

def test_fit_is_deterministic_for_a_seed():
    feats, _, _ = _blobs()
    a = C.fit_codebook(feats, 6, seed=7)
    b = C.fit_codebook(feats, 6, seed=7)
    assert np.array_equal(a.centroids, b.centroids)
    assert a.k == 6 and a.dim == feats.shape[1]


def test_code_dtype_is_uint8_small_uint16_large():
    feats, _, _ = _blobs(dim=8)
    assert C.fit_codebook(feats, 6, seed=0).code_dtype == np.uint8
    big = np.random.default_rng(0).normal(size=(400, 4)).astype(np.float32)
    cb = C.fit_codebook(big, 300, seed=0)
    assert cb.code_dtype == np.uint16


def test_assign_and_reconstruct_low_error_on_clusters():
    feats, labels, _ = _blobs()
    cb = C.fit_codebook(feats, 6, seed=0)
    codes = C.assign_codes(feats, cb)
    assert codes.shape == (feats.shape[0],)
    assert codes.dtype == cb.code_dtype
    report = C.compression_report(feats, cb, codes)
    # Well-separated blobs reconstruct from 6 atoms with small relative error.
    assert report["reconstruction_relative_error"] < 0.1
    # Carriers sharing a true cluster should share a codebook index.
    for c in np.unique(labels):
        assert len(set(codes[labels == c].tolist())) == 1


def test_compression_ratio_beats_one_and_matches_formula():
    feats, _, _ = _blobs(k=6, per=50, dim=32)
    cb = C.fit_codebook(feats, 6, seed=0)
    codes = C.assign_codes(feats, cb)
    r = C.compression_report(feats, cb, codes)
    n, dim = feats.shape
    assert r["original_bytes"] == n * dim * 4
    assert r["compressed_bytes"] == 6 * dim * 4 + n * 1  # uint8 index
    assert r["compression_ratio"] > 1.0
    assert r["compression_ratio"] == pytest.approx(r["original_bytes"] / r["compressed_bytes"])


def test_k_larger_than_carriers_rejected():
    feats, _, _ = _blobs(k=2, per=3, dim=4)  # N = 6
    with pytest.raises(ValueError):
        C.fit_codebook(feats, 10, seed=0)


# --- open-vocab query -----------------------------------------------------------

def test_open_vocab_query_returns_in_cluster_carriers():
    feats, labels, centers = _blobs(k=6, per=50, dim=32)
    cb = C.fit_codebook(feats, 6, seed=0)
    codes = C.assign_codes(feats, cb)
    target = 3
    top = C.open_vocab_query(centers[target], cb, codes, top_k=10)
    assert len(top) == 10
    assert all(labels[i] == target for i in top)


def test_query_scores_are_code_indexed_fanout():
    feats, _, centers = _blobs()
    cb = C.fit_codebook(feats, 6, seed=0)
    codes = C.assign_codes(feats, cb)
    out = C.query_codebook(centers[0], cb, codes, normalize=True)
    # carrier_scores must be exactly the per-code score gathered by index (O(K*d + N)).
    assert out["code_scores"].shape == (cb.k,)
    assert out["carrier_scores"].shape == (feats.shape[0],)
    assert np.allclose(out["carrier_scores"], out["code_scores"][codes.astype(np.int64)])


def test_query_dim_mismatch_rejected():
    feats, _, _ = _blobs(dim=16)
    cb = C.fit_codebook(feats, 4, seed=0)
    codes = C.assign_codes(feats, cb)
    with pytest.raises(ValueError):
        C.query_codebook(np.zeros(8, np.float32), cb, codes)


# --- serialization (carriers.npz-aligned sidecar) -------------------------------

def test_save_load_round_trip(tmp_path):
    feats, _, _ = _blobs()
    cb = C.fit_codebook(feats, 6, seed=0)
    codes = C.assign_codes(feats, cb)
    report = C.compression_report(feats, cb, codes)
    target = C.save_codebook(tmp_path, cb, codes, report=report)
    assert target.name == C.CODEBOOK_NPZ
    assert (tmp_path / "codebook.json").exists()
    cb2, codes2 = C.load_codebook(tmp_path)
    assert np.array_equal(cb.centroids, cb2.centroids)
    assert np.array_equal(codes, codes2)
    assert cb2.seed == cb.seed and cb2.iters == cb.iters


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_codebook(tmp_path)


# --- real distilled features (GPU-gated upstream; validated here if present) -----

def _distilled_feature_candidates():
    base = Path("/tmp/dbs_out")
    cands = [base / "truck_beta" / "carrier_features.npz"]
    if base.exists():
        cands += sorted(base.glob("*/carrier_features.npz"))
    return cands


@pytest.mark.local_data
def test_codebook_on_real_distilled_features():
    src = next((p for p in _distilled_feature_candidates() if p.exists()), None)
    if src is None:
        pytest.skip("no distilled carrier_features.npz present (run experiments/semantic_distill.py on GPU)")
    z = np.load(src)
    feats = np.asarray(z["features"], dtype=np.float32)
    feats = feats[np.linalg.norm(feats, axis=1) > 0]  # drop unseen carriers
    assert feats.shape[0] > 32 and feats.ndim == 2
    cb = C.fit_codebook(feats, 64, seed=0)
    codes = C.assign_codes(feats, cb)
    r = C.compression_report(feats, cb, codes)
    assert r["compression_ratio"] > 1.0
    assert r["reconstruction_relative_error"] < 1.0
    # Query with a real carrier's own feature: it should rank itself in the top set.
    probe = feats[0]
    top = C.open_vocab_query(probe, cb, codes, top_k=50)
    assert codes[0] in codes[top]
