"""Tests for the per-scene train/eval view split guard (aura.split_guard).

The historical P0 leak: carriers trained on ALL frames while "held-out" reliability
views were a subset of those same frames. These tests lock in that a synthetic
leaked split FAILS the guard and a clean llffhold split passes.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aura.split_guard import (  # noqa: E402
    SplitLeakError,
    assert_disjoint_views,
    holdout_split,
    verify_recorded_split,
)


def test_holdout_split_is_disjoint_and_covering():
    split = holdout_split(251, holdout=8)
    assert split.eval_count == 32  # ceil(251/8)
    assert split.train_count == 219
    assert set(split.train_views).isdisjoint(split.eval_views)
    assert set(split.train_views) | set(split.eval_views) == set(range(251))
    # index 0 and every 8th are eval; the rest are train
    assert 0 in split.eval_views and 8 in split.eval_views
    assert 1 in split.train_views and 7 in split.train_views


def test_holdout_split_rejects_degenerate_stride():
    with pytest.raises(ValueError):
        holdout_split(100, holdout=1)  # every frame eval -> no training views
    with pytest.raises(ValueError):
        holdout_split(0, holdout=8)


def test_assert_disjoint_views_passes_on_disjoint_sets():
    assert assert_disjoint_views([1, 2, 3], [0, 8, 16]) is True


def test_assert_disjoint_views_fails_on_overlap():
    with pytest.raises(SplitLeakError):
        assert_disjoint_views([0, 1, 2, 3], [2, 8], context="probe")


def test_assert_disjoint_views_fails_on_empty_eval():
    with pytest.raises(SplitLeakError):
        assert_disjoint_views([0, 1, 2], [])


def test_verify_recorded_split_accepts_clean_llffhold_counts():
    # Committed real-scene counts (p2_summary.json full-res arms) are clean.
    for train_count, eval_count in ((219, 32), (161, 24), (244, 35), (272, 39)):
        split = verify_recorded_split(train_count=train_count, eval_count=eval_count, holdout=8)
        assert split.train_count == train_count
        assert split.eval_count == eval_count


def test_verify_recorded_split_catches_train_on_all_frames_leak():
    # The P0 leak fingerprint: carriers trained on ALL 251 frames while eval reused
    # the 32 held-out views -> recorded train_count == total frames.
    with pytest.raises(SplitLeakError) as excinfo:
        verify_recorded_split(train_count=251, eval_count=32, holdout=8, context="truck")
    assert "leak" in str(excinfo.value).lower()


def test_verify_recorded_split_rejects_negative_counts():
    with pytest.raises(SplitLeakError):
        verify_recorded_split(train_count=-1, eval_count=32, holdout=8)
