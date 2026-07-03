"""Per-scene train/eval view split guard.

AURA's P0 result once leaked: carriers were trained on ALL frames while the
"held-out" reliability/calibration views were a subset of those SAME frames, so
the reliability numbers were evaluated on views the carriers had already seen
(docs/P2_FULLRES_RENDERLOSS.md, "P0 eval leak found+fixed"). This module makes
that class of leak mechanically detectable and prevents it at the source:

  * ``holdout_split`` is the single source of truth for the llffhold / ``test_every``
    partition every reliability producer must use — every ``holdout``-th frame is an
    EVAL view, the rest are TRAIN views, and the two are disjoint by construction.
  * ``assert_disjoint_views`` refuses any explicit split in which an evaluation view
    is also a training view.
  * ``verify_recorded_split`` reconstructs the partition from the per-scene view
    COUNTS a producer recorded and rejects counts that are not a clean disjoint
    holdout partition — the fingerprint of the historical leak (train_count == all
    frames while eval reused a held-out subset).

A gate that consumes reliability/calibration artifacts can call
``verify_recorded_split`` so a re-introduced leak fails the gate instead of quietly
inflating the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass


class SplitLeakError(AssertionError):
    """Raised when evaluation views are not disjoint from training views."""


@dataclass(frozen=True)
class ViewSplit:
    """A disjoint train/eval partition of ``num_frames`` frames."""

    train_views: tuple[int, ...]
    eval_views: tuple[int, ...]
    num_frames: int
    holdout: int

    @property
    def train_count(self) -> int:
        return len(self.train_views)

    @property
    def eval_count(self) -> int:
        return len(self.eval_views)


def assert_disjoint_views(train_views, eval_views, *, context: str = "") -> bool:
    """Assert that ``eval_views`` are held out — non-empty and disjoint from training.

    Raises :class:`SplitLeakError` when the evaluation set is empty (nothing held
    out) or shares any view with the training set (held-out views seen in training,
    the P0 leak class).
    """
    prefix = f"{context}: " if context else ""
    train = set(train_views)
    evals = set(eval_views)
    if not evals:
        raise SplitLeakError(f"{prefix}evaluation view set is empty — nothing is held out")
    if not train:
        raise SplitLeakError(f"{prefix}training view set is empty — no carriers were trained")
    overlap = train & evals
    if overlap:
        sample = sorted(overlap)[:8]
        raise SplitLeakError(
            f"{prefix}{len(overlap)} evaluation view(s) also used for training "
            f"(e.g. {sample}) — held-out views were seen in training (P0 leak class)"
        )
    return True


def holdout_split(num_frames: int, holdout: int = 8) -> ViewSplit:
    """Deterministic llffhold partition: index ``% holdout == 0`` -> EVAL, else TRAIN.

    This is the single source of truth for the reliability/calibration split; the
    returned sets are guaranteed disjoint and covering.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if holdout <= 1:
        raise ValueError(f"holdout must be >= 2 (else no training views), got {holdout}")
    eval_views = tuple(i for i in range(num_frames) if i % holdout == 0)
    train_views = tuple(i for i in range(num_frames) if i % holdout != 0)
    split = ViewSplit(train_views, eval_views, num_frames, holdout)
    assert_disjoint_views(
        split.train_views, split.eval_views,
        context=f"holdout_split(num_frames={num_frames}, holdout={holdout})",
    )
    return split


def verify_recorded_split(*, train_count: int, eval_count: int, holdout: int = 8,
                          context: str = "") -> ViewSplit:
    """Reject recorded per-scene view counts that are not a clean holdout partition.

    A clean llffhold partition of ``N`` frames has exactly ``ceil(N/holdout)`` eval
    views and ``N - that`` train views with ``train_count + eval_count == N``. If
    carriers were trained on ALL ``N`` frames while eval reused the held-out subset,
    the producer records ``train_count == N`` and ``train_count + eval_count > N``, so
    the reconstructed partition of ``train_count + eval_count`` frames no longer
    matches the recorded counts and the leak is caught. Returns the reconstructed
    :class:`ViewSplit` on success.
    """
    prefix = f"{context}: " if context else ""
    if train_count < 0 or eval_count < 0:
        raise SplitLeakError(f"{prefix}negative view counts ({train_count}/{eval_count})")
    total = train_count + eval_count
    if total <= 0:
        raise SplitLeakError(f"{prefix}no views recorded")
    split = holdout_split(total, holdout)
    if split.eval_count != eval_count or split.train_count != train_count:
        raise SplitLeakError(
            f"{prefix}recorded train/eval counts ({train_count}/{eval_count}) are not a "
            f"disjoint llffhold-{holdout} partition of {total} frames "
            f"(a clean partition is {split.train_count}/{split.eval_count}); "
            f"held-out views overlap the training set (P0 leak class)"
        )
    return split
