"""Finite-family certificates for distortion-controlled AURA streaming.

The existing :mod:`aura.lod` certificate controls discarded *reliability*.
That is useful trust metadata, but it is not a rendered-image guarantee.  This
module adds the missing, deliberately narrower contract: a consumer declares a
finite, nested ladder of already-encoded representations and calibrates the
measured distortion of every level against a frozen full asset on a disjoint
calibration set.  A Bonferroni split over that predeclared family permits a
level to be selected after calibration without silently claiming validity for
an unseen codec, camera distribution, or post-hoc candidate.

The loss supplied here is a bounded, normalized per-view loss.  It can be a
rendered full-asset difference (the primary certificate target), but the
module does not pretend that a reliability proxy or a sum of carrier scores is
the same thing.  Optional real-image losses are retained as a separate
held-out diagnostic and never get relabelled as the certificate target.

This is a finite-family empirical risk-control primitive, not a universal
compression theorem.  Production consumers should validate the emitted
metadata against the exact asset, renderer and codec before accepting a level.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


FORMAT = "AURA_STREAM_DISTORTION_CERTIFICATE"
SCHEMA_VERSION = 1


def _bounded_vector(values: Any, name: str) -> Any:
    """Return a finite float vector and reject silently clipped measurements."""
    import numpy as np

    arr = np.asarray(values, dtype="float64").ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite observations")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must be normalized to the closed interval [0, 1]")
    return arr


def _finite_scalar(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _canonical(value: Any) -> Any:
    """Make nested metadata deterministic and JSON-compatible."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        return value
    # Numpy scalars are common in benchmark metadata.
    if hasattr(value, "item"):
        return _canonical(value.item())
    return str(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _array_digest(values: Any) -> str:
    import numpy as np

    arr = np.asarray(values, dtype="<f8").ravel()
    payload = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
    }
    return _digest(payload)


def _p95(values: Any) -> float:
    import numpy as np

    return float(np.quantile(values, 0.95, method="linear"))


@dataclass(frozen=True)
class StreamCandidate:
    """One member of a predeclared nested representation ladder.

    Candidates are supplied to :func:`calibrate_distortion_budget` from the
    least retained representation to the full asset.  ``retained_fraction``
    is descriptive when a level combines pruning and quantization; the input
    order is still the authoritative nesting order.  ``encoded_bytes`` and
    ``render_time_ms`` are measured costs, not estimates used by the risk
    bound.
    """

    level_id: str
    encoded_bytes: int
    render_time_ms: float
    retained_fraction: float | None = None
    family: str = "finite-ladder"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    is_full_asset: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.level_id, str) or not self.level_id.strip():
            raise ValueError("level_id must be a non-empty string")
        if isinstance(self.encoded_bytes, bool) or int(self.encoded_bytes) != self.encoded_bytes:
            raise ValueError("encoded_bytes must be a non-negative integer")
        if int(self.encoded_bytes) < 0:
            raise ValueError("encoded_bytes must be a non-negative integer")
        object.__setattr__(self, "encoded_bytes", int(self.encoded_bytes))
        render_time = _finite_scalar(self.render_time_ms, "render_time_ms")
        if render_time < 0.0:
            raise ValueError("render_time_ms must be non-negative")
        object.__setattr__(self, "render_time_ms", render_time)
        if self.retained_fraction is not None:
            retained = _finite_scalar(self.retained_fraction, "retained_fraction")
            if not 0.0 < retained <= 1.0:
                raise ValueError("retained_fraction must be in (0, 1]")
            object.__setattr__(self, "retained_fraction", retained)
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StreamCandidate":
        if not isinstance(value, Mapping):
            raise TypeError("candidate must be a StreamCandidate or mapping")
        return cls(
            level_id=str(value.get("level_id", value.get("id", ""))),
            encoded_bytes=value.get("encoded_bytes", value.get("bytes", -1)),
            render_time_ms=value.get("render_time_ms", value.get("render_time", -1.0)),
            retained_fraction=value.get("retained_fraction", value.get("keep_fraction")),
            family=str(value.get("family", "finite-ladder")),
            parameters=value.get("parameters", {}),
            is_full_asset=bool(value.get("is_full_asset", value.get("full_asset", False))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "encoded_bytes": self.encoded_bytes,
            "render_time_ms": self.render_time_ms,
            "retained_fraction": self.retained_fraction,
            "family": self.family,
            "parameters": _canonical(self.parameters),
            "is_full_asset": self.is_full_asset,
        }


def _normalise_candidates(candidates: Sequence[StreamCandidate | Mapping[str, Any]]) -> list[StreamCandidate]:
    if not candidates:
        raise ValueError("candidates must be a non-empty finite ladder")
    out = [
        candidate if isinstance(candidate, StreamCandidate) else StreamCandidate.from_mapping(candidate)
        for candidate in candidates
    ]
    ids = [candidate.level_id for candidate in out]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate level_id values must be unique")
    full_indices = [i for i, candidate in enumerate(out) if candidate.is_full_asset]
    if len(full_indices) > 1:
        raise ValueError("the finite ladder may contain at most one full-asset level")
    if full_indices and full_indices[0] != len(out) - 1:
        raise ValueError("the full-asset level must be the final member of the nested ladder")
    retained = [candidate.retained_fraction for candidate in out]
    if all(value is not None for value in retained):
        for previous, current in zip(retained, retained[1:]):
            if float(current) <= float(previous):
                raise ValueError(
                    "retained_fraction must increase strictly in the declared nested ladder"
                )
    # A representation that is declared as progressively retained should not
    # become cheaper in encoded bytes as more content is retained.  Render time
    # is allowed to be non-monotone because decoder kernels can have different
    # occupancy at different levels.
    byte_sizes = [candidate.encoded_bytes for candidate in out]
    if any(current < previous for previous, current in zip(byte_sizes, byte_sizes[1:])):
        raise ValueError("encoded_bytes must be non-decreasing in the nested ladder")
    return out


def _validate_ids(ids: Sequence[str] | None, expected_length: int, name: str) -> tuple[str, ...] | None:
    if ids is None:
        return None
    values = tuple(str(value) for value in ids)
    if len(values) != expected_length:
        raise ValueError(f"{name} must have one entry per observation")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique observation IDs")
    return values


def _loss_stats(values: Any) -> dict[str, float | int]:
    import numpy as np

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p95": _p95(values),
        "max": float(values.max()),
        "digest": _array_digest(values),
    }


def calibrate_distortion_budget(
    candidates: Sequence[StreamCandidate | Mapping[str, Any]],
    calibration_distortions: Mapping[str, Any],
    *,
    alpha: float = 0.1,
    risk_budget: float = 0.05,
    asset_id: str = "",
    renderer_id: str = "",
    renderer_version: str = "",
    codec_id: str = "",
    codec_version: str = "",
    calibration_view_ids: Sequence[str] | None = None,
    evaluation_view_ids: Sequence[str] | None = None,
    evaluation_image_losses: Mapping[str, Any] | None = None,
    source_digest: str | None = None,
) -> dict[str, Any]:
    """Calibrate a finite nested stream ladder on a disjoint calibration set.

    ``calibration_distortions[level_id]`` must contain bounded per-view
    distortion against a frozen full asset.  The same calibration observations
    are used for every declared level, and ``alpha`` is split over non-full
    levels only.  A full-asset level is deterministic zero distortion and does
    not spend error budget.  If ``evaluation_image_losses`` is supplied, those
    values are summarized only as a separate held-out diagnostic; they do not
    enlarge or replace the certificate target.

    The output is JSON-able and contains a digest of the calibration arrays,
    candidate costs, and risk parameters.  This digest is intended to be bound
    into :func:`make_stream_metadata`.
    """
    import numpy as np

    candidates = _normalise_candidates(candidates)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 <= float(risk_budget) <= 1.0:
        raise ValueError("risk_budget must be in [0, 1]")
    alpha = float(alpha)
    risk_budget = float(risk_budget)
    if not isinstance(calibration_distortions, Mapping):
        raise TypeError("calibration_distortions must map level IDs to arrays")
    expected_ids = {candidate.level_id for candidate in candidates}
    supplied_ids = {str(key) for key in calibration_distortions}
    missing = expected_ids - supplied_ids
    extra = supplied_ids - expected_ids
    if missing:
        raise ValueError(f"missing calibration distortion levels: {sorted(missing)}")
    if extra:
        raise ValueError(f"unknown calibration distortion levels: {sorted(extra)}")

    arrays: dict[str, Any] = {}
    n_cal: int | None = None
    for candidate in candidates:
        values = _bounded_vector(
            calibration_distortions[candidate.level_id],
            f"calibration_distortions[{candidate.level_id!r}]",
        )
        if n_cal is None:
            n_cal = int(values.size)
        elif values.size != n_cal:
            raise ValueError("all candidate levels must use the same calibration observations")
        if candidate.is_full_asset and not np.allclose(values, 0.0, atol=0.0, rtol=0.0):
            raise ValueError("a full-asset level must have exactly zero full-asset distortion")
        arrays[candidate.level_id] = values
    assert n_cal is not None

    cal_ids = _validate_ids(calibration_view_ids, n_cal, "calibration_view_ids")
    eval_ids = None
    if evaluation_image_losses is not None:
        if not isinstance(evaluation_image_losses, Mapping):
            raise TypeError("evaluation_image_losses must map level IDs to arrays")
        eval_keys = {str(key) for key in evaluation_image_losses}
        if eval_keys != expected_ids:
            raise ValueError("evaluation_image_losses must contain exactly the candidate level IDs")
        eval_arrays = {
            candidate.level_id: _bounded_vector(
                evaluation_image_losses[candidate.level_id],
                f"evaluation_image_losses[{candidate.level_id!r}]",
            )
            for candidate in candidates
        }
        n_eval = len(next(iter(eval_arrays.values())))
        if any(len(values) != n_eval for values in eval_arrays.values()):
            raise ValueError("all candidate levels must use the same evaluation observations")
        eval_ids = _validate_ids(evaluation_view_ids, n_eval, "evaluation_view_ids")
        if cal_ids is not None and eval_ids is not None and set(cal_ids) & set(eval_ids):
            raise ValueError("calibration_view_ids and evaluation_view_ids must be disjoint")
    elif evaluation_view_ids is not None:
        raise ValueError("evaluation_view_ids require evaluation_image_losses")

    nontrivial = [candidate for candidate in candidates if not candidate.is_full_asset]
    alpha_per_level = alpha / len(nontrivial) if nontrivial else alpha
    from aura.calibration import conformal_mean_upper_bound

    levels: list[dict[str, Any]] = []
    for candidate in candidates:
        values = arrays[candidate.level_id]
        trivial = candidate.is_full_asset
        epsilon = 0.0 if trivial else min(1.0, conformal_mean_upper_bound(values, alpha_per_level))
        level: dict[str, Any] = {
            **candidate.to_dict(),
            "trivial": trivial,
            "calibration": _loss_stats(values),
            "epsilon_certified": float(epsilon),
            "alpha_per_level": 0.0 if trivial else alpha_per_level,
            "risk_budget": risk_budget,
            "qualifies": bool(trivial or epsilon <= risk_budget),
        }
        if evaluation_image_losses is not None:
            evaluation_values = eval_arrays[candidate.level_id]
            level["real_image_evaluation"] = _loss_stats(evaluation_values)
        else:
            level["real_image_evaluation"] = None
        levels.append(level)

    evidence_payload = {
        "calibration_view_ids": list(cal_ids) if cal_ids is not None else None,
        "calibration_levels": {
            candidate.level_id: _array_digest(arrays[candidate.level_id])
            for candidate in candidates
        },
        "evaluation_view_ids": list(eval_ids) if eval_ids is not None else None,
        "source_digest": source_digest,
    }
    plan: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "asset_id": str(asset_id),
        "renderer": {"id": str(renderer_id), "version": str(renderer_version)},
        "codec": {"id": str(codec_id), "version": str(codec_version)},
        "alpha": alpha,
        "alpha_per_level": alpha_per_level,
        "family_wise_confidence": 1.0 - alpha,
        "risk_budget": risk_budget,
        "loss": {
            "name": "normalized_full_asset_per_view_distortion",
            "bounded": [0.0, 1.0],
            "certificate_target": "frozen_full_asset_render_distortion",
            "real_image_quality_is_separate_diagnostic": True,
        },
        "correction": "bonferroni_over_nontrivial_levels",
        "n_calibration": n_cal,
        "n_levels": len(candidates),
        "n_nontrivial_levels": len(nontrivial),
        "calibration_evidence": evidence_payload,
        "calibration_set_digest": _digest(evidence_payload),
        "levels": levels,
        "nested_order": [candidate.level_id for candidate in candidates],
        "selection_rule": "choose_only_from_predeclared_levels_with_certified_epsilon_within_budget",
        "claim_boundary": [
            "valid only for this finite declared candidate family and calibration protocol",
            "does not certify arbitrary unseen codecs, camera distributions, or post-hoc levels",
            "real-image evaluation is reported separately and is not covered by the full-asset certificate",
        ],
    }
    plan["plan_digest"] = _digest(plan)
    return plan


def evaluate_stream_certificate(
    plan: Mapping[str, Any],
    heldout_distortions: Mapping[str, Any],
    *,
    heldout_view_ids: Sequence[str] | None = None,
    heldout_real_image_losses: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every declared level on a fresh, disjoint observation block.

    This function does not refit or change the certificate.  It reports the
    empirical mean full-asset distortion and whether that mean is below the
    level's precomputed bound.  Optional real-image losses are summarized under
    a separate key so a user can see reconstruction quality against actual
    images without confusing it with the certificate target.
    """
    if not isinstance(plan, Mapping) or plan.get("format") != FORMAT:
        raise ValueError("plan is not an AURA stream distortion certificate")
    if not isinstance(heldout_distortions, Mapping):
        raise TypeError("heldout_distortions must map level IDs to arrays")
    levels = list(plan.get("levels", []))
    level_ids = {str(level["level_id"]) for level in levels}
    supplied_ids = {str(key) for key in heldout_distortions}
    if supplied_ids != level_ids:
        raise ValueError("heldout_distortions must contain exactly the certificate level IDs")
    arrays = {
        str(level["level_id"]): _bounded_vector(
            heldout_distortions[level["level_id"]],
            f"heldout_distortions[{level['level_id']!r}]",
        )
        for level in levels
    }
    n_eval = len(next(iter(arrays.values())))
    if any(len(values) != n_eval for values in arrays.values()):
        raise ValueError("all levels must use the same held-out observations")
    eval_ids = _validate_ids(heldout_view_ids, n_eval, "heldout_view_ids")
    cal_ids = plan.get("calibration_evidence", {}).get("calibration_view_ids")
    if cal_ids is not None and eval_ids is not None and set(cal_ids) & set(eval_ids):
        raise ValueError("heldout_view_ids overlap the calibration_view_ids")

    real_arrays: dict[str, Any] | None = None
    if heldout_real_image_losses is not None:
        if not isinstance(heldout_real_image_losses, Mapping):
            raise TypeError("heldout_real_image_losses must map level IDs to arrays")
        if {str(key) for key in heldout_real_image_losses} != level_ids:
            raise ValueError("heldout_real_image_losses must contain exactly the certificate level IDs")
        real_arrays = {
            str(level["level_id"]): _bounded_vector(
                heldout_real_image_losses[level["level_id"]],
                f"heldout_real_image_losses[{level['level_id']!r}]",
            )
            for level in levels
        }
        n_real = len(next(iter(real_arrays.values())))
        if any(len(values) != n_real for values in real_arrays.values()):
            raise ValueError("all levels must use the same real-image held-out observations")

    reports: list[dict[str, Any]] = []
    for level in levels:
        level_id = str(level["level_id"])
        values = arrays[level_id]
        if bool(level.get("is_full_asset", False)) and not all(value == 0.0 for value in values):
            raise ValueError("a full-asset held-out distortion vector must be exactly zero")
        stats = _loss_stats(values)
        report: dict[str, Any] = {
            "level_id": level_id,
            "is_full_asset": bool(level.get("is_full_asset", False)),
            "epsilon_certified": float(level["epsilon_certified"]),
            "heldout_full_asset_distortion": stats,
            "holds": bool(stats["mean"] <= float(level["epsilon_certified"]) + 1e-12),
        }
        if real_arrays is not None:
            report["heldout_real_image_loss"] = _loss_stats(real_arrays[level_id])
        else:
            report["heldout_real_image_loss"] = None
        reports.append(report)
    evidence = {
        "heldout_view_ids": list(eval_ids) if eval_ids is not None else None,
        "levels": {level_id: _array_digest(values) for level_id, values in arrays.items()},
        "real_image_evaluation": real_arrays is not None,
    }
    return {
        "format": "AURA_STREAM_DISTORTION_EVALUATION",
        "schema_version": SCHEMA_VERSION,
        "plan_digest": plan.get("plan_digest"),
        "n_evaluation": n_eval,
        "evaluation_set_digest": _digest(evidence),
        "levels": reports,
        "all_levels_hold": all(report["holds"] for report in reports),
        "claim_boundary": [
            "held-out checks are empirical validation of the frozen finite-family plan",
            "real-image loss remains a separate diagnostic unless separately calibrated",
        ],
    }


def choose_stream_level(
    plan: Mapping[str, Any],
    *,
    distortion_budget: float | None = None,
    max_bytes: int | None = None,
    max_render_time_ms: float | None = None,
    objective: str = "bytes",
    allow_full_asset: bool = True,
) -> dict[str, Any]:
    """Choose the cheapest qualifying predeclared level, or abstain safely.

    The decision is made only over levels already present in ``plan``.  If no
    non-full level qualifies, a full-asset level is selected when permitted and
    it fits the operational cost constraints; otherwise the returned action is
    ``"abstain"``.  This makes a missing certificate a visible state rather
    than an implicit unsafe early stop.
    """
    if not isinstance(plan, Mapping) or plan.get("format") != FORMAT:
        raise ValueError("plan is not an AURA stream distortion certificate")
    if objective not in {"bytes", "render_time_ms"}:
        raise ValueError("objective must be 'bytes' or 'render_time_ms'")
    budget = plan.get("risk_budget") if distortion_budget is None else distortion_budget
    budget = _finite_scalar(budget, "distortion_budget")
    if not 0.0 <= budget <= 1.0:
        raise ValueError("distortion_budget must be in [0, 1]")
    if max_bytes is not None and (isinstance(max_bytes, bool) or int(max_bytes) != max_bytes or int(max_bytes) < 0):
        raise ValueError("max_bytes must be a non-negative integer")
    if max_render_time_ms is not None:
        max_render_time_ms = _finite_scalar(max_render_time_ms, "max_render_time_ms")
        if max_render_time_ms < 0.0:
            raise ValueError("max_render_time_ms must be non-negative")

    levels = list(plan.get("levels", []))
    if not levels:
        return {"action": "abstain", "selected_level_id": None, "reason": "empty_certificate"}

    def fits(level: Mapping[str, Any]) -> bool:
        if float(level.get("epsilon_certified", 1.0)) > budget + 1e-12:
            return False
        if max_bytes is not None and int(level.get("encoded_bytes", -1)) > int(max_bytes):
            return False
        if max_render_time_ms is not None and float(level.get("render_time_ms", math.inf)) > max_render_time_ms:
            return False
        return True

    nonfull = [level for level in levels if not bool(level.get("is_full_asset", False))]
    qualifying = [level for level in nonfull if fits(level)]
    if qualifying:
        key = (
            (lambda level: (int(level["encoded_bytes"]), float(level["render_time_ms"])))
            if objective == "bytes"
            else (lambda level: (float(level["render_time_ms"]), int(level["encoded_bytes"])))
        )
        selected = min(qualifying, key=key)
        return {
            "action": "select",
            "selected_level_id": selected["level_id"],
            "reason": "smallest_predeclared_level_within_certified_risk_and_cost",
            "distortion_budget": budget,
            "objective": objective,
            "used_full_asset_fallback": False,
        }

    full = next((level for level in levels if bool(level.get("is_full_asset", False))), None)
    if allow_full_asset and full is not None and fits(full):
        return {
            "action": "select_full_asset",
            "selected_level_id": full["level_id"],
            "reason": "no_nontrivial_level_qualifies_or_costs_are_infeasible",
            "distortion_budget": budget,
            "objective": objective,
            "used_full_asset_fallback": True,
        }
    return {
        "action": "abstain",
        "selected_level_id": None,
        "reason": "no_predeclared_level_meets_risk_and_cost_constraints",
        "distortion_budget": budget,
        "objective": objective,
        "used_full_asset_fallback": False,
    }


def make_stream_metadata(
    plan: Mapping[str, Any],
    *,
    asset_digest: str,
    renderer_id: str,
    renderer_version: str,
    codec_id: str,
    codec_version: str = "",
    expires_at: float | None = None,
    permitted_levels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bind a certificate to the exact asset, renderer, codec and level set."""
    if not isinstance(plan, Mapping) or plan.get("format") != FORMAT:
        raise ValueError("plan is not an AURA stream distortion certificate")
    if not isinstance(asset_digest, str) or not asset_digest.strip():
        raise ValueError("asset_digest is required")
    level_ids = [str(level["level_id"]) for level in plan.get("levels", [])]
    permitted = level_ids if permitted_levels is None else [str(level) for level in permitted_levels]
    if not permitted or any(level not in level_ids for level in permitted):
        raise ValueError("permitted_levels must be a non-empty subset of the certificate levels")
    if len(set(permitted)) != len(permitted):
        raise ValueError("permitted_levels must be unique")
    if expires_at is not None:
        expires_at = _finite_scalar(expires_at, "expires_at")
        if expires_at <= 0.0:
            raise ValueError("expires_at must be a positive epoch timestamp")
    metadata: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "binding": {
            "asset_id": str(plan.get("asset_id", "")),
            "asset_digest": asset_digest,
            "renderer_id": str(renderer_id),
            "renderer_version": str(renderer_version),
            "codec_id": str(codec_id),
            "codec_version": str(codec_version),
        },
        "risk": {
            "loss": plan["loss"],
            "alpha": plan["alpha"],
            "alpha_per_level": plan["alpha_per_level"],
            "family_wise_confidence": plan["family_wise_confidence"],
            "risk_budget": plan["risk_budget"],
            "correction": plan["correction"],
        },
        "calibration": {
            "n_calibration": plan["n_calibration"],
            "calibration_set_digest": plan["calibration_set_digest"],
            "plan_digest": plan["plan_digest"],
        },
        "permitted_levels": permitted,
        "levels": [level for level in plan["levels"] if level["level_id"] in permitted],
        "claim_boundary": list(plan.get("claim_boundary", [])),
    }
    if expires_at is not None:
        metadata["expires_at"] = expires_at
    metadata["certificate_digest"] = _digest(metadata)
    return metadata


def validate_stream_metadata(
    metadata: Mapping[str, Any],
    *,
    asset_id: str | None = None,
    asset_digest: str | None = None,
    renderer_id: str | None = None,
    renderer_version: str | None = None,
    codec_id: str | None = None,
    codec_version: str | None = None,
    level_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Reject a stream certificate that is stale, mismatched or unauthorized."""
    if not isinstance(metadata, Mapping):
        raise ValueError("stream metadata must be a mapping")
    if metadata.get("format") != FORMAT or int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported or invalid AURA stream metadata format")
    expected_digest = metadata.get("certificate_digest")
    unsigned = dict(metadata)
    unsigned.pop("certificate_digest", None)
    if expected_digest != _digest(unsigned):
        raise ValueError("stream metadata certificate_digest does not match its contents")
    binding = metadata.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("stream metadata has no binding block")
    expected = {
        "asset_id": asset_id,
        "asset_digest": asset_digest,
        "renderer_id": renderer_id,
        "renderer_version": renderer_version,
        "codec_id": codec_id,
        "codec_version": codec_version,
    }
    for key, value in expected.items():
        if value is not None and str(binding.get(key)) != str(value):
            raise ValueError(f"stream metadata binding mismatch for {key}")
    permitted = {str(value) for value in metadata.get("permitted_levels", [])}
    if level_id is not None and str(level_id) not in permitted:
        raise ValueError(f"level {level_id!r} is not permitted by this certificate")
    expires_at = metadata.get("expires_at")
    if expires_at is not None:
        current = time.time() if now is None else _finite_scalar(now, "now")
        if current >= float(expires_at):
            raise ValueError("stream metadata certificate has expired")
    return {
        "valid": True,
        "asset_id": binding.get("asset_id"),
        "permitted_levels": sorted(permitted),
        "level_id": None if level_id is None else str(level_id),
        "expires_at": expires_at,
    }


def metadata_overhead_ratio(metadata: Mapping[str, Any], asset_bytes: int) -> float:
    """Return JSON certificate bytes divided by the encoded asset size."""
    if isinstance(asset_bytes, bool) or int(asset_bytes) != asset_bytes or int(asset_bytes) <= 0:
        raise ValueError("asset_bytes must be a positive integer")
    return len(_json_bytes(metadata)) / int(asset_bytes)


__all__ = [
    "FORMAT",
    "SCHEMA_VERSION",
    "StreamCandidate",
    "calibrate_distortion_budget",
    "choose_stream_level",
    "evaluate_stream_certificate",
    "make_stream_metadata",
    "metadata_overhead_ratio",
    "validate_stream_metadata",
]
