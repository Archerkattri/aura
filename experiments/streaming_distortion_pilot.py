"""CPU pilot for AURA's finite-family streaming distortion certificate.

This is a protocol smoke test, not a real-scene or SOTA benchmark.  It keeps
the calibration and held-out image-quality channels separate and writes a
fully provenance-labelled JSON report so a later GPU run can replace the
synthetic arrays without changing the selection contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aura.distortion_budget import (  # noqa: E402
    StreamCandidate,
    calibrate_distortion_budget,
    choose_stream_level,
    evaluate_stream_certificate,
    make_stream_metadata,
    metadata_overhead_ratio,
)


def _source_digest() -> str:
    digest = hashlib.sha256()
    for relative in ("src/aura/distortion_budget.py", "src/aura/calibration.py", Path(__file__).relative_to(ROOT)):
        path = ROOT / relative
        digest.update(str(relative).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(seed: int = 17, n_calibration: int = 240, n_evaluation: int = 240) -> dict:
    rng = np.random.default_rng(seed)
    candidates = [
        StreamCandidate("prune_4bit", 120_000, 1.0, retained_fraction=0.10, parameters={"bits": 4}),
        StreamCandidate("prune_quant_8bit", 420_000, 1.5, retained_fraction=0.50, parameters={"bits": 8}),
        StreamCandidate("full_asset", 5_000_000, 2.2, retained_fraction=1.0, is_full_asset=True),
    ]

    cal_hardness = rng.uniform(0.0, 1.0, n_calibration)
    eval_hardness = rng.uniform(0.0, 1.0, n_evaluation)
    # Bounded distortion against a frozen full asset.  The small noise term
    # models view-dependent renderer differences while remaining synthetic.
    cal = {
        "prune_4bit": np.clip(0.015 + 0.045 * cal_hardness + rng.normal(0, 0.003, n_calibration), 0, 1),
        "prune_quant_8bit": np.clip(0.006 + 0.014 * cal_hardness + rng.normal(0, 0.002, n_calibration), 0, 1),
        "full_asset": np.zeros(n_calibration),
    }
    heldout = {
        "prune_4bit": np.clip(0.018 + 0.050 * eval_hardness + rng.normal(0, 0.003, n_evaluation), 0, 1),
        "prune_quant_8bit": np.clip(0.007 + 0.016 * eval_hardness + rng.normal(0, 0.002, n_evaluation), 0, 1),
        "full_asset": np.zeros(n_evaluation),
    }
    # This diagnostic is intentionally on a different scale: it represents
    # image loss against held-out real images, not loss against the full asset.
    real_image = {
        "prune_4bit": np.clip(0.10 + 0.08 * eval_hardness + rng.normal(0, 0.004, n_evaluation), 0, 1),
        "prune_quant_8bit": np.clip(0.025 + 0.025 * eval_hardness + rng.normal(0, 0.003, n_evaluation), 0, 1),
        "full_asset": np.zeros(n_evaluation),
    }
    source_digest = _source_digest()
    plan = calibrate_distortion_budget(
        candidates,
        cal,
        alpha=0.10,
        risk_budget=0.10,
        asset_id="synthetic-scene",
        renderer_id="synthetic-renderer",
        renderer_version="pilot",
        codec_id="synthetic-quantizer",
        codec_version="pilot",
        calibration_view_ids=[f"cal-{i}" for i in range(n_calibration)],
        source_digest=source_digest,
    )
    evaluation = evaluate_stream_certificate(
        plan,
        heldout,
        heldout_view_ids=[f"eval-{i}" for i in range(n_evaluation)],
        heldout_real_image_losses=real_image,
    )
    choice = choose_stream_level(plan, distortion_budget=0.10)
    metadata = make_stream_metadata(
        plan,
        asset_digest="synthetic-asset-digest",
        renderer_id="synthetic-renderer",
        renderer_version="pilot",
        codec_id="synthetic-quantizer",
        codec_version="pilot",
    )
    return {
        "format": "AURA_STREAM_DISTORTION_PILOT",
        "protocol": "synthetic_cpu_finite_family_full_asset_vs_real_image_split",
        "seed": seed,
        "n_calibration": n_calibration,
        "n_evaluation": n_evaluation,
        "source_digest": source_digest,
        "plan": plan,
        "evaluation": evaluation,
        "choice": choice,
        "metadata_overhead_ratio_at_5mb": metadata_overhead_ratio(metadata, 5_000_000),
        "claim_boundary": [
            "synthetic protocol only; no real scene, model, GPU or external baseline",
            "finite-family full-asset distortion certificate, not arbitrary-codec or distribution-free image quality",
            "real-image loss is diagnostic only in this pilot",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "streaming_distortion_pilot.json")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    payload = run(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "source_digest": payload["source_digest"],
        "choice": payload["choice"],
        "all_levels_hold": payload["evaluation"]["all_levels_hold"],
        "metadata_overhead_ratio_at_5mb": payload["metadata_overhead_ratio_at_5mb"],
    }, indent=2))


if __name__ == "__main__":
    main()
