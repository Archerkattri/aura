# AU — distortion-controlled streaming

`aura.lod` answers a trust question: how much calibrated carrier reliability is
discarded at a published keep level. It does not answer whether the delivered
asset still looks like the full asset. `aura.distortion_budget` supplies that
second, narrower contract.

## Contract

The producer freezes a finite nested ladder of already-encoded representations.
For every level it measures a bounded, normalized per-view distortion against a
frozen full asset on a calibration split. The module computes a Hoeffding upper
bound for each non-full level and applies Bonferroni correction across the
declared family. The full asset is a deterministic zero-distortion fallback and
does not consume error budget.

The certificate therefore supports a statement of the form:

> For this declared level family, renderer, codec and calibration protocol, the
> selected level's mean full-asset distortion is bounded by its published
> `epsilon_certified` at family-wise confidence `1 - alpha`.

This is not a guarantee for arbitrary unseen codecs, post-hoc levels, unseen
camera distributions, or a real-image target that was not separately calibrated.
The API keeps real-image losses under `real_image_evaluation` so an image-quality
diagnostic cannot accidentally be presented as the full-asset certificate.

## Minimal API

```python
import numpy as np
from aura import (
    StreamCandidate,
    calibrate_distortion_budget,
    choose_stream_level,
    evaluate_stream_certificate,
    make_stream_metadata,
    validate_stream_metadata,
)

candidates = [
    StreamCandidate("pruned-4bit", 120_000, 1.0, retained_fraction=.10),
    StreamCandidate("pruned-8bit", 420_000, 1.5, retained_fraction=.50),
    StreamCandidate("full", 5_000_000, 2.2, retained_fraction=1.0,
                    is_full_asset=True),
]

plan = calibrate_distortion_budget(
    candidates,
    {level.level_id: measured_distortion[level.level_id]
     for level in candidates},
    alpha=.10,
    risk_budget=.10,
    asset_id="scene-a",
    renderer_id="aura",
    renderer_version="<commit>",
    codec_id="spz",
    codec_version="4",
    calibration_view_ids=calibration_view_ids,
)

decision = choose_stream_level(plan, distortion_budget=.10,
                               max_bytes=1_000_000)
heldout = evaluate_stream_certificate(
    plan, heldout_distortions, heldout_view_ids=evaluation_view_ids,
    heldout_real_image_losses=heldout_real_image_losses,
)

metadata = make_stream_metadata(
    plan, asset_digest=asset_sha256, renderer_id="aura",
    renderer_version="<commit>", codec_id="spz", codec_version="4",
)
validate_stream_metadata(
    metadata, asset_id="scene-a", asset_digest=asset_sha256,
    renderer_id="aura", renderer_version="<commit>",
    codec_id="spz", codec_version="4",
    level_id=decision["selected_level_id"],
)
```

`calibration_view_ids` and `heldout_view_ids` are optional but strongly
recommended. When supplied, the implementation rejects overlap and duplicate
IDs, making split mistakes visible. Measurements must be finite and already
normalized to `[0, 1]`; out-of-range values are rejected rather than clipped.

## Reproduce the CPU protocol pilot

```powershell
python experiments/streaming_distortion_pilot.py
```

The pilot writes `experiments/results/streaming_distortion_pilot.json`. It is a
synthetic regime intended to exercise the family correction, held-out check,
full-asset fallback and metadata validation. It is not evidence of real-scene
quality, GPU throughput or a comparison with PCGS/other compression systems.

## Release gate

A real release or paper result must replace the pilot arrays with scene-disjoint
views, count all duplicated computation in the cost, freeze the renderer and
codec before calibration, compare against the current confidence LOD and a
strong fixed progressive/compression ladder, and report both full-asset
distortion and held-out real-image quality. If no non-full level qualifies, the
consumer must use the full asset or abstain. Metadata must be rejected when the
asset, renderer, codec, permitted level set, certificate contents or expiry no
longer match.
