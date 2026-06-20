# Convergence TODO — carrier gradient starvation

**Status of this document:** root-cause analysis + a *proposed* fix for the GPU
operator. I (the offline, no-GPU agent) deliberately did **not** implement the
core rotation fix, because the change touches a tightly-coupled
plan↔packed-batch invariant whose target/sample-offset accounting cannot be
verified correct without running the torch/GPU training path. Guessing here
would risk a silent correctness bug (mis-aligned targets) that is worse than the
current honest under-convergence. A safe, decoupled, CPU-verified **coverage
diagnostic** was implemented instead (see "What WAS implemented" below) so the
GPU run can *measure* the starvation directly.

## Symptom

`truck-3k-run6` evaluates at ~6.89 dB / SSIM 0.044 — essentially the untrained
floor — despite the writeback working (~23k/129k carriers updated). The
per-iteration loss on the sampled tiles stays noisy-flat (~0.03). Roughly **82%
of the 129,531 carriers never receive a gradient**, so they keep their seed
values for the whole run.

## Root cause (correct-by-inspection)

The training targets are sampled **once**, before the optimizer loop, and the
**same fixed pixel subset is reused every iteration**. With
`--max-targets-per-frame 16` (run6's flag), only the first 16 valid pixels of
each frame — in raster order from the top-left — ever become supervision
targets, and they are identical on every iteration. Carriers that none of those
few rays hit never appear in any backward pass.

Concretely:

- `src/aura/training_targets.py` → `plan_capture_tensor_sampling()`
  (around **lines 470–552**). The nested scan
  ```python
  for tile_y in range(0, H, tile_size):
      for tile_x in range(0, W, tile_size):
          for y in range(tile_y, tile_y + height, pixel_stride):
              for x in range(tile_x, tile_x + width, pixel_stride):
                  ...
                  produced += 1
                  if max_targets_per_frame is not None and produced >= max_targets_per_frame:
                      stop_frame = True
                      break          # <-- lines ~522-524
  ```
  hard-stops the whole frame after `max_targets_per_frame` valid pixels. With
  `max_targets_per_frame=16` and `tile_size=256`, only the **first tile** of
  each frame contributes, and only its first 16 (contiguous, top-left) valid
  pixels. There is **no per-iteration variation** and **no spread** across the
  frame.

- `src/aura/torch_optimizer.py` → `_optimize_torch_batches()`. The packed
  batches are built once at **line ~862**
  (`prepared_batches = tuple(...)`) and the iteration loop at **line ~893**
  re-uses *the same* `prepared_batches` every iteration (**line ~940**:
  `for batch ... in prepared_batches:`). No shuffle, no rotation.

So the set of supervised pixels — and therefore the set of carriers that get
gradients — is frozen for the entire run.

## Why this is subtle to fix safely (why I did not guess)

The plan and the packed-batch builder are **coupled by a replayed scan**.
`_append_tile_samples_to_packed_batch()` (`training_targets.py`, **lines
643–689**) independently re-walks the *identical* raster scan and selects pixels
whose running `sampled_offset` falls in `[tile.target_offset, effective_stop)`.
The `CaptureSamplingPlan.__post_init__` invariants (**lines ~237–266**) require
every tile's `target_offset` to be exactly the running prefix sum of prior
tiles' `sampled_pixel_count`, contiguous and starting at 0.

Any change to *which* pixels are sampled, or to the visit order of tiles, must
keep the plan's `sampled_pixel_count`/`target_offset` bookkeeping and the
builder's `sampled_offset` replay **byte-for-byte in agreement**, or the packed
batches silently bind the wrong colors/rays to the wrong target slots. That
alignment is exactly the kind of invariant that cannot be confirmed by reading
alone; it needs the torch path to assert sample↔target correspondence on real
tensors. Hence: documented, not guessed.

## Proposed fix for the GPU operator

Goal: over the course of training, **every carrier eventually receives
gradients**, while still respecting the per-iteration memory cap
(`max_targets_per_batch`, `max_targets_per_frame`). Two viable designs, in
increasing order of invasiveness:

### Option A (preferred): cross-iteration target rotation via an epoch phase
1. Add a deterministic `sample_phase: int` parameter to
   `plan_capture_tensor_sampling()` and `capture_tensors_to_packed_render_batches()`
   (and the matching replay in `_append_tile_samples_to_packed_batch()`), default
   `0` so existing behavior is unchanged. The phase shifts the **starting index
   into the per-frame valid-pixel stream** (i.e. skip the first
   `sample_phase * max_targets_per_frame` valid pixels, wrapping around) before
   selecting the next `max_targets_per_frame`. Because both the planner and the
   builder replay the same scan, applying the identical phase shift in both keeps
   them in sync.
2. In the CLI (`_train_capture_manifest_command`), build `K` plans for
   `sample_phase in range(K)` where `K = ceil(total_valid_pixels_per_frame /
   max_targets_per_frame)`, OR build them lazily.
3. In `_optimize_torch_batches`, select the plan/batch-group for
   `phase = absolute_iteration % K` each iteration. After enough iterations every
   valid pixel — and thus every carrier its rays hit — is supervised.

**Validation gate (must pass on GPU before trusting numbers):** assert that the
union of supervised pixels over one full phase cycle equals the full
`pixel_stride`-decimated valid-pixel set per frame, and that
`sample_phase=0,K=1` is bit-identical to today's output. Then confirm the
fraction of carriers with non-zero accumulated gradient over a cycle rises from
~18% toward ~100%, and that PSNR climbs.

### Option B (simpler, weaker): spread-decimate within the cap
Instead of taking the first `max_targets_per_frame` *contiguous* valid pixels,
take `max_targets_per_frame` valid pixels **evenly strided across the whole
valid-pixel stream** (decimation factor
`floor(num_valid / max_targets_per_frame)`). This spreads supervision over the
entire frame each iteration (better carrier coverage immediately) but the subset
is still fixed across iterations, so it only mitigates, not eliminates, the
starvation. It also still requires the planner/builder replay to agree on the
decimated indices — same coupling caveat as Option A.

Either way: raising `--max-targets-per-frame` (e.g. 16 → 4096) and adding more
iterations is the brute-force mitigation that needs no code change, at the cost
of memory/time.

## What WAS implemented (safe, CPU-verified)

`src/aura/training_targets.py` gains a pure, side-effect-free diagnostic
(exported from `aura`):

- `sampling_coverage_report(plan)` → returns per-frame and overall coverage:
  `sampledPixelCount`, `capacityPixelCount`, `maskedPixelCount`, `tileCount`,
  and `coverageFraction = sampled / capacity`. The denominator
  `capacityPixelCount` is each *present* tile's FULL stride-decimated grid
  capacity (from its `size` + the plan's `pixel_stride`), NOT the planner's
  scan-truncated `candidate_pixel_count` — otherwise an early cap would hide the
  starvation by simply not counting the pixels it skipped. So within-tile
  truncation shows as `coverageFraction` < 1.0, and `tileCount` shrinking flags
  whole tiles dropped by an early frame-level stop. It reads only already-
  computed plan fields, so it cannot perturb the plan↔builder invariant.

This lets the GPU operator quantify the starvation before/after applying Option
A/B and is covered by CPU-only unit tests
(`tests/test_training_targets.py::test_sampling_coverage_report_*`). It does
**not** change any sampled target, so all existing numbers and tests are
unaffected.

## File/line index

| What | File | Lines |
|---|---|---|
| Per-frame hard-stop sampling | `src/aura/training_targets.py` | ~470–552 |
| Packed-batch replay (must stay in sync) | `src/aura/training_targets.py` | ~643–689 |
| Plan contiguity invariants | `src/aura/training_targets.py` | ~237–266 |
| Batches built once | `src/aura/torch_optimizer.py` | ~862 |
| Same batches reused each iter | `src/aura/torch_optimizer.py` | ~893, ~940 |
| Coverage diagnostic (new) | `src/aura/training_targets.py` | `sampling_coverage_report` |
