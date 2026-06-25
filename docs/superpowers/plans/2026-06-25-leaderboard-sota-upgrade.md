# Leaderboard SOTA Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair leaderboard-grade A/B harness and use it to evaluate every SOTA upgrade candidate before claiming leaderboard readiness.

**Architecture:** Add a small `aura.leaderboard` report layer with strict promotion and coverage logic, then connect scripts and result artifacts to that layer one upgrade family at a time. The first implementation slice is schema/comparison only; later slices launch long GPU jobs for gsplat-main, MCMC, 3DGUT, RadSplat-style pruning, and inference-stack ablations.

**Tech Stack:** Python 3.11, dataclasses, JSON, pytest, CUDA PyTorch, gsplat, 3DGRUT, DBS-Beta isolated environment.

---

### Task 1: Leaderboard Report Schema

**Files:**
- Create: `src/aura/leaderboard.py`
- Test: `tests/test_leaderboard.py`

- [ ] **Step 1: Write the failing tests**

```python
from aura.leaderboard import (
    LeaderboardMetric,
    LeaderboardReport,
    LeaderboardRun,
    MethodSpec,
    SceneSpec,
)


def test_leaderboard_report_blocks_missing_scene_from_readiness():
    scenes = (
        SceneSpec(scene_id="truck", dataset="Tanks and Temples", split="llffhold8", image_scale="native"),
        SceneSpec(scene_id="room", dataset="Mip-NeRF 360", split="llffhold8", image_scale="images_2"),
    )
    baseline = MethodSpec(method_id="aura_beta", role="baseline", backend="dbs", command="existing")
    candidate = MethodSpec(method_id="gsplat_main_mcmc", role="candidate", backend="gsplat-main", command="pending")
    report = LeaderboardReport(
        benchmark_id="aura_sota_v1",
        task="novel_view_synthesis",
        scenes=scenes,
        methods=(baseline, candidate),
        runs=(
            LeaderboardRun(
                scene_id="truck",
                method_id="aura_beta",
                metrics=(LeaderboardMetric("psnr", 26.0, higher_is_better=True),),
                artifacts=("experiments/results/multiscene.json",),
                measured=True,
            ),
            LeaderboardRun(
                scene_id="truck",
                method_id="gsplat_main_mcmc",
                metrics=(LeaderboardMetric("psnr", 26.5, higher_is_better=True),),
                artifacts=("experiments/results/multiscene.json",),
                measured=True,
            ),
        ),
        primary_metric="psnr",
    )

    payload = report.to_dict()

    assert payload["leaderboardReady"] is False
    assert payload["missingScenes"] == ["room"]
    assert payload["claimBoundary"]["cannotClaim"]


def test_leaderboard_report_promotes_only_measured_candidate_with_artifact():
    scene = SceneSpec(scene_id="truck", dataset="Tanks and Temples", split="llffhold8", image_scale="native")
    report = LeaderboardReport(
        benchmark_id="aura_sota_v1",
        task="novel_view_synthesis",
        scenes=(scene,),
        methods=(
            MethodSpec(method_id="aura_beta", role="baseline", backend="dbs", command="existing"),
            MethodSpec(method_id="candidate_fixture", role="candidate", backend="fixture", command="none"),
            MethodSpec(method_id="candidate_real", role="candidate", backend="gsplat-main", command="python train.py"),
        ),
        runs=(
            LeaderboardRun(
                scene_id="truck",
                method_id="aura_beta",
                metrics=(LeaderboardMetric("psnr", 26.0, higher_is_better=True),),
                artifacts=("experiments/results/multiscene.json",),
                measured=True,
            ),
            LeaderboardRun(
                scene_id="truck",
                method_id="candidate_fixture",
                metrics=(LeaderboardMetric("psnr", 99.0, higher_is_better=True),),
                artifacts=(),
                measured=False,
            ),
            LeaderboardRun(
                scene_id="truck",
                method_id="candidate_real",
                metrics=(LeaderboardMetric("psnr", 26.5, higher_is_better=True),),
                artifacts=("experiments/results/real_candidate.json",),
                measured=True,
            ),
        ),
        primary_metric="psnr",
    )

    payload = report.to_dict()

    assert payload["leaderboardReady"] is True
    assert payload["promotedMethodIds"] == ["candidate_real"]
    assert payload["comparisons"][0]["winnerMethodId"] == "candidate_real"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.gpu_venv/bin/python -m pytest tests/test_leaderboard.py -q`

Expected: FAIL because `aura.leaderboard` does not exist.

- [ ] **Step 3: Implement minimal schema and comparison logic**

Create `src/aura/leaderboard.py` with dataclasses for `SceneSpec`, `MethodSpec`,
`LeaderboardMetric`, `LeaderboardRun`, and `LeaderboardReport`. Implement
`to_dict()`, missing-scene detection, measured/artifact gating, primary-metric
comparison, and claim-boundary output.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.gpu_venv/bin/python -m pytest tests/test_leaderboard.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aura/leaderboard.py tests/test_leaderboard.py
git commit -m "feat: add leaderboard ablation schema"
```

### Task 2: First Report Script

**Files:**
- Create: `experiments/leaderboard_ablation.py`
- Test: `tests/test_leaderboard_scripts.py`

- [ ] **Step 1: Write failing CLI/script tests**

Write tests that call the script with fixture rows and assert that fixture rows
cannot make `leaderboardReady` true unless every scene has measured baseline and
candidate artifacts.

- [ ] **Step 2: Run failing tests**

Run: `.gpu_venv/bin/python -m pytest tests/test_leaderboard_scripts.py -q`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement script**

Create a deterministic script that writes
`experiments/results/leaderboard_ablation_2026-06-25.json` from existing
multi-scene and official-baseline artifacts. Keep long job launching out of this
script.

- [ ] **Step 4: Run tests**

Run: `.gpu_venv/bin/python -m pytest tests/test_leaderboard.py tests/test_leaderboard_scripts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/leaderboard_ablation.py tests/test_leaderboard_scripts.py experiments/results/leaderboard_ablation_2026-06-25.json
git commit -m "exp: add leaderboard ablation report"
```

### Task 3: gsplat-main Upgrade A/B

**Files:**
- Modify: `experiments/leaderboard_ablation.py`
- Create: `experiments/gsplat_main_ablation.py`
- Test: `tests/test_leaderboard_scripts.py`

- [ ] Add a launcher contract for installed `gsplat 1.5.3` vs source/main.
- [ ] Record exact git SHA, package version, CUDA device, command, and output.
- [ ] Run Truck first; promote only if PSNR/LPIPS/FPS beat baseline without
  missing artifacts.
- [ ] Expand to all 8 scenes only after Truck passes.
- [ ] Commit measured JSON rows.

### Task 4: MCMC Densification A/B

**Files:**
- Modify: `src/aura/gsplat_renderer.py`
- Create: `experiments/mcmc_densification_ablation.py`
- Test: `tests/test_leaderboard_scripts.py`

- [ ] Add a config-level strategy selector with baseline `default` and candidate
  `mcmc`.
- [ ] Run TDD for strategy selection without changing default behavior.
- [ ] Run Truck A/B, then all 8 scenes if Truck improves.
- [ ] Commit only measured rows that include artifacts.

### Task 5: 3DGUT Backend Promotion

**Files:**
- Create: `src/aura/backends/three_dgut.py`
- Modify: `src/aura/publication.py`
- Test: `tests/test_leaderboard.py`

- [ ] Wrap official 3DGRUT outputs as AURA backend rows.
- [ ] Add distorted-camera and secondary-ray metadata fields.
- [ ] Keep PRISM additive; do not route Gaussian/Beta primary quality through
  PRISM.
- [ ] Benchmark query/render paths and commit measured artifacts.

### Task 6: RadSplat-Style Teacher/Pruning A/B

**Files:**
- Create: `experiments/radsplat_pruning_ablation.py`
- Test: `tests/test_leaderboard_scripts.py`

- [ ] Add teacher-supervision artifact inputs.
- [ ] Add pruning rows with PSNR/SSIM/LPIPS/FPS/size.
- [ ] Promote only if quality does not regress and FPS/size improves.

### Task 7: Inference Stack A/B

**Files:**
- Create: `experiments/inference_stack_ablation.py`
- Test: `tests/test_leaderboard_scripts.py`

- [ ] Compare standard gsplat inference, HiGS source/main, StopThePop-style
  culling/sorting, and OMG-style compact output where runnable.
- [ ] Record FPS, memory, size, and quality retention.
- [ ] Promote per task: quality leaderboard and runtime leaderboard are separate.

### Task 8: Final Verification And Claim Update

**Files:**
- Modify: `README.md`
- Modify: `docs/submission_readiness_2026-06-25.md`
- Modify: `src/aura/publication.py`

- [ ] Run focused leaderboard tests.
- [ ] Run publication validation tests.
- [ ] Run full pytest.
- [ ] Update claims only after measured report supports them.
- [ ] Push `main`.

## Self-Review

- Spec coverage: tasks cover schema, report generation, gsplat-main, MCMC,
  3DGUT, RadSplat-style pruning, inference stack, and claim updates.
- Placeholder scan: later tasks are intentionally higher-level because they
  require measured GPU outcomes and external repo state; Task 1 and Task 2 are
  fully executable now.
- Type consistency: Task 1 defines the dataclass names used by Task 2.
