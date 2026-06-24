# CUDA Production Backend Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an artifact-backed CUDA production gate that only passes when AURA executes compiled CUDA dispatch without CPU/torch fallback.

**Architecture:** Keep the existing production-readiness model, but add a durable CUDA validation artifact and a small loader/gate in `readiness.py`. The gate must fail when CUDA execution is unavailable, when fallback is used, or when required parity/throughput fields are missing.

**Tech Stack:** Python 3.11, PyTorch/CUDA, pytest, existing AURA CLI/report modules.

---

### Task 1: Add CUDA Production Artifact Contract

**Files:**
- Modify: `src/aura/readiness.py`
- Test: `tests/test_readiness.py`

- [x] **Step 1: Write failing tests**

Add tests that create a temporary artifact directory and verify readiness changes only when a strict CUDA artifact passes:

```python
def test_cuda_production_gate_requires_compiled_nonfallback_artifact(tmp_path, monkeypatch):
    from aura import readiness

    monkeypatch.setattr(readiness, "RESULTS", tmp_path)
    report = readiness.production_readiness_report().to_dict()
    cuda = next(p for p in report["pillars"] if p["id"] == "cuda_backend")
    assert cuda["productionReady"] is False
    assert "production CUDA validation artifact is missing" in cuda["gaps"]


def test_cuda_production_gate_accepts_compiled_cuda_artifact(tmp_path, monkeypatch):
    import json
    from aura import readiness

    (tmp_path / "cuda_production_backend_2026-06-24.json").write_text(json.dumps({
        "format": "AURA_CUDA_PRODUCTION_BACKEND_REPORT",
        "passed": True,
        "compiledCudaDispatch": True,
        "fallbackUsed": False,
        "device": "cuda",
        "parity": {"maxAbsError": 0.0001, "threshold": 0.001},
        "throughput": {"raysPerSecond": 1000.0, "minRaysPerSecond": 1.0},
    }))
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)

    report = readiness.production_readiness_report().to_dict()
    cuda = next(p for p in report["pillars"] if p["id"] == "cuda_backend")
    assert cuda["productionReady"] is True
    assert any("compiled CUDA dispatch" in item for item in cuda["evidence"])
```

- [x] **Step 2: Run tests and verify failure**

Run: `.gpu_venv/bin/python -m pytest tests/test_readiness.py -q`

Expected: failure because `readiness.RESULTS` and artifact loading do not exist.

- [x] **Step 3: Implement artifact loader and CUDA pillar gate**

Add `RESULTS`, `_latest_json`, `_cuda_production_artifact`, and use it in the `cuda_backend` pillar. Production-ready must require:

```python
passed
compiledCudaDispatch is True
fallbackUsed is False
parity.maxAbsError <= parity.threshold
throughput.raysPerSecond >= throughput.minRaysPerSecond
```

- [x] **Step 4: Run tests and verify pass**

Run: `.gpu_venv/bin/python -m pytest tests/test_readiness.py -q`

Expected: pass.

### Task 2: Add CUDA Production Validation Script

**Files:**
- Create: `experiments/cuda_production_backend_validation.py`
- Test: `tests/test_publication_validation_scripts.py`

- [x] **Step 1: Write failing tests**

Add tests for artifact pass/fail semantics with pure helper functions:

```python
def test_cuda_production_validation_rejects_fallback_payload():
    from experiments.cuda_production_backend_validation import summarize_cuda_gate

    report = summarize_cuda_gate(
        compiled_cuda_dispatch=False,
        fallback_used=True,
        device="cpu",
        max_abs_error=0.0,
        parity_threshold=0.001,
        rays_per_second=10.0,
        min_rays_per_second=1.0,
    )

    assert report["passed"] is False
    assert "compiled CUDA dispatch was not used" in report["failures"]
    assert "fallback backend was used" in report["failures"]


def test_cuda_production_validation_accepts_compiled_payload():
    from experiments.cuda_production_backend_validation import summarize_cuda_gate

    report = summarize_cuda_gate(
        compiled_cuda_dispatch=True,
        fallback_used=False,
        device="cuda",
        max_abs_error=0.0001,
        parity_threshold=0.001,
        rays_per_second=1000.0,
        min_rays_per_second=1.0,
    )

    assert report["passed"] is True
    assert report["failures"] == []
```

- [x] **Step 2: Run tests and verify failure**

Run: `.gpu_venv/bin/python -m pytest tests/test_publication_validation_scripts.py -q`

Expected: import failure because script does not exist.

- [x] **Step 3: Implement script**

Create `summarize_cuda_gate(...)` and a CLI that writes `experiments/results/cuda_production_backend_2026-06-24.json`. The CLI should call existing CUDA boundary utilities and use `fallback_backend="none"` / `require_cuda=True` so fallback cannot pass as production.

- [x] **Step 4: Run tests and verify pass**

Run: `.gpu_venv/bin/python -m pytest tests/test_publication_validation_scripts.py tests/test_readiness.py -q`

Expected: pass.

### Task 3: Run Real CUDA Validation And Update Readiness

**Files:**
- Create or update: `experiments/results/cuda_production_backend_2026-06-24.json`

- [x] **Step 1: Execute CUDA validation**

Run: `.gpu_venv/bin/python experiments/cuda_production_backend_validation.py --output experiments/results/cuda_production_backend_2026-06-24.json`

Expected: JSON artifact is written. If compiled CUDA dispatch is unavailable, artifact exists with `passed: false`.

- [x] **Step 2: Check readiness report**

Run:

```bash
.gpu_venv/bin/python - <<'PY'
from aura.readiness import production_readiness_report
r = production_readiness_report().to_dict()
print(r["productionReady"], r["productionReadyPillarCount"], r["pillarCount"])
print(next(p for p in r["pillars"] if p["id"] == "cuda_backend"))
PY
```

Expected: CUDA pillar production status matches the artifact. If the artifact fails, do not claim CUDA is production-ready.

- [x] **Step 3: Full verification**

Run: `.gpu_venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add src/aura/readiness.py experiments/cuda_production_backend_validation.py tests/test_readiness.py tests/test_publication_validation_scripts.py experiments/results/cuda_production_backend_2026-06-24.json docs/superpowers/plans/2026-06-24-cuda-production-backend-validation.md
git commit -m "feat: add cuda production readiness gate"
git push origin main
```

Expected: AURA `main` pushed.
