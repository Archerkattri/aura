import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GSPLAT_MAIN = Path("/tmp/aura_sota_repos/gsplat-main")

pytestmark = pytest.mark.skipif(
    not GSPLAT_MAIN.exists(),
    reason="gsplat-main source checkout is required for this local CUDA smoke",
)


def test_gsplat_main_mcmc_smoke_script_writes_artifact(tmp_path):
    out = tmp_path / "mcmc_smoke.json"

    result = subprocess.run(
        [sys.executable, "experiments/gsplat_main_mcmc_smoke.py", "--out", str(out)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text())
    printed = json.loads(result.stdout)
    assert payload["format"] == "AURA_GSPLAT_MAIN_MCMC_SMOKE"
    assert payload["passed"] is True
    assert printed["passed"] is True
    assert payload["smoke"]["strategy"] == "MCMCStrategy"
