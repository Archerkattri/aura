import json
import subprocess
import sys
from pathlib import Path

from experiments.leaderboard_ablation import leaderboard_ablation_report


ROOT = Path(__file__).resolve().parents[1]


def test_leaderboard_ablation_report_uses_current_artifacts_without_sota_claim(tmp_path):
    out = tmp_path / "leaderboard.json"

    payload = leaderboard_ablation_report(out)

    assert out.exists()
    assert payload["format"] == "AURA_LEADERBOARD_ABLATION_REPORT"
    assert payload["benchmarkId"] == "aura_leaderboard_v1"
    assert payload["leaderboardReady"] is False
    assert "leaderboard SOTA" in " ".join(payload["claimBoundary"]["cannotClaim"])
    assert payload["runs"]
    assert any(run["methodId"] == "gsplat_main_mcmc" for run in payload["runs"])
    assert any(run["methodId"] == "higs_inference" for run in payload["runs"])
    assert "gsplat_main_mcmc" not in payload["promotedMethodIds"]
    assert "higs_inference" not in payload["promotedMethodIds"]
    assert all(
        "gsplat_main_mcmc_truck_smoke" not in artifact
        for run in payload["runs"]
        for artifact in run["artifacts"]
    )


def test_leaderboard_ablation_cli_writes_json(tmp_path):
    out = tmp_path / "leaderboard_cli.json"

    result = subprocess.run(
        [sys.executable, "experiments/leaderboard_ablation.py", "--out", str(out)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text())
    printed = json.loads(result.stdout)
    assert payload["format"] == "AURA_LEADERBOARD_ABLATION_REPORT"
    assert printed["leaderboardReady"] is False
    assert printed["missingScenes"] == payload["missingScenes"]
