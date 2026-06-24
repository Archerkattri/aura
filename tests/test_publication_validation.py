import json
import subprocess
import sys

from aura.publication import publication_validation_report


def test_publication_validation_report_aggregates_current_artifacts():
    payload = publication_validation_report().to_dict()
    gates = {gate["id"]: gate for gate in payload["gates"]}

    assert payload["format"] == "AURA_PUBLICATION_VALIDATION_REPORT"
    assert payload["publicationReady"] is False
    assert gates["local_multiscene_quality"]["passed"] is True
    assert gates["dataset_audit"]["passed"] is True
    assert gates["prism_additive_contract"]["passed"] is True
    assert gates["prism_cuda_fps"]["passed"] is True
    assert gates["learned_lpips_cuda"]["passed"] is True
    assert gates["external_method_baselines"]["passed"] is False
    assert gates["secondary_ray_reflection"]["passed"] is True
    assert gates["inverse_materials"]["passed"] is True
    assert "external_method_baselines" in payload["remainingGateIds"]
    assert "secondary_ray_reflection" not in payload["remainingGateIds"]
    assert "inverse_materials" not in payload["remainingGateIds"]
    assert payload["claimBoundary"]["canClaim"]
    assert "superiority over COLMAP/NeRF/2DGS/ray-traced-GS baselines" in payload["claimBoundary"]["cannotClaim"]


def test_publication_validation_report_cli_prints_json():
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "publication-validation-report"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["format"] == "AURA_PUBLICATION_VALIDATION_REPORT"
    assert payload["publicationReady"] is False
    assert payload["gates"]
