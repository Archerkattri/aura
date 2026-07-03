import json
import subprocess
import sys

import pytest

from aura.publication import publication_validation_report


# The aggregate report asserts the relight/secondary-ray gates PASS, which needs
# the gitignored real trained asset (outputs/truck-sidecar.aura) on disk.
@pytest.mark.local_data
def test_publication_validation_report_aggregates_current_artifacts():
    payload = publication_validation_report().to_dict()
    gates = {gate["id"]: gate for gate in payload["gates"]}

    assert payload["format"] == "AURA_PUBLICATION_VALIDATION_REPORT"
    assert payload["publicationReady"] is True
    assert gates["local_multiscene_quality"]["passed"] is True
    assert gates["dataset_audit"]["passed"] is True
    assert gates["prism_additive_contract"]["passed"] is True
    assert gates["prism_cuda_fps"]["passed"] is True
    assert gates["real_scene_fps"]["passed"] is True
    assert gates["engine_integration_exports"]["passed"] is True
    assert gates["viewer_compatibility_exports"]["passed"] is True
    assert gates["learned_lpips_cuda"]["passed"] is True
    assert gates["external_method_baselines"]["passed"] is True
    # Real committed-artifact, content-checked calibration killer-property gates.
    assert gates["calibration_ece"]["passed"] is True
    assert gates["pruning_certificate"]["passed"] is True
    assert gates["cross_scene_transfer"]["passed"] is True
    assert gates["fullres_renderloss"]["passed"] is True
    # Relight / secondary-ray gates are bound to the real trained asset, not the
    # synthetic native demo scene, and report a concrete pass status.
    assert gates["secondary_ray_reflection"]["passed"] is True
    assert gates["secondary_ray_reflection"]["status"] == "passed"
    assert gates["inverse_materials"]["passed"] is True
    assert gates["inverse_materials"]["status"] == "passed"
    assert "external_method_baselines" not in payload["remainingGateIds"]
    assert "secondary_ray_reflection" not in payload["remainingGateIds"]
    assert "inverse_materials" not in payload["remainingGateIds"]
    assert payload["claimBoundary"]["canClaim"]
    assert "AURA has same-split external baseline metrics for COLMAP, NeRF, 3DGS, 2DGS, and ray-traced GS" in payload["claimBoundary"]["canClaim"]
    assert "AURA has completed official 2DGS and 3DGUT same-split rows for all 8 audited scenes" in payload["claimBoundary"]["canClaim"]
    assert any("official 2DGS/3DGUT same-split rows complete: 8/8 and 8/8" in line for line in gates["external_method_baselines"]["evidence"])


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
    assert payload["publicationReady"] is True
    assert payload["gates"]
