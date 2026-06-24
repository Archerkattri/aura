import json
import subprocess
import sys


def test_secondary_reflection_validation_script_writes_passing_artifact(tmp_path):
    out = tmp_path / "secondary.json"
    subprocess.run(
        [sys.executable, "experiments/secondary_reflection_validation.py", "--out", str(out)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["format"] == "AURA_SECONDARY_RAY_REFLECTION_VALIDATION"
    assert payload["passed"] is True
    assert payload["shadowTransmittanceReadyRate"] == 1.0
    assert payload["reflectionVectorReadyRate"] > 0.0


def test_inverse_material_validation_script_writes_passing_artifact(tmp_path):
    out = tmp_path / "materials.json"
    subprocess.run(
        [sys.executable, "experiments/inverse_material_validation.py", "--out", str(out), "--device", "cpu"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["format"] == "AURA_INVERSE_MATERIAL_VALIDATION"
    assert payload["passed"] is True
    assert payload["albedoSource"] == "explicit_payload"
    assert payload["roughnessSource"] == "explicit_payload"
    assert payload["metallicSource"] == "explicit_payload"
    assert payload["differentLightingChangesOutput"] is True
