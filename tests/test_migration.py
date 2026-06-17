import json
import subprocess
import sys

import pytest

from aura import AURA_SCHEMA_VERSION, migration_report, package_scene
from aura.cli import native_demo_scene


def test_migration_report_marks_current_schema_as_noop():
    report = migration_report(AURA_SCHEMA_VERSION)

    assert report.supported is True
    assert report.current_version == AURA_SCHEMA_VERSION
    assert report.actions == ("none",)


def test_migration_report_rejects_unknown_major():
    report = migration_report("99.0")

    assert report.supported is False
    assert report.actions == ("manual migration required for unsupported major version 99",)


def test_migration_report_rejects_malformed_versions():
    with pytest.raises(ValueError, match="major.minor"):
        migration_report("dev")


def test_migration_plan_cli_reports_loaded_package_status(tmp_path):
    package_scene(native_demo_scene()).write(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "migration-plan", str(tmp_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["supported"] is True
    assert payload["actions"] == ["none"]
