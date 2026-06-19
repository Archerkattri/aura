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


def test_migration_report_rejects_malformed_target_versions():
    with pytest.raises(ValueError, match="target version must use major.minor"):
        migration_report(AURA_SCHEMA_VERSION, target_version="next")


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


def test_migration_report_supported_different_versions_returns_rewrite_actions():
    """Line 44: when both versions are supported but differ, returns rewrite+revalidate actions."""
    from aura.schema import AURA_SUPPORTED_MAJOR_VERSIONS
    from aura import migration_report
    # We need a supported major version that differs from the current.
    # Find supported major versions and build a fake "old" version.
    supported = sorted(AURA_SUPPORTED_MAJOR_VERSIONS)
    if len(supported) < 2:
        # Only one major version supported: use same major with different minor
        major = supported[0]
        current = f"{major}.0"
        target = f"{major}.1"
    else:
        current = f"{supported[0]}.0"
        target = f"{supported[1]}.0"

    # If current == target (same version), just make minor differ
    if current == target:
        current = f"{supported[0]}.0"
        target = f"{supported[0]}.99"

    report = migration_report(current, target_version=target)
    assert report.supported is True
    assert len(report.actions) == 2
    assert any("rewrite" in a for a in report.actions)
    assert any("revalidate" in a for a in report.actions)
