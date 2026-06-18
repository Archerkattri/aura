"""AURA package schema versioning constants and helpers."""

from __future__ import annotations

from dataclasses import dataclass

AURA_FORMAT = "AURA"
AURA_SCHEMA_VERSION = "0.1"
AURA_SUPPORTED_MAJOR_VERSIONS = {0}


@dataclass(frozen=True)
class AuraSchemaVersion:
    """Parsed major.minor AURA schema version."""

    major: int
    minor: int


def parse_aura_schema_version(version: str, *, label: str = "version") -> AuraSchemaVersion:
    """Parse a ``major.minor`` version string into an :class:`AuraSchemaVersion`."""
    parts = version.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{label} must use major.minor format: {version}")
    return AuraSchemaVersion(major=int(parts[0]), minor=int(parts[1]))
