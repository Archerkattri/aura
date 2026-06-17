from __future__ import annotations

from dataclasses import asdict, dataclass

from aura.schema import AURA_SCHEMA_VERSION, AURA_SUPPORTED_MAJOR_VERSIONS


@dataclass(frozen=True)
class MigrationReport:
    current_version: str
    target_version: str
    supported: bool
    actions: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["actions"] = list(self.actions)
        return payload


def migration_report(current_version: str, *, target_version: str = AURA_SCHEMA_VERSION) -> MigrationReport:
    current_major = _major(current_version)
    target_major = _major(target_version)
    supported = current_major in AURA_SUPPORTED_MAJOR_VERSIONS and target_major in AURA_SUPPORTED_MAJOR_VERSIONS
    if not supported:
        return MigrationReport(
            current_version=current_version,
            target_version=target_version,
            supported=False,
            actions=(f"manual migration required for unsupported major version {current_major}",),
        )
    if current_version == target_version:
        return MigrationReport(
            current_version=current_version,
            target_version=target_version,
            supported=True,
            actions=("none",),
        )
    return MigrationReport(
        current_version=current_version,
        target_version=target_version,
        supported=True,
        actions=(f"rewrite manifest version to {target_version}", "revalidate package schemas"),
    )


def _major(version: str) -> int:
    parts = version.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"version must use major.minor format: {version}")
    return int(parts[0])
