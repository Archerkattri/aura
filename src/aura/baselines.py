from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aura.package import AuraPackage, package_scene
from aura.splats import load_3dgs_scene


@dataclass(frozen=True)
class BaselineExport:
    baseline: str
    source_path: Path
    splat_path: Path
    scene_name: str

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "sourcePath": str(self.source_path),
            "splatPath": str(self.splat_path),
            "sceneName": self.scene_name,
        }


def discover_3dgs_export(path: Path | str, *, name: str | None = None) -> BaselineExport:
    source = Path(path)
    if source.is_file():
        _require_supported_splat_file(source)
        return BaselineExport(
            baseline="3dgs",
            source_path=source,
            splat_path=source,
            scene_name=name or source.stem,
        )
    if not source.is_dir():
        raise FileNotFoundError(source)

    splat_path = _discover_3dgs_ply(source)
    return BaselineExport(
        baseline="3dgs",
        source_path=source,
        splat_path=splat_path,
        scene_name=name or _scene_name_from_dir(source),
    )


def package_3dgs_export(
    path: Path | str,
    *,
    name: str | None = None,
    radius_sigma: float = 2.0,
) -> AuraPackage:
    export = discover_3dgs_export(path, name=name)
    scene = load_3dgs_scene(export.splat_path, name=export.scene_name, radius_sigma=radius_sigma)
    return package_scene(
        scene,
        fallbacks={
            "splat": str(export.splat_path),
            "baseline": export.baseline,
            "source": str(export.source_path),
        },
    )


def _require_supported_splat_file(path: Path) -> None:
    if path.suffix.lower() not in {".ply", ".json"}:
        raise ValueError(f"unsupported 3DGS export file: {path}")


def _discover_3dgs_ply(root: Path) -> Path:
    direct = root / "point_cloud.ply"
    if direct.exists():
        return direct

    candidates = list(root.glob("point_cloud/iteration_*/point_cloud.ply"))
    candidates.extend(root.glob("output/point_cloud/iteration_*/point_cloud.ply"))
    candidates.extend(root.glob("outputs/point_cloud/iteration_*/point_cloud.ply"))
    if candidates:
        return sorted(candidates, key=_iteration_sort_key)[-1]

    recursive = sorted(root.rglob("*.ply"))
    if len(recursive) == 1:
        return recursive[0]
    if not recursive:
        raise FileNotFoundError(f"no 3DGS PLY export found under {root}")
    raise ValueError(f"multiple PLY files found under {root}; pass the intended export file directly")


def _iteration_sort_key(path: Path) -> tuple[int, str]:
    iteration = -1
    for part in path.parts:
        if part.startswith("iteration_"):
            suffix = part.removeprefix("iteration_")
            if suffix.isdigit():
                iteration = int(suffix)
    return (iteration, str(path))


def _scene_name_from_dir(path: Path) -> str:
    return path.resolve().name or "3dgs_scene"
