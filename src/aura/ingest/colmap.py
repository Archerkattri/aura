from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Sequence

from aura.ingest.capture import CaptureManifest, write_capture_manifest

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ColmapCamera:
    id: str
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def intrinsics(self) -> dict[str, float]:
        if self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params[:3]
            return {"fx": f, "fy": f, "cx": cx, "cy": cy, "width": float(self.width), "height": float(self.height)}
        if self.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            fx, fy, cx, cy = self.params[:4]
            return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": float(self.width), "height": float(self.height)}
        if self.model in {"SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = self.params[:3]
            return {"fx": f, "fy": f, "cx": cx, "cy": cy, "width": float(self.width), "height": float(self.height)}
        raise ValueError(f"unsupported COLMAP camera model: {self.model}")


@dataclass(frozen=True)
class ColmapImage:
    id: str
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: str
    name: str

    @property
    def camera_origin(self) -> Vec3:
        rotation = _quaternion_to_rotation((self.qw, self.qx, self.qy, self.qz))
        translation = (self.tx, self.ty, self.tz)
        return tuple(-sum(rotation[row][axis] * translation[row] for row in range(3)) for axis in range(3))  # type: ignore[return-value]

    @property
    def forward(self) -> Vec3:
        rotation = _quaternion_to_rotation((self.qw, self.qx, self.qy, self.qz))
        return _normalize(tuple(rotation[row][2] for row in range(3)))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ColmapPoint3D:
    id: str
    xyz: Vec3
    rgb: Vec3


def load_colmap_text_model(path: Path | str) -> tuple[dict[str, ColmapCamera], tuple[ColmapImage, ...], tuple[ColmapPoint3D, ...]]:
    root = Path(path)
    cameras = _read_cameras(root / "cameras.txt")
    images = _read_images(root / "images.txt")
    points = _read_points3d(root / "points3D.txt") if (root / "points3D.txt").exists() else tuple()
    return cameras, images, points


def colmap_text_to_capture_manifest(
    path: Path | str,
    *,
    root: str,
    image_dir: str = "images",
    target_color: Vec3 = (0.5, 0.5, 0.5),
    default_depth: float = 2.0,
) -> CaptureManifest:
    cameras, images, points = load_colmap_text_model(path)
    if not images:
        raise ValueError("COLMAP images.txt did not contain any registered images")
    centroid = _point_centroid(points)
    frames = []
    for image in images:
        camera = cameras.get(image.camera_id)
        if camera is None:
            raise ValueError(f"COLMAP image {image.id} references unknown camera {image.camera_id}")
        origin = image.camera_origin
        look_at = centroid if centroid is not None else _add(origin, image.forward)
        depth = _distance(origin, look_at) if centroid is not None else default_depth
        frames.append(
            {
                "id": f"colmap_image_{image.id}",
                "image_path": str(Path(image_dir) / image.name),
                "depth_path": None,
                "mask_path": None,
                "camera_model": camera.model,
                "intrinsics": camera.intrinsics(),
                "camera_origin": list(origin),
                "look_at": list(look_at),
                "target_color": list(_point_average_color(points) or target_color),
                "target_depth": max(depth, 1e-6),
                "semantic_label": None,
            }
        )
    regions = [_sparse_prior_region(frames[0]["id"], points, centroid, default_depth)]
    payload = {"format": "AURA_CAPTURE_MANIFEST", "root": root, "frames": frames, "regions": regions}
    return CaptureManifest.from_dict(payload)


def write_colmap_capture_manifest(
    path: Path | str,
    output: Path | str,
    *,
    root: str,
    image_dir: str = "images",
) -> Path:
    manifest = colmap_text_to_capture_manifest(path, root=root, image_dir=image_dir)
    return write_capture_manifest(manifest, output)


def _read_cameras(path: Path) -> dict[str, ColmapCamera]:
    if not path.exists():
        raise FileNotFoundError(path)
    cameras = {}
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"malformed COLMAP camera line: {line}")
        camera_id, model, width, height, *params = parts
        cameras[camera_id] = ColmapCamera(
            id=camera_id,
            model=model,
            width=int(width),
            height=int(height),
            params=tuple(float(item) for item in params),
        )
    return cameras


def _read_images(path: Path) -> tuple[ColmapImage, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = _image_record_lines(path)
    images = []
    for index in range(0, len(lines), 2):
        if not lines[index]:
            continue
        parts = lines[index].split()
        if len(parts) < 10:
            raise ValueError(f"malformed COLMAP image line: {lines[index]}")
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, *name_parts = parts
        images.append(
            ColmapImage(
                id=image_id,
                qw=float(qw),
                qx=float(qx),
                qy=float(qy),
                qz=float(qz),
                tx=float(tx),
                ty=float(ty),
                tz=float(tz),
                camera_id=camera_id,
                name=" ".join(name_parts),
            )
        )
    return tuple(images)


def _read_points3d(path: Path) -> tuple[ColmapPoint3D, ...]:
    points = []
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"malformed COLMAP points3D line: {line}")
        point_id, x, y, z, r, g, b, *_rest = parts
        points.append(
            ColmapPoint3D(
                id=point_id,
                xyz=(float(x), float(y), float(z)),
                rgb=(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0),
            )
        )
    return tuple(points)


def _data_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _image_record_lines(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
    while lines and not lines[0]:
        lines.pop(0)
    if len(lines) % 2 != 0:
        lines.append("")
    return lines


def _sparse_prior_region(frame_id: str, points: Sequence[ColmapPoint3D], centroid: Vec3 | None, default_depth: float) -> dict:
    if points:
        min_corner = tuple(min(point.xyz[index] for point in points) for index in range(3))
        max_corner = tuple(max(point.xyz[index] for point in points) for index in range(3))
        center = centroid or (0.0, 0.0, default_depth)
    else:
        center = (0.0, 0.0, default_depth)
        min_corner = tuple(value - 0.25 for value in center)
        max_corner = tuple(value + 0.25 for value in center)
    min_corner = tuple(value - 1e-3 for value in min_corner)
    max_corner = tuple(value + 1e-3 for value in max_corner)
    return {
        "id": "colmap_sparse_prior",
        "frame_id": frame_id,
        "bounds": {"min": list(min_corner), "max": list(max_corner)},
        "evidence": {"geometry_confidence": 0.65, "ray_need": 0.6, "edit_need": 0.25},
        "color": list(_point_average_color(points) or (0.5, 0.5, 0.5)),
        "opacity": 0.45,
        "confidence": 0.65 if points else 0.3,
        "normal": None,
        "material_id": "mat_colmap_sparse_prior",
        "semantic_label": "colmap_sparse_prior",
        "fallback_source": "colmap-text",
    }


def _point_centroid(points: Sequence[ColmapPoint3D]) -> Vec3 | None:
    if not points:
        return None
    return tuple(sum(point.xyz[index] for point in points) / len(points) for index in range(3))  # type: ignore[return-value]


def _point_average_color(points: Sequence[ColmapPoint3D]) -> Vec3 | None:
    if not points:
        return None
    return tuple(sum(point.rgb[index] for point in points) / len(points) for index in range(3))  # type: ignore[return-value]


def _quaternion_to_rotation(quaternion: tuple[float, float, float, float]) -> tuple[Vec3, Vec3, Vec3]:
    qw, qx, qy, qz = _normalize4(quaternion)
    return (
        (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)),
        (2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)),
        (2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)),
    )


def _normalize4(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("COLMAP quaternion must be non-zero")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _normalize(values: Vec3) -> Vec3:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("vector must be non-zero")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _add(left: Vec3, right: Vec3) -> Vec3:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _distance(left: Vec3, right: Vec3) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
