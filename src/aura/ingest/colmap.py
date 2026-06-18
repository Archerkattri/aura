"""COLMAP binary and text model ingest: camera, image, and point-cloud parsing into AURA capture manifests."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Sequence

from aura.ingest.capture import CaptureManifest, write_capture_manifest

Vec3 = tuple[float, float, float]

_COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class ColmapCamera:
    """A parsed COLMAP camera entry with model name, image dimensions, and intrinsic parameters."""

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
        if self.model in {"FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
            f, cx, cy = self.params[:3]
            return {"fx": f, "fy": f, "cx": cx, "cy": cy, "width": float(self.width), "height": float(self.height)}
        raise ValueError(f"unsupported COLMAP camera model: {self.model}")


@dataclass(frozen=True)
class ColmapImage:
    """A parsed COLMAP registered image with quaternion rotation, translation, and camera reference."""

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
    """A parsed COLMAP sparse 3-D point with world position and RGB color."""

    id: str
    xyz: Vec3
    rgb: Vec3


def load_colmap_text_model(path: Path | str) -> tuple[dict[str, ColmapCamera], tuple[ColmapImage, ...], tuple[ColmapPoint3D, ...]]:
    """Load a COLMAP text-format sparse model from ``path`` containing ``cameras.txt`` and ``images.txt``."""
    root = Path(path)
    cameras = _read_cameras(root / "cameras.txt")
    images = _read_images(root / "images.txt")
    points = _read_points3d(root / "points3D.txt") if (root / "points3D.txt").exists() else tuple()
    return cameras, images, points


def load_colmap_binary_model(path: Path | str) -> tuple[dict[str, ColmapCamera], tuple[ColmapImage, ...], tuple[ColmapPoint3D, ...]]:
    """Load a COLMAP binary sparse model from ``path`` containing ``cameras.bin`` and ``images.bin``."""
    root = Path(path)
    cameras = _read_cameras_binary(root / "cameras.bin")
    images = _read_images_binary(root / "images.bin")
    points = _read_points3d_binary(root / "points3D.bin") if (root / "points3D.bin").exists() else tuple()
    return cameras, images, points


def load_colmap_model(path: Path | str) -> tuple[dict[str, ColmapCamera], tuple[ColmapImage, ...], tuple[ColmapPoint3D, ...], str]:
    """Auto-detect and load a COLMAP sparse model from ``path``, returning cameras, images, points, and format label."""
    root = Path(path)
    if (root / "cameras.bin").exists() and (root / "images.bin").exists():
        cameras, images, points = load_colmap_binary_model(root)
        return cameras, images, points, "colmap-binary"
    if (root / "cameras.txt").exists() and (root / "images.txt").exists():
        cameras, images, points = load_colmap_text_model(root)
        return cameras, images, points, "colmap-text"
    raise FileNotFoundError(f"{root} must contain COLMAP cameras/images .bin or .txt files")


def colmap_to_capture_manifest(
    path: Path | str,
    *,
    root: str,
    image_dir: str = "images",
    target_color: Vec3 = (0.5, 0.5, 0.5),
    default_depth: float = 2.0,
    max_seed_regions: int = 2048,
    point_seeded: bool = False,
) -> CaptureManifest:
    """Convert a COLMAP sparse model directory (binary or text) into an AURA capture manifest."""
    cameras, images, points, source = load_colmap_model(path)
    return _colmap_to_capture_manifest(
        Path(path),
        cameras,
        images,
        points,
        root=root,
        image_dir=image_dir,
        target_color=target_color,
        default_depth=default_depth,
        source=source,
        max_seed_regions=max_seed_regions,
        point_seeded=point_seeded,
    )


def colmap_text_to_capture_manifest(
    path: Path | str,
    *,
    root: str,
    image_dir: str = "images",
    target_color: Vec3 = (0.5, 0.5, 0.5),
    default_depth: float = 2.0,
    max_seed_regions: int = 2048,
    point_seeded: bool = False,
) -> CaptureManifest:
    """Convert a COLMAP text-format sparse model into an AURA capture manifest."""
    cameras, images, points = load_colmap_text_model(path)
    return _colmap_to_capture_manifest(
        Path(path),
        cameras,
        images,
        points,
        root=root,
        image_dir=image_dir,
        target_color=target_color,
        default_depth=default_depth,
        source="colmap-text",
        max_seed_regions=max_seed_regions,
        point_seeded=point_seeded,
    )


def colmap_binary_to_capture_manifest(
    path: Path | str,
    *,
    root: str,
    image_dir: str = "images",
    target_color: Vec3 = (0.5, 0.5, 0.5),
    default_depth: float = 2.0,
    max_seed_regions: int = 2048,
    point_seeded: bool = False,
) -> CaptureManifest:
    """Convert a COLMAP binary sparse model into an AURA capture manifest."""
    cameras, images, points = load_colmap_binary_model(path)
    return _colmap_to_capture_manifest(
        Path(path),
        cameras,
        images,
        points,
        root=root,
        image_dir=image_dir,
        target_color=target_color,
        default_depth=default_depth,
        source="colmap-binary",
        max_seed_regions=max_seed_regions,
        point_seeded=point_seeded,
    )


def _probe_image_size(path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` of an image file without decoding all pixels.

    Returns ``None`` when the file is missing or no backend can read it, so the
    caller can fall back to the COLMAP camera dimensions unchanged.
    """
    if not path.exists():
        return None
    try:
        import imageio.v3 as imageio  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        props = imageio.improps(path)
        shape = tuple(int(item) for item in props.shape)
    except Exception:
        try:
            shape = tuple(int(item) for item in imageio.imread(path).shape)
        except Exception:
            return None
    if len(shape) < 2:
        return None
    height, width = shape[0], shape[1]
    if width <= 0 or height <= 0:
        return None
    return width, height


def _intrinsics_for_image(intrinsics: dict[str, float], image_file: Path) -> dict[str, float]:
    """Scale COLMAP intrinsics to the resolution of the actual image file.

    Real capture datasets (Tanks and Temples, Mip-NeRF 360, Deep Blending) ship
    images downsampled from the resolution COLMAP was run on, so the sparse model
    reports e.g. 1957x1091 while ``images/`` holds 979x546 frames. Rays built
    from the unscaled intrinsics would be wrong; this rescales fx/fy/cx/cy and
    sets width/height to match the image actually loaded. No-op when the image
    is absent or already matches the camera resolution.
    """
    size = _probe_image_size(image_file)
    if size is None:
        return intrinsics
    actual_w, actual_h = size
    cam_w = intrinsics.get("width", 0.0)
    cam_h = intrinsics.get("height", 0.0)
    if cam_w <= 0 or cam_h <= 0 or (actual_w == int(cam_w) and actual_h == int(cam_h)):
        return intrinsics
    sx = actual_w / cam_w
    sy = actual_h / cam_h
    return {
        "fx": intrinsics["fx"] * sx,
        "fy": intrinsics["fy"] * sy,
        "cx": intrinsics["cx"] * sx,
        "cy": intrinsics["cy"] * sy,
        "width": float(actual_w),
        "height": float(actual_h),
    }


def _colmap_to_capture_manifest(
    model_path: Path,
    cameras: dict[str, ColmapCamera],
    images: Sequence[ColmapImage],
    points: Sequence[ColmapPoint3D],
    *,
    root: str,
    image_dir: str,
    target_color: Vec3,
    default_depth: float,
    source: str,
    max_seed_regions: int = 2048,
    point_seeded: bool = False,
) -> CaptureManifest:
    if not images:
        raise ValueError(f"{source} model did not contain any registered images")
    centroid = _point_centroid(points)
    frames = []
    for image in images:
        camera = cameras.get(image.camera_id)
        if camera is None:
            raise ValueError(f"COLMAP image {image.id} references unknown camera {image.camera_id}")
        origin = image.camera_origin
        look_at = centroid if centroid is not None else _add(origin, image.forward)
        depth = _distance(origin, look_at) if centroid is not None else default_depth
        relative_image_path = Path(image_dir) / image.name
        intrinsics = _intrinsics_for_image(
            camera.intrinsics(), Path(root) / relative_image_path
        )
        frames.append(
            {
                "id": f"colmap_image_{image.id}",
                "image_path": str(relative_image_path),
                "depth_path": _find_colmap_depth_path(model_path, image.name),
                "mask_path": None,
                "normal_path": _find_colmap_normal_path(model_path, image.name),
                "camera_model": camera.model,
                "intrinsics": intrinsics,
                "camera_origin": list(origin),
                "look_at": list(look_at),
                "target_color": list(_point_average_color(points) or target_color),
                "target_depth": max(depth, 1e-6),
                "semantic_label": None,
            }
        )
    regions = _sparse_prior_regions(
        frames[0]["id"], points, centroid, default_depth, source, max_seed_regions,
        point_seeded=point_seeded,
    )
    payload = {"format": "AURA_CAPTURE_MANIFEST", "root": root, "frames": frames, "regions": regions}
    return CaptureManifest.from_dict(payload)


def write_colmap_capture_manifest(
    path: Path | str,
    output: Path | str,
    *,
    root: str,
    image_dir: str = "images",
    max_seed_regions: int = 2048,
    point_seeded: bool = False,
) -> Path:
    """Convert a COLMAP sparse model to a capture manifest and write it to ``output``."""
    manifest = colmap_to_capture_manifest(
        path, root=root, image_dir=image_dir, max_seed_regions=max_seed_regions,
        point_seeded=point_seeded,
    )
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


def _find_colmap_depth_path(model_path: Path, image_name: str) -> str | None:
    candidates = (
        model_path.parent / "stereo" / "depth_maps" / f"{image_name}.photometric.bin",
        model_path.parent / "stereo" / "depth_maps" / f"{image_name}.geometric.bin",
        model_path / "depth_maps" / f"{image_name}.photometric.bin",
        model_path / "depth_maps" / f"{image_name}.geometric.bin",
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.relative_to(model_path.parent).as_posix()
            except ValueError:
                return candidate.as_posix()
    return None


def _find_colmap_normal_path(model_path: Path, image_name: str) -> str | None:
    candidates = (
        model_path.parent / "stereo" / "normal_maps" / f"{image_name}.photometric.bin",
        model_path.parent / "stereo" / "normal_maps" / f"{image_name}.geometric.bin",
        model_path / "normal_maps" / f"{image_name}.photometric.bin",
        model_path / "normal_maps" / f"{image_name}.geometric.bin",
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.relative_to(model_path.parent).as_posix()
            except ValueError:
                return candidate.as_posix()
    return None


def _read_cameras_binary(path: Path) -> dict[str, ColmapCamera]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    offset = 0
    (camera_count,), offset = _unpack(data, offset, "<Q")
    cameras = {}
    for _index in range(camera_count):
        camera_id, model_id, width, height, offset = _unpack_camera_header(data, offset)
        if model_id not in _COLMAP_CAMERA_MODELS:
            raise ValueError(f"unsupported COLMAP binary camera model id: {model_id}")
        model, param_count = _COLMAP_CAMERA_MODELS[model_id]
        params, offset = _unpack(data, offset, "<" + "d" * param_count)
        cameras[str(camera_id)] = ColmapCamera(
            id=str(camera_id),
            model=model,
            width=int(width),
            height=int(height),
            params=tuple(float(item) for item in params),
        )
    _require_consumed(path, data, offset)
    return cameras


def _read_images_binary(path: Path) -> tuple[ColmapImage, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    offset = 0
    (image_count,), offset = _unpack(data, offset, "<Q")
    images = []
    for _index in range(image_count):
        values, offset = _unpack(data, offset, "<idddddddi")
        image_id = values[0]
        qw, qx, qy, qz, tx, ty, tz = values[1:8]
        camera_id = values[8]
        name_bytes, offset = _read_null_terminated(data, offset, path)
        (point2d_count,), offset = _unpack(data, offset, "<Q")
        offset += point2d_count * struct.calcsize("<ddq")
        if offset > len(data):
            raise ValueError(f"{path} has a truncated POINTS2D block")
        images.append(
            ColmapImage(
                id=str(image_id),
                qw=float(qw),
                qx=float(qx),
                qy=float(qy),
                qz=float(qz),
                tx=float(tx),
                ty=float(ty),
                tz=float(tz),
                camera_id=str(camera_id),
                name=name_bytes.decode("utf-8"),
            )
        )
    _require_consumed(path, data, offset)
    return tuple(images)


def _read_points3d_binary(path: Path) -> tuple[ColmapPoint3D, ...]:
    data = path.read_bytes()
    offset = 0
    (point_count,), offset = _unpack(data, offset, "<Q")
    points = []
    for _index in range(point_count):
        values, offset = _unpack(data, offset, "<QdddBBBdQ")
        point_id = values[0]
        x, y, z = values[1:4]
        r, g, b = values[4:7]
        track_length = values[8]
        offset += track_length * struct.calcsize("<ii")
        if offset > len(data):
            raise ValueError(f"{path} has a truncated TRACK block")
        points.append(
            ColmapPoint3D(
                id=str(point_id),
                xyz=(float(x), float(y), float(z)),
                rgb=(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0),
            )
        )
    _require_consumed(path, data, offset)
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


#: Point clouds with at least this many points are seeded as a dense voxel grid
#: (one carrier-region per occupied voxel) instead of the coarse near/far split.
#: Real COLMAP models have tens of thousands of points; small synthetic test
#: models stay on the legacy near/far path so their seeding is unchanged.
_DENSE_SEED_MIN_POINTS = 64


def _robust_point_subset(points: Sequence[ColmapPoint3D]) -> tuple[ColmapPoint3D, ...]:
    """Drop COLMAP outlier points that lie outside the central 1st-99th percentile
    range on any axis.

    Sparse models routinely contain a few stray points hundreds of units from the
    scene (reconstruction noise). Left in, they inflate the bounding box so the
    voxel grid becomes coarse and every seeded gaussian gets an enormous
    covariance. This keeps the dense central cloud and discards the tails.
    """
    points = tuple(points)
    if len(points) < 16:
        return points
    bounds = []
    for axis in range(3):
        values = sorted(p.xyz[axis] for p in points)
        n = len(values)
        bounds.append((values[max(0, n // 100)], values[min(n - 1, 99 * n // 100)]))
    kept = tuple(
        p for p in points if all(bounds[a][0] <= p.xyz[a] <= bounds[a][1] for a in range(3))
    )
    return kept or points


def _voxel_prior_regions(
    frame_id: str,
    points: Sequence[ColmapPoint3D],
    default_depth: float,
    source: str,
    max_regions: int,
) -> list[dict]:
    """Seed one carrier-region per occupied voxel of the sparse point cloud.

    The COLMAP sparse points are bucketed into a 3D voxel grid sized so the
    number of occupied cells is bounded by ``max_regions``. Each occupied voxel
    becomes a locally-tight region (bounds + average colour from its own points),
    giving dense, spatially-local carrier initialisation comparable to seeding a
    primitive per cluster of SfM points (cf. 3DGS one Gaussian per point).

    Outlier points are filtered first so the voxel grid tracks the real scene
    extent rather than reconstruction noise.
    """
    points = _robust_point_subset(points)
    mins = tuple(min(p.xyz[i] for p in points) for i in range(3))
    maxs = tuple(max(p.xyz[i] for p in points) for i in range(3))
    extents = tuple(max(maxs[i] - mins[i], 1e-6) for i in range(3))
    grid = max(1, round(max(max_regions, 1) ** (1.0 / 3.0)))
    buckets: dict[tuple[int, int, int], list[ColmapPoint3D]] = {}
    for point in points:
        key = tuple(
            min(grid - 1, int((point.xyz[i] - mins[i]) / extents[i] * grid))
            for i in range(3)
        )
        buckets.setdefault(key, []).append(point)
    # Prefer the most-populated voxels when the occupancy exceeds the budget.
    ordered = sorted(buckets.values(), key=len, reverse=True)[:max_regions]
    busiest = max((len(v) for v in ordered), default=1)
    regions = []
    for index, voxel_points in enumerate(ordered):
        confidence = min(0.95, 0.5 + 0.45 * len(voxel_points) / busiest)
        regions.append(
            _sparse_prior_region(
                frame_id,
                tuple(voxel_points),
                _point_centroid(voxel_points),
                default_depth,
                source,
                f"colmap_sparse_voxel_{index}",
                confidence,
            )
        )
    return regions


def _sfm_point_seeded_regions(
    frame_id: str,
    points: Sequence[ColmapPoint3D],
    source: str,
    max_regions: int,
) -> list[dict]:
    """Seed one carrier per SfM point — the 3DGS initialization strategy.

    Each carrier is placed exactly at the reconstructed 3D point with a
    tight bounding box whose half-extent is the mean inter-point spacing.
    This ensures every carrier overlaps with real scene geometry, unlike
    voxel seeding where carrier centres sit at voxel centroids that may be
    far from any surface.
    """
    points = _robust_point_subset(points)
    if not points:
        return []
    # Estimate typical inter-point spacing from scene extent and density.
    mins = tuple(min(p.xyz[i] for p in points) for i in range(3))
    maxs = tuple(max(p.xyz[i] for p in points) for i in range(3))
    diag = sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(3)))
    half_scale = max(1e-3, diag / max(1, sqrt(float(len(points)))) * 0.5)
    # If more points than budget, sample uniformly by stepping through sorted points.
    sampled: Sequence[ColmapPoint3D]
    if len(points) > max_regions:
        step = len(points) / max_regions
        sampled = [points[int(i * step)] for i in range(max_regions)]
    else:
        sampled = points
    regions = []
    for index, point in enumerate(sampled):
        x, y, z = point.xyz
        r, g, b = point.rgb
        half = half_scale
        regions.append({
            "id": f"colmap_sfm_point_{index}",
            "frame_id": frame_id,
            "bounds": {
                "min": [x - half, y - half, z - half],
                "max": [x + half, y + half, z + half],
            },
            "evidence": {"geometry_confidence": 0.75, "ray_need": 0.6, "edit_need": 0.2},
            "color": [r, g, b],
            "opacity": 0.1,
            "confidence": 0.75,
            "normal": None,
            "material_id": "mat_colmap_sfm_point",
            "semantic_label": "colmap_sfm_point",
            "fallback_source": source,
        })
    return regions


def _sparse_prior_regions(
    frame_id: str,
    points: Sequence[ColmapPoint3D],
    centroid: Vec3 | None,
    default_depth: float,
    source: str,
    max_seed_regions: int = 2048,
    point_seeded: bool = False,
) -> list[dict]:
    if not points:
        return [_sparse_prior_region(frame_id, tuple(), centroid, default_depth, source, "colmap_sparse_prior", 0.3)]
    if point_seeded:
        return _sfm_point_seeded_regions(frame_id, points, source, max_seed_regions)
    if len(points) >= _DENSE_SEED_MIN_POINTS and max_seed_regions > 2:
        return _voxel_prior_regions(frame_id, points, default_depth, source, max_seed_regions)
    layers = _sparse_depth_layers(points)
    if len(layers) == 1:
        return [_sparse_prior_region(frame_id, layers[0], centroid, default_depth, source, "colmap_sparse_prior", 0.65)]
    regions = []
    for index, layer in enumerate(layers):
        name = "near" if index == 0 else "far"
        confidence = min(0.9, 0.55 + 0.2 * len(layer) / len(points))
        regions.append(_sparse_prior_region(frame_id, layer, _point_centroid(layer), default_depth, source, f"colmap_sparse_prior_{name}", confidence))
    return regions


def _sparse_prior_region(
    frame_id: str,
    points: Sequence[ColmapPoint3D],
    centroid: Vec3 | None,
    default_depth: float,
    source: str,
    region_id: str,
    confidence: float,
) -> dict:
    if points:
        min_corner = tuple(min(point.xyz[index] for point in points) for index in range(3))
        max_corner = tuple(max(point.xyz[index] for point in points) for index in range(3))
    else:
        center = centroid or (0.0, 0.0, default_depth)
        min_corner = tuple(value - 0.25 for value in center)
        max_corner = tuple(value + 0.25 for value in center)
    min_corner = tuple(value - 1e-3 for value in min_corner)
    max_corner = tuple(value + 1e-3 for value in max_corner)
    return {
        "id": region_id,
        "frame_id": frame_id,
        "bounds": {"min": list(min_corner), "max": list(max_corner)},
        "evidence": {"geometry_confidence": confidence, "ray_need": 0.6, "edit_need": 0.25},
        "color": list(_point_average_color(points) or (0.5, 0.5, 0.5)),
        "opacity": 0.1,
        "confidence": confidence,
        "normal": None,
        "material_id": "mat_colmap_sparse_prior",
        "semantic_label": "colmap_sparse_prior",
        "fallback_source": source,
    }


def _sparse_depth_layers(points: Sequence[ColmapPoint3D]) -> tuple[tuple[ColmapPoint3D, ...], ...]:
    if len(points) < 2:
        return (tuple(points),)
    min_z = min(point.xyz[2] for point in points)
    max_z = max(point.xyz[2] for point in points)
    if max_z - min_z <= max(abs(max_z) * 0.05, 1e-4):
        return (tuple(points),)
    midpoint = (min_z + max_z) / 2.0
    near = tuple(point for point in points if point.xyz[2] <= midpoint)
    far = tuple(point for point in points if point.xyz[2] > midpoint)
    if not near or not far:
        return (tuple(points),)
    return near, far


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


def _unpack(data: bytes, offset: int, fmt: str) -> tuple[tuple, int]:
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError("truncated COLMAP binary model file")
    return struct.unpack_from(fmt, data, offset), offset + size


def _unpack_camera_header(data: bytes, offset: int) -> tuple[int, int, int, int, int]:
    values, offset = _unpack(data, offset, "<iiQQ")
    camera_id, model_id, width, height = values
    return int(camera_id), int(model_id), int(width), int(height), offset


def _read_null_terminated(data: bytes, offset: int, path: Path) -> tuple[bytes, int]:
    end = data.find(b"\x00", offset)
    if end < 0:
        raise ValueError(f"{path} has an unterminated image name")
    return data[offset:end], end + 1


def _require_consumed(path: Path, data: bytes, offset: int) -> None:
    if offset != len(data):
        raise ValueError(f"{path} has {len(data) - offset} trailing bytes")
