from __future__ import annotations

from dataclasses import dataclass
from math import isinf, log10
from pathlib import Path
from typing import Any, Literal, Sequence

from aura.elements import Bounds
from aura.ray import Ray, Vec3
from aura.scene import AuraScene

Pixel = tuple[float, float, float]


@dataclass(frozen=True)
class RenderImage:
    width: int
    height: int
    pixels: tuple[Pixel, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(self.pixels) != self.width * self.height:
            raise ValueError("pixel count must equal width * height")

    def pixel(self, x: int, y: int) -> Pixel:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError("pixel coordinate out of bounds")
        return self.pixels[y * self.width + x]

    def write_ppm(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["P3", f"{self.width} {self.height}", "255"]
        for pixel in self.pixels:
            lines.append(" ".join(str(_to_u8(channel)) for channel in pixel))
        output.write_text("\n".join(lines) + "\n", encoding="ascii")
        return output


def read_ppm(path: Path | str) -> RenderImage:
    tokens = _ppm_tokens(Path(path).read_text(encoding="ascii"))
    if len(tokens) < 4 or tokens[0] != "P3":
        raise ValueError("PPM must use ASCII P3 format")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    if max_value <= 0:
        raise ValueError("PPM max value must be positive")
    values = [int(item) for item in tokens[4:]]
    expected = width * height * 3
    if len(values) != expected:
        raise ValueError(f"PPM expected {expected} channel values but found {len(values)}")
    pixels = []
    for index in range(0, len(values), 3):
        pixels.append((values[index] / max_value, values[index + 1] / max_value, values[index + 2] / max_value))
    return RenderImage(width=width, height=height, pixels=tuple(pixels))


def compare_images(left: RenderImage, right: RenderImage, *, min_psnr: float | None = None) -> dict:
    mse = image_mse(left, right)
    psnr = image_psnr(left, right)
    ssim = image_ssim(left, right)
    perceptual = image_lpips_proxy(left, right)
    passed = True if min_psnr is None else psnr >= min_psnr
    return {
        "width": left.width,
        "height": left.height,
        "mse": mse,
        "psnr": None if isinf(psnr) else psnr,
        "psnrInfinite": isinf(psnr),
        "ssim": ssim,
        "lpipsProxy": perceptual,
        "minPsnr": min_psnr,
        "passed": passed,
    }


def render_orthographic(
    scene: AuraScene,
    *,
    width: int = 64,
    height: int = 64,
    bounds: Bounds | None = None,
    camera_z: float | None = None,
) -> RenderImage:
    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    frame = bounds or _scene_bounds(scene)
    z = camera_z if camera_z is not None else frame.min_corner[2] - 1.0
    pixels: list[Pixel] = []
    for row in range(height):
        y = _lerp(frame.max_corner[1], frame.min_corner[1], _center_fraction(row, height))
        for col in range(width):
            x = _lerp(frame.min_corner[0], frame.max_corner[0], _center_fraction(col, width))
            result = scene.ray_query(Ray(origin=(x, y, z), direction=(0.0, 0.0, 1.0)))
            pixels.append(result.color)
    return RenderImage(width=width, height=height, pixels=tuple(pixels))


def render_orthographic_cuda(
    scene: AuraScene,
    *,
    width: int = 64,
    height: int = 64,
    bounds: Bounds | None = None,
    camera_z: float | None = None,
    fallback_backend: Literal["cpu", "torch", "auto", "none"] = "auto",
    device: str | None = None,
    require_cuda: bool = False,
    threads_per_block: int = 128,
    max_hits: int = 8,
    extension: Any | None = None,
    extension_module: Any | None = None,
) -> RenderImage:
    """Render an orthographic preview through the batched CUDA renderer boundary."""

    ray_origins, ray_directions = orthographic_camera_rays(
        scene,
        width=width,
        height=height,
        bounds=bounds,
        camera_z=camera_z,
    )
    from aura.cuda_renderer import cuda_render_rays

    batch = cuda_render_rays(
        scene,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        fallback_backend=fallback_backend,
        device=device,
        require_cuda=require_cuda,
        threads_per_block=threads_per_block,
        max_hits=max_hits,
        extension=extension,
        extension_module=extension_module,
    )
    return RenderImage(width=width, height=height, pixels=batch.color)


def render_orthographic_torch(
    scene: AuraScene,
    *,
    width: int = 64,
    height: int = 64,
    bounds: Bounds | None = None,
    camera_z: float | None = None,
    device: str | None = None,
    require_cuda: bool = False,
    scene_tensors: Any | None = None,
) -> RenderImage:
    """Render an orthographic preview through the native tensor torch renderer."""

    from aura.torch_renderer import torch_render_rays, torch_renderer_status

    status = torch_renderer_status()
    resolved_device = device or status.default_device or "cpu"
    if require_cuda and not str(resolved_device).startswith("cuda"):
        raise RuntimeError("torch orthographic render requires a CUDA torch device")
    ray_origins, ray_directions = orthographic_camera_rays(
        scene,
        width=width,
        height=height,
        bounds=bounds,
        camera_z=camera_z,
    )

    batch = torch_render_rays(
        scene,
        ray_origins,
        ray_directions,
        device=resolved_device,
        frame_id_prefix="orthographic",
        scene_tensors=scene_tensors,
    )
    return RenderImage(width=width, height=height, pixels=batch.predicted_color)


def orthographic_camera_rays(
    scene: AuraScene,
    *,
    width: int = 64,
    height: int = 64,
    bounds: Bounds | None = None,
    camera_z: float | None = None,
) -> tuple[tuple[Vec3, ...], tuple[Vec3, ...]]:
    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    frame = bounds or _scene_bounds(scene)
    z = camera_z if camera_z is not None else frame.min_corner[2] - 1.0
    origins: list[Vec3] = []
    directions: list[Vec3] = []
    for row in range(height):
        y = _lerp(frame.max_corner[1], frame.min_corner[1], _center_fraction(row, height))
        for col in range(width):
            x = _lerp(frame.min_corner[0], frame.max_corner[0], _center_fraction(col, width))
            origins.append((x, y, z))
            directions.append((0.0, 0.0, 1.0))
    return tuple(origins), tuple(directions)


def image_mse(left: RenderImage, right: RenderImage) -> float:
    if left.width != right.width or left.height != right.height:
        raise ValueError("images must have matching dimensions")
    total = 0.0
    count = 0
    for a, b in zip(left.pixels, right.pixels):
        for channel_a, channel_b in zip(a, b):
            total += (channel_a - channel_b) ** 2
            count += 1
    return total / count


def image_psnr(left: RenderImage, right: RenderImage) -> float:
    mse = image_mse(left, right)
    if mse == 0.0:
        return float("inf")
    return 10.0 * log10(1.0 / mse)


def image_ssim(left: RenderImage, right: RenderImage) -> float:
    _require_matching_dimensions(left, right)
    left_values = _flat_channels(left)
    right_values = _flat_channels(right)
    mu_left = sum(left_values) / len(left_values)
    mu_right = sum(right_values) / len(right_values)
    var_left = sum((value - mu_left) ** 2 for value in left_values) / len(left_values)
    var_right = sum((value - mu_right) ** 2 for value in right_values) / len(right_values)
    covariance = sum((a - mu_left) * (b - mu_right) for a, b in zip(left_values, right_values)) / len(left_values)
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_left * mu_right + c1) * (2.0 * covariance + c2)
    denominator = (mu_left**2 + mu_right**2 + c1) * (var_left + var_right + c2)
    if denominator == 0.0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def image_lpips_proxy(left: RenderImage, right: RenderImage) -> float:
    """Deterministic perceptual-distance proxy, not learned LPIPS."""

    _require_matching_dimensions(left, right)
    total = 0.0
    count = 0
    for a, b in zip(left.pixels, right.pixels):
        for channel_a, channel_b in zip(a, b):
            total += abs(channel_a - channel_b)
            count += 1
    return total / count


def _require_matching_dimensions(left: RenderImage, right: RenderImage) -> None:
    if left.width != right.width or left.height != right.height:
        raise ValueError("images must have matching dimensions")


def _flat_channels(image: RenderImage) -> tuple[float, ...]:
    return tuple(channel for pixel in image.pixels for channel in pixel)


def _scene_bounds(scene: AuraScene) -> Bounds:
    if scene.chunks:
        return _union_bounds([chunk.bounds for chunk in scene.chunks])
    if scene.elements:
        return _union_bounds([element.bounds for element in scene.elements])
    raise ValueError("cannot render an empty scene")


def _union_bounds(bounds: Sequence[Bounds]) -> Bounds:
    return Bounds(
        min_corner=(
            min(item.min_corner[0] for item in bounds),
            min(item.min_corner[1] for item in bounds),
            min(item.min_corner[2] for item in bounds),
        ),
        max_corner=(
            max(item.max_corner[0] for item in bounds),
            max(item.max_corner[1] for item in bounds),
            max(item.max_corner[2] for item in bounds),
        ),
    )


def _center_fraction(index: int, size: int) -> float:
    return (index + 0.5) / size


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _to_u8(value: float) -> int:
    return max(0, min(255, round(float(value) * 255.0)))


def _ppm_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        tokens.extend(line.split())
    return tokens
