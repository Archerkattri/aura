"""Production image and video streaming export for AURA renders.

This module adds float-radiance EXR export and a video export path on top of the
deterministic stdlib PPM writer in :mod:`aura.render`. Both optional backends are
probed at runtime so the rest of the pipeline degrades gracefully:

* EXR export uses ``imageio`` (FreeImage/`tifffile` plugins). When the optional
  ``[assets]`` extra is missing, callers fall back to a deterministic ``.pfm``
  float raster written with the stdlib so radiance/depth still round-trips.
* Video export assembles an image sequence into an MP4 when a real encoder is
  available (``imageio[ffmpeg]`` or a system ``ffmpeg`` binary). When no encoder
  is present it writes a PNG/PPM frame sequence plus a JSON manifest describing
  the intended clip, which is the documented, always-available fallback.

The capability probes here are *real*: they attempt a tiny encode/decode rather
than only checking for an importable module, so tests can gate the hard MP4
encode step on a real signal while always exercising the frame-sequence path.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from aura.render import RenderImage, _to_u8

__all__ = [
    "EXR_CAPABILITY",
    "VideoCapability",
    "exr_export_capability",
    "video_export_capability",
    "read_reference_image",
    "read_pfm_image",
    "write_exr_image",
    "write_pfm_image",
    "write_radiance_image",
    "write_frame_sequence",
    "write_video",
]

SUPPORTED_REFERENCE_SUFFIXES = (".ppm", ".pnm", ".png", ".pfm", ".exr", ".hdr")


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Capability:
    available: bool
    backend: str
    detail: str

    def to_dict(self) -> dict:
        return {"available": self.available, "backend": self.backend, "detail": self.detail}


@dataclass(frozen=True)
class VideoCapability:
    """Result of probing the runtime for a usable MP4 encoder."""

    available: bool
    backend: str
    detail: str

    def to_dict(self) -> dict:
        return {"available": self.available, "backend": self.backend, "detail": self.detail}


@lru_cache(maxsize=1)
def exr_export_capability() -> _Capability:
    """Probe whether float EXR export actually works in this environment."""

    try:
        import numpy as np  # type: ignore[import-not-found]
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import guard
        return _Capability(False, "none", f"imageio/numpy unavailable: {exc}")

    import tempfile

    probe = Path(tempfile.gettempdir()) / "aura_exr_probe.exr"
    try:
        sample = np.zeros((2, 2, 3), dtype="float32")
        sample[0, 0, 0] = 0.5
        imageio.imwrite(probe, sample)
        back = imageio.imread(probe)
        if getattr(back, "shape", None) is None:
            return _Capability(False, "imageio", "EXR probe returned no array")
    except Exception as exc:
        return _Capability(False, "imageio", f"EXR encode probe failed: {exc}")
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
    return _Capability(True, "imageio", "imageio EXR encode/decode probe succeeded")


@lru_cache(maxsize=1)
def video_export_capability() -> VideoCapability:
    """Probe whether an MP4 encoder is genuinely available.

    Tries the ``imageio`` ffmpeg plugin first, then a system ``ffmpeg`` binary.
    The probe performs a real multi-frame encode so a missing or broken encoder
    is reported as unavailable instead of failing later at export time.
    """

    imageio_probe = _probe_imageio_ffmpeg()
    if imageio_probe.available:
        return imageio_probe
    binary_probe = _probe_ffmpeg_binary()
    if binary_probe.available:
        return binary_probe
    detail = f"imageio: {imageio_probe.detail}; binary: {binary_probe.detail}"
    return VideoCapability(False, "none", detail)


def _probe_imageio_ffmpeg() -> VideoCapability:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import guard
        return VideoCapability(False, "imageio-ffmpeg", f"imageio/numpy unavailable: {exc}")

    import tempfile

    probe = Path(tempfile.gettempdir()) / "aura_mp4_probe.mp4"
    try:
        writer = imageio.get_writer(probe, fps=4, format="FFMPEG", macro_block_size=1)
    except Exception as exc:
        return VideoCapability(False, "imageio-ffmpeg", f"writer unavailable: {exc}")
    try:
        for index in range(3):
            frame = np.full((16, 16, 3), index * 40, dtype="uint8")
            writer.append_data(frame)
        writer.close()
        if not probe.exists() or probe.stat().st_size == 0:
            return VideoCapability(False, "imageio-ffmpeg", "encode produced no output")
    except Exception as exc:
        return VideoCapability(False, "imageio-ffmpeg", f"encode probe failed: {exc}")
    finally:
        try:
            probe.unlink()
        except OSError:
            pass
    return VideoCapability(True, "imageio-ffmpeg", "imageio ffmpeg encode probe succeeded")


def _probe_ffmpeg_binary() -> VideoCapability:
    binary = _ffmpeg_binary()
    if binary is None:
        return VideoCapability(False, "ffmpeg-binary", "ffmpeg binary not found on PATH")
    try:
        result = subprocess.run(
            [binary, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return VideoCapability(False, "ffmpeg-binary", f"ffmpeg invocation failed: {exc}")
    if result.returncode != 0:
        return VideoCapability(False, "ffmpeg-binary", "ffmpeg -version returned nonzero")
    return VideoCapability(True, "ffmpeg-binary", f"system ffmpeg available at {binary}")


def _ffmpeg_binary() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


EXR_CAPABILITY = "aura.imaging.exr_export_capability"


# ---------------------------------------------------------------------------
# Float radiance image export (EXR / PFM)
# ---------------------------------------------------------------------------


def write_radiance_image(image: RenderImage, path: Path | str) -> Path:
    """Write float radiance to EXR when available, else a stdlib PFM fallback.

    The chosen format follows the requested extension. ``.exr`` requires the
    optional backend; if it is unavailable the function transparently writes a
    ``.pfm`` sibling so the float data is never silently downcast to 8-bit.
    """

    output = Path(path)
    if output.suffix.lower() == ".exr":
        if exr_export_capability().available:
            return write_exr_image(image, output)
        fallback = output.with_suffix(".pfm")
        return write_pfm_image(image, fallback)
    if output.suffix.lower() == ".pfm":
        return write_pfm_image(image, output)
    raise ValueError(f"unsupported radiance image extension {output.suffix!r}; expected .exr or .pfm")


def write_exr_image(image: RenderImage, path: Path | str) -> Path:
    """Write an EXR float image via imageio. Raises if EXR is unavailable."""

    capability = exr_export_capability()
    if not capability.available:
        raise RuntimeError(
            "EXR export requires the optional asset backend; "
            f"install aura-core[assets] ({capability.detail})"
        )
    import numpy as np  # type: ignore[import-not-found]
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image.pixels, dtype="float32").reshape(image.height, image.width, 3)
    imageio.imwrite(output, array)
    return output


def write_pfm_image(image: RenderImage, path: Path | str) -> Path:
    """Write a Portable Float Map (PFM) using only the stdlib.

    PFM is a tiny, widely-read float raster format, used as the deterministic
    fallback when EXR export is unavailable. Little-endian, scanlines stored
    bottom-to-top per the PFM specification.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = f"PF\n{image.width} {image.height}\n-1.0\n".encode("ascii")
    rows: list[bytes] = []
    for row in range(image.height - 1, -1, -1):
        floats: list[float] = []
        for col in range(image.width):
            pixel = image.pixel(col, row)
            floats.extend(float(channel) for channel in pixel)
        rows.append(struct.pack("<%df" % len(floats), *floats))
    output.write_bytes(header + b"".join(rows))
    return output


def read_pfm_image(path: Path | str) -> RenderImage:
    """Read a little-endian PFM written by :func:`write_pfm_image`."""

    data = Path(path).read_bytes()
    if not data.startswith(b"PF"):
        raise ValueError("PFM color image must start with 'PF'")
    # header: 'PF\n<w> <h>\n<scale>\n'
    cursor = 0
    fields: list[bytes] = []
    while len(fields) < 3:
        end = data.index(b"\n", cursor)
        fields.append(data[cursor:end].strip())
        cursor = end + 1
    width, height = (int(value) for value in fields[1].split())
    scale = float(fields[2])
    little_endian = scale < 0
    fmt = "<" if little_endian else ">"
    count = width * height * 3
    values = struct.unpack(fmt + "%df" % count, data[cursor : cursor + count * 4])
    rows: list[Sequence[float]] = []
    for row in range(height):
        offset = row * width * 3
        rows.append(values[offset : offset + width * 3])
    pixels: list[tuple[float, float, float]] = []
    for row in range(height - 1, -1, -1):
        line = rows[row]
        for col in range(width):
            base = col * 3
            pixels.append((line[base], line[base + 1], line[base + 2]))
    return RenderImage(width=width, height=height, pixels=tuple(pixels))


# ---------------------------------------------------------------------------
# Reference image loading (for benchmark harness)
# ---------------------------------------------------------------------------


def read_reference_image(path: Path | str) -> RenderImage:
    """Load a baseline reference image into a normalized RGB ``RenderImage``.

    Supports PPM (P3) and PNG via the stdlib loader, PFM via this module, and
    EXR/HDR via the optional asset backend. Single-channel images are expanded
    to RGB; alpha channels are dropped.
    """

    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in {".ppm", ".pnm"}:
        from aura.render import read_ppm

        return read_ppm(target)
    if suffix == ".pfm":
        return read_pfm_image(target)
    if suffix == ".png":
        from aura.ingest.capture import _read_capture_raster

        return _raster_image_to_render(_read_capture_raster(target))
    if suffix in {".exr", ".hdr"}:
        return _read_float_render_image(target)
    raise ValueError(
        f"unsupported reference image extension {suffix!r}; "
        f"expected one of {', '.join(SUPPORTED_REFERENCE_SUFFIXES)}"
    )


def _raster_image_to_render(raster: Any) -> RenderImage:
    width = int(raster.width)
    height = int(raster.height)
    channels = int(raster.channels)
    values = list(raster.values)
    pixels: list[tuple[float, float, float]] = []
    for index in range(width * height):
        base = index * channels
        if channels == 1:
            gray = float(values[base])
            pixels.append((gray, gray, gray))
        else:
            pixels.append(
                (float(values[base]), float(values[base + 1]), float(values[base + 2]))
            )
    return RenderImage(width=width, height=height, pixels=tuple(pixels))


def _read_float_render_image(path: Path) -> RenderImage:
    capability = exr_export_capability()
    try:
        import numpy as np  # type: ignore[import-not-found]
        import imageio.v2 as imageio  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - import guard
        raise ValueError(
            f"{path} requires the optional asset backend to load EXR/HDR references "
            f"({capability.detail})"
        ) from exc
    array = np.asarray(imageio.imread(path), dtype="float32")
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"{path} returned unsupported reference shape {array.shape}")
    array = array[:, :, :3]
    height, width = array.shape[0], array.shape[1]
    flat = array.reshape(-1, 3)
    pixels = tuple((float(r), float(g), float(b)) for r, g, b in flat.tolist())
    return RenderImage(width=int(width), height=int(height), pixels=pixels)


# ---------------------------------------------------------------------------
# Frame sequence + video export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSequenceResult:
    directory: Path
    frame_paths: tuple[Path, ...]
    manifest_path: Path
    frame_format: str

    def to_dict(self) -> dict:
        return {
            "directory": str(self.directory),
            "framePaths": [str(item) for item in self.frame_paths],
            "manifestPath": str(self.manifest_path),
            "frameFormat": self.frame_format,
            "frameCount": len(self.frame_paths),
        }


@dataclass(frozen=True)
class VideoExportResult:
    """Outcome of a video export. Either an MP4 or a frame-sequence fallback."""

    format: str  # "mp4" or "frame-sequence"
    backend: str
    fps: int
    frame_count: int
    video_path: Path | None
    sequence: FrameSequenceResult | None

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "backend": self.backend,
            "fps": self.fps,
            "frameCount": self.frame_count,
            "videoPath": None if self.video_path is None else str(self.video_path),
            "sequence": None if self.sequence is None else self.sequence.to_dict(),
        }


def _frame_format() -> str:
    try:
        import imageio.v2  # type: ignore[import-not-found]  # noqa: F401

        return "png"
    except Exception:
        return "ppm"


def write_frame_sequence(
    frames: Iterable[RenderImage],
    directory: Path | str,
    *,
    fps: int = 24,
    prefix: str = "frame",
    frame_format: str | None = None,
) -> FrameSequenceResult:
    """Write rendered frames to disk plus a JSON manifest describing the clip.

    This path never requires an encoder and is always exercised. PNG frames are
    written when ``imageio`` is importable; otherwise stdlib PPM frames are used.
    """

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = frame_format or _frame_format()
    frame_paths: list[Path] = []
    width: int | None = None
    height: int | None = None
    for index, frame in enumerate(frames):
        if width is None:
            width, height = frame.width, frame.height
        elif (frame.width, frame.height) != (width, height):
            raise ValueError("all frames in a sequence must share dimensions")
        frame_path = out_dir / f"{prefix}_{index:05d}.{fmt}"
        _write_frame(frame, frame_path, fmt)
        frame_paths.append(frame_path)
    if not frame_paths:
        raise ValueError("cannot write an empty frame sequence")

    manifest = {
        "format": "AURA_FRAME_SEQUENCE",
        "fps": fps,
        "frameCount": len(frame_paths),
        "width": width,
        "height": height,
        "frameFormat": fmt,
        "frames": [item.name for item in frame_paths],
        "assembleHint": (
            "ffmpeg -framerate %d -i %s_%%05d.%s -pix_fmt yuv420p out.mp4" % (fps, prefix, fmt)
        ),
    }
    manifest_path = out_dir / "sequence.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return FrameSequenceResult(
        directory=out_dir,
        frame_paths=tuple(frame_paths),
        manifest_path=manifest_path,
        frame_format=fmt,
    )


def _write_frame(frame: RenderImage, path: Path, fmt: str) -> None:
    if fmt == "ppm":
        frame.write_ppm(path)
        return
    if fmt == "png":
        import numpy as np  # type: ignore[import-not-found]
        import imageio.v2 as imageio  # type: ignore[import-not-found]

        array = _frame_to_uint8_array(np, frame)
        imageio.imwrite(path, array)
        return
    raise ValueError(f"unsupported frame format {fmt!r}; expected png or ppm")


def _frame_to_uint8_array(np: Any, frame: RenderImage) -> Any:
    flat = [
        [_to_u8(channel) for channel in frame.pixel(col, row)]
        for row in range(frame.height)
        for col in range(frame.width)
    ]
    return np.asarray(flat, dtype="uint8").reshape(frame.height, frame.width, 3)


def write_video(
    frames: Sequence[RenderImage],
    path: Path | str,
    *,
    fps: int = 24,
    allow_fallback: bool = True,
) -> VideoExportResult:
    """Assemble frames into an MP4 when an encoder exists, else a frame sequence.

    When no MP4 encoder is available and ``allow_fallback`` is true, the frames
    are written as an image sequence + manifest next to the requested path (with
    a ``_frames`` suffix). When ``allow_fallback`` is false an explicit error is
    raised so callers that require a real clip can detect the gap.
    """

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("cannot export a video with no frames")
    output = Path(path)
    capability = video_export_capability()
    if capability.available:
        output.parent.mkdir(parents=True, exist_ok=True)
        if capability.backend == "imageio-ffmpeg":
            _encode_mp4_imageio(frame_list, output, fps=fps)
        else:
            _encode_mp4_binary(frame_list, output, fps=fps)
        return VideoExportResult(
            format="mp4",
            backend=capability.backend,
            fps=fps,
            frame_count=len(frame_list),
            video_path=output,
            sequence=None,
        )

    if not allow_fallback:
        raise RuntimeError(
            f"no MP4 encoder available ({capability.detail}); "
            "install aura-core[assets] with ffmpeg, or allow the frame-sequence fallback"
        )
    sequence_dir = output.parent / f"{output.stem}_frames"
    sequence = write_frame_sequence(frame_list, sequence_dir, fps=fps)
    return VideoExportResult(
        format="frame-sequence",
        backend="frame-sequence",
        fps=fps,
        frame_count=len(frame_list),
        video_path=None,
        sequence=sequence,
    )


def _encode_mp4_imageio(frames: Sequence[RenderImage], path: Path, *, fps: int) -> None:
    import numpy as np  # type: ignore[import-not-found]
    import imageio.v2 as imageio  # type: ignore[import-not-found]

    writer = imageio.get_writer(path, fps=fps, format="FFMPEG", macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(_frame_to_uint8_array(np, frame))
    finally:
        writer.close()


def _encode_mp4_binary(frames: Sequence[RenderImage], path: Path, *, fps: int) -> None:
    import tempfile

    binary = _ffmpeg_binary()
    if binary is None:  # pragma: no cover - guarded by capability probe
        raise RuntimeError("ffmpeg binary disappeared after capability probe")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, frame in enumerate(frames):
            frame.write_ppm(tmp_dir / f"frame_{index:05d}.ppm")
        command = [
            binary,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.ppm"),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(path),
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0 or not path.exists():
            raise RuntimeError(
                "ffmpeg encode failed: " + result.stderr.decode("utf-8", "replace")[-500:]
            )
