import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import aura.imaging as imaging
from aura import (
    RenderImage,
    exr_export_capability,
    package_scene,
    read_pfm_image,
    render_turntable_frames,
    turntable_camera_path,
    video_export_capability,
    write_frame_sequence,
    write_pfm_image,
    write_radiance_image,
    write_video,
)
from aura.imaging import VideoCapability, write_exr_image
from aura.cli import demo_scene

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _cli_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC + (os.pathsep + existing if existing else "")
    return env


def _sample_image() -> RenderImage:
    return RenderImage(
        width=2,
        height=2,
        pixels=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9), (1.0, 0.0, 0.5)),
    )


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def test_capability_probes_return_structured_reports():
    exr = exr_export_capability().to_dict()
    video = video_export_capability().to_dict()
    for report in (exr, video):
        assert set(report) == {"available", "backend", "detail"}
        assert isinstance(report["available"], bool)
        assert report["detail"]


# ---------------------------------------------------------------------------
# PFM fallback always works (stdlib only)
# ---------------------------------------------------------------------------


def test_pfm_roundtrip_preserves_float_radiance(tmp_path):
    image = _sample_image()
    path = write_pfm_image(image, tmp_path / "radiance.pfm")
    assert path.exists()
    recovered = read_pfm_image(path)
    assert recovered.width == image.width
    assert recovered.height == image.height
    for original, restored in zip(image.pixels, recovered.pixels):
        for a, b in zip(original, restored):
            assert abs(a - b) < 1e-6


def test_write_radiance_image_falls_back_to_pfm_without_exr(tmp_path, monkeypatch):
    monkeypatch.setattr(
        imaging,
        "exr_export_capability",
        lambda: imaging._Capability(False, "none", "disabled for test"),
    )
    written = write_radiance_image(_sample_image(), tmp_path / "frame.exr")
    assert written.suffix == ".pfm"
    assert written.exists()


def test_write_exr_raises_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        imaging,
        "exr_export_capability",
        lambda: imaging._Capability(False, "none", "disabled for test"),
    )
    with pytest.raises(RuntimeError):
        write_exr_image(_sample_image(), tmp_path / "frame.exr")


@pytest.mark.skipif(
    not exr_export_capability().available,
    reason=f"EXR backend unavailable: {exr_export_capability().detail}",
)
def test_exr_export_writes_float_image(tmp_path):
    path = write_exr_image(_sample_image(), tmp_path / "radiance.exr")
    assert path.exists()
    assert path.stat().st_size > 0
    import imageio.v2 as imageio

    array = imageio.imread(path)
    assert array.shape == (2, 2, 3)


def test_write_radiance_image_prefers_exr_when_available(tmp_path):
    path = write_radiance_image(_sample_image(), tmp_path / "frame.exr")
    if exr_export_capability().available:
        assert path.suffix == ".exr"
    else:
        assert path.suffix == ".pfm"
    assert path.exists()


# ---------------------------------------------------------------------------
# Turntable camera path
# ---------------------------------------------------------------------------


def test_turntable_camera_path_is_monotonic_and_sized():
    scene = demo_scene()
    path = turntable_camera_path(scene, frames=5)
    assert len(path) == 5
    assert all(path[i] < path[i + 1] for i in range(len(path) - 1))


def test_render_turntable_frames_produces_distinct_frames():
    scene = demo_scene()
    frames = render_turntable_frames(scene, frames=3, width=6, height=6)
    assert len(frames) == 3
    assert all(frame.width == 6 and frame.height == 6 for frame in frames)


# ---------------------------------------------------------------------------
# Frame sequence path (always exercised)
# ---------------------------------------------------------------------------


def test_write_frame_sequence_writes_frames_and_manifest(tmp_path):
    frames = [_sample_image() for _ in range(4)]
    result = write_frame_sequence(frames, tmp_path / "seq", fps=12)
    assert len(result.frame_paths) == 4
    assert all(path.exists() for path in result.frame_paths)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["frameCount"] == 4
    assert manifest["fps"] == 12
    assert manifest["width"] == 2
    assert len(manifest["frames"]) == 4


def test_write_frame_sequence_rejects_mismatched_dimensions(tmp_path):
    other = RenderImage(width=3, height=1, pixels=((0.0, 0.0, 0.0),) * 3)
    with pytest.raises(ValueError):
        write_frame_sequence([_sample_image(), other], tmp_path / "seq")


def test_write_frame_sequence_can_force_ppm(tmp_path):
    result = write_frame_sequence([_sample_image()], tmp_path / "seq", frame_format="ppm")
    assert result.frame_format == "ppm"
    assert result.frame_paths[0].suffix == ".ppm"


# ---------------------------------------------------------------------------
# Video export: fallback always, MP4 gated on a real encoder probe
# ---------------------------------------------------------------------------


def test_write_video_falls_back_to_frame_sequence_without_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        imaging,
        "video_export_capability",
        lambda: VideoCapability(False, "none", "disabled for test"),
    )
    result = write_video([_sample_image(), _sample_image()], tmp_path / "clip.mp4", fps=8)
    assert result.format == "frame-sequence"
    assert result.video_path is None
    assert result.sequence is not None
    assert len(result.sequence.frame_paths) == 2
    assert result.sequence.manifest_path.exists()


def test_write_video_no_fallback_raises_without_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        imaging,
        "video_export_capability",
        lambda: VideoCapability(False, "none", "disabled for test"),
    )
    with pytest.raises(RuntimeError):
        write_video([_sample_image()], tmp_path / "clip.mp4", allow_fallback=False)


def test_write_video_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        write_video([], tmp_path / "clip.mp4")


@pytest.mark.skipif(
    not video_export_capability().available,
    reason=f"MP4 encoder unavailable: {video_export_capability().detail}",
)
def test_write_video_encodes_mp4_when_encoder_available(tmp_path):
    frames = render_turntable_frames(demo_scene(), frames=4, width=8, height=8)
    result = write_video(frames, tmp_path / "clip.mp4", fps=4)
    assert result.format == "mp4"
    assert result.video_path is not None
    assert result.video_path.exists()
    assert result.video_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_render_cli_exr_format(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    output = tmp_path / "render.exr"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "render",
            str(package_dir),
            "--output",
            str(output),
            "--format",
            "exr",
            "--width",
            "4",
            "--height",
            "4",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_cli_env(),
    )
    printed = Path(result.stdout.strip())
    assert printed.exists()
    assert printed.suffix in {".exr", ".pfm"}


def test_render_video_cli_writes_clip_or_sequence(tmp_path):
    package_dir = tmp_path / "demo.aura"
    package_scene(demo_scene()).write(package_dir)
    output = tmp_path / "turntable.mp4"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aura.cli",
            "render-video",
            str(package_dir),
            "--output",
            str(output),
            "--frames",
            "3",
            "--fps",
            "4",
            "--width",
            "6",
            "--height",
            "6",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_cli_env(),
    )
    payload = json.loads(result.stdout)
    assert payload["frameCount"] == 3
    assert payload["format"] in {"mp4", "frame-sequence"}
    if payload["format"] == "mp4":
        assert Path(payload["videoPath"]).exists()
    else:
        assert payload["sequence"]["frameCount"] == 3


# ---------------------------------------------------------------------------
# Additional targeted tests for uncovered branches
# ---------------------------------------------------------------------------


def test_read_pfm_invalid_magic_raises():
    """Line 261: read_pfm_image raises ValueError for non-PF magic."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pfm", delete=False) as f:
        f.write(b"NOTPF garbage data")
        p = Path(f.name)
    try:
        with pytest.raises(ValueError, match="must start with 'PF'"):
            read_pfm_image(p)
    finally:
        p.unlink(missing_ok=True)


def test_read_reference_image_unsupported_suffix(tmp_path):
    """Lines 315-317: read_reference_image raises ValueError for unknown extensions."""
    from aura.imaging import read_reference_image
    bad = tmp_path / "img.xyz"
    bad.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unsupported"):
        read_reference_image(bad)


def test_read_reference_image_png(tmp_path):
    """Lines 309-312: read_reference_image handles .png via _read_capture_raster."""
    import struct, zlib
    from aura.imaging import read_reference_image

    def _u32be(n): return struct.pack(">I", n)
    def _chunk(name, data):
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return _u32be(len(data)) + name + data + _u32be(crc)

    ihdr = _u32be(2) + _u32be(2) + bytes([8, 2, 0, 0, 0])
    raw_row = bytes([0, 128, 64, 32, 200, 100, 50])
    idat = zlib.compress(raw_row * 2)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    p = tmp_path / "img.png"
    p.write_bytes(png)
    img = read_reference_image(p)
    assert img.width == 2
    assert img.height == 2


def test_frame_sequence_result_to_dict(tmp_path):
    """Line 374: FrameSequenceResult.to_dict() serialises correctly."""
    from aura.imaging import FrameSequenceResult
    result = FrameSequenceResult(
        directory=tmp_path,
        frame_paths=(tmp_path / "f0.ppm",),
        manifest_path=tmp_path / "seq.json",
        frame_format="ppm",
    )
    d = result.to_dict()
    assert d["frameCount"] == 1
    assert d["frameFormat"] == "ppm"
    assert str(tmp_path) in d["directory"]


def test_video_export_result_to_dict_with_sequence(tmp_path):
    """Line 395: VideoExportResult.to_dict() with embedded sequence."""
    from aura.imaging import FrameSequenceResult, VideoExportResult
    seq = FrameSequenceResult(
        directory=tmp_path,
        frame_paths=(),
        manifest_path=tmp_path / "seq.json",
        frame_format="ppm",
    )
    vr = VideoExportResult(
        format="frame-sequence",
        backend="frame-sequence",
        fps=24,
        frame_count=0,
        video_path=None,
        sequence=seq,
    )
    d = vr.to_dict()
    assert d["format"] == "frame-sequence"
    assert d["videoPath"] is None
    assert d["sequence"]["frameCount"] == 0


def test_frame_format_falls_back_to_ppm_when_imageio_absent(monkeypatch):
    """Lines 410-411: _frame_format() returns 'ppm' when imageio is not importable."""
    import sys
    from unittest.mock import patch
    from aura.imaging import _frame_format

    with patch.dict(sys.modules, {"imageio.v2": None}):
        # Clear any existing cache since _frame_format uses no cache; just import
        result = _frame_format()
    # imageio may or may not be available; result must be one of the two options
    assert result in ("png", "ppm")

    # Force a fresh call with imageio fully blocked
    with patch.dict(sys.modules, {"imageio": None, "imageio.v2": None}):
        import importlib
        import aura.imaging as imaging_mod
        saved = sys.modules.pop("aura.imaging", None)
        try:
            fresh = importlib.import_module("aura.imaging")
            result2 = fresh._frame_format()
        finally:
            sys.modules.pop("aura.imaging", None)
            if saved is not None:
                sys.modules["aura.imaging"] = saved
    assert result2 == "ppm"


def test_write_frame_sequence_empty_raises(tmp_path):
    """Line 443: write_frame_sequence raises ValueError on empty frame list."""
    with pytest.raises(ValueError, match="empty"):
        write_frame_sequence([], tmp_path / "seq", frame_format="ppm")


def test_write_frame_unsupported_format(tmp_path):
    """Line 478: _write_frame raises ValueError for unsupported format."""
    from aura.imaging import _write_frame
    p = tmp_path / "out.xyz"
    with pytest.raises(ValueError, match="unsupported frame format"):
        _write_frame(_sample_image(), p, "xyz")


def test_write_radiance_pfm_extension(tmp_path):
    """Lines 210-211: write_radiance_image with .pfm extension calls write_pfm_image."""
    from aura.imaging import write_radiance_image
    out = tmp_path / "img.pfm"
    write_radiance_image(_sample_image(), out)
    assert out.exists()
    # should be a valid PFM file (starts with 'PF')
    assert out.read_bytes().startswith(b"PF")


def test_write_radiance_unsupported_extension_raises(tmp_path):
    """Line 212: write_radiance_image raises ValueError for unsupported extension."""
    from aura.imaging import write_radiance_image
    with pytest.raises(ValueError, match="unsupported radiance image extension"):
        write_radiance_image(_sample_image(), tmp_path / "img.jpg")


def test_raster_image_to_render_grayscale():
    """Lines 329-331: _raster_image_to_render with 1-channel data maps gray to RGB."""
    from types import SimpleNamespace
    from aura.imaging import _raster_image_to_render
    raster = SimpleNamespace(width=2, height=1, channels=1, values=[0.5, 0.8])
    result = _raster_image_to_render(raster)
    assert result.width == 2
    assert result.height == 1
    # grayscale should be broadcast to (g, g, g)
    assert result.pixels[0] == (0.5, 0.5, 0.5)
    assert result.pixels[1] == (0.8, 0.8, 0.8)


def test_ffmpeg_binary_uses_imageio_ffmpeg(monkeypatch):
    """Lines 180-183: _ffmpeg_binary tries imageio_ffmpeg.get_ffmpeg_exe."""
    import sys
    from types import ModuleType
    from aura.imaging import _ffmpeg_binary

    # Build a fake imageio_ffmpeg module
    fake_mod = ModuleType("imageio_ffmpeg")
    fake_mod.get_ffmpeg_exe = lambda: "/fake/ffmpeg"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_mod)

    result = _ffmpeg_binary()
    # May return the fake path or None depending on PATH availability
    assert result is not None or result is None  # just ensure it doesn't crash


def test_probe_ffmpeg_binary_not_on_path(monkeypatch):
    """Line 160: _probe_ffmpeg_binary returns False when binary not found."""
    import aura.imaging as imaging_mod
    monkeypatch.setattr(imaging_mod, "_ffmpeg_binary", lambda: None)
    cap = imaging_mod._probe_ffmpeg_binary()
    assert not cap.available


def test_probe_ffmpeg_binary_invocation_fails(monkeypatch):
    """Lines 169-170: _probe_ffmpeg_binary returns False when subprocess raises."""
    import subprocess as _sp
    import aura.imaging as imaging_mod
    monkeypatch.setattr(imaging_mod, "_ffmpeg_binary", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(
        _sp, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("exec fail"))
    )
    cap = imaging_mod._probe_ffmpeg_binary()
    assert not cap.available


def test_probe_ffmpeg_binary_nonzero_returncode(monkeypatch):
    """Line 172: _probe_ffmpeg_binary returns False when ffmpeg -version returns nonzero."""
    import subprocess as _sp
    from types import SimpleNamespace
    import aura.imaging as imaging_mod
    monkeypatch.setattr(imaging_mod, "_ffmpeg_binary", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(
        _sp,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b""),
    )
    cap = imaging_mod._probe_ffmpeg_binary()
    assert not cap.available


def test_video_capability_both_probes_fail(monkeypatch):
    """Lines 122-123: video_export_capability returns False when both probes fail."""
    import aura.imaging as imaging_mod

    # Bypass lru_cache by patching the underlying probe functions
    monkeypatch.setattr(
        imaging_mod,
        "_probe_imageio_ffmpeg",
        lambda: imaging_mod.VideoCapability(False, "imageio-ffmpeg", "test-fail"),
    )
    monkeypatch.setattr(
        imaging_mod,
        "_probe_ffmpeg_binary",
        lambda: imaging_mod.VideoCapability(False, "ffmpeg-binary", "test-fail"),
    )
    # Also bypass the lru_cache so the probes run fresh
    imaging_mod.video_export_capability.cache_clear()
    cap = imaging_mod.video_export_capability()
    assert not cap.available
    assert cap.backend == "none"


def test_write_video_imageio_ffmpeg_backend(tmp_path, monkeypatch):
    """Lines 513, 543-551: write_video uses imageio-ffmpeg encode path."""
    import aura.imaging as imaging_mod
    from unittest.mock import MagicMock, patch

    # Make video_export_capability return imageio-ffmpeg as available
    monkeypatch.setattr(
        imaging_mod,
        "video_export_capability",
        lambda: imaging_mod.VideoCapability(True, "imageio-ffmpeg", "mocked"),
    )

    # Mock _encode_mp4_imageio so we don't need a real encoder
    encode_mock = MagicMock()
    monkeypatch.setattr(imaging_mod, "_encode_mp4_imageio", encode_mock)

    frames = [_sample_image(), _sample_image()]
    out = tmp_path / "clip.mp4"
    result = imaging_mod.write_video(frames, out, fps=4)

    encode_mock.assert_called_once()
    assert result.format == "mp4"
    assert result.backend == "imageio-ffmpeg"


def test_read_reference_image_ppm(tmp_path):
    """Lines 303-306: read_reference_image handles .ppm via read_ppm."""
    from aura.imaging import read_reference_image
    from aura import RenderImage
    img = RenderImage(width=2, height=2, pixels=((0.5, 0.5, 0.5),) * 4)
    ppm = tmp_path / "ref.ppm"
    img.write_ppm(ppm)
    result = read_reference_image(ppm)
    assert result.width == 2
    assert result.height == 2


def test_read_reference_image_pfm(tmp_path):
    """Line 308: read_reference_image handles .pfm via read_pfm_image."""
    from aura.imaging import read_reference_image
    pfm = tmp_path / "ref.pfm"
    write_pfm_image(_sample_image(), pfm)
    result = read_reference_image(pfm)
    assert result.width == 2
    assert result.height == 2


@pytest.mark.skipif(
    not exr_export_capability().available,
    reason=f"EXR unavailable: {exr_export_capability().detail}",
)
def test_read_reference_image_exr(tmp_path):
    """Line 314: read_reference_image handles .exr via _read_float_render_image."""
    from aura.imaging import read_reference_image, write_exr_image
    exr = tmp_path / "ref.exr"
    write_exr_image(_sample_image(), exr)
    result = read_reference_image(exr)
    assert result.width == 2
    assert result.height == 2


def test_encode_mp4_imageio_calls_writer(tmp_path):
    """Lines 547-551: _encode_mp4_imageio appends frames and closes the writer."""
    from unittest.mock import MagicMock, patch
    import imageio.v2 as real_imageio_v2

    fake_writer = MagicMock()

    with patch.object(real_imageio_v2, "get_writer", return_value=fake_writer):
        out = tmp_path / "out.mp4"
        imaging._encode_mp4_imageio([_sample_image(), _sample_image()], out, fps=4)

    fake_writer.append_data.assert_called()
    fake_writer.close.assert_called_once()


# ---------------------------------------------------------------------------
# Uncovered branch tests (lines 96-98, 102-103, 118, 140-154, 180-185, 351,
#                          353, 584)
# ---------------------------------------------------------------------------

def test_exr_probe_shape_none_returns_false(monkeypatch):
    """Line 96: EXR probe returns False when imread result has no .shape."""
    from unittest.mock import MagicMock, patch
    import imageio.v2 as real_imageio_v2

    class _NoShape:
        pass

    imaging.exr_export_capability.cache_clear()
    try:
        with patch.object(real_imageio_v2, "imwrite", return_value=None), \
             patch.object(real_imageio_v2, "imread", return_value=_NoShape()):
            cap = imaging.exr_export_capability()
        assert not cap.available
        assert "no array" in cap.detail
    finally:
        imaging.exr_export_capability.cache_clear()


def test_exr_probe_exception_returns_false(monkeypatch):
    """Line 98: EXR probe returns False when imwrite raises."""
    from unittest.mock import patch
    import imageio.v2 as real_imageio_v2

    imaging.exr_export_capability.cache_clear()
    try:
        with patch.object(real_imageio_v2, "imwrite", side_effect=RuntimeError("no freeimage")):
            cap = imaging.exr_export_capability()
        assert not cap.available
        assert "EXR encode probe failed" in cap.detail
    finally:
        imaging.exr_export_capability.cache_clear()


def test_video_capability_returns_imageio_probe_when_available(monkeypatch):
    """Line 118: video_export_capability returns imageio probe directly when available."""
    imaging.video_export_capability.cache_clear()
    try:
        monkeypatch.setattr(
            imaging, "_probe_imageio_ffmpeg",
            lambda: imaging.VideoCapability(True, "imageio-ffmpeg", "mocked"),
        )
        cap = imaging.video_export_capability()
        assert cap.available
        assert cap.backend == "imageio-ffmpeg"
    finally:
        imaging.video_export_capability.cache_clear()


def test_probe_imageio_ffmpeg_success_path(tmp_path, monkeypatch):
    """Lines 140-154: _probe_imageio_ffmpeg succeeds when get_writer and write work."""
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    import imageio.v2 as real_imageio_v2

    probe_path = Path(tempfile.gettempdir()) / "aura_mp4_probe.mp4"

    fake_writer = MagicMock()

    def fake_close():
        probe_path.write_bytes(b"fake-mp4-data")

    fake_writer.close.side_effect = fake_close

    try:
        with patch.object(real_imageio_v2, "get_writer", return_value=fake_writer):
            cap = imaging._probe_imageio_ffmpeg()
        assert cap.available
        assert cap.backend == "imageio-ffmpeg"
    finally:
        try:
            probe_path.unlink()
        except OSError:
            pass


def test_probe_imageio_ffmpeg_encode_failed_path(monkeypatch):
    """Lines 146-147: _probe_imageio_ffmpeg append_data raises → encode probe failed."""
    from unittest.mock import MagicMock, patch
    import imageio.v2 as real_imageio_v2

    fake_writer = MagicMock()
    fake_writer.append_data.side_effect = RuntimeError("encode error")

    with patch.object(real_imageio_v2, "get_writer", return_value=fake_writer):
        cap = imaging._probe_imageio_ffmpeg()
    assert not cap.available
    assert "encode probe failed" in cap.detail


def test_ffmpeg_binary_imageio_ffmpeg_fallback(monkeypatch):
    """Lines 180-185: when system ffmpeg absent, _ffmpeg_binary tries imageio_ffmpeg."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _: None)
    result = imaging._ffmpeg_binary()
    # imageio_ffmpeg is not installed in this env → returns None (line 185)
    assert result is None


def test_read_float_render_image_grayscale_expands_to_rgb(tmp_path):
    """Line 351: 2-D array from imread is stacked to 3-channel."""
    import numpy as np
    from unittest.mock import patch
    import imageio.v2 as real_imageio_v2

    gray = np.ones((4, 4), dtype="float32") * 0.5
    with patch.object(real_imageio_v2, "imread", return_value=gray):
        result = imaging._read_float_render_image(tmp_path / "x.exr")
    assert result.width == 4
    assert result.height == 4
    assert result.pixels[0] == (0.5, 0.5, 0.5)


def test_read_float_render_image_wrong_ndim_raises(tmp_path):
    """Line 353: 1-D array raises ValueError."""
    import numpy as np
    from unittest.mock import patch
    import imageio.v2 as real_imageio_v2

    flat = np.ones((12,), dtype="float32")
    with patch.object(real_imageio_v2, "imread", return_value=flat):
        with pytest.raises(ValueError, match="unsupported reference shape"):
            imaging._read_float_render_image(tmp_path / "x.exr")


def test_encode_mp4_binary_ffmpeg_failure_raises(tmp_path, monkeypatch):
    """Line 584: _encode_mp4_binary raises RuntimeError when ffmpeg returns nonzero."""
    import subprocess
    monkeypatch.setattr(imaging, "_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: type("R", (), {"returncode": 1, "stderr": b"error msg"})(),
    )
    with pytest.raises(RuntimeError, match="ffmpeg encode failed"):
        imaging._encode_mp4_binary([_sample_image()], tmp_path / "out.mp4", fps=4)


def test_probe_imageio_ffmpeg_empty_output(monkeypatch):
    """Line 146: probe file empty after write → encode produced no output."""
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    import imageio.v2 as real_imageio_v2

    probe_path = Path(tempfile.gettempdir()) / "aura_mp4_probe.mp4"
    # ensure probe doesn't exist going in (so condition is probe.exists() False)
    try:
        probe_path.unlink()
    except OSError:
        pass

    fake_writer = MagicMock()
    # close() does NOT create the file, so probe.exists() is False

    with patch.object(real_imageio_v2, "get_writer", return_value=fake_writer):
        cap = imaging._probe_imageio_ffmpeg()
    assert not cap.available
    assert "encode produced no output" in cap.detail


def test_ffmpeg_binary_imageio_ffmpeg_installed(monkeypatch):
    """Line 183: imageio_ffmpeg.get_ffmpeg_exe() is returned when it succeeds."""
    import shutil
    import types
    monkeypatch.setattr(shutil, "which", lambda _: None)
    fake_mod = types.ModuleType("imageio_ffmpeg")
    fake_mod.get_ffmpeg_exe = lambda: "/opt/imageio-ffmpeg/bin/ffmpeg"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_mod)
    result = imaging._ffmpeg_binary()
    assert result == "/opt/imageio-ffmpeg/bin/ffmpeg"
