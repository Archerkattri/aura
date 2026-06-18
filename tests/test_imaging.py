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
