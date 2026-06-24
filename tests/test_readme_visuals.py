import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from make_readme_visuals import fit_media_frame


def test_fit_media_frame_cover_scales_small_gif_frame():
    small = Image.new("RGB", (245, 137), "red")

    fitted = fit_media_frame(small, (832, 450), mode="cover")

    assert fitted.size == (832, 450)
    assert fitted.getbbox() == (0, 0, 832, 450)


def test_fit_media_frame_contain_preserves_static_chart_shape():
    chart = Image.new("RGB", (1248, 468), "blue")

    fitted = fit_media_frame(chart, (832, 450), mode="contain")

    assert fitted.size == (832, 450)
    assert fitted.getbbox() is not None


def test_primary_readme_gifs_keep_source_width():
    for path in (
        ROOT / "docs" / "truck_orbit.gif",
        ROOT / "docs" / "truck_depth_orbit.gif",
        ROOT / "docs" / "relight_sweep.gif",
        ROOT / "docs" / "train_orbit.gif",
        ROOT / "docs" / "train_depth_orbit.gif",
    ):
        img = Image.open(path)
        assert img.width >= 900, path
        assert img.height >= 500, path


def _gif_total_duration_ms(img: Image.Image) -> int:
    return int(img.info.get("duration", 0)) * int(getattr(img, "n_frames", 1))


def test_orbit_readme_gifs_are_smooth_without_low_fps_slowdown():
    for path in (
        ROOT / "docs" / "truck_orbit.gif",
        ROOT / "docs" / "truck_depth_orbit.gif",
        ROOT / "docs" / "train_orbit.gif",
        ROOT / "docs" / "train_depth_orbit.gif",
    ):
        img = Image.open(path)
        assert getattr(img, "n_frames", 1) >= 240, path
        assert img.info.get("duration", 0) <= 80, path
        assert _gif_total_duration_ms(img) >= 15000, path


def test_relight_readme_gif_play_at_inspectable_speed():
    img = Image.open(ROOT / "docs" / "relight_sweep.gif")
    assert img.info.get("duration", 0) >= 100


def test_readme_includes_local_truck_and_train_media_only():
    readme = (ROOT / "README.md").read_text()
    assert "docs/truck_orbit.gif" in readme
    assert "docs/train_orbit.gif" in readme
    assert "docs/train_depth_orbit.gif" in readme
    assert "docs/capability_board.png" not in readme
    gallery = readme.split("## Gallery", 1)[1]
    assert "docs/train_orbit.gif" not in gallery
    assert "docs/train_depth_orbit.gif" not in gallery
    assert "Tanks and Temples" not in readme
    assert "Temple scene" not in readme
    assert "docs/temple" not in readme.lower()
    assert "M60" not in readme
    assert "tank scene" not in readme.lower()
