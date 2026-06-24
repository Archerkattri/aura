import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

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
