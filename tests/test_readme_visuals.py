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
    # The train image-sweep / sparse-depth GIFs are training-evidence media and
    # must live under the "Training backends" section, never as top-level hero art.
    training = readme.split("## Training backends", 1)
    assert len(training) == 2, "README must have a '## Training backends' section"
    assert "docs/train_orbit.gif" in training[1]
    assert "docs/train_depth_orbit.gif" in training[1]
    assert "docs/train_orbit.gif" not in training[0]
    assert "docs/train_depth_orbit.gif" not in training[0]
    assert "Tanks and Temples" not in readme
    assert "Temple scene" not in readme
    assert "docs/temple" not in readme.lower()
    assert "M60" not in readme
    assert "tank scene" not in readme.lower()


def test_readme_embeds_p0_p2_result_figures():
    """The four hardening result figures generated from committed data must be
    embedded, and the pruning-sweep animation must be featured."""
    readme = (ROOT / "README.md").read_text()
    for asset in (
        "assets/reliability_diagram.png",
        "assets/selection_curves.png",
        "assets/transfer_ece_heatmap.png",
        "assets/proxy_vs_renderloss.png",
        "assets/pruning_sweep.gif",
    ):
        assert asset in readme, asset
        assert (ROOT / asset).exists(), asset


def test_readme_states_the_p0_eval_leak_correction():
    """The honest P0 -> P2 leak correction (Truck certificate 1.00 -> 0.77 on the
    clean split) is load-bearing and must be stated plainly, not hidden."""
    readme = (ROOT / "README.md").read_text()
    assert "1.00 → 0.77" in readme or "1.00 -> 0.77" in readme
    assert "leak" in readme.lower()


def test_readme_sota_claim_boundary_matches_current_artifact():
    readme = (ROOT / "README.md").read_text()

    # The deliberately-removed "sotaReady" marker must never come back.
    assert "sotaReady" not in readme
    assert "full SOTA claim still gated" not in readme
    assert "short same-split GPU validation rows" not in readme
    # The honest boundary the repo actually ships.
    assert "no official-leaderboard SOTA claim" in readme
