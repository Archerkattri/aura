import json
import struct
import zlib
from array import array
from pathlib import Path

import pytest

from aura import CaptureTensor
from aura.ingest.capture import (
    CaptureFrameAssets,
    CaptureFrameTensors,
    CaptureManifest,
    PackedFloatBuffer,
    _average_normal,
    _average_rgb,
    _average_scalar,
    _depth_bins,
    _depth_summary,
    _frame_with_asset_summaries,
    _normalized_channel_buffer,
    _paeth_predictor,
    _png_unfilter,
    _read_capture_raster,
    _read_capture_tensor,
    _read_colmap_dense_map,
    _read_netpbm,
    _read_png,
    _RasterImage,
    _resolve_capture_path,
    _skip_netpbm_space_and_comments,
    _validate_capture_tensor_frame_set,
    _validate_manifest_links,
    _vec3,
    load_capture_asset_tensors,
    load_capture_manifest,
    write_capture_manifest,
)
from aura.core import TrainingDataset, TrainingFrame, TrainingRegion
from aura.elements import Bounds
from aura.assignment import RegionEvidence


def test_capture_tensor_preserves_packed_buffer_and_exposes_direct_tile_samples():
    values = PackedFloatBuffer(array("d", (0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2)))
    tensor = CaptureTensor("image.ppm", "Netpbm", "stdlib", 2, 2, 3, values)

    assert tensor.values is values
    assert tensor.value_offset(1, 1, 2) == 11
    assert tensor.value_at(1, 0, 1) == 1.1
    assert tensor.pixel(0, 1, channels=3) == (2.0, 2.1, 2.2)
    assert list(tensor.iter_tile_samples((0, 0), (2, 2), pixel_stride=1)) == [
        (0, 0, 0),
        (1, 0, 3),
        (0, 1, 6),
        (1, 1, 9),
    ]
    assert isinstance(tensor.values[:3], PackedFloatBuffer)
    assert tensor.values[:3] == (0.0, 0.1, 0.2)


def test_capture_tensor_tile_access_rejects_invalid_windows():
    tensor = CaptureTensor("mask.pgm", "Netpbm", "stdlib", 2, 1, 1, (1.0, 0.0))

    with pytest.raises(ValueError, match="outside tensor bounds"):
        list(tensor.iter_tile_samples((1, 0), (2, 1)))
    with pytest.raises(ValueError, match="pixel_stride must be positive"):
        list(tensor.iter_tile_samples((0, 0), (1, 1), pixel_stride=0))


def test_read_capture_tensor_loads_jpeg(tmp_path):
    """Real datasets (Tanks and Temples, Mip-NeRF 360) ship 8-bit JPEG frames."""
    import importlib.util
    if importlib.util.find_spec("imageio") is None:
        import pytest
        pytest.skip("imageio not installed")
    import numpy as np
    import imageio.v3 as iio
    from aura.ingest.capture import _read_capture_tensor

    # A solid-red 8x8 RGB image: JPEG preserves uniform color well (unlike a
    # single sharp pixel), so we can assert the channel content survives.
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :] = (255, 0, 0)
    path = tmp_path / "frame.jpg"
    iio.imwrite(path, pixels)

    tensor = _read_capture_tensor(path)
    assert tensor.width == 8
    assert tensor.height == 8
    assert tensor.channels == 3
    assert tensor.backend == "imageio"
    # Values are normalized to [0, 1]
    assert all(0.0 <= v <= 1.0 for v in tensor.values)
    # Red dominates green/blue at the first pixel (R, G, B order).
    assert tensor.values[0] > 0.5
    assert tensor.values[0] > tensor.values[1]
    assert tensor.values[0] > tensor.values[2]


# ---------------------------------------------------------------------------
# PackedFloatBuffer — equality, .array, .memoryview
# ---------------------------------------------------------------------------

def test_packed_float_buffer_eq_two_packed_buffers():
    a = PackedFloatBuffer([1.0, 2.0, 3.0])
    b = PackedFloatBuffer([1.0, 2.0, 3.0])
    c = PackedFloatBuffer([1.0, 2.0, 9.0])
    assert a == b          # line 47: True branch
    assert a != c


def test_packed_float_buffer_eq_non_sequence_returns_false():
    buf = PackedFloatBuffer([1.0, 2.0])
    assert buf.__eq__(42) is False   # line 50: non-sequence → False
    assert buf.__eq__(object()) is False


def test_packed_float_buffer_array_property():
    buf = PackedFloatBuffer([0.5, 1.5])
    arr = buf.array                  # line 60
    assert isinstance(arr, array)
    assert list(arr) == [0.5, 1.5]


def test_packed_float_buffer_memoryview():
    buf = PackedFloatBuffer([0.25, 0.75])
    mv = buf.memoryview()            # line 63
    assert isinstance(mv, memoryview)
    assert len(mv) == 2


# ---------------------------------------------------------------------------
# CaptureManifest.to_dict() and from_dict(validate=False)
# ---------------------------------------------------------------------------

def _minimal_manifest_dict(frame_id="f1", region_id="r1"):
    return {
        "format": "AURA_CAPTURE_MANIFEST",
        "root": ".",
        "frames": [
            {
                "id": frame_id,
                "image_path": "images/frame.png",
                "camera_origin": [0.0, 0.0, -1.0],
                "look_at": [0.0, 0.0, 0.0],
                "target_color": [0.5, 0.5, 0.5],
                "target_depth": 1.0,
                "semantic_label": None,
                "camera_model": "PINHOLE",
                "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0, "width": 100.0, "height": 100.0},
            }
        ],
        "regions": [
            {
                "id": region_id,
                "frame_id": frame_id,
                "bounds": {"min": [-0.1, -0.1, 0.9], "max": [0.1, 0.1, 1.1]},
                "evidence": {"geometry_confidence": 0.8},
                "color": [0.5, 0.5, 0.5],
                "opacity": 0.9,
                "confidence": 0.8,
                "normal": [0.0, 0.0, -1.0],
                "fallback_source": "capture-manifest",
            }
        ],
    }


def test_capture_manifest_to_dict():
    payload = _minimal_manifest_dict()
    manifest = CaptureManifest.from_dict(payload)
    d = manifest.to_dict()           # line 83
    assert d["format"] == "AURA_CAPTURE_MANIFEST"
    assert d["root"] == "."
    assert len(d["frames"]) == 1
    assert len(d["regions"]) == 1


def test_capture_manifest_from_dict_validate_false():
    payload = _minimal_manifest_dict()
    # validate=False skips jsonschema (line 93)
    manifest = CaptureManifest.from_dict(payload, validate=False)
    assert manifest.root == "."
    assert len(manifest.frames) == 1


# ---------------------------------------------------------------------------
# CaptureFrameAssets.to_dict()
# ---------------------------------------------------------------------------

def test_capture_frame_assets_to_dict():
    assets = CaptureFrameAssets(
        frame_id="f1",
        image_path="images/frame.png",
        width=4,
        height=4,
        average_color=(0.5, 0.5, 0.5),
        depth_path="depth/frame.bin",
        average_depth=2.0,
        min_depth=1.0,
        max_depth=3.0,
        depth_coverage=0.9,
        depth_bins=({"id": 0.0, "average": 2.0, "minimum": 1.0, "maximum": 3.0, "coverage": 0.9},),
        average_normal=(0.0, 0.0, -1.0),
    )
    d = assets.to_dict()             # line 125
    assert d["frameId"] == "f1"
    assert d["averageDepth"] == 2.0
    assert d["averageNormal"] == [0.0, 0.0, -1.0]


# ---------------------------------------------------------------------------
# CaptureTensor.__post_init__ validation errors
# ---------------------------------------------------------------------------

def test_capture_tensor_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):  # line 160
        CaptureTensor("t.png", "PNG", "stdlib", 0, 1, 1, [])
    with pytest.raises(ValueError, match="dimensions must be positive"):
        CaptureTensor("t.png", "PNG", "stdlib", 1, -1, 1, [])


def test_capture_tensor_rejects_non_positive_channels():
    with pytest.raises(ValueError, match="channels must be positive"):   # line 162
        CaptureTensor("t.png", "PNG", "stdlib", 1, 1, 0, [])


def test_capture_tensor_rejects_payload_mismatch():
    with pytest.raises(ValueError, match="payload does not match dimensions"):  # line 164
        CaptureTensor("t.png", "PNG", "stdlib", 2, 2, 3, [0.0] * 5)


# ---------------------------------------------------------------------------
# CaptureTensor.value_offset out-of-bounds
# ---------------------------------------------------------------------------

def test_capture_tensor_value_offset_out_of_bounds():
    t = CaptureTensor("t.png", "PNG", "stdlib", 2, 2, 1, [0.0, 0.0, 0.0, 0.0])
    with pytest.raises(IndexError, match="outside tensor bounds"):  # line 183
        t.value_offset(5, 0)
    with pytest.raises(IndexError, match="outside tensor bounds"):
        t.value_offset(0, 5)
    with pytest.raises(IndexError, match="channel.*outside"):        # line 185
        t.value_offset(0, 0, 3)


# ---------------------------------------------------------------------------
# CaptureTensor.iter_tile_samples size validation
# ---------------------------------------------------------------------------

def test_capture_tensor_iter_tile_samples_rejects_zero_size():
    t = CaptureTensor("t.png", "PNG", "stdlib", 2, 2, 1, [0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="tile size must be positive"):  # line 208
        list(t.iter_tile_samples((0, 0), (0, 1)))
    with pytest.raises(ValueError, match="tile size must be positive"):
        list(t.iter_tile_samples((0, 0), (1, 0)))


# ---------------------------------------------------------------------------
# load_capture_manifest error paths
# ---------------------------------------------------------------------------

def test_load_capture_manifest_rejects_non_dict_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):  # line 275
        load_capture_manifest(bad)


def test_load_capture_manifest_adds_root_from_parent(tmp_path):
    payload = _minimal_manifest_dict()
    del payload["root"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_capture_manifest(manifest_path, validate=False)  # line 280
    assert manifest.root == str(tmp_path)


# ---------------------------------------------------------------------------
# write_capture_manifest
# ---------------------------------------------------------------------------

def test_write_capture_manifest(tmp_path):
    payload = _minimal_manifest_dict()
    manifest = CaptureManifest.from_dict(payload, validate=False)
    out = tmp_path / "out.json"
    result = write_capture_manifest(manifest, out)   # lines 446-449
    assert result == out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["format"] == "AURA_CAPTURE_MANIFEST"


# ---------------------------------------------------------------------------
# _validate_capture_tensor_frame_set error paths
# ---------------------------------------------------------------------------

def _make_frame(frame_id):
    return TrainingFrame(
        id=frame_id,
        camera_origin=(0.0, 0.0, -1.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.5, 0.5, 0.5),
        target_depth=1.0,
        image_path="images/frame.png",
    )


def _make_manifest_with_frame(frame_id="f1"):
    frame = _make_frame(frame_id)
    return CaptureManifest(root=".", frames=(frame,), regions=())


def _make_tensor_frame(frame_id):
    values = PackedFloatBuffer([0.5] * 3)
    image = CaptureTensor("img.png", "PNG", "stdlib", 1, 1, 3, values)
    return CaptureFrameTensors(frame_id=frame_id, image=image)


def test_validate_capture_tensor_frame_set_raises_on_duplicates():
    manifest = _make_manifest_with_frame("f1")
    tf1 = _make_tensor_frame("f1")
    tf2 = _make_tensor_frame("f1")
    with pytest.raises(ValueError, match="duplicate frame ids"):  # line 348
        _validate_capture_tensor_frame_set(manifest, [tf1, tf2])


def test_validate_capture_tensor_frame_set_raises_on_unknown():
    manifest = _make_manifest_with_frame("f1")
    tf_unknown = _make_tensor_frame("unknown_frame")
    # batch has a frame not in manifest → "missing" + "unknown" errors; first is missing
    with pytest.raises(ValueError, match="missing manifest frame ids|unknown manifest frame ids"):  # lines 351, 354
        _validate_capture_tensor_frame_set(manifest, [tf_unknown])


# ---------------------------------------------------------------------------
# load_capture_asset_tensors validation errors
# ---------------------------------------------------------------------------

def _write_png_file(path, *, width=2, height=1, channels=3, values=None):
    if values is None:
        values = [128] * (width * height * channels)
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    scanline_width = width * channels
    rows = []
    for row in range(height):
        start = row * scanline_width
        rows.append(b"\x00" + bytes(values[start: start + scanline_width]))
    raw = b"".join(rows)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk_helper(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    payload += _png_chunk_helper(b"IDAT", zlib.compress(raw))
    payload += _png_chunk_helper(b"IEND", b"")
    path.write_bytes(payload)


def _png_chunk_helper(kind, data):
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _manifest_with_assets(root, frame_id="f1", image_path=None, depth_path=None, mask_path=None, normal_path=None):
    frame = TrainingFrame(
        id=frame_id,
        camera_origin=(0.0, 0.0, -1.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.5, 0.5, 0.5),
        target_depth=1.0,
        image_path=image_path,
        depth_path=depth_path,
        mask_path=mask_path,
        normal_path=normal_path,
    )
    return CaptureManifest(root=str(root), frames=(frame,), regions=())


def test_load_capture_asset_tensors_raises_if_image_path_is_none(tmp_path):
    manifest = _manifest_with_assets(tmp_path, image_path=None)
    with pytest.raises(ValueError, match="missing image_path"):  # line 380
        load_capture_asset_tensors(manifest)


def test_load_capture_asset_tensors_raises_if_image_has_fewer_than_3_channels(tmp_path):
    img = tmp_path / "gray.png"
    _write_png_file(img, width=2, height=1, channels=1, values=[128, 200])
    manifest = _manifest_with_assets(tmp_path, image_path="gray.png")
    with pytest.raises(ValueError, match="RGB/RGBA"):  # line 393
        load_capture_asset_tensors(manifest)


def test_load_capture_asset_tensors_raises_if_depth_not_single_channel(tmp_path):
    img = tmp_path / "frame.png"
    _write_png_file(img, width=2, height=1, channels=3)
    # Write a depth as a 3-channel PNG (wrong)
    depth = tmp_path / "depth.png"
    _write_png_file(depth, width=2, height=1, channels=3)
    manifest = _manifest_with_assets(tmp_path, image_path="frame.png", depth_path="depth.png")
    with pytest.raises(ValueError, match="single-channel"):  # line 406
        load_capture_asset_tensors(manifest)


def test_load_capture_asset_tensors_raises_if_mask_not_single_channel(tmp_path):
    img = tmp_path / "frame.png"
    _write_png_file(img, width=2, height=1, channels=3)
    mask = tmp_path / "mask.png"
    _write_png_file(mask, width=2, height=1, channels=3)
    manifest = _manifest_with_assets(tmp_path, image_path="frame.png", mask_path="mask.png")
    with pytest.raises(ValueError, match="single-channel"):  # line 419
        load_capture_asset_tensors(manifest)


def test_load_capture_asset_tensors_raises_if_normal_not_3_channels(tmp_path):
    img = tmp_path / "frame.png"
    _write_png_file(img, width=2, height=1, channels=3)
    normal = tmp_path / "normal.png"
    _write_png_file(normal, width=2, height=1, channels=1, values=[128, 200])
    manifest = _manifest_with_assets(tmp_path, image_path="frame.png", normal_path="normal.png")
    with pytest.raises(ValueError, match="3-channel normal map"):  # line 432
        load_capture_asset_tensors(manifest)


# ---------------------------------------------------------------------------
# _validate_manifest_links error paths
# ---------------------------------------------------------------------------

def test_validate_manifest_links_raises_on_duplicate_frame_ids():
    f1 = _make_frame("f1")
    f2 = _make_frame("f1")  # same id
    dataset = TrainingDataset(frames=(f1, f2), regions=())
    with pytest.raises(ValueError, match="duplicate frame ids"):  # line 542
        _validate_manifest_links(dataset)


def test_validate_manifest_links_raises_on_duplicate_region_ids():
    f1 = _make_frame("f1")
    r = TrainingRegion(
        id="r1", frame_id="f1",
        bounds=Bounds(min_corner=(-0.1, -0.1, 0.9), max_corner=(0.1, 0.1, 1.1)),
        evidence=RegionEvidence(),
    )
    dataset = TrainingDataset(frames=(f1,), regions=(r, r))
    with pytest.raises(ValueError, match="duplicate region ids"):  # line 545
        _validate_manifest_links(dataset)


# ---------------------------------------------------------------------------
# _vec3 validation
# ---------------------------------------------------------------------------

def test_vec3_raises_on_non_3_vector():
    with pytest.raises(ValueError, match="must be a 3-vector"):  # line 553
        _vec3([1.0, 2.0], "test")
    with pytest.raises(ValueError, match="must be a 3-vector"):
        _vec3("xyz", "test")


# ---------------------------------------------------------------------------
# _RasterImage validation
# ---------------------------------------------------------------------------

def test_raster_image_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):  # line 569
        _RasterImage("PNG", 0, 1, 1, [])


def test_raster_image_rejects_payload_mismatch():
    with pytest.raises(ValueError, match="payload does not match dimensions"):  # line 571
        _RasterImage("PNG", 2, 2, 3, [0.5] * 5)


# ---------------------------------------------------------------------------
# _frame_with_asset_summaries — assets is None
# ---------------------------------------------------------------------------

def test_frame_with_asset_summaries_returns_frame_unchanged_when_assets_none():
    frame = _make_frame("f1")
    result = _frame_with_asset_summaries(frame, None)   # line 585
    assert result is frame


# ---------------------------------------------------------------------------
# _resolve_capture_path — absolute path
# ---------------------------------------------------------------------------

def test_resolve_capture_path_returns_absolute_path_as_is():
    root = Path("/some/root")
    abs_path = Path("/absolute/path/image.png")
    result = _resolve_capture_path(root, abs_path)       # lines 716-717
    assert result == abs_path


# ---------------------------------------------------------------------------
# _average_rgb validation
# ---------------------------------------------------------------------------

def test_average_rgb_raises_for_fewer_than_3_channels():
    img = _RasterImage("PNG", 2, 1, 1, PackedFloatBuffer([0.5, 0.6]))
    with pytest.raises(ValueError, match="at least a 3-channel"):  # line 752
        _average_rgb(img)


# ---------------------------------------------------------------------------
# _average_scalar — image is None
# ---------------------------------------------------------------------------

def test_average_scalar_returns_none_for_none_image():
    assert _average_scalar(None) is None   # line 763


def test_average_scalar_raises_for_non_1_channel():
    img = _RasterImage("PNG", 2, 1, 3, PackedFloatBuffer([0.5] * 6))
    with pytest.raises(ValueError, match="1-channel"):  # line 765
        _average_scalar(img)


# ---------------------------------------------------------------------------
# _average_normal — None and non-3-channel
# ---------------------------------------------------------------------------

def test_average_normal_returns_none_for_none_image():
    assert _average_normal(None) is None   # line 773


def test_average_normal_raises_for_non_3_channel():
    img = _RasterImage("PNG", 2, 1, 1, PackedFloatBuffer([0.5, 0.5]))
    with pytest.raises(ValueError, match="3-channel"):  # line 782
        _average_normal(img)


def test_average_normal_returns_none_for_zero_vector():
    # All-zero normal values → norm is zero → returns None (line 790)
    img = _RasterImage("PNG", 1, 1, 3, PackedFloatBuffer([0.0, 0.0, 0.0]))
    assert _average_normal(img) is None


# ---------------------------------------------------------------------------
# _depth_summary — None and no positive samples
# ---------------------------------------------------------------------------

def test_depth_summary_returns_none_for_none_image():
    assert _depth_summary(None) is None   # line 797


def test_depth_summary_raises_when_no_positive_samples():
    img = _RasterImage("COLMAP_DENSE_MAP", 2, 1, 1, PackedFloatBuffer([0.0, 0.0]))
    with pytest.raises(ValueError, match="no positive samples"):  # line 803
        _depth_summary(img)


# ---------------------------------------------------------------------------
# _depth_bins — single-bin edge cases
# ---------------------------------------------------------------------------

def test_depth_bins_single_bin_when_fewer_than_2_values():
    result = _depth_bins(
        PackedFloatBuffer([2.0]),
        total_count=1, valid_count=1, minimum=2.0, maximum=2.0,
    )
    assert len(result) == 1   # lines 822-825


def test_depth_bins_single_bin_when_range_too_small():
    result = _depth_bins(
        PackedFloatBuffer([1.0, 1.0001]),
        total_count=2, valid_count=2, minimum=1.0, maximum=1.0001,
    )
    assert len(result) == 1


def test_depth_bins_skips_empty_bin():
    # All values on one side of midpoint → the empty half bin is skipped (line 843, 851)
    values = PackedFloatBuffer([1.0, 1.1, 1.2])
    result = _depth_bins(values, total_count=3, valid_count=3, minimum=1.0, maximum=3.0)
    # All values are below midpoint (2.0), so far bin is empty and skipped
    assert len(result) == 1
    assert result[0]["id"] == 0.0


# ---------------------------------------------------------------------------
# _read_capture_raster — extension errors
# ---------------------------------------------------------------------------

def test_read_capture_raster_raises_for_gpu_only_extension(tmp_path):
    p = tmp_path / "frame.exr"
    p.write_bytes(b"fake exr data")
    with pytest.raises(ValueError, match="GPU tensor asset backend"):  # lines 874-875
        _read_capture_raster(p)

    p2 = tmp_path / "video.mp4"
    p2.write_bytes(b"fake")
    with pytest.raises(ValueError, match="GPU tensor asset backend"):
        _read_capture_raster(p2)


def test_read_capture_raster_raises_for_truly_unsupported_extension(tmp_path):
    p = tmp_path / "frame.xyz"
    p.write_bytes(b"fake data")
    with pytest.raises(ValueError, match="unsupported capture asset extension"):  # lines 877-878
        _read_capture_raster(p)


# ---------------------------------------------------------------------------
# _read_capture_tensor — unsupported extension
# ---------------------------------------------------------------------------

def test_read_capture_tensor_raises_for_unsupported_extension(tmp_path):
    p = tmp_path / "frame.xyz"
    p.write_bytes(b"fake data")
    with pytest.raises(ValueError, match="unsupported capture asset extension"):  # line 898
        _read_capture_tensor(p)


# ---------------------------------------------------------------------------
# _read_netpbm — FileNotFoundError and binary format
# ---------------------------------------------------------------------------

def test_read_netpbm_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):  # line 975
        _read_netpbm(tmp_path / "missing.ppm")


def test_read_netpbm_raises_for_unsupported_magic(tmp_path):
    p = tmp_path / "frame.ppm"
    p.write_bytes(b"P1 1 1 1\n1\n")
    with pytest.raises(ValueError, match="unsupported capture asset format"):  # line 986
        _read_netpbm(p)


def test_read_netpbm_reads_binary_p5_pgm(tmp_path):
    # P5: binary grayscale, lines 993-999
    p = tmp_path / "gray.pgm"
    width, height, max_val = 2, 1, 255
    raw = b"P5\n2 1\n255\n" + bytes([128, 64])
    p.write_bytes(raw)
    img = _read_netpbm(p)
    assert img.width == 2
    assert img.height == 1
    assert img.channels == 1
    assert abs(img.values[0] - 128 / 255.0) < 1e-9


def test_read_netpbm_reads_binary_p6_ppm(tmp_path):
    # P6: binary RGB, lines 993-999
    p = tmp_path / "color.ppm"
    p.write_bytes(b"P6\n1 1\n255\n" + bytes([255, 128, 0]))
    img = _read_netpbm(p)
    assert img.width == 1
    assert img.channels == 3
    assert abs(img.values[0] - 1.0) < 1e-9
    assert abs(img.values[2] - 0.0) < 1e-9


def test_read_netpbm_skips_comments(tmp_path):
    # Comment lines starting with # (lines 1167-1169)
    p = tmp_path / "gray.pgm"
    p.write_bytes(b"P2\n# this is a comment\n1 1\n255\n200\n")
    img = _read_netpbm(p)
    assert img.width == 1
    assert abs(img.values[0] - 200 / 255.0) < 1e-9


# ---------------------------------------------------------------------------
# _read_png — error cases
# ---------------------------------------------------------------------------

def test_read_png_raises_for_non_png_file(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a png file at all")
    with pytest.raises(ValueError, match="not a PNG file"):  # line 1013
        _read_png(p)


def test_read_png_raises_for_truncated_chunk(tmp_path):
    # Write valid signature + incomplete chunk data
    p = tmp_path / "trunc.png"
    sig = b"\x89PNG\r\n\x1a\n"
    # Only 3 bytes after signature → triggers offset+12 > len(data) check
    p.write_bytes(sig + b"\x00\x00")
    with pytest.raises(ValueError, match="truncated PNG chunk"):  # line 1019
        _read_png(p)


def test_read_png_raises_for_non_8bit_depth(tmp_path):
    p = tmp_path / "16bit.png"
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 2, 1, 16, 2, 0, 0, 0)  # bit_depth=16
    chunk = _png_chunk_helper(b"IHDR", ihdr_data)
    p.write_bytes(sig + chunk)
    with pytest.raises(ValueError, match="8-bit"):  # line 1031
        _read_png(p)


def test_read_png_raises_for_unsupported_color_type(tmp_path):
    p = tmp_path / "indexed.png"
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 2, 1, 8, 3, 0, 0, 0)  # color_type=3 (indexed)
    chunk = _png_chunk_helper(b"IHDR", ihdr_data)
    p.write_bytes(sig + chunk)
    with pytest.raises(ValueError, match="grayscale, RGB"):  # line 1033
        _read_png(p)


def test_read_png_raises_for_interlaced(tmp_path):
    p = tmp_path / "interlaced.png"
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 1)  # interlace=1
    chunk = _png_chunk_helper(b"IHDR", ihdr_data)
    p.write_bytes(sig + chunk)
    with pytest.raises(ValueError, match="non-interlaced"):  # line 1035
        _read_png(p)


def test_read_png_raises_for_missing_ihdr(tmp_path):
    p = tmp_path / "no_ihdr.png"
    sig = b"\x89PNG\r\n\x1a\n"
    end_chunk = _png_chunk_helper(b"IEND", b"")
    p.write_bytes(sig + end_chunk)
    with pytest.raises(ValueError, match="missing a PNG IHDR chunk"):  # line 1041
        _read_png(p)


# ---------------------------------------------------------------------------
# _read_colmap_dense_map — error cases
# ---------------------------------------------------------------------------

def test_read_colmap_dense_map_raises_for_missing_header(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"nohttpheader")
    with pytest.raises(ValueError, match="missing a COLMAP dense-map header"):  # line 1071
        _read_colmap_dense_map(p)


def test_read_colmap_dense_map_raises_for_invalid_header_values(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"abc&def&1&")  # non-integer width/height
    with pytest.raises(ValueError, match="invalid COLMAP dense-map header"):  # lines 1076-1077
        _read_colmap_dense_map(p)


def test_read_colmap_dense_map_raises_for_invalid_channel_count(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"2&1&2&")  # 2 channels = invalid
    with pytest.raises(ValueError, match="single-channel.*or 3-channel"):  # line 1079
        _read_colmap_dense_map(p)


def test_read_colmap_dense_map_raises_for_wrong_byte_count(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"2&1&1&" + b"\x00\x00\x00")  # 3 bytes instead of 8 (2*1*4)
    with pytest.raises(ValueError, match="expected.*float32 depth bytes"):  # line 1084
        _read_colmap_dense_map(p)


def test_read_colmap_dense_map_raises_for_negative_depth(tmp_path):
    p = tmp_path / "neg.bin"
    values = struct.pack("<ff", 1.0, -0.5)
    p.write_bytes(b"2&1&1&" + values)
    with pytest.raises(ValueError, match="negative depth values"):  # line 1092
        _read_colmap_dense_map(p)


# ---------------------------------------------------------------------------
# _normalized_channel_buffer — error cases
# ---------------------------------------------------------------------------

def test_normalized_channel_buffer_raises_for_non_positive_max():
    with pytest.raises(ValueError, match="max channel value must be positive"):  # line 1104
        _normalized_channel_buffer([128], max_value=0, expected=1, path=Path("t.ppm"))


def test_normalized_channel_buffer_raises_for_out_of_range_value():
    with pytest.raises(ValueError, match="channel values outside"):  # line 1109
        _normalized_channel_buffer([256], max_value=255, expected=1, path=Path("t.ppm"))


def test_normalized_channel_buffer_raises_for_count_mismatch():
    with pytest.raises(ValueError, match="expected.*channel values"):  # line 1113
        _normalized_channel_buffer([128, 64], max_value=255, expected=5, path=Path("t.ppm"))


# ---------------------------------------------------------------------------
# _png_unfilter — all filter types and paeth predictor
# ---------------------------------------------------------------------------

def test_png_unfilter_type_0_none():
    result = _png_unfilter(0, bytes([100, 200]), bytes([0, 0]), 1)
    assert result == bytes([100, 200])


def test_png_unfilter_type_1_sub():
    # filter=1 (Sub): each byte += left byte  (line 1126)
    result = _png_unfilter(1, bytes([10, 20]), bytes([0, 0]), 1)
    assert result[0] == 10
    assert result[1] == (10 + 20) & 0xFF


def test_png_unfilter_type_2_up():
    # filter=2 (Up): each byte += up byte  (line 1128)
    result = _png_unfilter(2, bytes([10, 20]), bytes([5, 15]), 1)
    assert result == bytes([(10 + 5) & 0xFF, (20 + 15) & 0xFF])


def test_png_unfilter_type_3_average():
    # filter=3 (Average)  (line 1130)
    result = _png_unfilter(3, bytes([10]), bytes([4]), 1)
    # predictor = (left=0 + up=4) // 2 = 2; out = (10+2) & 0xFF
    assert result == bytes([(10 + 2) & 0xFF])


def test_png_unfilter_type_4_paeth():
    # filter=4 (Paeth)  (line 1132)
    result = _png_unfilter(4, bytes([5]), bytes([3]), 1)
    # paeth(0, 3, 0)=3; out=(5+3)=8
    assert result == bytes([8])


def test_png_unfilter_raises_for_unknown_filter_type():
    with pytest.raises(ValueError, match="unsupported PNG filter type"):  # line 1134
        _png_unfilter(7, bytes([0]), bytes([0]), 1)


def test_paeth_predictor_all_branches():
    # left wins: estimate=left+up-up_left; abs(estimate-left)<=abs(estimate-up) and <=abs(estimate-up_left)
    assert _paeth_predictor(10, 10, 10) == 10   # all equal → left
    # up wins: left=5,up=10,up_left=0 → estimate=15; left_dist=10,up_dist=5,upleft_dist=15 → up
    assert _paeth_predictor(5, 10, 0) == 10     # line 1146: up branch
    # up_left wins: left=5,up=10,up_left=8 → estimate=7; left_dist=2,up_dist=3,upleft_dist=1 → up_left
    assert _paeth_predictor(5, 10, 8) == 8      # line 1148: up_left branch


# ---------------------------------------------------------------------------
# _skip_netpbm_space_and_comments — comment skipping
# ---------------------------------------------------------------------------

def test_skip_netpbm_space_and_comments_skips_comments():
    data = b"# comment line\n128"
    offset = _skip_netpbm_space_and_comments(data, 0)
    assert data[offset:] == b"128"


# ---------------------------------------------------------------------------
# _validate_capture_tensor_frame_set — batch has extra unknown frame (line 354)
# ---------------------------------------------------------------------------

def test_validate_capture_tensor_frame_set_raises_on_extra_unknown_frame():
    """Trigger line 354: batch provides f1 (expected) + extra_frame (unknown)."""
    manifest = _make_manifest_with_frame("f1")
    tf1 = _make_tensor_frame("f1")
    tf_extra = _make_tensor_frame("extra_frame")
    with pytest.raises(ValueError, match="unknown manifest frame ids"):  # line 354
        _validate_capture_tensor_frame_set(manifest, [tf1, tf_extra])


# ---------------------------------------------------------------------------
# _validate_manifest_links — region references unknown frame id (line 548)
# ---------------------------------------------------------------------------

def test_validate_manifest_links_raises_on_region_with_unknown_frame():
    f1 = _make_frame("f1")
    r = TrainingRegion(
        id="r1", frame_id="ghost_frame",  # frame_id not in manifest
        bounds=Bounds(min_corner=(-0.1, -0.1, 0.9), max_corner=(0.1, 0.1, 1.1)),
        evidence=RegionEvidence(),
    )
    dataset = TrainingDataset(frames=(f1,), regions=(r,))
    with pytest.raises(ValueError, match="unknown frame ids"):  # line 548
        _validate_manifest_links(dataset)


# ---------------------------------------------------------------------------
# _read_capture_raster — FileNotFoundError and PPM/PGM path
# ---------------------------------------------------------------------------

def test_read_capture_raster_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):  # line 866
        _read_capture_raster(tmp_path / "missing.png")


def test_read_capture_raster_reads_pgm_via_netpbm_path(tmp_path):
    # Exercise line 869: .pgm suffix → _read_netpbm
    p = tmp_path / "gray.pgm"
    p.write_bytes(b"P5\n2 1\n255\n" + bytes([128, 64]))
    img = _read_capture_raster(p)  # line 869
    assert img.format == "Netpbm"
    assert img.width == 2


# ---------------------------------------------------------------------------
# _read_netpbm binary — max_value > 255 error (line 994)
# ---------------------------------------------------------------------------

def test_read_netpbm_binary_raises_for_large_max_value(tmp_path):
    p = tmp_path / "gray.pgm"
    # P5 with max_value=1000
    p.write_bytes(b"P5\n1 1\n1000\n\xff")
    with pytest.raises(ValueError, match="max_value <= 255"):  # line 994
        _read_netpbm(p)


# ---------------------------------------------------------------------------
# _read_netpbm binary — truncated raw bytes (line 998)
# ---------------------------------------------------------------------------

def test_read_netpbm_binary_raises_for_truncated_raw_bytes(tmp_path):
    p = tmp_path / "gray.pgm"
    # P5: 3×1 grayscale, max=255 — but only 2 bytes of data
    p.write_bytes(b"P5\n3 1\n255\n" + bytes([128, 64]))
    with pytest.raises(ValueError, match="expected.*binary channel values"):  # line 998
        _read_netpbm(p)


# ---------------------------------------------------------------------------
# _read_png — truncated chunk payload (line 1024)
# ---------------------------------------------------------------------------

def test_read_png_raises_for_truncated_chunk_payload(tmp_path):
    p = tmp_path / "trunc_payload.png"
    sig = b"\x89PNG\r\n\x1a\n"
    # Build a chunk that claims length=10 but only has 5 bytes of data
    kind = b"IHDR"
    fake_data = b"\x00" * 5
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(fake_data, checksum) & 0xFFFFFFFF
    chunk = struct.pack(">I", 10) + kind + fake_data + struct.pack(">I", checksum)
    p.write_bytes(sig + chunk)
    with pytest.raises(ValueError, match="truncated PNG chunk payload"):  # line 1024
        _read_png(p)


# ---------------------------------------------------------------------------
# _read_png — wrong inflated size (line 1047)
# ---------------------------------------------------------------------------

def test_read_png_raises_for_wrong_inflated_size(tmp_path):
    p = tmp_path / "wrong_idat.png"
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)  # 2x1 RGB
    # Compress only 1 byte instead of the expected height*(1+width*channels)=1*(1+6)=7 bytes
    idat_data = zlib.compress(b"\x00" * 3)
    payload = sig + _png_chunk_helper(b"IHDR", ihdr_data) + _png_chunk_helper(b"IDAT", idat_data) + _png_chunk_helper(b"IEND", b"")
    p.write_bytes(payload)
    with pytest.raises(ValueError, match="expected.*inflated PNG bytes"):  # line 1047
        _read_png(p)


# ---------------------------------------------------------------------------
# _read_png — truncated scanline (line 1055)
# ---------------------------------------------------------------------------

def test_read_png_raises_for_truncated_scanline(tmp_path):
    p = tmp_path / "trunc_scanline.png"
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)  # 2x1 RGB = 6 bytes + 1 filter
    # Provide correct total size but short scanline by using wrong filter row
    # We'll compress 7 bytes (correct size) but use data that when decompressed
    # the scanline will be too short — use 6 bytes (filter + 5 instead of 7)
    # Actually, to hit line 1055, we need len(raw) == expected but a scanline short.
    # That's impossible with correct raw length. The scanline check is in a loop.
    # If we have raw=[filter_byte] + only 5 bytes for a 2-pixel RGB row (need 6),
    # then raw length is 6 but expected is 7 → hits line 1047 first.
    # To test line 1055, we'd need to somehow pass 1047 but fail the scanline.
    # This branch seems unreachable in practice. Skip with a note.
    pytest.skip("truncated scanline branch (1055) unreachable when len(raw)==expected")


# ---------------------------------------------------------------------------
# _netpbm_token — unexpected end of header (line 1157)
# ---------------------------------------------------------------------------

def test_netpbm_token_raises_on_empty_data():
    from aura.ingest.capture import _netpbm_token
    with pytest.raises(ValueError, match="unexpected end of Netpbm header"):  # line 1157
        _netpbm_token(b"   ", 0)  # all whitespace, no token


# ---------------------------------------------------------------------------
# _average_scalar — coverage for raise path (line 766) using _RasterImage directly
# ---------------------------------------------------------------------------

def test_average_scalar_raises_with_raster_image_non_1_channel():
    img = _RasterImage("PNG", 1, 1, 3, PackedFloatBuffer([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="1-channel"):  # line 766 via _average_scalar
        _average_scalar(img)


# ---------------------------------------------------------------------------
# _depth_bins with zero valid_count
# ---------------------------------------------------------------------------

def test_depth_bins_empty_valid_count():
    result = _depth_bins(PackedFloatBuffer([0.0, -1.0]), total_count=2, valid_count=0, minimum=0.0, maximum=0.0)
    assert result == tuple()


# ---------------------------------------------------------------------------
# _average_scalar — successful return path (line 766)
# ---------------------------------------------------------------------------

def test_average_scalar_returns_correct_value():
    img = _RasterImage("PNG", 2, 1, 1, PackedFloatBuffer([0.25, 0.75]))
    result = _average_scalar(img)   # line 766
    assert abs(result - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _depth_summary — raises for multi-channel image (line 790)
# ---------------------------------------------------------------------------

def test_depth_summary_raises_for_non_1_channel_image():
    img = _RasterImage("PNG", 1, 1, 3, PackedFloatBuffer([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="1-channel image"):  # line 790
        _depth_summary(img)


# ---------------------------------------------------------------------------
# _depth_bins — zero values skip branch (line 843)
# ---------------------------------------------------------------------------

def test_depth_bins_two_bin_with_zero_values_skips_zero():
    # Trigger line 843: mix of zero and positive in a two-bin scenario
    values = PackedFloatBuffer([0.0, 1.0, 0.0, 3.0])  # zeros + values on both sides
    result = _depth_bins(values, total_count=4, valid_count=2, minimum=1.0, maximum=3.0)
    # midpoint=2.0; near=[1.0], far=[3.0] → both bins present
    assert len(result) == 2


# ---------------------------------------------------------------------------
# write_capture_manifest_template (lines 439-442)
# ---------------------------------------------------------------------------

def test_write_capture_manifest_template(tmp_path):
    from aura.ingest.capture import write_capture_manifest_template
    out = tmp_path / "template.json"
    result = write_capture_manifest_template(out)   # lines 439-442
    assert result == out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["format"] == "AURA_CAPTURE_MANIFEST"
    assert "frames" in data


# ---------------------------------------------------------------------------
# imageio paths — 2D shape and uint8/uint16 normalization (if imageio available)
# ---------------------------------------------------------------------------

def test_read_imageio_tensor_2d_grayscale_shape(tmp_path):
    """2D array shape → single channel (lines 941-943)."""
    import importlib.util
    if importlib.util.find_spec("imageio") is None:
        pytest.skip("imageio not installed")
    import numpy as np
    import imageio.v3 as iio
    from aura.ingest.capture import _read_imageio_tensor

    pixels = np.array([[128, 64]], dtype=np.uint8)  # shape (1, 2) → 2D grayscale
    path = tmp_path / "gray.png"
    iio.imwrite(path, pixels)
    tensor = _read_imageio_tensor(path)
    assert tensor.channels == 1   # lines 941-943
    assert tensor.width == 2
    assert tensor.height == 1


def test_normalize_tensor_values_uint8(tmp_path):
    """uint8 dtype → divide by 255 (lines 968-970)."""
    from aura.ingest.capture import _normalize_tensor_values
    buf = _normalize_tensor_values([0, 128, 255], dtype="uint8")
    assert abs(buf[0] - 0.0) < 1e-9
    assert abs(buf[1] - 128 / 255.0) < 1e-9
    assert abs(buf[2] - 1.0) < 1e-9


def test_normalize_tensor_values_uint16():
    """uint16 dtype → divide by 65535 (line 969)."""
    from aura.ingest.capture import _normalize_tensor_values
    buf = _normalize_tensor_values([0, 32767, 65535], dtype="uint16")
    assert abs(buf[0] - 0.0) < 1e-9
    assert abs(buf[2] - 1.0) < 1e-9


def test_normalize_tensor_values_float():
    """Non-uint dtype → pass through as-is (line 970)."""
    from aura.ingest.capture import _normalize_tensor_values
    buf = _normalize_tensor_values([0.25, 0.75], dtype="float32")
    assert abs(buf[0] - 0.25) < 1e-9
    assert abs(buf[1] - 0.75) < 1e-9


def test_read_imageio_tensor_filenotfound(tmp_path):
    """FileNotFoundError for missing file (line 928)."""
    from aura.ingest.capture import _read_imageio_tensor
    with pytest.raises(FileNotFoundError):
        _read_imageio_tensor(tmp_path / "missing.jpg")


# ---------------------------------------------------------------------------
# PackedFloatBuffer.sample() (line 66)
# ---------------------------------------------------------------------------

def test_packed_float_buffer_sample():
    buf = PackedFloatBuffer([1.0, 2.0, 3.0, 4.0, 5.0])
    assert buf.sample(3) == (1.0, 2.0, 3.0)   # line 66
    assert buf.sample(0) == ()
    assert buf.sample(-1) == ()


# ---------------------------------------------------------------------------
# CaptureManifest.from_dict — non-dict payload raises (line 93)
# ---------------------------------------------------------------------------

def test_capture_manifest_from_dict_raises_for_non_dict():
    with pytest.raises(ValueError, match="must be an object"):  # line 93
        CaptureManifest.from_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# CaptureTensor.to_dict() (line 217)
# ---------------------------------------------------------------------------

def test_capture_tensor_to_dict():
    t = CaptureTensor("img.png", "PNG", "stdlib", 2, 1, 3, [0.5] * 6)
    d = t.to_dict()   # line 217
    assert d["width"] == 2
    assert d["channels"] == 3
    assert d["storageDtype"] == "float64"


# ---------------------------------------------------------------------------
# CaptureTensor.shape property (line 168)
# ---------------------------------------------------------------------------

def test_capture_tensor_shape_property():
    t = CaptureTensor("img.png", "PNG", "stdlib", 3, 2, 4, [0.5] * 24)
    assert t.shape == (2, 3, 4)   # (height, width, channels) — line 168


# ---------------------------------------------------------------------------
# validate_capture_manifest_document (lines 496-499)
# ---------------------------------------------------------------------------

def test_validate_capture_manifest_document_passes_valid():
    from aura.ingest.capture import validate_capture_manifest_document
    payload = _minimal_manifest_dict()
    validate_capture_manifest_document(payload)  # should not raise


def test_validate_capture_manifest_document_raises_for_invalid():
    from aura.ingest.capture import validate_capture_manifest_document
    with pytest.raises(ValueError, match="validation failed"):  # lines 496-499
        validate_capture_manifest_document({"format": "WRONG"})


# ---------------------------------------------------------------------------
# CaptureFrameTensors.byte_count and to_dict (lines 244, 251)
# ---------------------------------------------------------------------------

def test_capture_frame_tensors_byte_count_and_to_dict():
    values = PackedFloatBuffer([0.5] * 3)
    image = CaptureTensor("img.png", "PNG", "stdlib", 1, 1, 3, values)
    frame_tensors = CaptureFrameTensors(frame_id="f1", image=image)
    assert frame_tensors.byte_count == 24   # 3 floats × 8 bytes (line 244)
    d = frame_tensors.to_dict()             # line 251
    assert d["frameId"] == "f1"
    assert d["depth"] is None


# ---------------------------------------------------------------------------
# _depth_region_half_extent — no intrinsics path (line 665)
# ---------------------------------------------------------------------------

def test_depth_region_half_extent_no_intrinsics():
    from aura.ingest.capture import _depth_region_half_extent
    frame = TrainingFrame(
        id="f1",
        camera_origin=(0.0, 0.0, -1.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.5, 0.5, 0.5),
        target_depth=2.0,
        image_path="img.png",
        intrinsics=None,
    )
    result = _depth_region_half_extent(frame, 1.0)   # line 665
    assert result == max(0.05, 1.0 * 0.05)


# ---------------------------------------------------------------------------
# _mask_regions_from_assets — exercise lines 684-688
# ---------------------------------------------------------------------------

def test_mask_regions_from_assets_creates_region_when_coverage_positive():
    from aura.ingest.capture import _mask_regions_from_assets
    frame = TrainingFrame(
        id="f1",
        camera_origin=(0.0, 0.0, -1.0),
        look_at=(0.0, 0.0, 0.0),
        target_color=(0.5, 0.5, 0.5),
        target_depth=2.0,
        image_path="img.png",
        semantic_label="wall",
    )
    asset = CaptureFrameAssets(
        frame_id="f1",
        image_path="img.png",
        width=2, height=1,
        average_color=(0.5, 0.5, 0.5),
        mask_coverage=0.75,        # > 0 → creates a mask region (lines 684-688)
    )
    regions = _mask_regions_from_assets((frame,), {"f1": asset})
    assert len(regions) == 1
    assert regions[0].id == "f1_mask_semantic"
    assert regions[0].fallback_source == "capture-mask-prior"


# ---------------------------------------------------------------------------
# _validate_byte_limit — raises for non-positive value (line 723)
# ---------------------------------------------------------------------------

def test_validate_byte_limit_raises_for_non_positive():
    from aura.ingest.capture import _validate_byte_limit
    with pytest.raises(ValueError, match="must be positive"):  # line 723
        _validate_byte_limit(0, "max_loaded_bytes")
    with pytest.raises(ValueError, match="must be positive"):
        _validate_byte_limit(-5, "max_frame_bytes")


# ---------------------------------------------------------------------------
# load_capture_asset_tensors — max_frame_bytes and max_loaded_bytes exceeded (lines 737, 743)
# ---------------------------------------------------------------------------

def test_load_capture_asset_tensors_raises_when_max_frame_bytes_exceeded(tmp_path):
    img = tmp_path / "frame.png"
    _write_png_file(img, width=4, height=4, channels=3)  # 48 values × 8 bytes = 384 bytes
    manifest = _manifest_with_assets(tmp_path, image_path="frame.png")
    with pytest.raises(ValueError, match="max_frame_bytes"):  # line 737
        load_capture_asset_tensors(manifest, max_frame_bytes=1)


def test_load_capture_asset_tensors_raises_when_max_loaded_bytes_exceeded(tmp_path):
    img = tmp_path / "frame.png"
    _write_png_file(img, width=4, height=4, channels=3)
    manifest = _manifest_with_assets(tmp_path, image_path="frame.png")
    with pytest.raises(ValueError, match="max_loaded_bytes"):  # line 743
        load_capture_asset_tensors(manifest, max_loaded_bytes=1)


# ---------------------------------------------------------------------------
# imageio-backed paths — require imageio (lines 931-932, 937-938, 947-952)
# ---------------------------------------------------------------------------

def test_read_imageio_tensor_raises_on_load_error(tmp_path):
    """Exercise line 937-938: imageio.imread raises → ValueError."""
    import importlib.util
    if importlib.util.find_spec("imageio") is None:
        pytest.skip("imageio not installed")
    from aura.ingest.capture import _read_imageio_tensor
    # Write a file that exists but is not a valid image so imageio raises
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"this is not a jpeg")
    with pytest.raises(ValueError, match="could not be loaded"):  # lines 937-938
        _read_imageio_tensor(p)


def test_read_imageio_tensor_4d_video_shape(tmp_path):
    """4D shape (video) → takes first frame (lines 947-952)."""
    import importlib.util
    if importlib.util.find_spec("imageio") is None:
        pytest.skip("imageio not installed")
    import numpy as np
    from unittest.mock import patch
    from aura.ingest.capture import _read_imageio_tensor

    # Mock imageio.imread to return a 4D array (video frames)
    fake_video = np.zeros((2, 4, 6, 3), dtype=np.uint8)  # 2 frames, 4x6 RGB
    fake_video[0, :, :] = 100  # first frame has values

    with patch("imageio.v3.imread", return_value=fake_video):
        p = tmp_path / "video.mp4"
        p.write_bytes(b"fake")  # file must exist
        tensor = _read_imageio_tensor(p)  # lines 947-952
    assert tensor.width == 6
    assert tensor.height == 4
    assert tensor.channels == 3


def test_read_imageio_tensor_unsupported_shape_raises(tmp_path):
    """0D/1D shape → raises ValueError (line 952)."""
    import importlib.util
    if importlib.util.find_spec("imageio") is None:
        pytest.skip("imageio not installed")
    import numpy as np
    from unittest.mock import patch
    from aura.ingest.capture import _read_imageio_tensor

    # Return a 1D array (shape with 1 dim) — unsupported
    fake_1d = np.zeros((6,), dtype=np.uint8)

    with patch("imageio.v3.imread", return_value=fake_1d):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"fake")
        with pytest.raises(ValueError, match="unsupported shape"):  # line 952
            _read_imageio_tensor(p)
