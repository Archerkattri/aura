from array import array

import pytest

from aura import CaptureTensor
from aura.ingest.capture import PackedFloatBuffer


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
