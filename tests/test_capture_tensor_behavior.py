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
