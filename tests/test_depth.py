"""Tests for aura.depth — MiDaS monocular depth hook and geometric fallback."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from aura.depth import geometric_depth_fallback, midas_depth


# ---------------------------------------------------------------------------
# midas_depth tests (no actual model download)
# ---------------------------------------------------------------------------

def test_midas_depth_returns_none_when_torch_unavailable() -> None:
    """midas_depth must return None when torch cannot be imported."""
    # Hide torch from the import system
    with patch.dict(sys.modules, {"torch": None, "torch.hub": None}):
        result = midas_depth("nonexistent_image.png")
    assert result is None, "Expected None when torch is unavailable"


def test_midas_depth_returns_none_on_hub_error() -> None:
    """midas_depth must return None when torch.hub.load raises any exception."""
    mock_torch = MagicMock()
    mock_torch.hub.load.side_effect = RuntimeError("hub network error")

    with patch.dict(sys.modules, {"torch": mock_torch, "torch.hub": mock_torch.hub,
                                   "PIL": MagicMock(), "PIL.Image": MagicMock(),
                                   "numpy": MagicMock()}):
        result = midas_depth("some_image.jpg")
    assert result is None, "Expected None when torch.hub.load raises"


# ---------------------------------------------------------------------------
# geometric_depth_fallback tests
# ---------------------------------------------------------------------------

def test_geometric_fallback_empty_points() -> None:
    """geometric_depth_fallback with no points must return an empty list."""
    result = geometric_depth_fallback([])
    assert result == [], f"Expected [], got {result!r}"


def test_geometric_fallback_single_point() -> None:
    """geometric_depth_fallback with a single point must return [0.5]."""
    result = geometric_depth_fallback([(1.0, 2.0, 3.0)])
    assert result == [0.5], f"Expected [0.5] for a single point, got {result!r}"


def test_geometric_fallback_normalized() -> None:
    """All values returned by geometric_depth_fallback must be in [0, 1]."""
    points = [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 3.0),
        (0.0, 0.0, 5.0),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
    ]
    result = geometric_depth_fallback(points)
    assert len(result) == len(points)
    for v in result:
        assert 0.0 <= v <= 1.0, f"Depth value {v} is outside [0, 1]"


def test_geometric_fallback_ordering() -> None:
    """A point closer to the camera origin should receive a lower depth value."""
    near = (0.0, 0.0, 1.0)   # distance = 1
    far  = (0.0, 0.0, 10.0)  # distance = 10
    result = geometric_depth_fallback([near, far])
    assert len(result) == 2
    assert result[0] < result[1], (
        f"Near point depth {result[0]} should be less than far point depth {result[1]}"
    )


def test_midas_depth_returns_list_on_successful_inference() -> None:
    """midas_depth returns a normalized float list when hub.load and inference succeed."""
    import numpy as _np

    mock_model = MagicMock()
    mock_transforms = MagicMock()
    mock_torch = MagicMock()
    # First hub.load → model, second → transforms object
    mock_torch.hub.load.side_effect = [mock_model, mock_transforms]
    # prediction.squeeze().cpu().numpy() must return a real numpy array so that
    # .min()/.max()/.flatten()/.tolist() work — numpy itself is NOT mocked here
    # because mocking sys.modules['numpy'] breaks ndarray's internal _NoValue sentinel
    depth_arr = _np.array([[0.2, 0.8], [0.4, 1.0]])
    mock_model.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = depth_arr

    mock_pil = MagicMock()

    with patch.dict(sys.modules, {
        "torch": mock_torch,
        "torch.hub": mock_torch.hub,
        "PIL": mock_pil,
        "PIL.Image": mock_pil.Image,
    }):
        result = midas_depth("dummy.png", device="cpu")

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 4, f"Expected 4 values from 2×2 array, got {len(result)}"
    for v in result:
        assert 0.0 <= v <= 1.0, f"Depth {v} outside [0, 1]"


def test_midas_depth_returns_none_on_inference_error() -> None:
    """midas_depth returns None when hub loads but model.eval() raises."""
    mock_model = MagicMock()
    mock_model.eval.side_effect = RuntimeError("GPU unavailable")
    mock_transforms = MagicMock()
    mock_torch = MagicMock()
    mock_torch.hub.load.side_effect = [mock_model, mock_transforms]

    mock_pil = MagicMock()

    with patch.dict(sys.modules, {
        "torch": mock_torch,
        "torch.hub": mock_torch.hub,
        "PIL": mock_pil,
        "PIL.Image": mock_pil.Image,
    }):
        result = midas_depth("dummy.png")

    assert result is None, "Expected None when model.eval() raises"
