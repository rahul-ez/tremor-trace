"""Tests for overlapping signal windows."""

import numpy as np
import pytest

from signal_processing.windowing import segment_windows


def test_segment_windows_uses_v1_geometry() -> None:
    signal = np.arange(450, dtype=np.float64)

    windows = segment_windows(signal, 100.0, 2.0, 50.0)

    assert windows.shape == (3, 200)
    np.testing.assert_array_equal(windows[0], signal[:200])
    np.testing.assert_array_equal(windows[1], signal[100:300])
    np.testing.assert_array_equal(windows[2], signal[200:400])


def test_segment_windows_preserves_multiaxis_shape() -> None:
    signal = np.arange(600, dtype=np.float64).reshape(200, 3)

    windows = segment_windows(signal, 100.0, 1.0, 50.0)

    assert windows.shape == (3, 100, 3)


def test_segment_windows_discards_incomplete_trailing_samples() -> None:
    signal = np.arange(250, dtype=np.float64)

    windows = segment_windows(signal, 100.0, 2.0, 50.0)

    assert windows.shape == (1, 200)


def test_segment_windows_returns_empty_for_short_signal() -> None:
    signal = np.arange(100, dtype=np.float64)

    windows = segment_windows(signal, 100.0, 2.0, 50.0)

    assert windows.shape == (0, 200)


def test_segment_windows_rejects_invalid_overlap() -> None:
    signal = np.arange(200, dtype=np.float64)

    with pytest.raises(ValueError, match="overlap_pct"):
        segment_windows(signal, 100.0, 2.0, 100.0)