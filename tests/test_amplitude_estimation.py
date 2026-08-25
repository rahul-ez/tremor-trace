"""Tests for tremor amplitude estimation."""

import numpy as np
import pytest

from estimation.amplitude_estimation import estimate_amplitude


def test_estimate_amplitude_known_sine_amplitude() -> None:
    time_s = np.arange(1000, dtype=np.float64) / 100.0
    amplitude = 2.0
    signal = amplitude * np.sin(2.0 * np.pi * 6.0 * time_s)

    assert estimate_amplitude(signal) == pytest.approx(amplitude / np.sqrt(2.0))


def test_estimate_amplitude_rejects_non_1d_signal() -> None:
    signal = np.zeros((10, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="Expected 1D"):
        estimate_amplitude(signal)
