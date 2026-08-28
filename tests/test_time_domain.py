"""Tests for time-domain signal features."""

import numpy as np
import pytest

from signal_processing.time_domain import compute_rms, compute_variance


def test_compute_rms_known_sine_amplitude() -> None:
    time_s = np.arange(1000, dtype=np.float64) / 100.0
    amplitude = 2.0
    signal = amplitude * np.sin(2.0 * np.pi * 6.0 * time_s)

    assert compute_rms(signal) == pytest.approx(amplitude / np.sqrt(2.0))


def test_compute_variance_known_signal() -> None:
    signal = np.array([-2.0, 0.0, 2.0], dtype=np.float64)

    assert compute_variance(signal) == pytest.approx(8.0 / 3.0)


def test_time_domain_features_reject_non_1d_signal() -> None:
    signal = np.zeros((10, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="Expected 1D"):
        compute_rms(signal)