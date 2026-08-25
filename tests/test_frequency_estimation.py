"""Tests for dominant tremor-frequency estimation."""

import numpy as np
import pytest

from estimation.frequency_estimation import estimate_frequency
from signal_processing.spectral_analysis import compute_welch_psd


def test_estimate_frequency_detects_six_hz_peak() -> None:
    sample_rate_hz = 100.0
    time_s = np.arange(200, dtype=np.float64) / sample_rate_hz
    signal = np.sin(2.0 * np.pi * 6.0 * time_s)
    freqs_hz, psd = compute_welch_psd(signal, sample_rate_hz, nperseg=200)

    frequency = estimate_frequency(freqs_hz, psd, tremor_band_hz=(4.0, 12.0))

    assert frequency == pytest.approx(6.0)


def test_estimate_frequency_returns_none_for_flat_spectrum() -> None:
    freqs_hz = np.linspace(0.0, 50.0, 101, dtype=np.float64)
    flat_psd = np.full_like(freqs_hz, 1.0)

    frequency = estimate_frequency(freqs_hz, flat_psd, tremor_band_hz=(4.0, 12.0))

    assert frequency is None


def test_estimate_frequency_returns_none_for_zero_power_band() -> None:
    freqs_hz = np.linspace(0.0, 50.0, 101, dtype=np.float64)
    zero_psd = np.zeros_like(freqs_hz)

    frequency = estimate_frequency(freqs_hz, zero_psd, tremor_band_hz=(4.0, 12.0))

    assert frequency is None


def test_estimate_frequency_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="1D arrays"):
        estimate_frequency(np.array([1.0, 2.0]), np.array([1.0]))
