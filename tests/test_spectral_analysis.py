"""Tests for Welch PSD and tremor-band spectral features."""

import numpy as np
import pytest

from signal_processing.spectral_analysis import (
    compute_welch_psd,
    dominant_frequency,
    power_ratio,
    spectral_entropy,
    total_power,
    tremor_band_power,
)


@pytest.fixture
def six_hz_psd() -> tuple[np.ndarray, np.ndarray]:
    sample_rate_hz = 100.0
    time_s = np.arange(200, dtype=np.float64) / sample_rate_hz
    signal = np.sin(2.0 * np.pi * 6.0 * time_s)
    return compute_welch_psd(signal, sample_rate_hz, nperseg=200)


def test_compute_welch_psd_detects_six_hz_peak(six_hz_psd) -> None:
    freqs_hz, psd = six_hz_psd

    assert freqs_hz.shape == psd.shape
    assert abs(freqs_hz[int(np.argmax(psd))] - 6.0) <= 0.5


def test_tremor_features_integrate_frequency_bins(six_hz_psd) -> None:
    freqs_hz, psd = six_hz_psd

    tremor_power = tremor_band_power(freqs_hz, psd, (4.0, 12.0))
    signal_power = total_power(freqs_hz, psd)

    assert tremor_power > 0.4
    assert signal_power > 0.4
    assert power_ratio(tremor_power, signal_power) > 0.95
    assert dominant_frequency(freqs_hz, psd, (4.0, 12.0)) == 6.0


def test_power_ratio_guards_zero_total_power() -> None:
    assert power_ratio(0.0, 0.0) == 0.0


def test_spectral_entropy_distinguishes_concentrated_and_distributed_psd() -> None:
    concentrated = np.zeros(16, dtype=np.float64)
    concentrated[6] = 1.0
    distributed = np.ones(16, dtype=np.float64)

    concentrated_entropy = spectral_entropy(concentrated)
    distributed_entropy = spectral_entropy(distributed)

    assert concentrated_entropy == pytest.approx(0.0)
    assert distributed_entropy == pytest.approx(1.0)
    assert concentrated_entropy < distributed_entropy


def test_dominant_frequency_ignores_out_of_band_peak() -> None:
    freqs_hz = np.arange(0.0, 21.0, 1.0, dtype=np.float64)
    psd = np.ones_like(freqs_hz)
    psd[2] = 100.0
    psd[6] = 10.0

    assert dominant_frequency(freqs_hz, psd, (4.0, 12.0)) == 6.0


def test_spectral_functions_reject_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        total_power(np.array([1.0, 2.0]), np.array([1.0]))