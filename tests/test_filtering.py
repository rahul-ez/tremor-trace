"""Tests for baseline removal and offline filtering."""

import numpy as np
import pytest

from signal_processing.filtering import bandpass_filter, remove_baseline


def test_remove_baseline_centers_each_axis() -> None:
    signal = np.array([[1.0, 10.0], [3.0, 14.0]], dtype=np.float64)

    result = remove_baseline(signal)

    np.testing.assert_allclose(result, [[-1.0, -2.0], [1.0, 2.0]])


def test_remove_baseline_centers_single_axis() -> None:
    signal = np.array([2.0, 4.0, 6.0], dtype=np.float64)

    result = remove_baseline(signal)

    np.testing.assert_allclose(result, [-2.0, 0.0, 2.0])


def test_bandpass_filter_preserves_tremor_and_attenuates_high_frequency() -> None:
    sample_rate_hz = 100.0
    time_s = np.arange(2000, dtype=np.float64) / sample_rate_hz
    tremor = np.sin(2.0 * np.pi * 6.0 * time_s)
    high_frequency = np.sin(2.0 * np.pi * 30.0 * time_s)
    signal = tremor + high_frequency

    result = bandpass_filter(signal, sample_rate_hz, (4.0, 12.0), order=4)

    central = slice(200, -200)
    correlation = np.corrcoef(result[central], tremor[central])[0, 1]
    assert correlation > 0.99
    assert np.std(result[central] - tremor[central]) < 0.1


@pytest.mark.parametrize(
    "signal",
    [np.zeros((2, 3, 1)), np.zeros(0)],
)
def test_filtering_rejects_invalid_signal_shapes(signal: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Expected 1D or 2D|at least one sample"):
        remove_baseline(signal)


def test_bandpass_filter_rejects_invalid_band() -> None:
    signal = np.ones(100, dtype=np.float64)

    with pytest.raises(ValueError, match="Nyquist"):
        bandpass_filter(signal, 100.0, (4.0, 60.0), order=4)