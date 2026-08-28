"""Tests for the synthetic tremor generator (Feature 32)."""

import numpy as np
import pytest

from simulation.tremor_model import generate_synthetic_tremor
from signal_processing.spectral_analysis import compute_welch_psd
from signal_processing.time_domain import compute_rms


def test_generate_synthetic_tremor_shape_and_dtype() -> None:
    signal = generate_synthetic_tremor(6.0, 0.4, 2.0, 100.0)

    assert signal.shape == (200,)
    assert signal.dtype == np.float64


def test_generate_synthetic_tremor_frequency_and_amplitude() -> None:
    signal = generate_synthetic_tremor(6.0, 0.4, 2.0, 100.0)
    freqs_hz, psd = compute_welch_psd(signal, 100.0, 200)

    assert freqs_hz[int(np.argmax(psd))] == pytest.approx(6.0)
    assert compute_rms(signal) == pytest.approx(0.4 / np.sqrt(2.0), abs=0.01)


def test_generate_synthetic_tremor_noise_is_seeded() -> None:
    first = generate_synthetic_tremor(6.0, 0.4, 2.0, 100.0, 0.05, seed=42)
    second = generate_synthetic_tremor(6.0, 0.4, 2.0, 100.0, 0.05, seed=42)
    different = generate_synthetic_tremor(6.0, 0.4, 2.0, 100.0, 0.05, seed=43)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency_hz": -1.0},
        {"amplitude": -1.0},
        {"duration_s": 0.0},
        {"sample_rate_hz": 0.0},
        {"noise_std": -1.0},
    ],
)
def test_generate_synthetic_tremor_rejects_invalid_parameters(kwargs: dict) -> None:
    defaults = {
        "frequency_hz": 6.0,
        "amplitude": 0.4,
        "duration_s": 2.0,
        "sample_rate_hz": 100.0,
        "noise_std": 0.0,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        generate_synthetic_tremor(**defaults)
