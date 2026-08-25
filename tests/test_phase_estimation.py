"""Tests for instantaneous phase estimation (Hilbert transform)."""

import numpy as np
import pytest

from estimation.phase_estimation import estimate_phase


def test_estimate_phase_known_sine_at_zero_crossing() -> None:
    # scipy.hilbert's instantaneous-phase convention treats the signal as
    # cos(phase) -- a cosine's analytic-signal phase equals its own
    # argument exactly, with no conventional offset to account for.
    sample_rate_hz = 100.0
    n_samples = 200
    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    signal = np.cos(2.0 * np.pi * 6.0 * time_s)

    phase = estimate_phase(signal)

    expected_phase = (2.0 * np.pi * 6.0 * time_s[-1] + np.pi) % (2.0 * np.pi) - np.pi
    assert phase is not None
    assert phase == pytest.approx(expected_phase, abs=0.05)


def test_estimate_phase_returns_none_for_constant_signal() -> None:
    signal = np.full(100, 0.5, dtype=np.float64)

    assert estimate_phase(signal) is None


def test_estimate_phase_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        estimate_phase(np.array([], dtype=np.float64))
