"""Instantaneous tremor phase estimation (advanced, experimental).

Feature 31 of the build plan. Per architecture.md -> Frequency & Phase
Estimation, this is disabled by default (config.estimation.phase_enabled)
and is not required for the initial controller, which operates on
frequency + severity + confidence only. This module always exposes the
optional-output contract (phase: Optional[float]) so callers degrade
gracefully to phase=None.

Gating is the caller's responsibility: this function does not check
config.estimation.phase_enabled itself, matching how estimate_frequency()/
estimate_amplitude() are also ungated helpers -- whoever wires this into
ml/inference.py or the controller loop must check the flag first.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert


def estimate_phase(filtered_signal: NDArray[np.float64]) -> float | None:
    """Estimate instantaneous phase via the Hilbert transform.

    Args:
        filtered_signal: shape (n_samples,), band-pass filtered tremor-band
            signal for one window, units: g.

    Returns:
        Instantaneous phase in radians (range (-pi, pi]) at the final
        sample of the window, or None if the signal is constant (phase is
        undefined for a non-oscillating signal).

    Raises:
        ValueError: If filtered_signal is not a non-empty 1D array.
    """
    if filtered_signal.ndim != 1 or filtered_signal.size == 0:
        raise ValueError(f"Expected non-empty 1D signal, got shape {filtered_signal.shape}")
    if np.allclose(filtered_signal, filtered_signal[0]):
        return None

    analytic_signal = hilbert(filtered_signal)
    instantaneous_phase = np.angle(analytic_signal)
    return float(instantaneous_phase[-1])
