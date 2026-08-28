"""Time-domain features for filtered tremor signals."""

import numpy as np
from numpy.typing import NDArray


def _validate_signal(signal: NDArray[np.float64]) -> None:
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal array, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError("Expected non-empty signal")


def compute_rms(signal: NDArray[np.float64]) -> float:
    """Compute unnormalized RMS amplitude for one filtered signal window.

    Args:
        signal: shape (n_samples,), units: g or deg/s.

    Returns:
        RMS amplitude in the same physical units as ``signal``.
    """
    _validate_signal(signal)
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_variance(signal: NDArray[np.float64]) -> float:
    """Compute variance for one filtered signal window.

    Args:
        signal: shape (n_samples,), units: g or deg/s.

    Returns:
        Variance in squared units of ``signal``.
    """
    _validate_signal(signal)
    return float(np.var(signal))