"""Baseline removal and offline frequency filtering for calibrated signals."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt


def _validate_signal(signal: NDArray[np.float64]) -> None:
    if signal.ndim not in (1, 2):
        raise ValueError(
            f"Expected 1D or 2D signal array, got shape {signal.shape}"
        )
    if signal.shape[0] == 0:
        raise ValueError("Expected signal with at least one sample")


def remove_baseline(
    signal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Remove the per-axis DC/gravity baseline from a calibrated signal.

    Args:
        signal: shape (n_samples,) or (n_samples, n_axes), units: g or deg/s.

    Returns:
        Baseline-removed signal with the same shape and units as ``signal``.
        For multi-axis input, each axis is centered independently.
    """
    _validate_signal(signal)
    baseline = np.mean(signal, axis=0, keepdims=signal.ndim == 2)
    return np.asarray(signal, dtype=np.float64) - baseline


def bandpass_filter(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    band_hz: tuple[float, float],
    order: int,
) -> NDArray[np.float64]:
    """Apply an offline, non-causal Butterworth band-pass filter.

    This implementation uses forward-backward filtering and is not valid for
    real-time, sample-by-sample on-device use without modification.

    Args:
        signal: shape (n_samples,) or (n_samples, n_axes), units: g or deg/s.
        sample_rate_hz: Sampling rate in Hz.
        band_hz: Inclusive pass band as ``(low_hz, high_hz)``.
        order: Positive Butterworth filter order.

    Returns:
        Filtered signal with the same shape and units as ``signal``.
    """
    _validate_signal(signal)
    if sample_rate_hz <= 0:
        raise ValueError(f"Expected positive sample_rate_hz, got {sample_rate_hz}")
    if order <= 0:
        raise ValueError(f"Expected positive filter order, got {order}")
    if len(band_hz) != 2:
        raise ValueError(f"Expected band_hz=(low, high), got {band_hz}")

    low_hz, high_hz = band_hz
    nyquist_hz = sample_rate_hz / 2.0
    if not 0 < low_hz < high_hz < nyquist_hz:
        raise ValueError(
            f"Expected 0 < low_hz < high_hz < Nyquist ({nyquist_hz}), "
            f"got {band_hz}"
        )

    normalized_band = (low_hz / nyquist_hz, high_hz / nyquist_hz)
    numerator, denominator = butter(order, normalized_band, btype="bandpass")
    return np.asarray(
        filtfilt(numerator, denominator, signal, axis=0),
        dtype=np.float64,
    )