"""Suppression measurement for simulated tremor mitigation (Feature 34).

Computes the percentage reduction in tremor-band power between a pre-mitigation
and post-mitigation signal using Welch PSD -- the same spectral method used
throughout the signal-processing pipeline.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from signal_processing.spectral_analysis import compute_welch_psd, tremor_band_power

logger = logging.getLogger(__name__)

# Division-by-zero guard consistent with power_ratio convention in spectral_analysis.py
_POWER_FLOOR = 1e-12

# Pre-signal power below this threshold is treated as near-zero (no meaningful tremor).
_NEAR_ZERO_POWER_THRESHOLD = 1e-8


def compute_suppression_pct(
    pre_signal: NDArray[np.float64],
    post_signal: NDArray[np.float64],
    sample_rate_hz: float,
    tremor_band_hz: tuple[float, float],
) -> float:
    """Compute the percentage reduction in tremor-band power after mitigation.

    Uses Welch PSD on both signals and integrates power over the tremor band.
    The result is clamped to [0.0, 100.0]: negative values (post > pre) are
    treated as 0.0 suppression, not as amplification.

    Offline/non-causal -- operates on a complete signal window, not sample-by-sample.

    Args:
        pre_signal: Shape (n_samples,), units g -- tremor signal before mitigation.
        post_signal: Shape (n_samples,), units g -- tremor signal after mitigation.
            Must have the same length as `pre_signal`.
        sample_rate_hz: Sampling rate in Hz, e.g. 100.0.
        tremor_band_hz: Inclusive (low_hz, high_hz) tremor band, e.g. (4.0, 12.0).

    Returns:
        Suppression percentage in [0.0, 100.0]. Returns 0.0 when pre-signal
        tremor-band power is near-zero (no meaningful baseline to suppress).

    Raises:
        ValueError: If either signal is not 1D, is empty, has mismatched length,
            or if sample_rate_hz or tremor_band_hz is invalid.
    """
    pre_signal = np.asarray(pre_signal, dtype=np.float64)
    post_signal = np.asarray(post_signal, dtype=np.float64)

    if pre_signal.ndim != 1 or pre_signal.size == 0:
        raise ValueError(
            f"pre_signal must be a non-empty 1D array, got shape {pre_signal.shape}"
        )
    if post_signal.ndim != 1 or post_signal.size == 0:
        raise ValueError(
            f"post_signal must be a non-empty 1D array, got shape {post_signal.shape}"
        )
    if pre_signal.size != post_signal.size:
        raise ValueError(
            f"pre_signal and post_signal must have the same length, "
            f"got {pre_signal.size} and {post_signal.size}"
        )
    if sample_rate_hz <= 0.0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

    # nperseg: use full window length, capped to signal size.
    # Consistent with feature extraction which uses window_samples = signal.size.
    nperseg = pre_signal.size

    pre_freqs, pre_psd = compute_welch_psd(pre_signal, sample_rate_hz, nperseg)
    post_freqs, post_psd = compute_welch_psd(post_signal, sample_rate_hz, nperseg)

    pre_power = tremor_band_power(pre_freqs, pre_psd, tremor_band_hz)
    post_power = tremor_band_power(post_freqs, post_psd, tremor_band_hz)

    if pre_power < _NEAR_ZERO_POWER_THRESHOLD:
        logger.warning(
            "Pre-signal tremor-band power is near-zero (%.2e); "
            "suppression is not meaningful -- returning 0.0.",
            pre_power,
        )
        return 0.0

    raw_pct = (pre_power - post_power) / (pre_power + _POWER_FLOOR) * 100.0
    return float(np.clip(raw_pct, 0.0, 100.0))
