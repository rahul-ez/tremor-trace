"""Overlapping fixed-length windows for calibrated or filtered signals."""

import numpy as np
from numpy.typing import NDArray


def segment_windows(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    window_length_s: float,
    overlap_pct: float,
) -> NDArray[np.float64]:
    """Segment a signal into overlapping complete windows.

    Args:
        signal: shape (n_samples,) or (n_samples, n_axes), units preserved.
        sample_rate_hz: Sampling rate in Hz.
        window_length_s: Window duration in seconds.
        overlap_pct: Percentage of each window shared with the next window.

    Returns:
        Windows with shape (n_windows, window_samples) for 1D input or
        (n_windows, window_samples, n_axes) for 2D input. Incomplete trailing
        samples are discarded rather than padded.
    """
    if signal.ndim not in (1, 2):
        raise ValueError(
            f"Expected 1D or 2D signal array, got shape {signal.shape}"
        )
    if signal.shape[0] == 0:
        raise ValueError("Expected signal with at least one sample")
    if sample_rate_hz <= 0:
        raise ValueError(f"Expected positive sample_rate_hz, got {sample_rate_hz}")
    if window_length_s <= 0:
        raise ValueError(
            f"Expected positive window_length_s, got {window_length_s}"
        )
    if not 0 <= overlap_pct < 100:
        raise ValueError(
            f"Expected overlap_pct in [0, 100), got {overlap_pct}"
        )

    window_samples = int(round(sample_rate_hz * window_length_s))
    overlap_samples = int(round(window_samples * overlap_pct / 100.0))
    step_samples = window_samples - overlap_samples
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Window length and step must be positive whole samples")

    n_samples = signal.shape[0]
    if n_samples < window_samples:
        output_shape = (
            0,
            window_samples,
        ) if signal.ndim == 1 else (
            0,
            window_samples,
            signal.shape[1],
        )
        return np.empty(output_shape, dtype=np.float64)

    n_windows = (n_samples - window_samples) // step_samples + 1
    windows = np.stack(
        [
            signal[start : start + window_samples]
            for start in range(0, n_windows * step_samples, step_samples)
        ],
        axis=0,
    )
    return np.asarray(windows, dtype=np.float64)