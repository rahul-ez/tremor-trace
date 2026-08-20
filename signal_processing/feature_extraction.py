"""Assembly of the documented per-window feature vector."""

import numpy as np
from numpy.typing import NDArray

from signal_processing.axis_handling import compute_magnitude
from signal_processing.spectral_analysis import (
    compute_welch_psd,
    dominant_frequency,
    power_ratio,
    spectral_entropy,
    total_power,
    tremor_band_power,
)
from signal_processing.time_domain import compute_rms, compute_variance
from tremor_system.config import load_config
from tremor_system.types import FeatureVector


def extract_features(
    analysis_window: NDArray[np.float64],
    sample_rate_hz: float,
    accel_window: NDArray[np.float64],
    gyro_window: NDArray[np.float64],
    subject_id: str = "",
    session_id: str = "",
    window_id: int = 0,
    nperseg: int = 200,
) -> FeatureVector:
    """Compute and assemble the documented features for one analysis window.

    Args:
        analysis_window: shape (n_samples,), filtered tremor signal used for
            spectral and time-domain features; units: g or deg/s.
        sample_rate_hz: Sampling rate in Hz.
        accel_window: shape (n_samples, 3), calibrated accelerometer window;
            units: g. Used only for the window-level magnitude feature.
        gyro_window: shape (n_samples, 3), calibrated gyroscope window;
            units: deg/s. Used only for the window-level magnitude feature.
        subject_id: Metadata identifying the subject.
        session_id: Metadata identifying the recording session.
        window_id: Metadata index of the window.
        nperseg: Welch segment length; v1 uses 200 samples.

    Returns:
        FeatureVector containing nine numeric features plus metadata.
    """
    if analysis_window.ndim != 1:
        raise ValueError(
            f"Expected analysis_window shape (n_samples,), got {analysis_window.shape}"
        )
    if accel_window.ndim != 2 or accel_window.shape[1] != 3:
        raise ValueError(
            f"Expected accel_window shape (n_samples, 3), got {accel_window.shape}"
        )
    if gyro_window.ndim != 2 or gyro_window.shape[1] != 3:
        raise ValueError(
            f"Expected gyro_window shape (n_samples, 3), got {gyro_window.shape}"
        )
    if accel_window.shape[0] != analysis_window.shape[0] or gyro_window.shape[0] != analysis_window.shape[0]:
        raise ValueError("All feature windows must have the same number of samples")

    config = load_config()
    tremor_band_hz = tuple(config.signal.tremor_band_hz)
    freqs_hz, psd = compute_welch_psd(analysis_window, sample_rate_hz, nperseg)
    tremor_power = tremor_band_power(freqs_hz, psd, tremor_band_hz)
    total_signal_power = total_power(freqs_hz, psd)

    return FeatureVector(
        subject_id=subject_id,
        session_id=session_id,
        window_id=window_id,
        tremor_band_power=tremor_power,
        total_power=total_signal_power,
        power_ratio=power_ratio(tremor_power, total_signal_power),
        dominant_frequency_hz=dominant_frequency(freqs_hz, psd, tremor_band_hz),
        rms_amplitude=compute_rms(analysis_window),
        variance=compute_variance(analysis_window),
        spectral_entropy=spectral_entropy(psd),
        accel_magnitude=float(np.mean(compute_magnitude(accel_window))),
        gyro_magnitude=float(np.mean(compute_magnitude(gyro_window))),
    )