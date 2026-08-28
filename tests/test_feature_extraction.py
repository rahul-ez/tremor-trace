"""Tests for documented feature-vector assembly."""

import numpy as np
import pytest

from signal_processing.feature_extraction import extract_features


def test_extract_features_schema_and_magnitude_units() -> None:
    sample_rate_hz = 100.0
    time_s = np.arange(200, dtype=np.float64) / sample_rate_hz
    analysis_window = np.sin(2.0 * np.pi * 6.0 * time_s)
    accel_window = np.tile(np.array([3.0, 4.0, 0.0]), (200, 1))
    gyro_window = np.tile(np.array([0.0, 0.0, 5.0]), (200, 1))

    features = extract_features(
        analysis_window,
        sample_rate_hz,
        accel_window,
        gyro_window,
        subject_id="subj01",
        session_id="sess01",
        window_id=2,
    )

    assert features.subject_id == "subj01"
    assert features.session_id == "sess01"
    assert features.window_id == 2
    assert list(features.to_dict()) == [
        "tremor_band_power",
        "total_power",
        "power_ratio",
        "dominant_frequency_hz",
        "rms_amplitude",
        "variance",
        "spectral_entropy",
        "accel_magnitude",
        "gyro_magnitude",
    ]
    assert features.accel_magnitude == pytest.approx(5.0)
    assert features.gyro_magnitude == pytest.approx(5.0)
    assert 0.0 <= features.spectral_entropy <= 1.0


def test_extract_features_rejects_wrong_accelerometer_shape() -> None:
    analysis_window = np.ones(200, dtype=np.float64)
    invalid_accel_window = np.ones((200, 2), dtype=np.float64)
    gyro_window = np.ones((200, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="accel_window shape"):
        extract_features(analysis_window, 100.0, invalid_accel_window, gyro_window)