"""Tests for calibration module (Feature 11)."""

import pytest
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from signal_processing.calibration import (
    estimate_offsets,
    validate_offsets,
    apply_calibration,
)
from signal_processing.data_loader import load_raw_csv
from tremor_system.config import load_config


class TestEstimateOffsets:
    """Test suite for estimate_offsets function."""

    def test_estimate_offsets_shape(self) -> None:
        """Test that estimated offsets have correct shape."""
        raw_signal = np.array([[100, 200, 16384, 10, 20, 30]], dtype=np.int16)
        offsets = estimate_offsets(raw_signal)

        assert offsets.shape == (6,), f"Expected shape (6,), got {offsets.shape}"
        assert offsets.dtype == np.float64, f"Expected float64, got {offsets.dtype}"

    def test_estimate_offsets_mean_value(self) -> None:
        """Test that offsets are computed as per-axis mean."""
        raw_signal = np.array(
            [
                [100, 200, 16384 + 300, 10, 20, 30],
                [200, 300, 16384 + 400, 20, 30, 40],
                [300, 400, 16384 + 500, 30, 40, 50],
            ],
            dtype=np.int16,
        )
        offsets = estimate_offsets(raw_signal)

        expected = np.array([200, 300, 400, 20, 30, 40], dtype=np.float64)
        np.testing.assert_array_almost_equal(offsets, expected)

    def test_estimate_offsets_on_recorded_data(self) -> None:
        """Test offset estimation on actual recorded stationary data."""
        csv_path = Path("data/raw/subj01/sess01/raw_stream.csv")
        if not csv_path.exists():
            pytest.skip(f"Test data not found at {csv_path}")

        timestamps_us, raw_signal = load_raw_csv(csv_path)
        offsets = estimate_offsets(raw_signal)

        # The z offset excludes +1g gravity, so the corrected signal retains +1g.
        assert abs(offsets[2]) < 6000, f"Accel z excess offset should be near 0 LSB, got {offsets[2]}"

        # Gyro offsets should be small (close to 0), allow some tolerance for real sensor bias
        assert abs(offsets[3]) < 1000, f"Gyro x offset should be small, got {offsets[3]}"
        assert abs(offsets[4]) < 1000, f"Gyro y offset should be small, got {offsets[4]}"
        assert abs(offsets[5]) < 1000, f"Gyro z offset should be small, got {offsets[5]}"


class TestValidateOffsets:
    """Test suite for validate_offsets function."""

    @pytest.fixture
    def config(self):
        """Load default config."""
        return load_config()

    def test_validate_offsets_good_orientation(self, config) -> None:
        """Test validation passes for device lying flat (z-axis up)."""
        # The estimated z offset excludes the expected +1g gravity component.
        good_offsets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Should not raise
        try:
            validate_offsets(good_offsets, config)
        except ValueError as e:
            pytest.fail(f"Validation unexpectedly failed: {e}")

    def test_validate_offsets_bad_z_axis(self, config) -> None:
        """Test validation fails when z-axis offset doesn't match expected gravity."""
        # z-axis excess offset outside the default tolerance
        bad_offsets = np.array([0.0, 0.0, 10000.0, 0.0, 0.0, 0.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Accel z-axis offset validation failed"):
            validate_offsets(bad_offsets, config)

    def test_validate_offsets_shape_check(self, config) -> None:
        """Test that ValueError is raised for wrong-shaped offset array."""
        wrong_shape = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 5 elements

        with pytest.raises(ValueError, match="Expected shape"):
            validate_offsets(wrong_shape, config)

    def test_validate_offsets_on_recorded_data(self, config) -> None:
        """Test validation on actual recorded stationary data."""
        csv_path = Path("data/raw/subj01/sess01/raw_stream.csv")
        if not csv_path.exists():
            pytest.skip(f"Test data not found at {csv_path}")

        timestamps_us, raw_signal = load_raw_csv(csv_path)
        offsets = estimate_offsets(raw_signal)

        # Should not raise (validation should pass for properly recorded stationary data)
        try:
            validate_offsets(offsets, config)
        except ValueError as e:
            pytest.fail(f"Validation unexpectedly failed: {e}")


class TestApplyCalibration:
    """Test suite for apply_calibration function."""

    @pytest.fixture
    def config(self):
        """Load default config."""
        return load_config()

    def test_apply_calibration_shape_and_dtype(self, config) -> None:
        """Test that calibrated signal has correct shape and dtype."""
        raw_signal = np.array([[100, 200, 16384, 10, 20, 30]], dtype=np.int16)
        offsets = np.array([100.0, 200.0, 16384.0, 10.0, 20.0, 30.0], dtype=np.float64)

        calibrated = apply_calibration(raw_signal, offsets, config)

        assert calibrated.shape == raw_signal.shape, "Shape should be preserved"
        assert calibrated.dtype == np.float64, f"Expected float64, got {calibrated.dtype}"

    def test_apply_calibration_zero_offset(self, config) -> None:
        """Test calibration with zero offsets (no offset subtraction)."""
        raw_signal = np.array([[100, 200, 16384, 10, 20, 30]], dtype=np.int16)
        offsets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        calibrated = apply_calibration(raw_signal, offsets, config)

        # With zero offset, calibrated = raw * scale
        # For accel_range_g = 2, accel_scale = 2 / 32768 ≈ 0.0000611
        # For gyro_range_dps = 250, gyro_scale = 250 / 32768 ≈ 0.00763
        accel_scale = config.sensor.accel_range_g / 32768.0
        gyro_scale = config.sensor.gyro_range_dps / 32768.0

        expected = np.array(
            [
                [100 * accel_scale, 200 * accel_scale, 16384 * accel_scale, 
                 10 * gyro_scale, 20 * gyro_scale, 30 * gyro_scale]
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(calibrated, expected, decimal=6)

    def test_apply_calibration_on_recorded_data(self, config) -> None:
        """Test full calibration pipeline on recorded data."""
        csv_path = Path("data/raw/subj01/sess01/raw_stream.csv")
        if not csv_path.exists():
            pytest.skip(f"Test data not found at {csv_path}")

        timestamps_us, raw_signal = load_raw_csv(csv_path)
        offsets = estimate_offsets(raw_signal)
        validate_offsets(offsets, config)

        calibrated = apply_calibration(raw_signal, offsets, config)

        # Calibration preserves the +1g gravity component for Feature 13.
        accel_z_calibrated = calibrated[:, 2]
        mean_accel_z = np.mean(accel_z_calibrated)
        assert 0.95 < mean_accel_z < 1.05, f"Expected accel z ≈ +1g, got {mean_accel_z:.3f}g"

        # Accel x/y should be close to 0 (horizontal axes)
        accel_x_calibrated = calibrated[:, 0]
        accel_y_calibrated = calibrated[:, 1]
        mean_accel_x = np.mean(np.abs(accel_x_calibrated))
        mean_accel_y = np.mean(np.abs(accel_y_calibrated))
        assert mean_accel_x < 0.2, f"Expected accel x ≈ 0g, got {mean_accel_x:.3f}g"
        assert mean_accel_y < 0.2, f"Expected accel y ≈ 0g, got {mean_accel_y:.3f}g"

        # Gyro should be close to 0 (at rest, no rotation)
        gyro_calibrated = calibrated[:, 3:6]
        mean_gyro = np.mean(np.abs(gyro_calibrated))
        assert mean_gyro < 10.0, f"Expected gyro ≈ 0 deg/s, got {mean_gyro:.3f} deg/s"

    @pytest.mark.parametrize(
        "raw_signal",
        [
            np.zeros(6, dtype=np.int16),
            np.zeros((2, 3, 2), dtype=np.int16),
            np.zeros((2, 5), dtype=np.int16),
        ],
        ids=["one_dimensional", "three_dimensional", "wrong_column_count"],
    )
    def test_apply_calibration_rejects_invalid_shapes(self, config, raw_signal: NDArray[np.int16]) -> None:
        offsets = np.zeros(6, dtype=np.float64)

        with pytest.raises(ValueError, match=r"Expected raw_signal shape \(n_samples, 6\)"):
            apply_calibration(raw_signal, offsets, config)
