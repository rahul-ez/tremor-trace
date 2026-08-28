"""Tests for data_loader module (Feature 10)."""

import pytest
from pathlib import Path
import numpy as np

from signal_processing.data_loader import load_raw_csv


class TestLoadRawCsv:
    """Test suite for load_raw_csv function."""

    @pytest.fixture
    def test_csv_path(self) -> Path:
        """Path to the recorded test data."""
        return Path("data/raw/subj01/sess01/raw_stream.csv")

    def test_load_raw_csv_shape_and_dtype(self, test_csv_path: Path) -> None:
        """Test that loaded data has correct shape and dtype."""
        if not test_csv_path.exists():
            pytest.skip(f"Test data not found at {test_csv_path}")

        timestamps_us, raw_signal = load_raw_csv(test_csv_path)

        # Check shapes
        assert timestamps_us.ndim == 1, "timestamps_us should be 1D"
        assert raw_signal.ndim == 2, "raw_signal should be 2D"
        assert raw_signal.shape[1] == 6, "raw_signal should have 6 columns (ax,ay,az,gx,gy,gz)"
        assert len(timestamps_us) == raw_signal.shape[0], "lengths should match"

        # Check dtypes
        assert timestamps_us.dtype == np.int64, f"Expected int64, got {timestamps_us.dtype}"
        assert raw_signal.dtype == np.int16, f"Expected int16, got {raw_signal.dtype}"

    def test_load_raw_csv_monotonic_timestamps(self, test_csv_path: Path) -> None:
        """Test that timestamps are monotonically increasing."""
        if not test_csv_path.exists():
            pytest.skip(f"Test data not found at {test_csv_path}")

        timestamps_us, _ = load_raw_csv(test_csv_path)

        # Check monotonicity
        diffs = np.diff(timestamps_us)
        assert np.all(diffs > 0), "Timestamps should be strictly increasing"

    def test_load_raw_csv_expected_sample_count(self, test_csv_path: Path) -> None:
        """Test that sample count is reasonable for recorded data."""
        if not test_csv_path.exists():
            pytest.skip(f"Test data not found at {test_csv_path}")

        timestamps_us, raw_signal = load_raw_csv(test_csv_path)

        # Should have at least 10 samples
        assert len(timestamps_us) >= 10, f"Expected at least 10 samples, got {len(timestamps_us)}"
        print(f"Loaded {len(timestamps_us)} samples from {test_csv_path}")

    def test_load_raw_csv_plausible_values(self, test_csv_path: Path) -> None:
        """Test that raw values are within plausible LSB range."""
        if not test_csv_path.exists():
            pytest.skip(f"Test data not found at {test_csv_path}")

        _, raw_signal = load_raw_csv(test_csv_path)

        # For a stationary device:
        # - Accel z-axis should be near +16384 LSB (≈1g at ±2g range)
        # - Accel x,y should be near 0
        # - Gyro x,y,z should be near 0
        # Allow generous tolerance for movement, noise
        assert np.abs(raw_signal[:, 0]).mean() < 5000, "Accel x should be relatively small"
        assert np.abs(raw_signal[:, 1]).mean() < 5000, "Accel y should be relatively small"
        assert 10000 < raw_signal[:, 2].mean() < 23000, "Accel z should be around 1g (≈16384 LSB)"

    def test_load_raw_csv_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        nonexistent_path = Path("data/raw/nonexistent/file.csv")
        with pytest.raises(FileNotFoundError):
            load_raw_csv(nonexistent_path)

    def test_load_raw_csv_empty_file(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for empty CSV."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        with pytest.raises(ValueError, match="No valid samples"):
            load_raw_csv(empty_csv)

    def test_load_raw_csv_header_skip(self, tmp_path: Path) -> None:
        """Test that CSV header line (if present) is skipped or handled."""
        csv_with_header = tmp_path / "with_header.csv"
        csv_with_header.write_text(
            "timestamp_us,ax,ay,az,gx,gy,gz\n"
            "1000,100,200,300,10,20,30\n"
            "2000,101,201,301,11,21,31\n"
        )

        # The loader should skip the header line (it's not numeric, will be marked malformed)
        # and load the 2 data lines
        timestamps_us, raw_signal = load_raw_csv(csv_with_header)
        assert len(timestamps_us) == 2, "Should load 2 data lines, skipping header"
        assert timestamps_us[0] == 1000
        assert raw_signal[0, 0] == 100

    def test_load_raw_csv_skip_err_lines(self, tmp_path: Path) -> None:
        """Test that ERR marker lines are skipped."""
        csv_with_errors = tmp_path / "with_errors.csv"
        csv_with_errors.write_text(
            "1000,100,200,300,10,20,30\n"
            "ERR,I2C_READ_FAILED\n"
            "2000,101,201,301,11,21,31\n"
        )

        timestamps_us, raw_signal = load_raw_csv(csv_with_errors)
        assert len(timestamps_us) == 2, "Should skip ERR line and load 2 data lines"
        assert timestamps_us[0] == 1000
        assert timestamps_us[1] == 2000
