"""Tests for axis_handling module (Feature 12)."""

import pytest
import numpy as np
from numpy.typing import NDArray

from signal_processing.axis_handling import (
    compute_magnitude,
    get_axis_representation,
    select_strongest_axis,
)


class TestComputeMagnitude:
    """Test suite for compute_magnitude function."""

    def test_compute_magnitude_2d_array(self) -> None:
        """Test magnitude computation for 2D multi-axis signal."""
        # Simple test case: [3, 4] has magnitude 5; [0, 5] has magnitude 5
        signal = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float64)
        magnitude = compute_magnitude(signal)

        expected = np.array([5.0, 5.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(magnitude, expected)

    def test_compute_magnitude_3d_signal(self) -> None:
        """Test magnitude for 3-axis signal."""
        # 3-4-5 right triangle in first sample: sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.071
        signal = np.array([[3.0, 4.0, 5.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        magnitude = compute_magnitude(signal)

        expected = np.array([np.sqrt(50), 1.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(magnitude, expected)

    def test_compute_magnitude_1d_array(self) -> None:
        """Test that 1D array is returned unchanged."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        magnitude = compute_magnitude(signal)

        expected = signal
        np.testing.assert_array_almost_equal(magnitude, expected)

    def test_compute_magnitude_shape_preservation(self) -> None:
        """Test output shape for multi-axis input."""
        n_samples = 100
        n_axes = 3
        signal = np.random.randn(n_samples, n_axes).astype(np.float64)

        magnitude = compute_magnitude(signal)

        assert magnitude.shape == (n_samples,), f"Expected shape ({n_samples},), got {magnitude.shape}"

    def test_compute_magnitude_invalid_shape(self) -> None:
        """Test that ValueError is raised for 3D+ arrays."""
        signal_3d = np.random.randn(10, 3, 2).astype(np.float64)
        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            compute_magnitude(signal_3d)


class TestSelectStrongestAxis:
    """Test suite for select_strongest_axis function."""

    def test_select_strongest_axis_not_implemented(self) -> None:
        """Test that NotImplementedError is raised (Feature 15 stub)."""
        signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

        with pytest.raises(NotImplementedError, match="Feature 15"):
            select_strongest_axis(signal)


class TestGetAxisRepresentation:
    """Test suite for get_axis_representation dispatcher."""

    def test_get_axis_representation_per_axis_strategy(self) -> None:
        """Test per_axis strategy returns signal unchanged."""
        signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        result = get_axis_representation(signal, strategy="per_axis")

        np.testing.assert_array_equal(result, signal)
        assert result.shape == signal.shape

    def test_get_axis_representation_magnitude_strategy(self) -> None:
        """Test magnitude strategy returns 1D magnitude array."""
        signal = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64)
        result = get_axis_representation(signal, strategy="magnitude")

        expected = np.array([5.0, 5.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)
        assert result.shape == (2,), f"Expected 1D output, got shape {result.shape}"

    def test_get_axis_representation_strongest_axis_not_implemented(self) -> None:
        """Test strongest_axis strategy raises NotImplementedError."""
        signal = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

        with pytest.raises(NotImplementedError):
            get_axis_representation(signal, strategy="strongest_axis")

    def test_get_axis_representation_invalid_strategy(self) -> None:
        """Test that invalid strategy name raises ValueError."""
        signal = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="Unknown axis strategy"):
            get_axis_representation(signal, strategy="invalid_strategy")

    def test_get_axis_representation_rejects_6axis_signal(self) -> None:
        """Test that 6-axis input (accel + gyro) is rejected."""
        signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="6 axes"):
            get_axis_representation(signal, strategy="magnitude")

    def test_get_axis_representation_rejects_1d_input(self) -> None:
        """Test that 1D input is rejected."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with pytest.raises(ValueError, match="2D signal array"):
            get_axis_representation(signal, strategy="magnitude")

    def test_get_axis_representation_rejects_wrong_axis_count(self) -> None:
        """Test that non-3-axis input is rejected."""
        signal = np.array([[1.0, 2.0]], dtype=np.float64)  # 2 axes only

        with pytest.raises(ValueError, match="Expected 3-axis signal"):
            get_axis_representation(signal, strategy="magnitude")

    def test_get_axis_representation_large_signal(self) -> None:
        """Test on a large realistic signal."""
        n_samples = 10000
        # Simulate tremor-like signal: dominant in z-axis, low in x-y
        signal = np.random.randn(n_samples, 3).astype(np.float64)
        signal[:, 0] *= 0.1  # Low x-axis
        signal[:, 1] *= 0.1  # Low y-axis
        signal[:, 2] *= 1.0  # High z-axis

        # Test both strategies
        per_axis = get_axis_representation(signal, strategy="per_axis")
        magnitude = get_axis_representation(signal, strategy="magnitude")

        assert per_axis.shape == (n_samples, 3)
        assert magnitude.shape == (n_samples,)

        # Magnitude should be dominated by z-axis values
        # For a mostly 1D signal in z, magnitude ≈ |z|
        assert np.mean(magnitude) > np.mean(np.abs(signal[:, 0]))
        assert np.mean(magnitude) > np.mean(np.abs(signal[:, 1]))
