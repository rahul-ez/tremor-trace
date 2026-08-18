"""Axis representation strategies for multi-axis IMU signals.

Supports per-axis, 3-axis magnitude, and strongest-tremor-axis selection.
"""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_magnitude(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Euclidean magnitude across axes.

    For a multi-axis signal, computes the L2 norm (magnitude) at each time step.

    Args:
        signal: shape (n_samples, n_axes) or (n_samples,).
        If 1D, returns the signal unchanged (already single-axis).

    Returns:
        magnitude: shape (n_samples,), dtype float64.
        Units: same as input signal.

    Example:
        >>> import numpy as np
        >>> signal = np.array([[3.0, 4.0], [0.0, 5.0]])  # 2 samples, 2 axes
        >>> compute_magnitude(signal)
        array([5., 5.])
    """
    if signal.ndim == 1:
        return signal.astype(np.float64)
    elif signal.ndim == 2:
        # Compute L2 norm (Euclidean distance) across axes (axis=1)
        magnitude = np.linalg.norm(signal, axis=1)
        return magnitude.astype(np.float64)
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {signal.shape}")


def select_strongest_axis(
    signal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Select the axis with the highest tremor-band power.

    PLACEHOLDER: This function is completed in Feature 15 after filtering is available.

    Args:
        signal: shape (n_samples, 3), multi-axis calibrated signal.

    Returns:
        NotImplementedError: Always raised in Feature 12.

    Note:
        This stub will be completed in Feature 15 (Strongest-Axis Selection).
        It will require access to filtering and PSD computation from Features 14, 17, 18.
    """
    raise NotImplementedError(
        "select_strongest_axis() is completed in Feature 15 after "
        "filtering and PSD analysis are implemented. "
        "For now, use strategy='per_axis' or strategy='magnitude'."
    )


def get_axis_representation(
    signal: NDArray[np.float64],
    strategy: str,
) -> NDArray[np.float64]:
    """Select and return the desired axis representation of the signal.

    Dispatcher that routes multi-axis signals to per-axis, magnitude, or
    strongest-tremor-axis strategies.

    Args:
        signal: shape (n_samples, 3) for multi-axis (accel or gyro),
                or shape (n_samples, 6) if containing both accel and gyro
                (strategy will select which to use).
                dtype: float64, units: g (accel) or °/s (gyro).
        strategy: One of:
            - "per_axis": Return per-axis signals. If input is (n, 3), output is (n, 3);
                         if input is (n, 6), raise error (caller must pre-select accel/gyro).
            - "magnitude": Return 3-axis magnitude as (n_samples,).
                          If input is (n, 3), compute magnitude; if (n, 6), raise error.
            - "strongest_axis": Select the axis with highest tremor-band power.
                               (Feature 15 implementation; raises NotImplementedError in Feature 12.)

    Returns:
        signal: shape (n_samples,) for magnitude/strongest_axis,
                or (n_samples, 3) for per_axis.
                dtype: float64.

    Raises:
        ValueError: If strategy is unrecognized or signal shape is incompatible with strategy.
        NotImplementedError: If strategy is "strongest_axis" (Feature 15 placeholder).

    Example:
        >>> import numpy as np
        >>> accel_3axis = np.array([[0.1, 0.2, 0.9], [0.15, 0.25, 0.95]])
        >>> get_axis_representation(accel_3axis, "magnitude")
        array([0.91651513, 0.97089362])
        >>> get_axis_representation(accel_3axis, "per_axis").shape
        (2, 3)
    """
    if signal.ndim != 2:
        raise ValueError(
            f"Expected 2D signal array (n_samples, n_axes), got shape {signal.shape}"
        )

    if signal.shape[1] == 6:
        raise ValueError(
            "signal has 6 axes (accel + gyro). "
            "Caller must pre-select accel (axes 0-2) or gyro (axes 3-5) before calling. "
            "Example: get_axis_representation(signal[:, :3], strategy) for accel only."
        )

    if signal.shape[1] != 3:
        raise ValueError(
            f"Expected 3-axis signal (n_samples, 3), got shape {signal.shape}"
        )

    if strategy == "per_axis":
        logger.debug("Axis strategy: per_axis, returning (n_samples, 3)")
        return signal

    elif strategy == "magnitude":
        magnitude = compute_magnitude(signal)
        logger.debug("Axis strategy: magnitude, returning (n_samples,)")
        return magnitude

    elif strategy == "strongest_axis":
        # This will raise NotImplementedError (stub for Feature 15)
        return select_strongest_axis(signal)

    else:
        raise ValueError(
            f"Unknown axis strategy: '{strategy}'. "
            f"Valid options: 'per_axis', 'magnitude', 'strongest_axis'."
        )
