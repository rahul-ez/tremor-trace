"""Axis representation strategies for multi-axis IMU signals.

Supports per-axis, 3-axis magnitude, and strongest-tremor-axis selection.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from signal_processing.filtering import bandpass_filter
from tremor_system.config import load_config

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
    sample_rate_hz: float | None = None,
    tremor_band_hz: tuple[float, float] | None = None,
    filter_order: int | None = None,
) -> NDArray[np.float64]:
    """Select the axis with the highest tremor-band power.

    Args:
        signal: shape (n_samples, 3), multi-axis calibrated signal.
        sample_rate_hz: Sampling rate in Hz. Defaults to system configuration.
        tremor_band_hz: Tremor pass band in Hz. Defaults to system configuration.
        filter_order: Butterworth order. Defaults to system configuration.

    Returns:
        The selected original calibrated axis, shape (n_samples,), with the
        same units as ``signal``.

    Note:
        Each axis is filtered in the tremor band and ranked by mean-square
        filtered power. The filtered signals are used only for ranking; the
        returned data is the original calibrated axis.
    """
    if signal.ndim != 2 or signal.shape[1] != 3:
        raise ValueError(f"Expected signal shape (n_samples, 3), got {signal.shape}")

    config = load_config()
    resolved_sample_rate = (
        config.sensor.sample_rate_hz if sample_rate_hz is None else sample_rate_hz
    )
    resolved_band = (
        tuple(config.signal.tremor_band_hz)
        if tremor_band_hz is None
        else tremor_band_hz
    )
    resolved_order = (
        config.signal.filter_order if filter_order is None else filter_order
    )
    if resolved_order is None:
        raise ValueError("filter_order must be configured for strongest-axis selection")

    filtered = bandpass_filter(
        signal,
        sample_rate_hz=resolved_sample_rate,
        band_hz=resolved_band,
        order=resolved_order,
    )
    axis_power = np.mean(np.square(filtered), axis=0)
    strongest_axis_index = int(np.argmax(axis_power))
    logger.debug(
        "Selected strongest tremor axis %d with powers %s",
        strongest_axis_index,
        axis_power,
    )
    return np.asarray(signal[:, strongest_axis_index], dtype=np.float64)


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
