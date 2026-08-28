"""Sensor calibration: offset estimation and unit conversion.

Handles stationary-offset estimation and conversion from raw LSB values to physical units.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from tremor_system.config import Config

logger = logging.getLogger(__name__)

def estimate_offsets(
    raw_signal: NDArray[np.int16],
    accel_range_g: float = 2.0,
) -> NDArray[np.float64]:
    """Estimate per-axis offsets from a stationary recording.

    For a device at rest (stationary), the raw signal should be approximately constant
    on each axis. Accelerometer x/y and gyroscope offsets are the stationary means.
    The accelerometer z offset excludes the expected +1g gravity component so that
    calibration preserves gravity for the later baseline-removal stage.

    Preconditions:
        The device must be lying flat with the z-axis vertical during the calibration
        recording. This is a hard requirement for separating the z-axis gravity
        component from the z-axis sensor offset.

    Gyro offset (axes gx, gy, gz): expected ~0 °/s at rest (all axes).
    Accel offset (axes ax, ay, az): expected 0g on horizontal (x, y), +1g on vertical
    (z, up) after calibration. Raw accel values are still in LSB at this stage.

    Args:
        raw_signal: shape (n_samples, 6), dtype int16, axis order [ax,ay,az,gx,gy,gz] in LSB.
        accel_range_g: Configured accelerometer full-scale range in g.

    Returns:
        offsets: shape (6,), dtype float64, per-axis mean offset in LSB.
        Axis order: [ax_offset, ay_offset, az_offset, gx_offset, gy_offset, gz_offset].
    """
    if raw_signal.ndim != 2 or raw_signal.shape[1] != 6:
        raise ValueError(f"Expected shape (n_samples, 6), got {raw_signal.shape}")
    if accel_range_g <= 0:
        raise ValueError(f"Expected positive accel_range_g, got {accel_range_g}")

    offsets = np.mean(raw_signal, axis=0, dtype=np.float64)
    sensitivity_lsb_per_g = 32768.0 / accel_range_g
    offsets[2] -= sensitivity_lsb_per_g
    logger.debug("Estimated offsets (LSB): %s", offsets)
    return offsets


def validate_offsets(
    offsets: NDArray[np.float64],
    config: Config,
    tolerance_lsb: float = 6000.0,
) -> None:
    """Validate that estimated offsets match expected device orientation.

        Expects:
        - Accel z-axis sensor offset ≈ 0 after the expected +1g gravity component is removed
            from the estimated offset (device lying flat, z-axis up).
    - Accel x/y-axis offsets ≈ 0 (horizontal axes, no gravity component).
    - Gyro x/y/z-axis offsets ≈ 0 (at rest, no rotation).

    Args:
        offsets: shape (6,), dtype float64, from estimate_offsets().
        config: Config object with sensor.accel_range_g.
        tolerance_lsb: Tolerance band around expected offset (default 6000 LSB ≈ 0.37g at ±2g range).

    Raises:
        ValueError: If any offset is significantly out of expected range.

    Note:
        The device orientation assumption is: z-axis pointing upward (perpendicular to ground).
        If calibration is performed with a different orientation, adjust expected values accordingly.
        The tolerance accounts for real-world sensor variations and minor orientation differences.
    """
    if offsets.shape != (6,):
        raise ValueError(f"Expected shape (6,), got {offsets.shape}")

    accel_range_g = config.sensor.accel_range_g
    # Compute LSB per g for the configured accel range
    # MPU6050 accel register: ±32768 LSB represents ±accel_range_g
    # So: 1g = 32768 / accel_range_g LSB
    # For accel_range_g = 2.0: 1g = 16384 LSB
    lsb_per_g = 32768.0 / accel_range_g

    # The estimated z offset excludes the expected +1g gravity component.
    expected_az_offset = 0.0
    actual_az_offset = offsets[2]
    az_error = abs(actual_az_offset - expected_az_offset)

    if az_error > tolerance_lsb:
        raise ValueError(
            f"Accel z-axis offset validation failed. "
            f"Expected excess offset {expected_az_offset:.0f} ± {tolerance_lsb:.0f} LSB, got {actual_az_offset:.0f} LSB. "
            f"Error: {az_error:.0f} LSB (~{az_error / lsb_per_g:.3f}g). "
            f"Device may not be lying flat (z-axis up) during calibration. "
            f"Check device orientation and re-run calibration."
        )

    # Accel x/y offsets should be near 0 (horizontal axes, no gravity)
    for axis_idx, axis_name in [(0, "x"), (1, "y")]:
        actual_offset = offsets[axis_idx]
        if abs(actual_offset) > tolerance_lsb:
            logger.warning(
                "Accel %s-axis offset is %.0f LSB (expected ~0). "
                "Device may be tilted during calibration.",
                axis_name,
                actual_offset,
            )

    # Gyro offsets should all be near 0 (at rest, no rotation)
    for axis_idx, axis_name in [(3, "x"), (4, "y"), (5, "z")]:
        actual_offset = offsets[axis_idx]
        if abs(actual_offset) > tolerance_lsb:
            logger.warning(
                "Gyro %s-axis offset is %.0f LSB (expected ~0). "
                "Device may have been moving during calibration.",
                axis_name,
                actual_offset,
            )

    logger.info("Offset validation passed.")


def apply_calibration(
    raw_signal: NDArray[np.int16],
    offsets: NDArray[np.float64],
    config: Config,
) -> NDArray[np.float64]:
    """Convert raw LSB values to physical units (g for accel, °/s for gyro).

    Args:
        raw_signal: shape (n_samples, 6), dtype int16, axis order [ax,ay,az,gx,gy,gz] in LSB.
        offsets: shape (6,), dtype float64, per-axis offset in LSB (from estimate_offsets).
        config: Config object with sensor range configurations.

    Returns:
        calibrated_signal: shape (n_samples, 6), dtype float64.
        Axis order: [ax,ay,az,gx,gy,gz] in physical units (g, g, g, °/s, °/s, °/s).

    Note:
        Units: accelerometer in g, gyroscope in °/s.
        Raw accel z-axis includes the constant gravity component (≈1g when device is flat).
    """
    if raw_signal.ndim != 2 or raw_signal.shape[1] != 6:
        raise ValueError(f"Expected raw_signal shape (n_samples, 6), got {raw_signal.shape}")

    if offsets.shape != (6,):
        raise ValueError(f"Expected offsets shape (6,), got {offsets.shape}")

    # Convert raw signal to float and subtract offsets
    calibrated = np.array(raw_signal, dtype=np.float64) - offsets

    # Compute conversion factors based on configured ranges
    accel_range_g = config.sensor.accel_range_g
    gyro_range_dps = config.sensor.gyro_range_dps

    # Full register range for MPU6050: ±32768 LSB
    accel_scale = accel_range_g / 32768.0  # g per LSB
    gyro_scale = gyro_range_dps / 32768.0  # °/s per LSB

    # Apply scale to axes
    # Accel axes (0, 1, 2): convert from LSB to g
    calibrated[:, :3] *= accel_scale
    # Gyro axes (3, 4, 5): convert from LSB to °/s
    calibrated[:, 3:6] *= gyro_scale

    logger.debug(
        "Applied calibration: accel_scale=%.6f g/LSB, gyro_scale=%.6f deg/s/LSB",
        accel_scale,
        gyro_scale,
    )

    return calibrated
