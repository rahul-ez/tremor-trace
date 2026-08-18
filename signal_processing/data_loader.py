"""Raw IMU data loader for offline signal processing.

Loads recorded ESP32 CSV streams and returns NumPy arrays in documented shape/dtype conventions.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_raw_csv(csv_path: Path) -> tuple[NDArray[np.int64], NDArray[np.int16]]:
    """Load raw IMU samples from a recorded ESP32 CSV stream.

    Expected CSV format (no header):
        timestamp_us,ax,ay,az,gx,gy,gz
    
    where timestamp_us is a microsecond counter (int), and ax/ay/az/gx/gy/gz
    are raw int16 LSB values (unconverted).

    Args:
        csv_path: Path to raw_stream.csv file.

    Returns:
        (timestamps_us, raw_signal) tuple:
            timestamps_us: shape (n_samples,), dtype int64, monotonic microsecond timestamps.
            raw_signal: shape (n_samples, 6), dtype int16, axis order [ax,ay,az,gx,gy,gz] in LSB.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is empty or all lines are malformed.

    Note:
        Malformed lines (incorrect field count, non-numeric fields) are silently skipped
        with a warning log entry. If a line is in the data stream, the gap in timestamp_us
        will be detectable downstream.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV file not found at {csv_path}")

    timestamps = []
    raw_samples = []
    malformed_count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Skip error marker lines (e.g., "ERR,I2C_READ_FAILED")
            if line.startswith("ERR,"):
                logger.warning("Line %d: error marker skipped: %s", line_num, line)
                malformed_count += 1
                continue

            try:
                fields = line.split(",")
                if len(fields) != 7:
                    raise ValueError(f"Expected 7 fields, got {len(fields)}")

                timestamp_us = int(fields[0])
                ax, ay, az = int(fields[1]), int(fields[2]), int(fields[3])
                gx, gy, gz = int(fields[4]), int(fields[5]), int(fields[6])

                timestamps.append(timestamp_us)
                raw_samples.append([ax, ay, az, gx, gy, gz])

            except (ValueError, IndexError) as e:
                logger.warning("Line %d: malformed, skipped: %s (error: %s)", line_num, line, e)
                malformed_count += 1
                continue

    if not raw_samples:
        raise ValueError(
            f"No valid samples loaded from {csv_path}. "
            f"Malformed lines: {malformed_count}. Check CSV format."
        )

    if malformed_count > 0:
        logger.warning(
            "Loaded %d valid samples from %s (%d malformed lines skipped)",
            len(raw_samples),
            csv_path,
            malformed_count,
        )
    else:
        logger.info("Loaded %d samples from %s", len(raw_samples), csv_path)

    timestamps_us = np.array(timestamps, dtype=np.int64)
    raw_signal = np.array(raw_samples, dtype=np.int16)

    return timestamps_us, raw_signal
