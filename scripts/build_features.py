"""Batch-process raw IMU recordings into feature vectors (Feature 20 applied
to every session under a raw-data directory).

Discovers every data/raw/<subject_id>/<session_id>/raw_stream.csv, runs the
Phase 3 signal-processing pipeline (calibration -> baseline removal ->
band-pass filter -> windowing -> feature extraction) on each, and writes:

  1. Per-session feature CSVs at data/features/<subject_id>/<session_id>/features.csv
     (matches architecture.md -> Storage).
  2. One consolidated features.csv (all sessions concatenated) for
     ml/dataset_builder.py -> load_dataset().

Usage:
    python scripts/build_features.py \\
        --input data/raw \\
        --output data/features/features.csv
"""

import argparse
from collections import defaultdict
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from ml.dataset_builder import build_feature_table
from signal_processing.axis_handling import compute_magnitude, select_strongest_axis
from signal_processing.calibration import apply_calibration, estimate_offsets, validate_offsets
from signal_processing.data_loader import load_raw_csv
from signal_processing.feature_extraction import extract_features
from signal_processing.filtering import bandpass_filter, remove_baseline
from signal_processing.windowing import segment_windows
from tremor_system.config import Config, load_config
from tremor_system.types import FeatureVector

logger = logging.getLogger(__name__)

# Session naming convention (confirmed): session_id ending in this suffix is
# the subject's stationary "hand resting in the glove" recording -- both the
# no-tremor label source (ml/dataset_builder.py::default_session_label) and
# the correct source recording for estimate_offsets(), which requires a
# stationary device. See estimate_subject_offsets() below.
CALIBRATION_SESSION_SUFFIX = "01"


def discover_raw_sessions(input_dir: Path) -> list[tuple[str, str, Path]]:
    """Find every raw_stream.csv under input_dir/<subject_id>/<session_id>/.

    Args:
        input_dir: Root of the raw-data tree (e.g. data/raw).

    Returns:
        Sorted list of (subject_id, session_id, csv_path) tuples.
    """
    sessions = []
    for csv_path in sorted(input_dir.glob("*/*/raw_stream.csv")):
        session_id = csv_path.parent.name
        subject_id = csv_path.parent.parent.name
        sessions.append((subject_id, session_id, csv_path))
    return sessions


def estimate_subject_offsets(
    subject_sessions: list[tuple[str, Path]],
    config: Config,
) -> Optional[NDArray[np.float64]]:
    """Estimate one offset vector per subject from their stationary session.

    session_id ending in CALIBRATION_SESSION_SUFFIX ("01") is the subject's
    stationary "hand resting in the glove" recording -- the correct source
    for estimate_offsets(), which requires the device to be stationary. The
    resulting offsets are reused for every session belonging to that
    subject (both "01" and "02"), matching the one-calibration-per-device
    design estimate_offsets()/validate_offsets() were built for.

    Args:
        subject_sessions: List of (session_id, csv_path) for one subject.
        config: Loaded system configuration.

    Returns:
        Offset vector, shape (6,), or None if no stationary session was
        found for this subject, or if it failed offset validation (caller
        should fall back to per-session zero-offset calibration and log a
        warning).
    """
    calibration_path = next(
        (path for session_id, path in subject_sessions if session_id.endswith(CALIBRATION_SESSION_SUFFIX)),
        None,
    )
    if calibration_path is None:
        return None

    _timestamps_us, raw_signal = load_raw_csv(calibration_path)
    try:
        offsets = estimate_offsets(raw_signal, config.sensor.accel_range_g)
        validate_offsets(offsets, config)
    except ValueError as exc:
        logger.warning(
            "Stationary calibration session %s failed offset validation "
            "(%s); this subject will fall back to per-session zero-offset "
            "calibration instead.",
            calibration_path,
            exc,
        )
        return None
    return offsets


def _calibrate_signal(
    raw_signal: NDArray[np.int16],
    config: Config,
    offsets: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Calibrate one session's raw signal.

    Prefers a precomputed per-subject offset (from estimate_subject_offsets(),
    derived from that subject's stationary recording). Falls back to
    per-session offset estimation, and then to zero offsets if that also
    fails validation, only when no stationary session is available for this
    subject.

    Args:
        raw_signal: shape (n_samples, 6), dtype int16, from load_raw_csv().
        config: Loaded system configuration.
        offsets: Precomputed offset vector for this subject, or None.

    Returns:
        Calibrated signal, shape (n_samples, 6), units g/deg-s.
    """
    if offsets is None:
        try:
            offsets = estimate_offsets(raw_signal, config.sensor.accel_range_g)
            validate_offsets(offsets, config)
        except ValueError as exc:
            logger.warning(
                "No stationary calibration source available and this "
                "session's own offsets failed validation (%s); falling "
                "back to zero offsets for this session.",
                exc,
            )
            offsets = np.zeros(6, dtype=np.float64)
    return apply_calibration(raw_signal, offsets, config)


def _select_analysis_signal(
    accel_signal: NDArray[np.float64],
    config: Config,
) -> NDArray[np.float64]:
    """Derive the single-channel tremor analysis signal per config.signal.axis_strategy.

    "per_axis" has no single-channel meaning for a scalar-per-window
    feature vector, so it falls back to strongest-axis selection -- the
    same choice already made by scripts/run_pipeline_baseline.py for the
    Milestone 2 verification script. Flagged as an assumption to confirm.

    Args:
        accel_signal: shape (n_samples, 3), calibrated accelerometer signal.
        config: Loaded system configuration.

    Returns:
        shape (n_samples,), the signal fed into filtering/windowing/features.
    """
    strategy = config.signal.axis_strategy
    tremor_band_hz = tuple(config.signal.tremor_band_hz)

    if strategy == "magnitude":
        return compute_magnitude(accel_signal)

    if strategy == "per_axis":
        logger.info(
            "axis_strategy='per_axis' has no single-channel meaning for "
            "dataset feature extraction; falling back to strongest-axis "
            "selection (see _select_analysis_signal docstring)."
        )
    elif strategy != "strongest_axis":
        raise ValueError(f"Unknown axis_strategy: {strategy!r}")

    if config.signal.filter_order is None:
        raise ValueError("signal.filter_order must be configured for strongest-axis selection")

    return select_strongest_axis(
        accel_signal,
        sample_rate_hz=config.sensor.sample_rate_hz,
        tremor_band_hz=tremor_band_hz,
        filter_order=config.signal.filter_order,
    )


def build_session_features(
    csv_path: Path,
    subject_id: str,
    session_id: str,
    config: Config,
    offsets: Optional[NDArray[np.float64]] = None,
) -> list[FeatureVector]:
    """Run the full Phase 3 pipeline on one session and return its feature vectors.

    Args:
        csv_path: Path to raw_stream.csv.
        subject_id: Subject identifier (from the parent-parent folder name).
        session_id: Session identifier (from the parent folder name).
        config: Loaded system configuration.
        offsets: Precomputed per-subject calibration offsets from
            estimate_subject_offsets(), or None to estimate per-session.

    Returns:
        One FeatureVector per complete analysis window (empty list if the
        recording is too short to contain a single complete window).
    """
    if config.signal.filter_order is None or config.signal.window_length_s is None:
        raise ValueError(
            "signal.filter_order and signal.window_length_s must be configured "
            "before building features"
        )

    _timestamps_us, raw_signal = load_raw_csv(csv_path)
    calibrated = _calibrate_signal(raw_signal, config, offsets=offsets)
    accel_signal = calibrated[:, :3]
    gyro_signal = calibrated[:, 3:6]

    sample_rate_hz = config.sensor.sample_rate_hz
    tremor_band_hz = tuple(config.signal.tremor_band_hz)

    analysis_signal = _select_analysis_signal(accel_signal, config)
    analysis_baseline_removed = remove_baseline(analysis_signal)
    analysis_filtered = bandpass_filter(
        analysis_baseline_removed,
        sample_rate_hz=sample_rate_hz,
        band_hz=tremor_band_hz,
        order=config.signal.filter_order,
    )

    window_length_s = config.signal.window_length_s
    overlap_pct = config.signal.window_overlap_pct

    analysis_windows = segment_windows(analysis_filtered, sample_rate_hz, window_length_s, overlap_pct)
    accel_windows = segment_windows(accel_signal, sample_rate_hz, window_length_s, overlap_pct)
    gyro_windows = segment_windows(gyro_signal, sample_rate_hz, window_length_s, overlap_pct)

    n_windows = analysis_windows.shape[0]
    if n_windows == 0:
        logger.warning(
            "%s/%s produced zero complete windows (recording shorter than "
            "one window); skipping",
            subject_id,
            session_id,
        )
        return []

    nperseg = analysis_windows.shape[1]
    records = []
    for window_id in range(n_windows):
        records.append(
            extract_features(
                analysis_windows[window_id],
                sample_rate_hz,
                accel_windows[window_id],
                gyro_windows[window_id],
                subject_id=subject_id,
                session_id=session_id,
                window_id=window_id,
                nperseg=nperseg,
            )
        )
    return records


def _feature_records_to_dicts(records: list[FeatureVector]) -> list[dict]:
    return [
        {
            "subject_id": fv.subject_id,
            "session_id": fv.session_id,
            "window_id": fv.window_id,
            **fv.to_dict(),
        }
        for fv in records
    ]


def run_build_features(input_dir: Path, output_path: Path) -> Path:
    """Process every raw session under input_dir and write a consolidated features.csv.

    Args:
        input_dir: Root of the raw-data tree (e.g. data/raw).
        output_path: Where to write the consolidated features.csv.

    Returns:
        output_path, on success.

    Raises:
        SystemExit: If no raw sessions or no feature windows were found.
    """
    config = load_config()
    sessions = discover_raw_sessions(input_dir)
    if not sessions:
        raise SystemExit(f"No raw_stream.csv files found under {input_dir}")

    logger.info("Discovered %d raw session(s) under %s", len(sessions), input_dir)

    sessions_by_subject: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for subject_id, session_id, csv_path in sessions:
        sessions_by_subject[subject_id].append((session_id, csv_path))

    subject_offsets = {
        subject_id: estimate_subject_offsets(subj_sessions, config)
        for subject_id, subj_sessions in sessions_by_subject.items()
    }
    for subject_id, offsets in subject_offsets.items():
        if offsets is None:
            logger.warning(
                "No usable stationary ('*%s') session found for %s; its "
                "sessions will fall back to per-session zero-offset "
                "calibration.",
                CALIBRATION_SESSION_SUFFIX,
                subject_id,
            )

    all_records: list[FeatureVector] = []
    for subject_id, session_id, csv_path in sessions:
        try:
            records = build_session_features(
                csv_path, subject_id, session_id, config, offsets=subject_offsets[subject_id]
            )
        except Exception as exc:  # noqa: BLE001 - keep processing remaining sessions
            logger.error("Failed to process %s/%s (%s): %s", subject_id, session_id, csv_path, exc)
            continue

        if not records:
            continue

        session_df = build_feature_table(_feature_records_to_dicts(records))
        session_output = input_dir.parent / "features" / subject_id / session_id / "features.csv"
        session_output.parent.mkdir(parents=True, exist_ok=True)
        session_df.to_csv(session_output, index=False)
        logger.info("Wrote %d window(s) to %s", len(records), session_output)

        all_records.extend(records)

    if not all_records:
        raise SystemExit("No feature windows were produced from any session")

    consolidated_df = build_feature_table(_feature_records_to_dicts(all_records))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_csv(output_path, index=False)
    logger.info(
        "Wrote consolidated dataset (%d windows across %d sessions) to %s",
        len(all_records),
        len(sessions),
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build features.csv from raw stream CSVs")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "features" / "features.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_build_features(args.input, args.output)


if __name__ == "__main__":
    main()
