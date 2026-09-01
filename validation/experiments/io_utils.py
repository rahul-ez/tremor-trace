"""Shared CLI arguments and input-signal loading for validation experiments.

Reuses scripts/run_closed_loop_simulation.py's loading/slicing functions
(Feature 42/43) rather than duplicating axis-selection/calibration logic --
Features 44-46 need exactly the same "synthetic or recorded, sliced into
consecutive windows" input as Feature 42's multi-cycle runner.
"""

import argparse
from pathlib import Path

from scripts.run_closed_loop_simulation import (
    load_recorded_signal,
    load_synthetic_signal,
    slice_into_windows,
)
from tremor_system.config import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the input-source/model/run-id arguments shared by all three experiments."""
    parser.add_argument("--source", choices=["synthetic", "recorded"], default="synthetic")
    parser.add_argument("--path", type=Path, help="Required when --source recorded")
    parser.add_argument("--frequency-hz", type=float, default=6.0)
    parser.add_argument("--amplitude", type=float, default=1.5)
    parser.add_argument("--n-cycles", type=int, default=30)
    parser.add_argument(
        "--model-path", type=Path, default=PROJECT_ROOT / "data/models/model_logistic_regression_v1.pkl"
    )
    parser.add_argument("--scaler-path", type=Path, default=PROJECT_ROOT / "data/models/scaler_v1.pkl")
    parser.add_argument("--run-id", type=str, default=None, help="Defaults to a UTC timestamp")


def load_input_signal(args: argparse.Namespace, config: Config) -> list:
    """Load and slice the requested input signal, per --source.

    Args:
        args: Parsed CLI args (from a parser that called add_common_args()).
        config: Loaded system configuration.

    Returns:
        List of (analysis_chunk, accel_chunk, gyro_chunk) tuples, one per
        cycle (see scripts/run_closed_loop_simulation.py::slice_into_windows).

    Raises:
        SystemExit: If --source recorded was chosen without --path, or the
            input is too short to produce a single window.
    """
    if args.source == "recorded" and args.path is None:
        raise SystemExit("--path is required when --source recorded")

    analysis_signal, accel_signal, gyro_signal = (
        load_recorded_signal(args, config) if args.source == "recorded" else load_synthetic_signal(args, config)
    )
    chunks = slice_into_windows(
        analysis_signal,
        config.signal.window_length_s,
        config.sensor.sample_rate_hz,
        accel_signal=accel_signal,
        gyro_signal=gyro_signal,
    )
    if args.n_cycles is not None:
        chunks = chunks[: args.n_cycles]
    if not chunks:
        raise SystemExit(f"Input signal too short to produce a single {config.signal.window_length_s}s window")
    return chunks
