"""Run the closed loop across many successive cycles, synthetic or recorded.

Features 42 and 43 of the build plan.

Feature 42 (--source synthetic, default): generates one long constant
frequency/amplitude synthetic tremor signal, slices it into consecutive
non-overlapping config.signal.window_length_s chunks, and steps
run_closed_loop_cycle() across them, logging ControllerState/StimParams/
SimResult per cycle.

Feature 43 (--source recorded --path <raw_stream.csv>): runs the full
Phase 3 pipeline (calibration -> strongest-axis selection) on a real
recorded session first, then steps the same closed loop across consecutive
windows of that real single-axis signal.

--source recorded now passes the session's real calibrated 3-axis
accel/gyro data through to run_closed_loop_cycle() (see
load_recorded_signal()), so accel_magnitude/gyro_magnitude features
reflect real sensor data, not a synthetic embedding. --source synthetic
still uses the synthetic-embedding fallback in
simulation/closed_loop_runner.py, since Feature 32 only generates a single
axis -- there is no real 3-axis motion to offer for synthetic input.

Usage:
    python scripts/run_closed_loop_simulation.py \\
        --source synthetic --frequency-hz 6.0 --amplitude 1.0 --n-cycles 30

    python scripts/run_closed_loop_simulation.py \\
        --source recorded --path data/raw/subj02/sess02/raw_stream.csv \\
        --n-cycles 30
"""

import argparse
import csv
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive, file-only backend -- avoids requiring
# a working Tcl/Tk installation (this script only ever saves figures to disk).
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from controller.controller_state import ControllerState
from signal_processing.axis_handling import select_strongest_axis
from signal_processing.calibration import apply_calibration, estimate_offsets, validate_offsets
from signal_processing.data_loader import load_raw_csv
from simulation.closed_loop_runner import run_closed_loop_cycle
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import Config, load_config

logger = logging.getLogger(__name__)

CYCLE_LOG_COLUMNS = [
    "cycle",
    "mitigate",
    "hysteresis_active",
    "achieved_suppression_pct",
    "stability_warning",
    "amplitude",
    "pulse_frequency_hz",
    "duty_cycle",
]


def slice_into_windows(
    analysis_signal: NDArray[np.float64],
    window_length_s: float,
    sample_rate_hz: float,
    accel_signal: NDArray[np.float64] | None = None,
    gyro_signal: NDArray[np.float64] | None = None,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None]]:
    """Slice signal(s) into consecutive, non-overlapping window_length_s chunks.

    Deliberately non-overlapping (unlike the 50%-overlap windowing used for
    ML training data): a real-time-style closed loop steps forward through
    new data each cycle, it does not re-process already-seen samples.

    Args:
        analysis_signal: shape (n_samples,), units g -- the dominant motion
            axis.
        window_length_s: Chunk duration in seconds.
        sample_rate_hz: Sampling rate in Hz.
        accel_signal: shape (n_samples, 3), units g, real calibrated data
            aligned with analysis_signal. None for synthetic input.
        gyro_signal: shape (n_samples, 3), units deg/s, real calibrated
            data aligned with analysis_signal. None for synthetic input.

    Returns:
        List of (analysis_chunk, accel_chunk, gyro_chunk) tuples; the
        latter two are None when accel_signal/gyro_signal were not
        provided. A trailing partial chunk (shorter than one window) is
        dropped.
    """
    window_samples = int(round(window_length_s * sample_rate_hz))
    n_chunks = analysis_signal.size // window_samples
    chunks = []
    for i in range(n_chunks):
        start, end = i * window_samples, (i + 1) * window_samples
        accel_chunk = accel_signal[start:end] if accel_signal is not None else None
        gyro_chunk = gyro_signal[start:end] if gyro_signal is not None else None
        chunks.append((analysis_signal[start:end], accel_chunk, gyro_chunk))
    return chunks


def load_synthetic_signal(
    args: argparse.Namespace, config: Config
) -> tuple[NDArray[np.float64], None, None]:
    total_duration_s = args.n_cycles * config.signal.window_length_s
    signal = generate_synthetic_tremor(
        frequency_hz=args.frequency_hz,
        amplitude=args.amplitude,
        duration_s=total_duration_s,
        sample_rate_hz=config.sensor.sample_rate_hz,
    )
    # No real 3-axis motion exists for synthetic input -- see
    # simulation/closed_loop_runner.py's synthetic-embedding fallback.
    return signal, None, None


def load_recorded_signal(
    args: argparse.Namespace, config: Config
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run calibration + strongest-axis selection on a real recorded session.

    Returns:
        (analysis_signal, accel_signal, gyro_signal) -- the dominant motion
        axis (for tremor-band feature extraction) plus the full real
        calibrated 3-axis accel/gyro data (for accel_magnitude/
        gyro_magnitude features), all aligned sample-for-sample.
    """
    _timestamps_us, raw_signal = load_raw_csv(args.path)
    try:
        offsets = estimate_offsets(raw_signal, config.sensor.accel_range_g)
        validate_offsets(offsets, config)
    except ValueError as exc:
        logger.warning(
            "Offset validation failed (%s); falling back to zero offsets for this session.",
            exc,
        )
        offsets = np.zeros(6, dtype=np.float64)
    calibrated = apply_calibration(raw_signal, offsets, config)
    accel_signal = calibrated[:, :3]
    gyro_signal = calibrated[:, 3:6]
    analysis_signal = select_strongest_axis(
        accel_signal,
        sample_rate_hz=config.sensor.sample_rate_hz,
        tremor_band_hz=tuple(config.signal.tremor_band_hz),
        filter_order=config.signal.filter_order,
    )
    return analysis_signal, accel_signal, gyro_signal


def run_multi_cycle_simulation(
    args: argparse.Namespace,
    config: Config,
    output_dir: Path,
) -> Path:
    """Run the closed loop across all available cycles and write log + plot.

    Args:
        args: Parsed CLI arguments.
        config: Loaded system configuration.
        output_dir: data/simulation/<experiment_id>/ directory to write into.

    Returns:
        Path to the written cycle_log.csv.
    """
    analysis_signal, accel_signal, gyro_signal = (
        load_recorded_signal(args, config)
        if args.source == "recorded"
        else load_synthetic_signal(args, config)
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

    state = ControllerState()
    rows = []
    for cycle_index, (analysis_chunk, accel_chunk, gyro_chunk) in enumerate(chunks):
        result, state = run_closed_loop_cycle(
            analysis_chunk,
            state,
            args.model_path,
            args.scaler_path,
            config=config,
            accel_signal=accel_chunk,
            gyro_signal=gyro_chunk,
        )
        rows.append(
            {
                "cycle": cycle_index,
                "mitigate": state.hysteresis_active,
                "hysteresis_active": state.hysteresis_active,
                "achieved_suppression_pct": result.achieved_suppression_pct,
                "stability_warning": result.stability_warning,
                "amplitude": state.current_params.amplitude if state.current_params else None,
                "pulse_frequency_hz": state.current_params.pulse_frequency_hz if state.current_params else None,
                "duty_cycle": state.current_params.duty_cycle if state.current_params else None,
            }
        )
        logger.info(
            "cycle=%d mitigate=%s suppression=%.1f%% amplitude=%s",
            cycle_index,
            state.hysteresis_active,
            result.achieved_suppression_pct,
            f"{rows[-1]['amplitude']:.3f}" if rows[-1]["amplitude"] is not None else "N/A",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "cycle_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CYCLE_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d cycle(s) to %s", len(rows), log_path)

    _plot_suppression_vs_cycle(rows, config, output_dir / "suppression_plot.png")

    return log_path


def _plot_suppression_vs_cycle(rows: list[dict], config: Config, output_path: Path) -> None:
    cycles = [row["cycle"] for row in rows]
    suppression = [row["achieved_suppression_pct"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(cycles, suppression, marker="o", markersize=3, label="achieved_suppression_pct")
    ax.axhline(
        config.controller.target_suppression_pct,
        color="red",
        linestyle="--",
        label=f"target ({config.controller.target_suppression_pct:.0f}%)",
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Suppression (%)")
    ax.set_title("Closed-loop suppression vs. cycle")
    ax.set_ylim(-5, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote suppression plot to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-cycle closed-loop simulation")
    parser.add_argument("--source", choices=["synthetic", "recorded"], default="synthetic")
    parser.add_argument("--path", type=Path, help="Required when --source recorded")
    parser.add_argument("--frequency-hz", type=float, default=6.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--n-cycles", type=int, default=30)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "data/models/model_logistic_regression_v1.pkl")
    parser.add_argument("--scaler-path", type=Path, default=PROJECT_ROOT / "data/models/scaler_v1.pkl")
    parser.add_argument("--experiment-id", type=str, default=None)
    args = parser.parse_args()

    if args.source == "recorded" and args.path is None:
        parser.error("--path is required when --source recorded")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config()
    experiment_id = args.experiment_id or f"{args.source}_run"
    output_dir = PROJECT_ROOT / "data" / "simulation" / experiment_id

    run_multi_cycle_simulation(args, config, output_dir)


if __name__ == "__main__":
    main()
