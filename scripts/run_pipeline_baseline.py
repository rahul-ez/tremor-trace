"""Run and visualize the Phase 3 signal-processing baseline pipeline."""
# Usage : python scripts/run_pipeline_baseline.py \
#  --input data/raw/subj01/sess01/raw_stream.csv \
# --output data/processed/subj01/sess01/pipeline_baseline.png \
#  --subject-id subj01 \
#  --session-id sess01 \
#  --window-index 0

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from signal_processing.axis_handling import select_strongest_axis
from signal_processing.calibration import apply_calibration, estimate_offsets, validate_offsets
from signal_processing.data_loader import load_raw_csv
from signal_processing.filtering import bandpass_filter, remove_baseline
from signal_processing.spectral_analysis import compute_welch_psd, dominant_frequency
from signal_processing.windowing import segment_windows
from tremor_system.config import load_config

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path,
    output_path: Path,
    subject_id: str,
    session_id: str,
    window_index: int,
) -> Path:
    """Process one recorded session and save the baseline verification plot.

    Args:
        input_path: Raw CSV path with timestamp and six raw IMU channels.
        output_path: PNG path for the generated verification plot.
        subject_id: Subject label used in the plot title.
        session_id: Session label used in the plot title.
        window_index: Complete window index to use for the PSD panel.

    Returns:
        The saved output path.
    """
    config = load_config()
    timestamps_us, raw_signal = load_raw_csv(input_path)
    offsets = estimate_offsets(raw_signal, config.sensor.accel_range_g)
    validate_offsets(offsets, config)
    calibrated_signal = apply_calibration(raw_signal, offsets, config)

    accel_signal = calibrated_signal[:, :3]
    sample_rate_hz = config.sensor.sample_rate_hz
    tremor_band_hz = tuple(config.signal.tremor_band_hz)
    filter_order = config.signal.filter_order
    window_length_s = config.signal.window_length_s
    if filter_order is None or window_length_s is None:
        raise ValueError(
            "signal.filter_order and signal.window_length_s must be configured "
            "before running the baseline pipeline"
        )

    baseline_removed = remove_baseline(accel_signal)
    filtered_accel = bandpass_filter(
        baseline_removed,
        sample_rate_hz=sample_rate_hz,
        band_hz=tremor_band_hz,
        order=filter_order,
    )
    filtered_windows = segment_windows(
        filtered_accel,
        sample_rate_hz=sample_rate_hz,
        window_length_s=window_length_s,
        overlap_pct=config.signal.window_overlap_pct,
    )
    if filtered_windows.shape[0] == 0:
        raise ValueError("Recording does not contain a complete analysis window")
    if not 0 <= window_index < filtered_windows.shape[0]:
        raise ValueError(
            f"window_index must be in [0, {filtered_windows.shape[0] - 1}], "
            f"got {window_index}"
        )

    strongest_axis = select_strongest_axis(
        accel_signal,
        sample_rate_hz=sample_rate_hz,
        tremor_band_hz=tremor_band_hz,
        filter_order=filter_order,
    )
    strongest_axis_baseline_removed = remove_baseline(strongest_axis)
    strongest_axis_filtered = bandpass_filter(
        strongest_axis_baseline_removed,
        sample_rate_hz=sample_rate_hz,
        band_hz=tremor_band_hz,
        order=filter_order,
    )
    window_samples = filtered_windows.shape[1]
    start_sample = window_index * int(
        round(window_samples * (1.0 - config.signal.window_overlap_pct / 100.0))
    )
    spectrum_window = strongest_axis_filtered[start_sample : start_sample + window_samples]
    freqs_hz, psd = compute_welch_psd(
        spectrum_window,
        sample_rate_hz=sample_rate_hz,
        nperseg=window_samples,
    )
    peak_frequency_hz = dominant_frequency(freqs_hz, psd, tremor_band_hz)

    time_s = (timestamps_us - timestamps_us[0]) / 1_000_000.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(time_s, calibrated_signal[:, 0], label="Accel X calibrated", alpha=0.65)
    axes[0].plot(time_s, calibrated_signal[:, 1], label="Accel Y calibrated", alpha=0.65)
    axes[0].plot(time_s, calibrated_signal[:, 2], label="Accel Z calibrated", alpha=0.65)
    axes[0].plot(
        time_s,
        strongest_axis_filtered,
        label="Strongest axis tremor-band filtered",
        linewidth=1.0,
    )
    axes[0].set_title("Calibrated acceleration and filtered tremor signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs_hz, psd, label="Welch PSD")
    axes[1].axvspan(
        tremor_band_hz[0], tremor_band_hz[1],
        alpha=0.2,
        label=f"Tremor band ({tremor_band_hz[0]:g}-{tremor_band_hz[1]:g} Hz)",
    )
    axes[1].axvline(
        peak_frequency_hz,
        color="tab:red",
        linestyle="--",
        label=f"Dominant: {peak_frequency_hz:.2f} Hz",
    )
    axes[1].set_title(f"Welch PSD, window {window_index}")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power spectral density")
    axes[1].set_xlim(0.0, sample_rate_hz / 2.0)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(f"Signal-processing baseline: {subject_id}/{session_id}")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    logger.info("Saved baseline plot to %s", output_path)
    logger.info("Dominant tremor frequency for window %d: %.2f Hz", window_index, peak_frequency_hz)
    return output_path


def _default_input_path(project_root: Path, subject_id: str, session_id: str) -> Path:
    return project_root / "data" / "raw" / subject_id / session_id / "raw_stream.csv"


def _default_output_path(project_root: Path, subject_id: str, session_id: str) -> Path:
    return project_root / "data" / "processed" / subject_id / session_id / "pipeline_baseline.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the Phase 3 signal-processing baseline")
    parser.add_argument("--input", type=Path, default=None, help="Raw recording CSV path")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--subject-id", type=str, default="subj01", help="Subject ID")
    parser.add_argument("--session-id", type=str, default="sess01", help="Session ID")
    parser.add_argument("--window-index", type=int, default=0, help="Complete window index for PSD")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    input_path = args.input or _default_input_path(project_root, args.subject_id, args.session_id)
    output_path = args.output or _default_output_path(project_root, args.subject_id, args.session_id)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pipeline(
        input_path=input_path,
        output_path=output_path,
        subject_id=args.subject_id,
        session_id=args.session_id,
        window_index=args.window_index,
    )


if __name__ == "__main__":
    main()