"""Demo script: raw vs. filtered signal + PSD plots (Feature 52).

Not a build-plan feature itself -- a runnable convenience script showing
visualization/signal_plots.py working end-to-end on a real recorded
session.

Usage:
    python scripts/demo_signal_plots.py
    python scripts/demo_signal_plots.py --subject subj02 --session sess02
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from signal_processing.axis_handling import select_strongest_axis
from signal_processing.calibration import apply_calibration, estimate_offsets, validate_offsets
from signal_processing.data_loader import load_raw_csv
from signal_processing.filtering import bandpass_filter, remove_baseline
from signal_processing.spectral_analysis import compute_welch_psd
from tremor_system.config import load_config
from visualization.signal_plots import plot_psd, plot_raw_vs_filtered

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: raw vs. filtered signal + PSD plots")
    parser.add_argument("--subject", type=str, default="subj05")
    parser.add_argument("--session", type=str, default="sess02")
    parser.add_argument("--n-samples", type=int, default=500, help="How many samples to plot (for readability)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config()
    raw_csv = PROJECT_ROOT / "data" / "raw" / args.subject / args.session / "raw_stream.csv"
    if not raw_csv.exists():
        raise SystemExit(f"Raw session not found: {raw_csv}")

    _timestamps_us, raw_signal = load_raw_csv(raw_csv)
    try:
        offsets = estimate_offsets(raw_signal, config.sensor.accel_range_g)
        validate_offsets(offsets, config)
    except ValueError as exc:
        logger.warning("Offset validation failed (%s); using zero offsets.", exc)
        offsets = np.zeros(6, dtype=np.float64)

    calibrated = apply_calibration(raw_signal, offsets, config)
    analysis = select_strongest_axis(
        calibrated[:, :3],
        sample_rate_hz=config.sensor.sample_rate_hz,
        tremor_band_hz=tuple(config.signal.tremor_band_hz),
        filter_order=config.signal.filter_order,
    )
    filtered = bandpass_filter(
        remove_baseline(analysis),
        sample_rate_hz=config.sensor.sample_rate_hz,
        band_hz=tuple(config.signal.tremor_band_hz),
        order=config.signal.filter_order,
    )

    n = args.n_samples
    output_dir = PROJECT_ROOT / "data" / "validation" / "demo_plots"

    plot_raw_vs_filtered(
        analysis[:n], filtered[:n], config.sensor.sample_rate_hz,
        output_path=output_dir / "raw_vs_filtered.png",
    )

    freqs_hz, psd = compute_welch_psd(filtered[:n], config.sensor.sample_rate_hz, nperseg=n)
    plot_psd(freqs_hz, psd, tuple(config.signal.tremor_band_hz), output_path=output_dir / "psd.png")

    logger.info("Done. Open %s and %s", output_dir / "raw_vs_filtered.png", output_dir / "psd.png")


if __name__ == "__main__":
    main()
