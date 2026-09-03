"""Demo script: detection dashboard over a real recorded session (Feature 53).

Not a build-plan feature itself -- a runnable convenience script showing
visualization/signal_plots.py::plot_detection_dashboard() working
end-to-end.

Usage:
    python scripts/demo_detection_dashboard.py
    python scripts/demo_detection_dashboard.py --subject subj02 --session sess02
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ml.inference import predict
from scripts.build_features import build_session_features, estimate_subject_offsets
from signal_processing.axis_handling import select_strongest_axis
from signal_processing.calibration import apply_calibration
from signal_processing.data_loader import load_raw_csv
from signal_processing.filtering import bandpass_filter, remove_baseline
from signal_processing.windowing import segment_windows
from tremor_system.config import load_config
from visualization.signal_plots import plot_detection_dashboard

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: detection dashboard over a real recorded session")
    parser.add_argument("--subject", type=str, default="subj05")
    parser.add_argument("--session", type=str, default="sess02")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "data/models/model_logistic_regression_v1.pkl")
    parser.add_argument("--scaler-path", type=Path, default=PROJECT_ROOT / "data/models/scaler_v1.pkl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config()
    raw_dir = PROJECT_ROOT / "data" / "raw" / args.subject
    target_csv = raw_dir / args.session / "raw_stream.csv"
    if not target_csv.exists():
        raise SystemExit(f"Raw session not found: {target_csv}")

    sess01_csv = raw_dir / "sess01" / "raw_stream.csv"
    subject_sessions = [("sess01", sess01_csv)] if sess01_csv.exists() else []
    subject_sessions.append((args.session, target_csv))
    offsets = estimate_subject_offsets(subject_sessions, config)

    records = build_session_features(target_csv, args.subject, args.session, config, offsets=offsets)
    if not records:
        raise SystemExit(f"No feature windows produced from {target_csv}")

    _timestamps_us, raw_signal = load_raw_csv(target_csv)
    calibrated = apply_calibration(raw_signal, offsets if offsets is not None else np.zeros(6), config)
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
    windows = segment_windows(
        filtered, config.sensor.sample_rate_hz, config.signal.window_length_s, config.signal.window_overlap_pct
    )

    results = [
        predict(records[i].to_dict(), windows[i], config.sensor.sample_rate_hz, args.model_path, args.scaler_path, config=config)
        for i in range(len(records))
    ]

    output_path = PROJECT_ROOT / "data" / "validation" / "demo_plots" / "detection_dashboard.png"
    plot_detection_dashboard(results, config.signal.window_length_s, output_path=output_path)

    n_detected = sum(1 for r in results if r.label)
    logger.info("%d/%d windows detected as tremor. Open %s", n_detected, len(results), output_path)


if __name__ == "__main__":
    main()
