"""Detection performance and voluntary-movement rejection evaluation.

Feature 48 of the build plan. Reuses Feature 25's evaluation functions
(ml/evaluate.py::evaluate_model) directly -- its false_positive_rate output
is exactly "false-positive rate on the voluntary-movement-rejection
sub-task" when the negative class is voluntary movement.

DATA NOTE: this project's recorded sessions (data/raw/<subject>/sess01,
sess02) only distinguish "stationary rest" vs "deliberate tremor-like
motion" -- there is no recorded, labeled voluntary-movement (0.5-4 Hz,
e.g. reaching/gesturing) session in this dataset. Real recorded tremor
sessions are used for the positive class; synthetic sine signals in the
0.5-4 Hz voluntary-movement band (Feature 32) stand in for the negative
class, clearly labeled as such below and in every report this module
writes. This is a documented limitation, not a hidden one -- see
memory.md's Phase 9 session update.

Usage:
    python -m validation.detection_performance --tremor-subject subj05
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from numpy.typing import NDArray

from ml.dataset_builder import FEATURE_COLUMNS, build_feature_table
from ml.evaluate import evaluate_model
from scripts.build_features import build_session_features, estimate_subject_offsets
from signal_processing.feature_extraction import extract_features
from simulation.closed_loop_runner import build_single_window
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import Config, load_config

logger = logging.getLogger(__name__)

DEFAULT_VOLUNTARY_FREQUENCIES_HZ = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
DEFAULT_VOLUNTARY_AMPLITUDE_G = 1.0


def _offsets_for_session(csv_path: Path, config: Config) -> NDArray[np.float64] | None:
    """Reuse the subject's real sess01-derived calibration offset, if available.

    Mirrors scripts/build_features.py::run_build_features's per-subject
    calibration, without needing the full batch-processing pass.
    """
    sess01_path = csv_path.parent.parent / "sess01" / "raw_stream.csv"
    if not sess01_path.exists():
        return None
    return estimate_subject_offsets([("sess01", sess01_path)], config)


def build_voluntary_movement_dataset(
    tremor_session_paths: list[Path],
    voluntary_frequencies_hz: list[float],
    voluntary_amplitude_g: float,
    config: Config,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Assemble a tremor-vs-voluntary-movement feature dataset.

    Args:
        tremor_session_paths: Paths to real recorded raw_stream.csv tremor
            sessions (positive class, label=1). subject_id/session_id are
            inferred from the parent folder names, matching
            scripts/build_features.py's convention.
        voluntary_frequencies_hz: Synthetic voluntary-movement frequencies
            (negative class, label=0) -- see module docstring for why
            these are synthetic rather than recorded.
        voluntary_amplitude_g: Peak sine amplitude for the synthetic
            voluntary-movement signals.
        config: Loaded system configuration.

    Returns:
        (X, y) -- X shape (n_windows, 9) in ml.dataset_builder.FEATURE_COLUMNS
        order, y shape (n_windows,) with 1=tremor, 0=voluntary movement.
    """
    records = []

    for csv_path in tremor_session_paths:
        session_id = csv_path.parent.name
        subject_id = csv_path.parent.parent.name
        offsets = _offsets_for_session(csv_path, config)
        feature_vectors = build_session_features(csv_path, subject_id, session_id, config, offsets=offsets)
        for fv in feature_vectors:
            records.append({"subject_id": fv.subject_id, "session_id": fv.session_id, "window_id": fv.window_id, **fv.to_dict()})

    voluntary_labels_start = len(records)
    for window_id, frequency_hz in enumerate(voluntary_frequencies_hz):
        signal = generate_synthetic_tremor(
            frequency_hz=frequency_hz,
            amplitude=voluntary_amplitude_g,
            duration_s=config.signal.window_length_s,
            sample_rate_hz=config.sensor.sample_rate_hz,
        )
        analysis_window, accel_window, gyro_window, sample_rate_hz = build_single_window(signal, config)
        nperseg = analysis_window.size
        fv = extract_features(
            analysis_window, sample_rate_hz, accel_window, gyro_window,
            subject_id="voluntary_synthetic", session_id=f"freq_{frequency_hz}hz", window_id=window_id, nperseg=nperseg,
        )
        records.append({"subject_id": fv.subject_id, "session_id": fv.session_id, "window_id": fv.window_id, **fv.to_dict()})

    df = build_feature_table(records)
    X = df[FEATURE_COLUMNS].to_numpy()
    y = np.zeros(len(records), dtype=np.int64)
    y[:voluntary_labels_start] = 1  # tremor sessions come first, per the loop order above
    return X, y


def run_detection_performance_report(
    tremor_session_paths: list[Path],
    model_path: Path,
    scaler_path: Path,
    run_id: str | None = None,
) -> Path:
    """Evaluate detection performance and voluntary-movement rejection, write a report.

    Returns:
        Path to the written report.json.
    """
    import json

    config = load_config()
    X, y = build_voluntary_movement_dataset(
        tremor_session_paths, DEFAULT_VOLUNTARY_FREQUENCIES_HZ, DEFAULT_VOLUNTARY_AMPLITUDE_G, config
    )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metrics = evaluate_model(model, scaler, X, y)
    # Rename for clarity in this report: "false_positive_rate" here specifically
    # means voluntary-movement misclassified as tremor.
    metrics["voluntary_movement_false_positive_rate"] = metrics.pop("false_positive_rate")

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"detection_performance_{resolved_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_type": "detection_performance",
        "n_tremor_windows": int(y.sum()),
        "n_voluntary_windows": int((y == 0).sum()),
        "metrics": metrics,
        "note": (
            "Voluntary-movement class is synthetic (0.5-4 Hz sine), not recorded -- "
            "see validation/detection_performance.py module docstring."
        ),
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "Detection performance: precision=%.3f recall=%.3f voluntary_fp_rate=%.3f",
        metrics["precision"], metrics["recall"], metrics["voluntary_movement_false_positive_rate"],
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the detection performance / voluntary-movement rejection evaluation")
    parser.add_argument("--tremor-subject", type=str, default="subj05")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "data/models/model_logistic_regression_v1.pkl")
    parser.add_argument("--scaler-path", type=Path, default=PROJECT_ROOT / "data/models/scaler_v1.pkl")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tremor_path = PROJECT_ROOT / "data" / "raw" / args.tremor_subject / "sess02" / "raw_stream.csv"
    run_detection_performance_report([tremor_path], args.model_path, args.scaler_path, args.run_id)


if __name__ == "__main__":
    main()
