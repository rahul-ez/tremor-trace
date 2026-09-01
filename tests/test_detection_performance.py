"""Tests for detection performance / voluntary-movement rejection (Feature 48)."""

from pathlib import Path

import pytest

from tremor_system.config import load_config
from validation.detection_performance import build_voluntary_movement_dataset, run_detection_performance_report

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl").resolve()
SCALER_PATH = Path("data/models/scaler_v1.pkl").resolve()
TREMOR_SESSION = Path("data/raw/subj05/sess02/raw_stream.csv").resolve()

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists() and TREMOR_SESSION.exists()),
    reason="Trained model/scaler/recorded session not found locally; run "
    "scripts/build_features.py and scripts/run_training.py first, and "
    "ensure data/raw/subj05/sess02 is present.",
)


def test_build_voluntary_movement_dataset_labels_correctly() -> None:
    config = load_config()

    X, y = build_voluntary_movement_dataset(
        [TREMOR_SESSION], voluntary_frequencies_hz=[1.0, 2.0], voluntary_amplitude_g=1.0, config=config
    )

    assert X.shape[1] == 9  # ml.dataset_builder.FEATURE_COLUMNS width
    assert X.shape[0] == y.shape[0]
    n_voluntary = int((y == 0).sum())
    n_tremor = int((y == 1).sum())
    assert n_voluntary == 2  # one per voluntary_frequencies_hz entry
    assert n_tremor > 0
    assert n_voluntary + n_tremor == len(y)


def test_run_detection_performance_report_no_target_asserted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per the build plan: report and review the false-positive rate, but do
    not assert a fixed numeric threshold -- acceptable rates are still
    experimentally determined (architecture.md Success Criteria #3, TBD).
    """
    import json

    monkeypatch.chdir(tmp_path)

    import validation.detection_performance as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "DEFAULT_VOLUNTARY_FREQUENCIES_HZ", [1.0, 2.0])

    report_path = run_detection_performance_report([TREMOR_SESSION], MODEL_PATH, SCALER_PATH, run_id="test_report")

    with open(report_path) as f:
        report = json.load(f)

    assert report["experiment_type"] == "detection_performance"
    assert "voluntary_movement_false_positive_rate" in report["metrics"]
    assert 0.0 <= report["metrics"]["voluntary_movement_false_positive_rate"] <= 1.0
    assert "precision" in report["metrics"] and "recall" in report["metrics"]
