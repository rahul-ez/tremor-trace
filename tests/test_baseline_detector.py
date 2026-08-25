"""Tests for the non-ML signal-processing threshold baseline detector."""

import pandas as pd
import pytest

from ml.baseline_detector import evaluate_baseline, predict_baseline


def test_predict_baseline_thresholds_on_power_ratio() -> None:
    assert predict_baseline({"power_ratio": 0.9}, threshold=0.5) is True
    assert predict_baseline({"power_ratio": 0.1}, threshold=0.5) is False


def test_predict_baseline_rejects_missing_feature() -> None:
    with pytest.raises(KeyError):
        predict_baseline({"rms_amplitude": 1.0}, threshold=0.5)


def test_evaluate_baseline_matches_hand_computed_metrics() -> None:
    # 2 true positives, 1 false negative, 1 false positive, 1 true negative.
    df = pd.DataFrame(
        {
            "power_ratio": [0.9, 0.8, 0.2, 0.6, 0.1],
            "label": [True, True, True, False, False],
        }
    )

    metrics = evaluate_baseline(df, threshold=0.5)

    # predictions: [True, True, False, True, False]
    # TP=2 (idx0,1), FN=1 (idx2), FP=1 (idx3), TN=1 (idx4)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_evaluate_baseline_requires_label_column() -> None:
    df = pd.DataFrame({"power_ratio": [0.5]})
    with pytest.raises(ValueError, match="label"):
        evaluate_baseline(df, threshold=0.5)
