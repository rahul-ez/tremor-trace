"""Tests for precision/recall/F1/sensitivity/false-positive-rate evaluation."""

import numpy as np
import pytest

from ml.evaluate import REQUIRED_METRICS, aggregate_fold_results, evaluate_candidates_across_folds, evaluate_model


class _FixedPredictionModel:
    """Stub classifier that returns a pre-set prediction sequence, ignoring input."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, X) -> np.ndarray:
        return self._predictions


class _IdentityScaler:
    def transform(self, X):
        return X


def test_evaluate_model_matches_hand_computed_confusion_matrix() -> None:
    # y_true:      [1, 1, 1, 0, 0, 0]
    # y_pred:      [1, 1, 0, 1, 0, 0]
    # -> TP=2, FN=1, FP=1, TN=2
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    model = _FixedPredictionModel(y_pred)
    scaler = _IdentityScaler()
    X_test = np.zeros((6, 9))

    metrics = evaluate_model(model, scaler, X_test, y_true)

    for name in REQUIRED_METRICS:
        assert name in metrics
        assert isinstance(metrics[name], float)
        assert 0.0 <= metrics[name] <= 1.0

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["sensitivity"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)


def test_evaluate_candidates_across_folds_and_aggregate() -> None:
    y_true_fold = np.array([1, 1, 0, 0])
    model_a = _FixedPredictionModel(np.array([1, 1, 0, 0]))  # perfect
    model_b = _FixedPredictionModel(np.array([0, 0, 0, 0]))  # misses both positives
    scaler = _IdentityScaler()
    X_test = np.zeros((4, 9))

    fitted_models_by_fold = [{"model_a": model_a, "model_b": model_b}] * 2
    scalers_by_fold = [scaler, scaler]
    test_folds = [(X_test, y_true_fold), (X_test, y_true_fold)]

    fold_results = evaluate_candidates_across_folds(fitted_models_by_fold, scalers_by_fold, test_folds)
    assert len(fold_results) == 4  # 2 models x 2 folds
    assert set(fold_results["model_name"]) == {"model_a", "model_b"}

    aggregated = aggregate_fold_results(fold_results)
    assert len(aggregated) == 2
    a_row = aggregated[aggregated["model_name"] == "model_a"].iloc[0]
    b_row = aggregated[aggregated["model_name"] == "model_b"].iloc[0]
    assert a_row["recall"] == pytest.approx(1.0)
    assert b_row["recall"] == pytest.approx(0.0)
