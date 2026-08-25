"""Precision/recall/F1/sensitivity/false-positive-rate evaluation.

Feature 25 of the build plan. Never selects a model on accuracy alone
(code-standards.md -> Machine Learning).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REQUIRED_METRICS = ["precision", "recall", "f1", "sensitivity", "false_positive_rate"]


def evaluate_model(
    model: ClassifierMixin,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Score one fitted model on one held-out fold.

    Args:
        model: A fitted classifier from ml/train.py.
        scaler: The scaler fit_transform-ed on the matching training fold
            (transform only here -- never re-fit).
        X_test: shape (n_windows, n_features), held-out feature matrix.
        y_test: shape (n_windows,), held-out labels.

    Returns:
        Dict with "precision", "recall", "f1", "sensitivity",
        "false_positive_rate" -- all floats in [0, 1]. sensitivity is an
        alias for recall on the positive (tremor) class, kept as a
        separate documented field per architecture.md -> ML Architecture.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_true = np.asarray(y_test).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": recall,
        "false_positive_rate": false_positive_rate,
    }


def evaluate_candidates_across_folds(
    fitted_models_by_fold: list[dict[str, ClassifierMixin]],
    scalers_by_fold: list[StandardScaler],
    test_folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Evaluate every candidate model across every GroupKFold fold.

    Args:
        fitted_models_by_fold: One dict[model_name, model] per fold, in the
            same order as test_folds (i.e. from calling train_and_scale()
            once per fold's training split).
        scalers_by_fold: One StandardScaler per fold, matching
            fitted_models_by_fold.
        test_folds: One (X_test, y_test) pair per fold.

    Returns:
        Long-format DataFrame with columns: fold, model_name, precision,
        recall, f1, sensitivity, false_positive_rate.
    """
    if not (len(fitted_models_by_fold) == len(scalers_by_fold) == len(test_folds)):
        raise ValueError(
            "fitted_models_by_fold, scalers_by_fold, and test_folds must be the same length"
        )

    rows = []
    for fold_index, (fitted_models, scaler, (X_test, y_test)) in enumerate(
        zip(fitted_models_by_fold, scalers_by_fold, test_folds)
    ):
        for model_name, model in fitted_models.items():
            metrics = evaluate_model(model, scaler, X_test, y_test)
            rows.append({"fold": fold_index, "model_name": model_name, **metrics})

    results = pd.DataFrame(rows)
    logger.info(
        "Evaluated %d model(s) across %d fold(s)",
        results["model_name"].nunique() if not results.empty else 0,
        len(test_folds),
    )
    return results


def aggregate_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Average per-fold metrics into one row per model_name.

    Args:
        fold_results: Output of evaluate_candidates_across_folds().

    Returns:
        DataFrame indexed by model_name with mean precision/recall/f1/
        sensitivity/false_positive_rate across folds.
    """
    return (
        fold_results.groupby("model_name")[REQUIRED_METRICS]
        .mean()
        .reset_index()
    )
