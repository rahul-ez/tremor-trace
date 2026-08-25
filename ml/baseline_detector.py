"""Non-ML signal-processing threshold baseline detector.

Per architecture.md -> Development Philosophy, a simple threshold rule is
established as the reference point to beat before comparing ML models
(Feature 23 of the build plan).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_FEATURE = "power_ratio"


def predict_baseline(
    features: dict,
    threshold: float,
    feature_name: str = DEFAULT_THRESHOLD_FEATURE,
) -> bool:
    """Classify a single window using a simple threshold rule.

    Args:
        features: Feature dict for one window (Feature 20 output), must
            contain feature_name.
        threshold: Decision threshold; tremor if features[feature_name] > threshold.
        feature_name: Which feature to threshold on. Defaults to
            "power_ratio" (tremor-band power / total power).

    Returns:
        True if the window is classified as tremor-positive.

    Raises:
        KeyError: If feature_name is not present in features.
    """
    if feature_name not in features:
        raise KeyError(f"features is missing required key {feature_name!r}")
    return bool(features[feature_name] > threshold)


def evaluate_baseline(
    df: pd.DataFrame,
    threshold: float,
    feature_name: str = DEFAULT_THRESHOLD_FEATURE,
) -> dict:
    """Compute precision/recall/F1 for the threshold rule on a labeled dataset.

    Args:
        df: Feature table with an added "label" column (see
            ml/dataset_builder.py -> assign_labels).
        threshold: Decision threshold passed through to predict_baseline().
        feature_name: Feature column to threshold on.

    Returns:
        Dict with "precision", "recall", "f1" as floats in [0, 1]. This is
        the reference point recorded for later comparison against the ML
        candidates in ml/evaluate.py -- no target score is asserted here.

    Raises:
        ValueError: If df has no "label" column.
        KeyError: If feature_name is not a column in df.
    """
    if "label" not in df.columns:
        raise ValueError("df must have a 'label' column; call assign_labels() first")
    if feature_name not in df.columns:
        raise KeyError(f"df is missing feature column {feature_name!r}")

    y_true = df["label"].astype(bool).to_numpy()
    y_pred = (df[feature_name].to_numpy() > threshold)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    logger.info(
        "Baseline threshold detector (%s > %.4f): precision=%.3f recall=%.3f f1=%.3f",
        feature_name,
        threshold,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics
