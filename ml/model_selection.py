"""Select the smallest model meeting the project's detection requirements.

Feature 26 of the build plan. Exact min_recall/min_precision thresholds are
an experimental Open Item (architecture.md -> Configuration); callers must
pass them explicitly rather than relying on a silent default.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"model_name", "recall", "precision", "artifact_size_bytes"}


def select_model(
    evaluation_results: pd.DataFrame,
    min_recall: float,
    min_precision: float,
) -> str | None:
    """Select the smallest-artifact model meeting the given thresholds.

    Args:
        evaluation_results: One row per candidate model with columns
            "model_name", "recall", "precision" (aggregated, e.g. from
            ml.evaluate.aggregate_fold_results), and "artifact_size_bytes"
            (from the saved .pkl file size).
        min_recall: Minimum acceptable recall (sensitivity).
        min_precision: Minimum acceptable precision.

    Returns:
        The model_name of the smallest model meeting both thresholds, or
        None if no candidate qualifies.

    Raises:
        ValueError: If evaluation_results is missing a required column.
    """
    missing = REQUIRED_COLUMNS - set(evaluation_results.columns)
    if missing:
        raise ValueError(f"evaluation_results missing required columns: {missing}")

    eligible = evaluation_results[
        (evaluation_results["recall"] >= min_recall)
        & (evaluation_results["precision"] >= min_precision)
    ]

    if eligible.empty:
        logger.warning(
            "No candidate model met min_recall=%.3f, min_precision=%.3f", min_recall, min_precision
        )
        return None

    selected = eligible.sort_values("artifact_size_bytes", ascending=True).iloc[0]
    logger.info(
        "Selected model=%s (recall=%.3f, precision=%.3f, size=%d bytes)",
        selected["model_name"],
        selected["recall"],
        selected["precision"],
        selected["artifact_size_bytes"],
    )
    return str(selected["model_name"])
