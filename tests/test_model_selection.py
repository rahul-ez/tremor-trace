"""Tests for smallest-model-meeting-thresholds selection."""

import pandas as pd
import pytest

from ml.model_selection import select_model


def _evaluation_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model_name": "logistic_regression", "recall": 0.80, "precision": 0.75, "artifact_size_bytes": 2_000},
            {"model_name": "svm", "recall": 0.85, "precision": 0.80, "artifact_size_bytes": 50_000},
            {"model_name": "random_forest", "recall": 0.90, "precision": 0.85, "artifact_size_bytes": 500_000},
        ]
    )


def test_select_model_returns_smallest_eligible_candidate() -> None:
    results = _evaluation_results()

    selected = select_model(results, min_recall=0.7, min_precision=0.7)

    assert selected == "logistic_regression"


def test_select_model_excludes_candidates_below_thresholds() -> None:
    results = _evaluation_results()

    # Only svm and random_forest meet min_recall=0.85; svm is the smaller of the two.
    selected = select_model(results, min_recall=0.85, min_precision=0.7)

    assert selected == "svm"


def test_select_model_returns_none_when_nothing_qualifies() -> None:
    results = _evaluation_results()

    selected = select_model(results, min_recall=0.99, min_precision=0.99)

    assert selected is None


def test_select_model_requires_expected_columns() -> None:
    incomplete_results = pd.DataFrame([{"model_name": "svm", "recall": 0.9}])

    with pytest.raises(ValueError, match="missing required columns"):
        select_model(incomplete_results, min_recall=0.7, min_precision=0.7)
