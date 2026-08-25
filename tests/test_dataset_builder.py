"""Tests for feature-table assembly, labeling, and subject-level splitting."""

import numpy as np
import pandas as pd
import pytest

from ml.dataset_builder import (
    FEATURE_COLUMNS,
    assign_labels,
    build_feature_table,
    default_session_label,
    get_subject_level_folds,
)


def _synthetic_records(n_subjects: int = 6, n_windows_per_session: int = 4) -> list[dict]:
    records = []
    rng = np.random.default_rng(42)
    for subject_index in range(n_subjects):
        subject_id = f"subj{subject_index:02d}"
        for session_id in ("sess01", "sess02"):
            for window_id in range(n_windows_per_session):
                records.append(
                    {
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "window_id": window_id,
                        "tremor_band_power": float(rng.random()),
                        "total_power": float(rng.random() + 1.0),
                        "power_ratio": float(rng.random()),
                        "dominant_frequency_hz": float(rng.uniform(4.0, 12.0)),
                        "rms_amplitude": float(rng.random()),
                        "variance": float(rng.random()),
                        "spectral_entropy": float(rng.random()),
                        "accel_magnitude": float(rng.random() + 1.0),
                        "gyro_magnitude": float(rng.random()),
                    }
                )
    return records


def test_build_feature_table_validates_required_columns() -> None:
    records = _synthetic_records(n_subjects=2, n_windows_per_session=1)
    df = build_feature_table(records)

    assert all(isinstance(value, str) for value in df["subject_id"])
    assert all(isinstance(value, str) for value in df["session_id"])
    assert len(df) == 4

    with pytest.raises(ValueError, match="missing required columns"):
        build_feature_table([{"subject_id": "subj01"}])


def test_build_feature_table_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_feature_table([])


def test_default_session_label_convention() -> None:
    assert default_session_label("sess01") is False
    assert default_session_label("sess02") is True

    with pytest.raises(ValueError, match="Cannot infer"):
        default_session_label("calibration")


def test_assign_labels_adds_boolean_column() -> None:
    df = build_feature_table(_synthetic_records(n_subjects=1, n_windows_per_session=2))

    labeled = assign_labels(df)

    assert labeled["label"].dtype == bool
    assert set(labeled.loc[labeled["session_id"] == "sess01", "label"]) == {False}
    assert set(labeled.loc[labeled["session_id"] == "sess02", "label"]) == {True}


def test_assign_labels_supports_custom_label_fn() -> None:
    df = build_feature_table(_synthetic_records(n_subjects=1, n_windows_per_session=1))

    labeled = assign_labels(df, label_fn=lambda session_id: True)

    assert labeled["label"].all()


def test_get_subject_level_folds_has_no_subject_leakage() -> None:
    records = _synthetic_records(n_subjects=8, n_windows_per_session=5)
    df = assign_labels(build_feature_table(records))

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label"].astype(int).to_numpy()
    groups = df["subject_id"].to_numpy()

    for train_idx, test_idx in get_subject_level_folds(X, y, groups, n_splits=4):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        assert train_subjects.isdisjoint(test_subjects)


def test_get_subject_level_folds_rejects_too_many_splits() -> None:
    records = _synthetic_records(n_subjects=2, n_windows_per_session=2)
    df = assign_labels(build_feature_table(records))
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label"].astype(int).to_numpy()
    groups = df["subject_id"].to_numpy()

    with pytest.raises(ValueError, match="n_splits"):
        list(get_subject_level_folds(X, y, groups, n_splits=5))
