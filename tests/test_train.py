"""Tests for candidate model fitting and artifact persistence."""

from pathlib import Path

import numpy as np
import pytest

from ml.train import save_artifacts, train_and_scale


def _synthetic_labeled_dataset(n_samples: int = 60, n_features: int = 9, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    # Make the label linearly separable-ish on the first feature so all
    # three candidates can fit without warnings.
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_train_and_scale_fits_all_three_candidates() -> None:
    X, y = _synthetic_labeled_dataset()

    fitted_models, scaler = train_and_scale(X, y, random_seed=42)

    assert set(fitted_models) == {"logistic_regression", "svm", "random_forest"}
    for model in fitted_models.values():
        # A fitted classifier can predict on transformed data without error.
        predictions = model.predict(scaler.transform(X))
        assert predictions.shape == (X.shape[0],)


def test_save_artifacts_writes_expected_files(tmp_path: Path) -> None:
    X, y = _synthetic_labeled_dataset()
    fitted_models, scaler = train_and_scale(X, y, random_seed=42)

    paths = save_artifacts(fitted_models, scaler, tmp_path, version="v1")

    assert paths["scaler"] == tmp_path / "scaler_v1.pkl"
    assert paths["scaler"].exists()
    for name in ("logistic_regression", "svm", "random_forest"):
        expected_path = tmp_path / f"model_{name}_v1.pkl"
        assert paths[name] == expected_path
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0


def test_save_artifacts_refuses_to_overwrite(tmp_path: Path) -> None:
    X, y = _synthetic_labeled_dataset()
    fitted_models, scaler = train_and_scale(X, y, random_seed=42)

    save_artifacts(fitted_models, scaler, tmp_path, version="v1")

    with pytest.raises(FileExistsError):
        save_artifacts(fitted_models, scaler, tmp_path, version="v1")
