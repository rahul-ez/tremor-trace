"""Tests for the sole ML inference entry point."""

from pathlib import Path

import numpy as np
import pytest

from ml.inference import FEATURE_ORDER, predict
from ml.train import save_artifacts, train_and_scale


def _feature_dict(power_ratio: float = 0.8) -> dict:
    return {
        "tremor_band_power": 0.5,
        "total_power": 1.0,
        "power_ratio": power_ratio,
        "dominant_frequency_hz": 6.0,
        "rms_amplitude": 0.3,
        "variance": 0.05,
        "spectral_entropy": 0.7,
        "accel_magnitude": 1.02,
        "gyro_magnitude": 0.4,
    }


def _train_synthetic_model(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(FEATURE_ORDER)))
    y = (X[:, 0] > 0).astype(int)
    fitted_models, scaler = train_and_scale(X, y, random_seed=42)
    paths = save_artifacts(fitted_models, scaler, tmp_path, version="v1")
    return paths["logistic_regression"], paths["scaler"]


def test_predict_returns_documented_output_contract(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)

    result = predict(_feature_dict(), model_path, scaler_path)

    assert isinstance(result.label, bool)
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.severity <= 1.0
    assert result.dominant_frequency_hz == pytest.approx(6.0)
    assert result.amplitude == pytest.approx(0.3)
    assert result.phase is None


def test_predict_severity_derived_from_power_ratio(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)

    result = predict(_feature_dict(power_ratio=0.42), model_path, scaler_path)

    assert result.severity == pytest.approx(0.42)


def test_predict_rejects_missing_feature(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)
    incomplete_features = _feature_dict()
    del incomplete_features["rms_amplitude"]

    with pytest.raises(KeyError):
        predict(incomplete_features, model_path, scaler_path)


def test_predict_on_recorded_deliberate_tremor_session() -> None:
    """Integration check against real recorded data, when available locally.

    Mirrors tests/test_pipeline_baseline.py's pattern: this repo does not
    ship committed sample recordings under data/raw/, so this test is
    skipped unless a real session (and a trained model) is present.
    """
    features_csv = Path("data/features/subj02/sess02/features.csv")
    model_path = Path("data/models/model_logistic_regression_v1.pkl")
    scaler_path = Path("data/models/scaler_v1.pkl")
    if not (features_csv.exists() and model_path.exists() and scaler_path.exists()):
        pytest.skip(
            "Recorded features/trained model not found locally; run "
            "scripts/build_features.py and scripts/run_training.py first "
            "to exercise this integration check."
        )

    import pandas as pd

    df = pd.read_csv(features_csv)
    features = df.iloc[len(df) // 2].to_dict()

    result = predict(features, model_path, scaler_path)

    assert result.label is True
    assert 0.0 <= result.confidence <= 1.0
