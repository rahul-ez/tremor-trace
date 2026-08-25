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


def _six_hz_window(sample_rate_hz: float = 100.0, n_samples: int = 200) -> np.ndarray:
    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    return 0.3 * np.sin(2.0 * np.pi * 6.0 * time_s)


def _train_synthetic_model(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(FEATURE_ORDER)))
    y = (X[:, 0] > 0).astype(int)
    fitted_models, scaler = train_and_scale(X, y, random_seed=42)
    paths = save_artifacts(fitted_models, scaler, tmp_path, version="v1")
    return paths["logistic_regression"], paths["scaler"]


def test_predict_returns_documented_output_contract(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)

    result = predict(_feature_dict(), _six_hz_window(), 100.0, model_path, scaler_path)

    assert isinstance(result.label, bool)
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.severity <= 1.0
    assert result.dominant_frequency_hz == pytest.approx(6.0, abs=0.5)
    assert result.amplitude == pytest.approx(0.3 / np.sqrt(2.0), abs=0.05)
    assert result.phase is None


def test_predict_severity_derived_from_power_ratio(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)

    result = predict(_feature_dict(power_ratio=0.42), _six_hz_window(), 100.0, model_path, scaler_path)

    assert result.severity == pytest.approx(0.42)


def test_predict_frequency_is_none_for_flat_window(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)
    flat_window = np.zeros(200, dtype=np.float64)

    result = predict(_feature_dict(), flat_window, 100.0, model_path, scaler_path)

    assert result.dominant_frequency_hz is None
    assert result.amplitude == pytest.approx(0.0)


def test_predict_phase_is_none_when_disabled_by_default(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)

    result = predict(_feature_dict(), _six_hz_window(), 100.0, model_path, scaler_path)

    assert result.phase is None


def test_predict_phase_populated_when_config_enables_it(tmp_path: Path) -> None:
    import dataclasses

    from tremor_system.config import load_config

    model_path, scaler_path = _train_synthetic_model(tmp_path)
    base_config = load_config()
    phase_enabled_config = dataclasses.replace(
        base_config, estimation=dataclasses.replace(base_config.estimation, phase_enabled=True)
    )

    result = predict(
        _feature_dict(), _six_hz_window(), 100.0, model_path, scaler_path, config=phase_enabled_config
    )

    assert result.phase is not None
    assert -np.pi <= result.phase <= np.pi


def test_predict_rejects_missing_feature(tmp_path: Path) -> None:
    model_path, scaler_path = _train_synthetic_model(tmp_path)
    incomplete_features = _feature_dict()
    del incomplete_features["rms_amplitude"]

    with pytest.raises(KeyError):
        predict(incomplete_features, _six_hz_window(), 100.0, model_path, scaler_path)


def test_predict_on_recorded_deliberate_tremor_session() -> None:
    """Integration check against real recorded data, when available locally.

    Mirrors tests/test_pipeline_baseline.py's pattern: this repo does not
    ship committed sample recordings under data/raw/, so this test is
    skipped unless a real session (and a trained model) is present.
    Regenerates the analysis window directly from the raw session (rather
    than reading features.csv, which only stores scalar features) so
    dominant_frequency_hz/amplitude are computed from the real window, per
    Feature 30's verification requirement.
    """
    raw_csv = Path("data/raw/subj02/sess02/raw_stream.csv")
    model_path = Path("data/models/model_logistic_regression_v1.pkl")
    scaler_path = Path("data/models/scaler_v1.pkl")
    if not (raw_csv.exists() and model_path.exists() and scaler_path.exists()):
        pytest.skip(
            "Recorded raw session/trained model not found locally; run "
            "scripts/build_features.py and scripts/run_training.py first "
            "to exercise this integration check."
        )

    from scripts.build_features import build_session_features, estimate_subject_offsets
    from signal_processing.axis_handling import select_strongest_axis
    from signal_processing.calibration import apply_calibration
    from signal_processing.data_loader import load_raw_csv
    from signal_processing.filtering import bandpass_filter, remove_baseline
    from signal_processing.windowing import segment_windows
    from tremor_system.config import load_config

    config = load_config()
    subject_sessions = [
        ("sess01", Path("data/raw/subj02/sess01/raw_stream.csv")),
        ("sess02", raw_csv),
    ]
    offsets = estimate_subject_offsets(subject_sessions, config)

    records = build_session_features(raw_csv, "subj02", "sess02", config, offsets=offsets)
    assert records, "expected at least one window from the recorded session"

    _timestamps_us, raw_signal = load_raw_csv(raw_csv)
    calibrated = apply_calibration(raw_signal, offsets if offsets is not None else np.zeros(6), config)
    accel_signal = calibrated[:, :3]
    analysis_signal = select_strongest_axis(
        accel_signal,
        sample_rate_hz=config.sensor.sample_rate_hz,
        tremor_band_hz=tuple(config.signal.tremor_band_hz),
        filter_order=config.signal.filter_order,
    )
    analysis_filtered = bandpass_filter(
        remove_baseline(analysis_signal),
        sample_rate_hz=config.sensor.sample_rate_hz,
        band_hz=tuple(config.signal.tremor_band_hz),
        order=config.signal.filter_order,
    )
    analysis_windows = segment_windows(
        analysis_filtered, config.sensor.sample_rate_hz, config.signal.window_length_s, config.signal.window_overlap_pct
    )
    mid_index = len(records) // 2
    mid_record = records[mid_index]
    analysis_window = analysis_windows[mid_index]

    result = predict(mid_record.to_dict(), analysis_window, config.sensor.sample_rate_hz, model_path, scaler_path)

    assert result.label is True
    assert 0.0 <= result.confidence <= 1.0
    assert result.dominant_frequency_hz is not None
    assert 4.0 <= result.dominant_frequency_hz <= 12.0
    assert result.amplitude is not None
    assert result.amplitude > 0.0
