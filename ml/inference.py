"""Sole ML inference entry point: feature vector -> label/severity/confidence.

Feature 27 of the build plan. Per code-standards.md -> Machine Learning,
this is the only place scikit-learn's predict()/predict_proba() may be
called; controller/ and simulation/ must never call them directly.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray

from estimation.amplitude_estimation import estimate_amplitude
from estimation.frequency_estimation import estimate_frequency
from estimation.phase_estimation import estimate_phase
from signal_processing.spectral_analysis import compute_welch_psd
from tremor_system.config import Config, load_config
from tremor_system.types import InferenceResult

logger = logging.getLogger(__name__)

# Must match FeatureVector.to_dict() key order exactly (tremor_system/types.py)
# and the Feature vector schema in architecture.md -> Interfaces & Data
# Contracts. Never reorder without updating both.
FEATURE_ORDER = [
    "tremor_band_power",
    "total_power",
    "power_ratio",
    "dominant_frequency_hz",
    "rms_amplitude",
    "variance",
    "spectral_entropy",
    "accel_magnitude",
    "gyro_magnitude",
]


def _estimate_severity(features: dict) -> float:
    """Rule-based severity estimate from tremor-band power ratio.

    OPEN ITEM (architecture.md -> ML Architecture): severity may end up
    being model-derived instead. Until that is decided, this reuses
    power_ratio (already a bounded tremor-power/total-power ratio) as a
    severity proxy, clipped defensively to [0, 1].

    Args:
        features: Feature dict for one window, must contain "power_ratio".

    Returns:
        Severity in [0.0, 1.0].
    """
    return float(min(max(features["power_ratio"], 0.0), 1.0))


def predict(
    features: dict,
    analysis_window: NDArray[np.float64],
    sample_rate_hz: float,
    model_path: Path,
    scaler_path: Path,
    nperseg: int | None = None,
    config: Config | None = None,
) -> InferenceResult:
    """Run inference for one window and return the documented output contract.

    Args:
        features: Feature dict for one window (Feature 20 output); must
            contain every key in FEATURE_ORDER. These are the classifier's
            *input* features -- distinct from the dominant_frequency_hz/
            amplitude *output* fields below, which are computed fresh by
            Features 28-29 rather than copied from this dict.
        analysis_window: shape (n_samples,), the same filtered tremor-band
            window used to produce `features` (Feature 20's analysis_window).
            Used here to derive dominant_frequency_hz, amplitude, and
            (if enabled) phase.
        sample_rate_hz: Sampling rate in Hz, for the Welch PSD.
        model_path: Path to a joblib-dumped classifier (ml/train.py output).
        scaler_path: Path to the joblib-dumped StandardScaler fit at
            training time for this exact model version. Never re-fit here.
        nperseg: Welch segment length; defaults to analysis_window's length
            (matches the whole-window PSD used throughout this project).
        config: Loaded system configuration; defaults to load_config().
            Exposed as a parameter so callers doing per-window inference in
            a loop can load it once and pass it through, rather than
            re-reading system_config.yaml on every call.

    Returns:
        InferenceResult{label, severity, confidence, dominant_frequency_hz,
        amplitude, phase}. dominant_frequency_hz is None if the window's
        spectrum has no clear in-band peak (see
        estimation/frequency_estimation.py). phase is None unless
        config.estimation.phase_enabled is True (Feature 31 is disabled by
        default, per architecture.md -> Frequency & Phase Estimation).

    Raises:
        KeyError: If features is missing any required key.
        ValueError: If the loaded model has no predict_proba (confidence
            cannot be derived without native or calibrated probability
            output -- code-standards.md -> Machine Learning).
    """
    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise KeyError(f"features is missing required keys: {missing}")

    resolved_config = config or load_config()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    x = [[features[name] for name in FEATURE_ORDER]]
    x_scaled = scaler.transform(x)  # never scaler.fit_transform at inference time

    label = bool(model.predict(x_scaled)[0])

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Model {type(model).__name__} does not expose predict_proba(); "
            "wrap it with sklearn.calibration.CalibratedClassifierCV before "
            "saving it, or select a different candidate."
        )
    confidence = float(model.predict_proba(x_scaled)[0].max())

    freqs_hz, psd = compute_welch_psd(
        analysis_window, sample_rate_hz, nperseg or analysis_window.size
    )
    dominant_frequency_hz = estimate_frequency(freqs_hz, psd, config=resolved_config)
    amplitude = estimate_amplitude(analysis_window)
    phase = estimate_phase(analysis_window) if resolved_config.estimation.phase_enabled else None

    result = InferenceResult(
        label=label,
        severity=_estimate_severity(features),
        confidence=confidence,
        dominant_frequency_hz=dominant_frequency_hz,
        amplitude=amplitude,
        phase=phase,
    )
    logger.debug("Inference result: %s", result)
    return result
