"""Sole ML inference entry point: feature vector -> label/severity/confidence.

Feature 27 of the build plan. Per code-standards.md -> Machine Learning,
this is the only place scikit-learn's predict()/predict_proba() may be
called; controller/ and simulation/ must never call them directly.
"""

import logging
from pathlib import Path

import joblib

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
    model_path: Path,
    scaler_path: Path,
) -> InferenceResult:
    """Run inference for one window and return the documented output contract.

    Args:
        features: Feature dict for one window (Feature 20 output); must
            contain every key in FEATURE_ORDER plus "dominant_frequency_hz"
            and "rms_amplitude" (both already included in FEATURE_ORDER).
        model_path: Path to a joblib-dumped classifier (ml/train.py output).
        scaler_path: Path to the joblib-dumped StandardScaler fit at
            training time for this exact model version. Never re-fit here.

    Returns:
        InferenceResult{label, severity, confidence, dominant_frequency_hz,
        amplitude, phase}. dominant_frequency_hz/amplitude are populated
        directly from the input feature vector as an interim measure --
        Phase 5's estimation/frequency_estimation.py and
        estimation/amplitude_estimation.py are expected to replace this
        once implemented (see architecture.md -> Frequency & Phase
        Estimation). phase is always None until Phase 5 phase estimation
        is enabled.

    Raises:
        KeyError: If features is missing any required key.
        ValueError: If the loaded model has no predict_proba (confidence
            cannot be derived without native or calibrated probability
            output -- code-standards.md -> Machine Learning).
    """
    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise KeyError(f"features is missing required keys: {missing}")

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

    result = InferenceResult(
        label=label,
        severity=_estimate_severity(features),
        confidence=confidence,
        dominant_frequency_hz=float(features["dominant_frequency_hz"]),
        amplitude=float(features["rms_amplitude"]),
        phase=None,
    )
    logger.debug("Inference result: %s", result)
    return result
