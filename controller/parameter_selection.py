"""Bounded stimulation parameter selection.

Feature 37 of the build plan.

Selects the *initial* StimParams for a fresh mitigate=True decision.  This is
NOT adaptation — it does not adjust parameters based on measured suppression.
Adaptation (increase/decrease/maintain) is Feature 38 (adaptation.py).

Parameter mapping:
- amplitude      : linear severity scaling within [min, max] bounds.
- pulse_frequency_hz: clamped dominant_frequency_hz from ml_output when
                  available; midpoint of [min, max] when dominant_frequency_hz
                  is None (flat spectrum or estimation failure, per
                  estimation/frequency_estimation.py error handling).
- pulse_width_us : minimum bound (conservative starting point).
- duty_cycle     : minimum bound.
- on_off_timing  : (on_time_ms.min, off_time_ms.min).
- phase          : None — phase-aware control is disabled by default
                  (architecture.md -> Frequency & Phase Estimation; Feature 31
                  is off until explicitly enabled and validated).

All emitted StimParams are clamped to param_bounds BEFORE being returned.  A
bound clamp is logged at WARNING level per code-standards.md -> Adaptive
Controller ("log a bound_clamped=True event — never silently clamp").

Per architecture.md -> Adaptive Controller and code-standards.md:
- Controller input is ONLY the documented InferenceResult contract.
- No raw signal, PSD, or feature vector is accepted here.
- No scikit-learn or signal-processing imports are allowed in this module.
- param_bounds must be configured; raises ValueError if absent.
"""

import logging
from typing import Optional

from tremor_system.config import Config, ParamBoundsConfig, load_config
from tremor_system.types import InferenceResult, StimParams

logger = logging.getLogger(__name__)


def _clamp(value: float, lo: float, hi: float, param_name: str) -> float:
    """Clamp value to [lo, hi], logging a warning if clamping occurs.

    Args:
        value: Candidate parameter value.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).
        param_name: Name of the parameter, used in the warning message.

    Returns:
        Clamped value within [lo, hi].
    """
    if value < lo:
        logger.warning(
            "bound_clamped=True: %s=%.4f clamped to min=%.4f",
            param_name,
            value,
            lo,
        )
        return lo
    if value > hi:
        logger.warning(
            "bound_clamped=True: %s=%.4f clamped to max=%.4f",
            param_name,
            value,
            hi,
        )
        return hi
    return value


def _require_param_bounds(config: Config) -> ParamBoundsConfig:
    """Return param_bounds, raising if it is not configured.

    Args:
        config: Loaded system configuration.

    Returns:
        Typed ParamBoundsConfig.

    Raises:
        ValueError: If config.controller.param_bounds is None.
    """
    bounds = config.controller.param_bounds
    if bounds is None:
        raise ValueError(
            "config.controller.param_bounds is None. "
            "Set all parameter bounds in config/system_config.yaml before "
            "running the controller (see architecture.md Open Question #7)."
        )
    return bounds


def select_initial_params(
    ml_output: InferenceResult,
    config: Optional[Config] = None,
) -> StimParams:
    """Select initial stimulation parameters for a fresh mitigate=True cycle.

    Args:
        ml_output: Output from ml/inference.py. Only ``severity`` and
            ``dominant_frequency_hz`` are consumed here; the controller never
            reads raw signal, PSD, or feature vectors (architecture.md
            invariant 8).
        config: Loaded system configuration. Defaults to load_config().

    Returns:
        StimParams with every field within the configured param_bounds.

    Raises:
        ValueError: If config.controller.param_bounds is None (fail-fast).
    """
    resolved_config = config or load_config()
    bounds = _require_param_bounds(resolved_config)

    severity = float(ml_output.severity)
    # Clamp severity to [0, 1] defensively — InferenceResult.severity should
    # already be in this range, but numerical noise can drift it slightly.
    severity = max(0.0, min(1.0, severity))

    # --- amplitude: linear severity scaling ---
    # Maps severity=0 -> min amplitude (no stimulus), severity=1 -> max amplitude.
    raw_amplitude = bounds.amplitude.min + severity * (
        bounds.amplitude.max - bounds.amplitude.min
    )
    amplitude = _clamp(raw_amplitude, bounds.amplitude.min, bounds.amplitude.max, "amplitude")

    # --- pulse_frequency_hz: frequency-informed ---
    # Use dominant_frequency_hz from estimation when available; fall back to
    # the midpoint of the configured range when the spectrum had no clear peak
    # (dominant_frequency_hz=None per estimation/frequency_estimation.py).
    freq_hz = ml_output.dominant_frequency_hz
    if freq_hz is not None:
        raw_pulse_freq = float(freq_hz)
    else:
        raw_pulse_freq = (
            bounds.pulse_frequency_hz.min + bounds.pulse_frequency_hz.max
        ) / 2.0
        logger.debug(
            "dominant_frequency_hz=None — using pulse_frequency_hz midpoint=%.2f Hz.",
            raw_pulse_freq,
        )
    pulse_frequency_hz = _clamp(
        raw_pulse_freq,
        bounds.pulse_frequency_hz.min,
        bounds.pulse_frequency_hz.max,
        "pulse_frequency_hz",
    )

    # --- remaining fields: start at minimum (conservative) ---
    # Adaptation (Feature 38) will increase these if suppression is insufficient.
    pulse_width_us = _clamp(
        bounds.pulse_width_us.min,
        bounds.pulse_width_us.min,
        bounds.pulse_width_us.max,
        "pulse_width_us",
    )
    duty_cycle = _clamp(
        bounds.duty_cycle.min,
        bounds.duty_cycle.min,
        bounds.duty_cycle.max,
        "duty_cycle",
    )
    on_ms = _clamp(
        bounds.on_time_ms.min,
        bounds.on_time_ms.min,
        bounds.on_time_ms.max,
        "on_time_ms",
    )
    off_ms = _clamp(
        bounds.off_time_ms.min,
        bounds.off_time_ms.min,
        bounds.off_time_ms.max,
        "off_time_ms",
    )

    params = StimParams(
        amplitude=amplitude,
        pulse_frequency_hz=pulse_frequency_hz,
        pulse_width_us=pulse_width_us,
        duty_cycle=duty_cycle,
        on_off_timing=(on_ms, off_ms),
        phase=None,  # Phase-aware control is disabled by default (Feature 31 off).
    )

    logger.debug(
        "select_initial_params: severity=%.3f -> amplitude=%.4f, "
        "pulse_frequency_hz=%.2f Hz, pulse_width_us=%.1f, duty_cycle=%.3f",
        severity,
        params.amplitude,
        params.pulse_frequency_hz,
        params.pulse_width_us,
        params.duty_cycle,
    )

    return params
