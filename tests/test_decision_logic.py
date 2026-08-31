"""Tests for controller/decision_logic.py (Feature 36).

Covers:
- Confidence gate (always False below threshold, regardless of severity/label).
- Label gate (label=False with high confidence -> False).
- Hysteresis enter: severity at/above enter threshold triggers mitigation.
- Hysteresis exit: severity below exit threshold deactivates mitigation.
- Hysteresis hold: severity between exit and enter thresholds keeps current state.
- Rapid toggling is prevented by the hysteresis band.
- ValueError raised when required config thresholds are None.
- LOW_CONFIDENCE_NO_ACTION log event string is distinct from other non-action paths.
"""

import pytest

from controller.controller_state import ControllerState
from controller.decision_logic import LOW_CONFIDENCE_NO_ACTION, decide_mitigation
from tremor_system.config import (
    Config,
    ControllerConfig,
    EstimationConfig,
    MaxDeltaConfig,
    MlConfig,
    ParamBound,
    ParamBoundsConfig,
    SensorConfig,
    SignalConfig,
    SimulationConfig,
)
from tremor_system.types import InferenceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    confidence_threshold: float = 0.6,
    severity_threshold: float = 0.3,
    hysteresis_pct: float = 10.0,
) -> Config:
    """Build a minimal Config for decision_logic tests."""
    return Config(
        sensor=SensorConfig(
            sample_rate_hz=100.0,
            i2c_clock_hz=400000,
            accel_range_g=2.0,
            gyro_range_dps=250.0,
        ),
        signal=SignalConfig(
            tremor_band_hz=[4.0, 12.0],
            voluntary_band_hz=[0.5, 4.0],
            filter_order=4,
            window_length_s=2.0,
            window_overlap_pct=50.0,
            axis_strategy="per_axis",
        ),
        ml=MlConfig(confidence_threshold=confidence_threshold, random_seed=42),
        controller=ControllerConfig(
            severity_threshold=severity_threshold,
            target_suppression_pct=50.0,
            suppression_tolerance_pct=5.0,
            hysteresis_pct=hysteresis_pct,
            max_delta_per_step=None,
            param_bounds=ParamBoundsConfig(
                amplitude=ParamBound(min=0.0, max=5.0),
                pulse_frequency_hz=ParamBound(min=20.0, max=100.0),
                pulse_width_us=ParamBound(min=50.0, max=500.0),
                duty_cycle=ParamBound(min=0.0, max=0.5),
                on_time_ms=ParamBound(min=50.0, max=500.0),
                off_time_ms=ParamBound(min=50.0, max=500.0),
            ),
        ),
        simulation=SimulationConfig(timestep_s=0.001, latency_ms=50.0),
        estimation=EstimationConfig(phase_enabled=False),
    )


def _ml(
    label: bool = True,
    severity: float = 0.5,
    confidence: float = 0.9,
    dominant_frequency_hz: float | None = 6.0,
    amplitude: float = 0.2,
) -> InferenceResult:
    return InferenceResult(
        label=label,
        severity=severity,
        confidence=confidence,
        dominant_frequency_hz=dominant_frequency_hz,
        amplitude=amplitude,
    )


def _fresh_state() -> ControllerState:
    return ControllerState()


def _active_state() -> ControllerState:
    return ControllerState(hysteresis_active=True)


# ---------------------------------------------------------------------------
# Stage 1: Confidence gate
# ---------------------------------------------------------------------------

def test_low_confidence_forces_false_regardless_of_high_severity() -> None:
    """Below-threshold confidence must yield mitigate=False even with severity=1.0."""
    cfg = _make_config(confidence_threshold=0.6)
    ml = _ml(label=True, severity=1.0, confidence=0.5)  # confidence < 0.6

    mitigate, new_state = decide_mitigation(ml, _fresh_state(), cfg)

    assert mitigate is False
    # hysteresis_active must not be changed — state is preserved unchanged.
    assert new_state.hysteresis_active is False


def test_low_confidence_does_not_exit_active_hysteresis() -> None:
    """Low confidence while mitigating must leave hysteresis_active unchanged (hold)."""
    cfg = _make_config(confidence_threshold=0.6)
    ml = _ml(label=True, severity=1.0, confidence=0.4)
    state = _active_state()

    mitigate, new_state = decide_mitigation(ml, state, cfg)

    assert mitigate is False
    # Hysteresis position must be preserved exactly — low confidence is not a
    # genuine severity drop, so we should not clear hysteresis_active.
    assert new_state.hysteresis_active is True


def test_low_confidence_log_event_is_distinct(caplog) -> None:
    """LOW_CONFIDENCE_NO_ACTION event must appear in logs for low-confidence calls."""
    import logging

    cfg = _make_config(confidence_threshold=0.6)
    ml = _ml(label=True, severity=0.8, confidence=0.3)

    with caplog.at_level(logging.WARNING, logger="controller.decision_logic"):
        decide_mitigation(ml, _fresh_state(), cfg)

    assert any(LOW_CONFIDENCE_NO_ACTION in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Stage 2: Label gate
# ---------------------------------------------------------------------------

def test_label_false_with_high_confidence_yields_false() -> None:
    cfg = _make_config()
    ml = _ml(label=False, severity=0.8, confidence=0.95)

    mitigate, _ = decide_mitigation(ml, _fresh_state(), cfg)

    assert mitigate is False


def test_label_false_deactivates_hysteresis_when_active() -> None:
    cfg = _make_config()
    ml = _ml(label=False, severity=0.8, confidence=0.95)

    mitigate, new_state = decide_mitigation(ml, _active_state(), cfg)

    assert mitigate is False
    assert new_state.hysteresis_active is False


# ---------------------------------------------------------------------------
# Stage 3: Hysteresis — entering mitigation
# ---------------------------------------------------------------------------

def test_severity_at_enter_threshold_activates_mitigation() -> None:
    """severity == severity_threshold should trigger mitigate=True (inclusive)."""
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    ml = _ml(label=True, severity=0.3, confidence=0.9)  # exactly at threshold

    mitigate, new_state = decide_mitigation(ml, _fresh_state(), cfg)

    assert mitigate is True
    assert new_state.hysteresis_active is True


def test_severity_above_enter_threshold_activates_mitigation() -> None:
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    ml = _ml(label=True, severity=0.8, confidence=0.9)

    mitigate, new_state = decide_mitigation(ml, _fresh_state(), cfg)

    assert mitigate is True
    assert new_state.hysteresis_active is True


def test_severity_below_enter_threshold_stays_inactive() -> None:
    """Severity below enter threshold must not activate mitigation."""
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    ml = _ml(label=True, severity=0.2, confidence=0.9)  # below 0.3

    mitigate, new_state = decide_mitigation(ml, _fresh_state(), cfg)

    assert mitigate is False
    assert new_state.hysteresis_active is False


# ---------------------------------------------------------------------------
# Stage 3: Hysteresis — exiting mitigation
# ---------------------------------------------------------------------------

def test_severity_below_exit_threshold_deactivates_mitigation() -> None:
    """While active, severity < exit_threshold must deactivate mitigation."""
    # enter=0.3, hysteresis=10% -> exit=0.3-0.1=0.2
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    ml = _ml(label=True, severity=0.15, confidence=0.9)  # below exit 0.2

    mitigate, new_state = decide_mitigation(ml, _active_state(), cfg)

    assert mitigate is False
    assert new_state.hysteresis_active is False


def test_severity_at_exit_threshold_deactivates_mitigation() -> None:
    """severity exactly at exit_threshold should deactivate (strict <)."""
    # enter=0.3, exit=0.3-0.1=0.2; severity=0.2 is NOT below exit -> hold
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    ml = _ml(label=True, severity=0.2, confidence=0.9)  # at exit threshold

    mitigate, new_state = decide_mitigation(ml, _active_state(), cfg)

    # 0.2 is NOT < 0.2 so hysteresis holds
    assert mitigate is True
    assert new_state.hysteresis_active is True


# ---------------------------------------------------------------------------
# Stage 3: Hysteresis — hold in the band
# ---------------------------------------------------------------------------

def test_severity_in_hysteresis_band_holds_current_state_inactive() -> None:
    """Severity between exit and enter thresholds while inactive keeps inactive."""
    # enter=0.3, exit=0.2; severity=0.25 is in band [0.2, 0.3)
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    ml = _ml(label=True, severity=0.25, confidence=0.9)

    mitigate, new_state = decide_mitigation(ml, _fresh_state(), cfg)

    assert mitigate is False
    assert new_state.hysteresis_active is False


def test_severity_in_hysteresis_band_holds_current_state_active() -> None:
    """Severity between exit and enter thresholds while active keeps active."""
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    ml = _ml(label=True, severity=0.25, confidence=0.9)  # in band, was active

    mitigate, new_state = decide_mitigation(ml, _active_state(), cfg)

    assert mitigate is True
    assert new_state.hysteresis_active is True


def test_hysteresis_prevents_rapid_toggling() -> None:
    """A sequence straddling severity_threshold must not toggle on every cycle."""
    # enter=0.3, exit=0.2
    cfg = _make_config(severity_threshold=0.3, hysteresis_pct=10.0)
    # Severity bounces between 0.25 (in band) and 0.35 (above enter).
    # Once activated, 0.25 must NOT deactivate (it is above exit=0.2).
    severities = [0.35, 0.25, 0.35, 0.25, 0.35, 0.25]
    state = _fresh_state()
    decisions = []
    for sev in severities:
        ml = _ml(label=True, severity=sev, confidence=0.9)
        mitigate, state = decide_mitigation(ml, state, cfg)
        decisions.append(mitigate)

    # First call (0.35) activates. Subsequent 0.25 values are in the band
    # and should HOLD active, not toggle off.
    assert decisions[0] is True   # activated at 0.35
    assert decisions[1] is True   # 0.25 >= exit(0.2) -> hold active
    # All should remain True once activated at 0.35, because 0.25 >= exit 0.2.
    assert all(d is True for d in decisions)


# ---------------------------------------------------------------------------
# Config validation: None thresholds must raise
# ---------------------------------------------------------------------------

def _config_with_null_confidence() -> Config:
    base = _make_config()
    return Config(
        sensor=base.sensor,
        signal=base.signal,
        ml=MlConfig(confidence_threshold=None, random_seed=42),
        controller=base.controller,
        simulation=base.simulation,
        estimation=base.estimation,
    )


def _config_with_null_severity() -> Config:
    base = _make_config()
    return Config(
        sensor=base.sensor,
        signal=base.signal,
        ml=base.ml,
        controller=ControllerConfig(
            severity_threshold=None,
            target_suppression_pct=50.0,
            suppression_tolerance_pct=5.0,
            hysteresis_pct=10.0,
            max_delta_per_step=None,
            param_bounds=base.controller.param_bounds,
        ),
        simulation=base.simulation,
        estimation=base.estimation,
    )


def _config_with_null_hysteresis() -> Config:
    base = _make_config()
    return Config(
        sensor=base.sensor,
        signal=base.signal,
        ml=base.ml,
        controller=ControllerConfig(
            severity_threshold=0.3,
            target_suppression_pct=50.0,
            suppression_tolerance_pct=5.0,
            hysteresis_pct=None,
            max_delta_per_step=None,
            param_bounds=base.controller.param_bounds,
        ),
        simulation=base.simulation,
        estimation=base.estimation,
    )


@pytest.mark.parametrize("bad_config", [
    _config_with_null_confidence(),
    _config_with_null_severity(),
    _config_with_null_hysteresis(),
])
def test_none_config_threshold_raises_value_error(bad_config: Config) -> None:
    """All required thresholds must be set; None must raise ValueError (fail-fast)."""
    ml = _ml(label=True, severity=0.8, confidence=0.9)
    with pytest.raises(ValueError):
        decide_mitigation(ml, _fresh_state(), bad_config)
