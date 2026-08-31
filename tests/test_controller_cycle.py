"""Integration tests for controller/cycle.py (Feature 40).

Covers:
- Low confidence -> mitigate=False -> (None, state)
- High confidence, low severity -> mitigate=False -> (None, state)
- First mitigate cycle (current_params=None) -> select_initial_params called,
  StimParams returned, state.current_params set.
- Later mitigate cycle (current_params already set) -> current_params returned
  unchanged (no select_initial_params or adapt_params call).
- current_params is PRESERVED (not cleared) in the returned state on
  mitigate=False — deliberate design choice, see cycle.py module docstring.
- Sequence of mixed InferenceResults: state transitions match expected
  decision logic derived from Features 36-38.
- Returned StimParams.amplitude is within param_bounds on first cycle.
"""

from typing import Optional

import pytest

from controller.cycle import run_controller_cycle
from controller.controller_state import ControllerState
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
from tremor_system.types import InferenceResult, StimParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOUNDS = ParamBoundsConfig(
    amplitude=ParamBound(min=0.0, max=5.0),
    pulse_frequency_hz=ParamBound(min=20.0, max=100.0),
    pulse_width_us=ParamBound(min=50.0, max=500.0),
    duty_cycle=ParamBound(min=0.0, max=0.5),
    on_time_ms=ParamBound(min=50.0, max=500.0),
    off_time_ms=ParamBound(min=50.0, max=500.0),
)


def _make_config(
    confidence_threshold: float = 0.6,
    severity_threshold: float = 0.3,
    hysteresis_pct: float = 10.0,
) -> Config:
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
            max_delta_per_step=MaxDeltaConfig(
                amplitude=0.5,
                pulse_frequency_hz=5.0,
                pulse_width_us=25.0,
                duty_cycle=0.05,
                on_time_ms=25.0,
                off_time_ms=25.0,
            ),
            param_bounds=_BOUNDS,
        ),
        simulation=SimulationConfig(timestep_s=0.001, latency_ms=50.0),
        estimation=EstimationConfig(phase_enabled=False),
    )


def _ml(
    label: bool = True,
    severity: float = 0.6,
    confidence: float = 0.9,
    dominant_frequency_hz: Optional[float] = 50.0,
    amplitude: float = 0.3,
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


def _state_with_params(amplitude: float = 2.5) -> ControllerState:
    """Return a ControllerState that already holds current_params (simulates a
    post-first-cycle state where adaptation has been applied by Feature 41)."""
    params = StimParams(
        amplitude=amplitude,
        pulse_frequency_hz=60.0,
        pulse_width_us=200.0,
        duty_cycle=0.2,
        on_off_timing=(100.0, 50.0),
        phase=None,
    )
    return ControllerState(hysteresis_active=True, current_params=params)


# ---------------------------------------------------------------------------
# mitigate=False cases
# ---------------------------------------------------------------------------

def test_low_confidence_returns_none_params() -> None:
    """Confidence below threshold -> mitigate=False -> params is None."""
    cfg = _make_config(confidence_threshold=0.6)
    ml = _ml(confidence=0.4, severity=0.9, label=True)
    params, _ = run_controller_cycle(ml, _fresh_state(), cfg)
    assert params is None


def test_low_severity_returns_none_params() -> None:
    """Severity below threshold -> mitigate=False -> params is None."""
    cfg = _make_config(severity_threshold=0.3)
    ml = _ml(confidence=0.9, severity=0.1, label=True)
    params, _ = run_controller_cycle(ml, _fresh_state(), cfg)
    assert params is None


def test_label_false_returns_none_params() -> None:
    """ML label=False (no tremor) -> mitigate=False -> params is None."""
    cfg = _make_config()
    ml = _ml(label=False, confidence=0.95, severity=0.8)
    params, _ = run_controller_cycle(ml, _fresh_state(), cfg)
    assert params is None


def test_mitigate_false_preserves_current_params_in_state() -> None:
    """Deliberate design: current_params must NOT be cleared on mitigate=False.

    If the controller was mid-adaptation when mitigation briefly deactivated
    (e.g. an intra-session severity dip below the hysteresis exit threshold),
    state.current_params should be preserved so re-activation can resume from
    the last converged amplitude rather than starting cold via
    select_initial_params().
    """
    cfg = _make_config()
    prior_params = StimParams(
        amplitude=3.2,
        pulse_frequency_hz=60.0,
        pulse_width_us=200.0,
        duty_cycle=0.2,
        on_off_timing=(100.0, 50.0),
        phase=None,
    )
    state = ControllerState(
        hysteresis_active=False,
        current_params=prior_params,
    )
    # Below-severity ml that produces mitigate=False
    ml = _ml(confidence=0.9, severity=0.1, label=True)
    _, new_state = run_controller_cycle(ml, state, cfg)

    assert new_state.current_params is not None, (
        "current_params was cleared on mitigate=False — this is a deliberate "
        "design violation. See cycle.py module docstring for rationale."
    )
    assert new_state.current_params.amplitude == pytest.approx(3.2)


def test_mitigate_false_returns_updated_hysteresis_state() -> None:
    """The returned state must reflect hysteresis_active=False after a
    mitigate=False decision."""
    cfg = _make_config()
    state = ControllerState(hysteresis_active=True)
    # Label=False forces mitigate=False even with high severity/confidence.
    ml = _ml(label=False, confidence=0.95, severity=0.8)
    _, new_state = run_controller_cycle(ml, state, cfg)
    assert new_state.hysteresis_active is False


# ---------------------------------------------------------------------------
# mitigate=True, first cycle (current_params is None)
# ---------------------------------------------------------------------------

def test_first_cycle_returns_stim_params() -> None:
    """First mitigate cycle with no prior params -> StimParams returned (not None)."""
    cfg = _make_config()
    ml = _ml(confidence=0.9, severity=0.6, label=True)
    params, _ = run_controller_cycle(ml, _fresh_state(), cfg)
    assert params is not None
    assert isinstance(params, StimParams)


def test_first_cycle_amplitude_within_bounds() -> None:
    """First-cycle StimParams.amplitude must be within param_bounds."""
    cfg = _make_config()
    ml = _ml(confidence=0.9, severity=0.6, label=True)
    params, _ = run_controller_cycle(ml, _fresh_state(), cfg)
    assert _BOUNDS.amplitude.min <= params.amplitude <= _BOUNDS.amplitude.max


def test_first_cycle_state_current_params_set() -> None:
    """After the first mitigate cycle, state.current_params must equal the
    returned StimParams."""
    cfg = _make_config()
    ml = _ml(confidence=0.9, severity=0.6, label=True)
    params, new_state = run_controller_cycle(ml, _fresh_state(), cfg)
    assert new_state.current_params is params


def test_first_cycle_hysteresis_active_set_true() -> None:
    """First mitigate cycle must set hysteresis_active=True."""
    cfg = _make_config()
    ml = _ml(confidence=0.9, severity=0.6, label=True)
    _, new_state = run_controller_cycle(ml, _fresh_state(), cfg)
    assert new_state.hysteresis_active is True


def test_first_cycle_phase_is_none() -> None:
    """Phase-aware control is disabled by default — phase must be None."""
    cfg = _make_config()
    ml = _ml(confidence=0.9, severity=0.6, label=True)
    params, _ = run_controller_cycle(ml, _fresh_state(), cfg)
    assert params.phase is None


# ---------------------------------------------------------------------------
# mitigate=True, later cycle (current_params already set)
# ---------------------------------------------------------------------------

def test_later_cycle_returns_existing_current_params_unchanged() -> None:
    """Later mitigate cycle with current_params set must return exactly those
    params (no re-selection, no adaptation — Feature 41 handles adaptation
    after simulation)."""
    cfg = _make_config()
    state = _state_with_params(amplitude=2.5)
    ml = _ml(confidence=0.9, severity=0.6, label=True)
    params, _ = run_controller_cycle(ml, state, cfg)

    assert params is state.current_params


def test_later_cycle_amplitude_unchanged() -> None:
    cfg = _make_config()
    state = _state_with_params(amplitude=3.7)
    ml = _ml(confidence=0.9, severity=0.9, label=True)
    params, _ = run_controller_cycle(ml, state, cfg)

    assert params.amplitude == pytest.approx(3.7)


def test_later_cycle_state_current_params_unchanged() -> None:
    """state.current_params in the returned state must still be the same object
    — run_controller_cycle must not replace it in the later-cycle path."""
    cfg = _make_config()
    state = _state_with_params(amplitude=2.5)
    ml = _ml(confidence=0.9, severity=0.6, label=True)
    _, new_state = run_controller_cycle(ml, state, cfg)

    assert new_state.current_params is state.current_params


# ---------------------------------------------------------------------------
# Sequence integration test
# ---------------------------------------------------------------------------

def test_sequence_of_decisions_matches_expected_state_transitions() -> None:
    """Run a 5-cycle synthetic sequence and assert the complete state evolution.

    Cycle 1: low confidence -> mitigate=False, params=None, current_params preserved (None).
    Cycle 2: high severity  -> mitigate=True, first cycle, params set by select_initial_params.
    Cycle 3: high severity  -> mitigate=True, later cycle, params returned unchanged.
    Cycle 4: label=False    -> mitigate=False, params=None, current_params preserved.
    Cycle 5: high severity  -> mitigate=True, later cycle (preserved params from cycle 3),
                               params returned unchanged — NOT re-selected via
                               select_initial_params because current_params survived cycle 4.
    """
    cfg = _make_config(
        confidence_threshold=0.6,
        severity_threshold=0.3,
        hysteresis_pct=10.0,
    )
    state = _fresh_state()

    # Cycle 1: low confidence -> mitigate=False
    ml1 = _ml(confidence=0.3, severity=0.8, label=True)
    params1, state = run_controller_cycle(ml1, state, cfg)
    assert params1 is None
    assert state.current_params is None    # nothing was ever set
    assert state.hysteresis_active is False

    # Cycle 2: high confidence, above-threshold severity, label=True -> mitigate=True, first cycle
    ml2 = _ml(confidence=0.9, severity=0.7, label=True)
    params2, state = run_controller_cycle(ml2, state, cfg)
    assert params2 is not None
    assert state.current_params is params2
    assert state.hysteresis_active is True
    first_cycle_amplitude = params2.amplitude

    # Cycle 3: still above threshold (hysteresis hold) -> mitigate=True, later cycle
    ml3 = _ml(confidence=0.9, severity=0.7, label=True)
    params3, state = run_controller_cycle(ml3, state, cfg)
    assert params3 is state.current_params  # same object — not re-selected
    assert params3.amplitude == pytest.approx(first_cycle_amplitude)

    # Cycle 4: label=False -> mitigate=False; current_params PRESERVED in state
    ml4 = _ml(label=False, confidence=0.95, severity=0.7)
    params4, state = run_controller_cycle(ml4, state, cfg)
    assert params4 is None
    assert state.current_params is not None              # preserved, not cleared
    assert state.current_params.amplitude == pytest.approx(first_cycle_amplitude)
    assert state.hysteresis_active is False

    # Cycle 5: high severity again -> mitigate=True; current_params is not None
    # (preserved from cycle 3) -> later-cycle path, not re-selected
    ml5 = _ml(confidence=0.9, severity=0.8, label=True)
    params5, state = run_controller_cycle(ml5, state, cfg)
    assert params5 is not None
    assert params5.amplitude == pytest.approx(first_cycle_amplitude), (
        "Re-activation after a brief off period must resume from preserved "
        "amplitude, not start fresh via select_initial_params()."
    )
