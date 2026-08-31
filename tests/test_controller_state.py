"""Tests for controller/controller_state.py (Feature 39).

Verifies that ControllerState:
1. Has all four documented fields with correct defaults.
2. reset() returns a fresh instance without mutating self.
3. adaptation_history is independent between instances (no shared mutable default).
4. Two runs through the full decide_mitigation -> select_initial_params ->
   adapt_params chain with identical inputs and fresh-but-identical states
   produce byte-identical outputs (determinism invariant from architecture.md
   and code-standards.md -> Adaptive Controller).
"""

import pytest

from controller.adaptation import adapt_params
from controller.controller_state import ControllerState
from controller.decision_logic import decide_mitigation
from controller.parameter_selection import select_initial_params
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
# Config helper
# ---------------------------------------------------------------------------

def _make_config() -> Config:
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
        ml=MlConfig(confidence_threshold=0.6, random_seed=42),
        controller=ControllerConfig(
            severity_threshold=0.3,
            target_suppression_pct=50.0,
            suppression_tolerance_pct=5.0,
            hysteresis_pct=10.0,
            max_delta_per_step=MaxDeltaConfig(
                amplitude=0.5,
                pulse_frequency_hz=5.0,
                pulse_width_us=25.0,
                duty_cycle=0.05,
                on_time_ms=25.0,
                off_time_ms=25.0,
            ),
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
    severity: float = 0.6,
    confidence: float = 0.9,
    dominant_frequency_hz: float | None = 50.0,
    amplitude: float = 0.3,
) -> InferenceResult:
    return InferenceResult(
        label=label,
        severity=severity,
        confidence=confidence,
        dominant_frequency_hz=dominant_frequency_hz,
        amplitude=amplitude,
    )


# ---------------------------------------------------------------------------
# 1. Field defaults
# ---------------------------------------------------------------------------

def test_controller_state_default_hysteresis_active() -> None:
    state = ControllerState()
    assert state.hysteresis_active is False


def test_controller_state_default_current_params_is_none() -> None:
    state = ControllerState()
    assert state.current_params is None


def test_controller_state_default_last_decision_timestamp_is_none() -> None:
    state = ControllerState()
    assert state.last_decision_timestamp is None


def test_controller_state_default_adaptation_history_is_empty_list() -> None:
    state = ControllerState()
    assert state.adaptation_history == []


# ---------------------------------------------------------------------------
# 2. reset() returns fresh instance, does not mutate self
# ---------------------------------------------------------------------------

def test_reset_returns_fresh_state_with_defaults() -> None:
    state = ControllerState(
        hysteresis_active=True,
        last_decision_timestamp=123.456,
        adaptation_history=[{"event": "test"}],
    )
    fresh = state.reset()

    assert fresh.hysteresis_active is False
    assert fresh.current_params is None
    assert fresh.last_decision_timestamp is None
    assert fresh.adaptation_history == []


def test_reset_does_not_mutate_original() -> None:
    state = ControllerState(hysteresis_active=True)
    _ = state.reset()

    assert state.hysteresis_active is True


# ---------------------------------------------------------------------------
# 3. adaptation_history is NOT shared between instances
# ---------------------------------------------------------------------------

def test_adaptation_history_is_independent_between_instances() -> None:
    """Appending to one instance's list must not affect another instance's list.

    This guards against the classic Python mutable-default-argument bug where
    a shared list reference causes cross-instance contamination.
    """
    state_a = ControllerState()
    state_b = ControllerState()

    state_a.adaptation_history.append({"event": "a"})

    assert state_b.adaptation_history == [], (
        "state_b.adaptation_history was mutated by appending to state_a — "
        "the two instances are sharing the same list object."
    )


# ---------------------------------------------------------------------------
# 4. Full-chain determinism: decide_mitigation -> select_initial_params -> adapt_params
# ---------------------------------------------------------------------------

def _run_full_chain(
    ml_output: InferenceResult,
    initial_state: ControllerState,
    config: Config,
    simulated_suppression_pct: float,
) -> tuple[bool, "StimParams", ControllerState]:
    """Run one complete controller cycle and return (mitigate, new_params, new_state)."""
    from tremor_system.types import StimParams

    mitigate, state_after_decision = decide_mitigation(ml_output, initial_state, config)

    if mitigate:
        if state_after_decision.current_params is None:
            # First mitigate cycle: select initial params.
            initial_params = select_initial_params(ml_output, config)
        else:
            initial_params = state_after_decision.current_params

        new_params, state_after_adapt = adapt_params(
            current_params=initial_params,
            achieved_suppression_pct=simulated_suppression_pct,
            config=config,
            state=state_after_decision,
        )
        return mitigate, new_params, state_after_adapt
    else:
        return mitigate, state_after_decision.current_params, state_after_decision


def test_full_chain_determinism_identical_inputs_produce_identical_outputs() -> None:
    """Two independent runs with the same inputs and fresh states must produce
    identical (mitigate, amplitude, hysteresis_active) values.

    This is the primary verification for the code-standards.md invariant:
    'Decisions must be deterministic given the same input and controller state
    — no randomness inside decision_logic.py or adaptation.py.'
    """
    cfg = _make_config()
    ml = _ml(label=True, severity=0.6, confidence=0.9, dominant_frequency_hz=50.0)
    simulated_suppression = 30.0  # below target-tolerance -> triggers INCREASE on next cycle

    state_a = ControllerState()
    state_b = ControllerState()

    mitigate_a, params_a, state_a = _run_full_chain(ml, state_a, cfg, simulated_suppression)
    mitigate_b, params_b, state_b = _run_full_chain(ml, state_b, cfg, simulated_suppression)

    assert mitigate_a == mitigate_b, "mitigate decision is non-deterministic"
    if params_a is not None and params_b is not None:
        assert params_a.amplitude == pytest.approx(params_b.amplitude), (
            "amplitude is non-deterministic"
        )
        assert params_a.pulse_frequency_hz == pytest.approx(params_b.pulse_frequency_hz)
        assert params_a.pulse_width_us == pytest.approx(params_b.pulse_width_us)
        assert params_a.duty_cycle == pytest.approx(params_b.duty_cycle)
        assert params_a.on_off_timing == params_b.on_off_timing
    assert state_a.hysteresis_active == state_b.hysteresis_active, (
        "hysteresis_active is non-deterministic"
    )


def test_full_chain_determinism_across_multiple_cycles() -> None:
    """Three-cycle run: two independent replicas must stay in lockstep every cycle."""
    cfg = _make_config()
    ml_sequence = [
        _ml(label=True, severity=0.6, confidence=0.9, dominant_frequency_hz=50.0),
        _ml(label=True, severity=0.7, confidence=0.85, dominant_frequency_hz=55.0),
        _ml(label=False, severity=0.1, confidence=0.95, dominant_frequency_hz=None),
    ]
    suppression_sequence = [30.0, 45.0, 0.0]

    state_a = ControllerState()
    state_b = ControllerState()

    for cycle, (ml, suppression) in enumerate(zip(ml_sequence, suppression_sequence)):
        mitigate_a, params_a, state_a = _run_full_chain(ml, state_a, cfg, suppression)
        mitigate_b, params_b, state_b = _run_full_chain(ml, state_b, cfg, suppression)

        assert mitigate_a == mitigate_b, f"cycle {cycle}: mitigate non-deterministic"
        assert state_a.hysteresis_active == state_b.hysteresis_active, (
            f"cycle {cycle}: hysteresis_active non-deterministic"
        )
        if params_a is not None and params_b is not None:
            assert params_a.amplitude == pytest.approx(params_b.amplitude), (
                f"cycle {cycle}: amplitude non-deterministic"
            )
