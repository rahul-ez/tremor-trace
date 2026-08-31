"""Tests for controller/adaptation.py (Feature 38).

Covers:
- achieved < target - tolerance  -> INCREASE: amplitude increases by max_delta.amplitude,
  clamped to amplitude.max.
- achieved > target + tolerance  -> DECREASE: amplitude decreases by max_delta.amplitude,
  clamped to amplitude.min.
- achieved within tolerance band -> MAINTAIN: params unchanged.
- stability_warning=True         -> MAINTAIN regardless of suppression value.
- achieved_suppression_pct=None  -> MAINTAIN regardless of suppression value.
- Non-amplitude fields are unchanged in all cases.
- adaptation_history grows by one entry per call.
- All returned StimParams.amplitude values are within param_bounds.
- None config fields (suppression_tolerance_pct, max_delta_per_step, param_bounds)
  raise ValueError (fail-fast).
"""

from typing import Optional

import pytest

from controller.adaptation import (
    ADAPTATION_DECREASE,
    ADAPTATION_INCREASE,
    ADAPTATION_MAINTAIN,
    adapt_params,
)
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
from tremor_system.types import StimParams


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

_MAX_DELTA = MaxDeltaConfig(
    amplitude=0.5,
    pulse_frequency_hz=5.0,
    pulse_width_us=25.0,
    duty_cycle=0.05,
    on_time_ms=25.0,
    off_time_ms=25.0,
)


def _make_config(
    target: float = 50.0,
    tolerance: float = 5.0,
    bounds: Optional[ParamBoundsConfig] = _BOUNDS,
    max_delta: Optional[MaxDeltaConfig] = _MAX_DELTA,
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
        ml=MlConfig(confidence_threshold=0.6, random_seed=42),
        controller=ControllerConfig(
            severity_threshold=0.3,
            target_suppression_pct=target,
            suppression_tolerance_pct=tolerance,
            hysteresis_pct=10.0,
            max_delta_per_step=max_delta,
            param_bounds=bounds,
        ),
        simulation=SimulationConfig(timestep_s=0.001, latency_ms=50.0),
        estimation=EstimationConfig(phase_enabled=False),
    )


def _make_params(amplitude: float = 2.0) -> StimParams:
    return StimParams(
        amplitude=amplitude,
        pulse_frequency_hz=60.0,
        pulse_width_us=200.0,
        duty_cycle=0.2,
        on_off_timing=(100.0, 50.0),
        phase=None,
    )


def _fresh_state() -> ControllerState:
    return ControllerState()


# ---------------------------------------------------------------------------
# INCREASE case
# ---------------------------------------------------------------------------

def test_increase_when_achieved_below_target_minus_tolerance() -> None:
    """Suppression well below target -> amplitude must increase by max_delta.amplitude."""
    cfg = _make_config(target=50.0, tolerance=5.0)  # band: [45, 55]
    params = _make_params(amplitude=2.0)
    # achieved=30 < 45 -> INCREASE
    new_params, _ = adapt_params(params, achieved_suppression_pct=30.0, config=cfg)

    assert new_params.amplitude == pytest.approx(2.0 + 0.5)


def test_increase_is_clamped_to_amplitude_max() -> None:
    """INCREASE must not push amplitude above param_bounds.amplitude.max."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=4.9)  # 4.9 + 0.5 = 5.4 -> clamped to 5.0
    new_params, _ = adapt_params(params, achieved_suppression_pct=10.0, config=cfg)

    assert new_params.amplitude == pytest.approx(5.0)


def test_increase_at_max_bound_stays_at_max() -> None:
    """INCREASE when already at max bound must hold at max (clamped, not raised beyond)."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=5.0)
    new_params, _ = adapt_params(params, achieved_suppression_pct=10.0, config=cfg)

    assert new_params.amplitude == pytest.approx(5.0)


def test_increase_logs_bound_clamp(caplog) -> None:
    """Exceeding amplitude.max on INCREASE must log a bound_clamped=True WARNING."""
    import logging

    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=4.9)
    with caplog.at_level(logging.WARNING, logger="controller.adaptation"):
        adapt_params(params, achieved_suppression_pct=10.0, config=cfg)

    assert any("bound_clamped=True" in r.message for r in caplog.records)


def test_history_records_increase_decision() -> None:
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=2.0)
    _, new_state = adapt_params(params, achieved_suppression_pct=30.0, config=cfg)

    assert len(new_state.adaptation_history) == 1
    assert new_state.adaptation_history[0]["decision"] == ADAPTATION_INCREASE


# ---------------------------------------------------------------------------
# DECREASE case
# ---------------------------------------------------------------------------

def test_decrease_when_achieved_above_target_plus_tolerance() -> None:
    """Suppression above target+tolerance -> amplitude must decrease by max_delta.amplitude."""
    cfg = _make_config(target=50.0, tolerance=5.0)  # band: [45, 55]
    params = _make_params(amplitude=3.0)
    # achieved=80 > 55 -> DECREASE
    new_params, _ = adapt_params(params, achieved_suppression_pct=80.0, config=cfg)

    assert new_params.amplitude == pytest.approx(3.0 - 0.5)


def test_decrease_is_clamped_to_amplitude_min() -> None:
    """DECREASE must not push amplitude below param_bounds.amplitude.min."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=0.2)  # 0.2 - 0.5 = -0.3 -> clamped to 0.0
    new_params, _ = adapt_params(params, achieved_suppression_pct=90.0, config=cfg)

    assert new_params.amplitude == pytest.approx(0.0)


def test_decrease_at_min_bound_stays_at_min() -> None:
    """DECREASE when already at min bound must hold at min."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=0.0)
    new_params, _ = adapt_params(params, achieved_suppression_pct=90.0, config=cfg)

    assert new_params.amplitude == pytest.approx(0.0)


def test_history_records_decrease_decision() -> None:
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=3.0)
    _, new_state = adapt_params(params, achieved_suppression_pct=80.0, config=cfg)

    assert new_state.adaptation_history[0]["decision"] == ADAPTATION_DECREASE


# ---------------------------------------------------------------------------
# MAINTAIN case (suppression within tolerance band)
# ---------------------------------------------------------------------------

def test_maintain_when_achieved_equals_target() -> None:
    """Achieved == target exactly -> MAINTAIN."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=2.0)
    new_params, _ = adapt_params(params, achieved_suppression_pct=50.0, config=cfg)

    assert new_params.amplitude == pytest.approx(2.0)


def test_maintain_at_lower_edge_of_band() -> None:
    """achieved == target - tolerance exactly is within band -> MAINTAIN."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=2.0)
    # 45.0 == 50 - 5, within [45, 55] -> MAINTAIN (not < 45)
    new_params, _ = adapt_params(params, achieved_suppression_pct=45.0, config=cfg)

    assert new_params.amplitude == pytest.approx(2.0)


def test_maintain_at_upper_edge_of_band() -> None:
    """achieved == target + tolerance exactly is within band -> MAINTAIN."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=2.0)
    new_params, _ = adapt_params(params, achieved_suppression_pct=55.0, config=cfg)

    assert new_params.amplitude == pytest.approx(2.0)


def test_history_records_maintain_decision() -> None:
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=2.0)
    _, new_state = adapt_params(params, achieved_suppression_pct=50.0, config=cfg)

    assert new_state.adaptation_history[0]["decision"] == ADAPTATION_MAINTAIN


# ---------------------------------------------------------------------------
# MAINTAIN on invalid input
# ---------------------------------------------------------------------------

def test_stability_warning_true_forces_maintain() -> None:
    """stability_warning=True must force MAINTAIN regardless of suppression value."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=2.0)
    # achieved=10 is well below target - tolerance, but stability_warning overrides
    new_params, new_state = adapt_params(
        params,
        achieved_suppression_pct=10.0,
        config=cfg,
        stability_warning=True,
    )

    assert new_params.amplitude == pytest.approx(2.0)
    assert new_state.adaptation_history[0]["decision"] == ADAPTATION_MAINTAIN
    assert new_state.adaptation_history[0]["stability_warning"] is True


def test_achieved_none_forces_maintain() -> None:
    """achieved_suppression_pct=None must force MAINTAIN."""
    cfg = _make_config(target=50.0, tolerance=5.0)
    params = _make_params(amplitude=2.0)
    new_params, new_state = adapt_params(
        params, achieved_suppression_pct=None, config=cfg
    )

    assert new_params.amplitude == pytest.approx(2.0)
    assert new_state.adaptation_history[0]["decision"] == ADAPTATION_MAINTAIN


# ---------------------------------------------------------------------------
# Non-amplitude fields are unchanged
# ---------------------------------------------------------------------------

def test_non_amplitude_fields_unchanged_on_increase() -> None:
    cfg = _make_config()
    params = _make_params(amplitude=2.0)
    new_params, _ = adapt_params(params, achieved_suppression_pct=10.0, config=cfg)

    assert new_params.pulse_frequency_hz == pytest.approx(params.pulse_frequency_hz)
    assert new_params.pulse_width_us == pytest.approx(params.pulse_width_us)
    assert new_params.duty_cycle == pytest.approx(params.duty_cycle)
    assert new_params.on_off_timing == params.on_off_timing
    assert new_params.phase is None


def test_non_amplitude_fields_unchanged_on_decrease() -> None:
    cfg = _make_config()
    params = _make_params(amplitude=3.0)
    new_params, _ = adapt_params(params, achieved_suppression_pct=90.0, config=cfg)

    assert new_params.pulse_frequency_hz == pytest.approx(params.pulse_frequency_hz)
    assert new_params.pulse_width_us == pytest.approx(params.pulse_width_us)
    assert new_params.duty_cycle == pytest.approx(params.duty_cycle)
    assert new_params.on_off_timing == params.on_off_timing
    assert new_params.phase is None


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def test_adaptation_history_grows_per_call() -> None:
    """Each call appends exactly one entry to adaptation_history."""
    cfg = _make_config()
    params = _make_params(amplitude=2.0)
    state = _fresh_state()

    _, state = adapt_params(params, 30.0, cfg, state)   # INCREASE
    _, state = adapt_params(state.current_params, 80.0, cfg, state)  # DECREASE
    _, state = adapt_params(state.current_params, 50.0, cfg, state)  # MAINTAIN

    assert len(state.adaptation_history) == 3
    assert state.adaptation_history[0]["decision"] == ADAPTATION_INCREASE
    assert state.adaptation_history[1]["decision"] == ADAPTATION_DECREASE
    assert state.adaptation_history[2]["decision"] == ADAPTATION_MAINTAIN


def test_state_current_params_updated_to_new_params() -> None:
    """updated_state.current_params must equal the returned new_params."""
    cfg = _make_config()
    params = _make_params(amplitude=2.0)
    new_params, new_state = adapt_params(params, 10.0, cfg)

    assert new_state.current_params is new_params


def test_state_hysteresis_active_preserved() -> None:
    """hysteresis_active must be preserved unchanged through adapt_params."""
    cfg = _make_config()
    params = _make_params(amplitude=2.0)
    state = ControllerState(hysteresis_active=True)
    _, new_state = adapt_params(params, 50.0, cfg, state)

    assert new_state.hysteresis_active is True


# ---------------------------------------------------------------------------
# Config validation: None fields must raise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_cfg", [
    _make_config(tolerance=None),
    _make_config(max_delta=None),
    _make_config(bounds=None),
])
def test_none_config_field_raises_value_error(bad_cfg: Config) -> None:
    """suppression_tolerance_pct, max_delta_per_step, or param_bounds being None
    must raise ValueError immediately (fail-fast)."""
    with pytest.raises(ValueError):
        adapt_params(_make_params(), 50.0, bad_cfg)
