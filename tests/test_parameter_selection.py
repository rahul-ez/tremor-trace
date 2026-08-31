"""Tests for controller/parameter_selection.py (Feature 37).

Covers:
- All returned StimParams fields are within param_bounds across a severity sweep.
- severity=0.0 maps amplitude to minimum bound.
- severity=1.0 maps amplitude to maximum bound.
- dominant_frequency_hz=None falls back to midpoint of pulse_frequency_hz bounds.
- dominant_frequency_hz outside bounds is clamped, not rejected.
- pulse_width_us, duty_cycle, and on_off_timing start at their minimums.
- phase is always None (phase-aware control disabled by default).
- ValueError raised when param_bounds is None in config.
"""

import pytest

from controller.parameter_selection import select_initial_params
from tremor_system.config import (
    Config,
    ControllerConfig,
    EstimationConfig,
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

_BOUNDS = ParamBoundsConfig(
    amplitude=ParamBound(min=0.0, max=5.0),
    pulse_frequency_hz=ParamBound(min=20.0, max=100.0),
    pulse_width_us=ParamBound(min=50.0, max=500.0),
    duty_cycle=ParamBound(min=0.0, max=0.5),
    on_time_ms=ParamBound(min=50.0, max=500.0),
    off_time_ms=ParamBound(min=50.0, max=500.0),
)


def _make_config(param_bounds: ParamBoundsConfig | None = _BOUNDS) -> Config:
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
            max_delta_per_step=None,
            param_bounds=param_bounds,
        ),
        simulation=SimulationConfig(timestep_s=0.001, latency_ms=50.0),
        estimation=EstimationConfig(phase_enabled=False),
    )


def _ml(
    severity: float = 0.5,
    dominant_frequency_hz: float | None = 6.0,
) -> InferenceResult:
    return InferenceResult(
        label=True,
        severity=severity,
        confidence=0.9,
        dominant_frequency_hz=dominant_frequency_hz,
        amplitude=0.2,
    )


# ---------------------------------------------------------------------------
# Bounds enforcement — all fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("severity", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_all_fields_within_bounds_across_severity_sweep(severity: float) -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(severity=severity), cfg)
    b = _BOUNDS

    assert b.amplitude.min <= params.amplitude <= b.amplitude.max
    assert b.pulse_frequency_hz.min <= params.pulse_frequency_hz <= b.pulse_frequency_hz.max
    assert b.pulse_width_us.min <= params.pulse_width_us <= b.pulse_width_us.max
    assert b.duty_cycle.min <= params.duty_cycle <= b.duty_cycle.max
    assert b.on_time_ms.min <= params.on_off_timing[0] <= b.on_time_ms.max
    assert b.off_time_ms.min <= params.on_off_timing[1] <= b.off_time_ms.max


# ---------------------------------------------------------------------------
# Amplitude: linear severity scaling
# ---------------------------------------------------------------------------

def test_severity_zero_maps_amplitude_to_minimum() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(severity=0.0), cfg)

    assert params.amplitude == pytest.approx(_BOUNDS.amplitude.min)


def test_severity_one_maps_amplitude_to_maximum() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(severity=1.0), cfg)

    assert params.amplitude == pytest.approx(_BOUNDS.amplitude.max)


def test_amplitude_scales_linearly_with_severity() -> None:
    """Amplitude at severity=0.5 should equal the midpoint of the bounds."""
    cfg = _make_config()
    params = select_initial_params(_ml(severity=0.5), cfg)
    expected = (_BOUNDS.amplitude.min + _BOUNDS.amplitude.max) / 2.0

    assert params.amplitude == pytest.approx(expected)


def test_amplitude_increases_monotonically_with_severity() -> None:
    cfg = _make_config()
    severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    amplitudes = [
        select_initial_params(_ml(severity=s), cfg).amplitude
        for s in severities
    ]

    for i in range(len(amplitudes) - 1):
        assert amplitudes[i] <= amplitudes[i + 1]


# ---------------------------------------------------------------------------
# pulse_frequency_hz: frequency-informed selection
# ---------------------------------------------------------------------------

def test_dominant_frequency_within_bounds_used_directly() -> None:
    # Use 30.0 Hz — within pulse_frequency_hz bounds [20, 100].
    # (Tremor band 4-12 Hz is the sensor frequency; pulse_frequency_hz bounds
    # are stimulation pulse rates, which are higher. A dominant_frequency_hz
    # below 20.0 would be correctly clamped to the minimum bound.)
    cfg = _make_config()
    params = select_initial_params(_ml(dominant_frequency_hz=30.0), cfg)

    assert params.pulse_frequency_hz == pytest.approx(30.0)


def test_dominant_frequency_none_falls_back_to_midpoint() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(dominant_frequency_hz=None), cfg)
    expected_midpoint = (_BOUNDS.pulse_frequency_hz.min + _BOUNDS.pulse_frequency_hz.max) / 2.0

    assert params.pulse_frequency_hz == pytest.approx(expected_midpoint)


def test_dominant_frequency_below_bounds_is_clamped_to_min() -> None:
    """Frequency below configured minimum must be clamped, not used as-is."""
    cfg = _make_config()
    params = select_initial_params(_ml(dominant_frequency_hz=5.0), cfg)
    # 5.0 < min=20.0 -> clamped to 20.0
    assert params.pulse_frequency_hz == pytest.approx(_BOUNDS.pulse_frequency_hz.min)


def test_dominant_frequency_above_bounds_is_clamped_to_max() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(dominant_frequency_hz=150.0), cfg)
    # 150.0 > max=100.0 -> clamped to 100.0
    assert params.pulse_frequency_hz == pytest.approx(_BOUNDS.pulse_frequency_hz.max)


# ---------------------------------------------------------------------------
# Conservative starting points for other fields
# ---------------------------------------------------------------------------

def test_pulse_width_us_starts_at_minimum() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(), cfg)

    assert params.pulse_width_us == pytest.approx(_BOUNDS.pulse_width_us.min)


def test_duty_cycle_starts_at_minimum() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(), cfg)

    assert params.duty_cycle == pytest.approx(_BOUNDS.duty_cycle.min)


def test_on_off_timing_starts_at_minimums() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(), cfg)

    assert params.on_off_timing[0] == pytest.approx(_BOUNDS.on_time_ms.min)
    assert params.on_off_timing[1] == pytest.approx(_BOUNDS.off_time_ms.min)


# ---------------------------------------------------------------------------
# Phase is always None (Feature 31 disabled by default)
# ---------------------------------------------------------------------------

def test_phase_is_always_none() -> None:
    cfg = _make_config()
    params = select_initial_params(_ml(), cfg)

    assert params.phase is None


# ---------------------------------------------------------------------------
# Config validation: None param_bounds must raise
# ---------------------------------------------------------------------------

def test_none_param_bounds_raises_value_error() -> None:
    """param_bounds=None must raise ValueError (fail-fast) before any computation."""
    cfg = _make_config(param_bounds=None)
    with pytest.raises(ValueError, match="param_bounds"):
        select_initial_params(_ml(), cfg)
