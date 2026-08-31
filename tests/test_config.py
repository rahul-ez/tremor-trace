"""Tests for system_config.yaml loader."""

from pathlib import Path
import pytest

from tremor_system.config import Config, load_config


def test_load_config_default() -> None:
    config = load_config()
    assert isinstance(config, Config)

    # Check sensor values
    assert config.sensor.sample_rate_hz == 100.0
    assert config.sensor.i2c_clock_hz == 400000
    assert config.sensor.accel_range_g == 2.0
    assert config.sensor.gyro_range_dps == 250.0

    # Check signal values
    assert config.signal.tremor_band_hz == [4.0, 12.0]
    assert config.signal.voluntary_band_hz == [0.5, 4.0]
    assert config.signal.window_overlap_pct == 50.0
    assert config.signal.filter_order == 4
    assert config.signal.window_length_s == 2.0
    assert config.signal.axis_strategy == "per_axis"

    # Check ML values
    assert config.ml.random_seed == 42
    # confidence_threshold was TBD (None); set to 0.6 provisional in Feature 36.
    assert config.ml.confidence_threshold == 0.6

    # Check controller values
    assert config.controller.target_suppression_pct == 50.0
    # Previously TBD (None); set to provisional values in Feature 36.
    assert config.controller.severity_threshold == 0.3
    assert config.controller.hysteresis_pct == 10.0
    # suppression_tolerance_pct added in Feature 38 pre-work — separate from hysteresis_pct.
    assert config.controller.suppression_tolerance_pct == 5.0
    # max_delta_per_step converted from scalar to per-field MaxDeltaConfig in Feature 38 pre-work.
    from tremor_system.config import MaxDeltaConfig, ParamBoundsConfig
    assert isinstance(config.controller.max_delta_per_step, MaxDeltaConfig)
    assert config.controller.max_delta_per_step.amplitude == 0.5
    assert config.controller.max_delta_per_step.duty_cycle == 0.05
    # param_bounds is now a typed ParamBoundsConfig (not None) since Feature 36.
    assert isinstance(config.controller.param_bounds, ParamBoundsConfig)

    # Check simulation values
    assert config.simulation.timestep_s == 0.001
    assert config.simulation.latency_ms == 50.0


def test_load_config_missing_file(tmp_path: Path) -> None:
    non_existent_path = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(non_existent_path)
