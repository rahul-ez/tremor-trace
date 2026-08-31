"""Configuration loader and typed dataclasses for system_config.yaml."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class SensorConfig:
    sample_rate_hz: float
    i2c_clock_hz: int
    accel_range_g: float
    gyro_range_dps: float


@dataclass(frozen=True)
class SignalConfig:
    tremor_band_hz: list[float]
    voluntary_band_hz: list[float]
    filter_order: Optional[int]
    window_length_s: Optional[float]
    window_overlap_pct: float
    axis_strategy: Optional[str]


@dataclass(frozen=True)
class MlConfig:
    confidence_threshold: Optional[float]
    random_seed: int


@dataclass(frozen=True)
class ParamBound:
    """Min/max bound for a single stimulation parameter."""

    min: float
    max: float


@dataclass(frozen=True)
class ParamBoundsConfig:
    """Bounds for every field in StimParams.

    All values are provisional (v1) — see architecture.md Open Question #7.
    """

    amplitude: ParamBound
    pulse_frequency_hz: ParamBound
    pulse_width_us: ParamBound
    duty_cycle: ParamBound
    on_time_ms: ParamBound
    off_time_ms: ParamBound


@dataclass(frozen=True)
class MaxDeltaConfig:
    """Per-parameter maximum change cap per adaptation cycle.

    Each field specifies the maximum absolute change allowed for the
    corresponding StimParams field in a single call to adapt_params().
    Structure mirrors ParamBoundsConfig so that iteration over fields is
    symmetric. All values are provisional (v1) — see architecture.md
    Open Question #5.

    Mapping to StimParams fields:
        on_time_ms  → StimParams.on_off_timing[0]
        off_time_ms → StimParams.on_off_timing[1]
    """

    amplitude: float
    pulse_frequency_hz: float
    pulse_width_us: float
    duty_cycle: float
    on_time_ms: float
    off_time_ms: float


@dataclass(frozen=True)
class ControllerConfig:
    severity_threshold: Optional[float]
    target_suppression_pct: float
    suppression_tolerance_pct: Optional[float]
    hysteresis_pct: Optional[float]
    max_delta_per_step: Optional[MaxDeltaConfig]
    param_bounds: Optional[ParamBoundsConfig]


@dataclass(frozen=True)
class SimulationConfig:
    timestep_s: Optional[float]
    latency_ms: Optional[float]


@dataclass(frozen=True)
class EstimationConfig:
    phase_enabled: bool


@dataclass(frozen=True)
class Config:
    sensor: SensorConfig
    signal: SignalConfig
    ml: MlConfig
    controller: ControllerConfig
    simulation: SimulationConfig
    estimation: EstimationConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "system_config.yaml"


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load and validate configuration from YAML file into typed Config object.

    Args:
        config_path: Optional Path to system_config.yaml. Defaults to project root config.

    Returns:
        Config object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If a required top-level section or required field is missing.
    """
    path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid YAML structure in {path}, expected dictionary")

    required_sections = {"sensor", "signal", "ml", "controller", "simulation", "estimation"}
    missing_sections = required_sections - set(raw_data.keys())
    if missing_sections:
        raise KeyError(f"Missing required configuration sections: {missing_sections}")

    sensor_raw = raw_data["sensor"]
    signal_raw = raw_data["signal"]
    ml_raw = raw_data["ml"]
    controller_raw = raw_data["controller"]
    simulation_raw = raw_data["simulation"]
    estimation_raw = raw_data["estimation"]

    sensor_cfg = SensorConfig(
        sample_rate_hz=float(sensor_raw["sample_rate_hz"]),
        i2c_clock_hz=int(sensor_raw["i2c_clock_hz"]),
        accel_range_g=float(sensor_raw["accel_range_g"]),
        gyro_range_dps=float(sensor_raw["gyro_range_dps"]),
    )

    signal_cfg = SignalConfig(
        tremor_band_hz=[float(x) for x in signal_raw["tremor_band_hz"]],
        voluntary_band_hz=[float(x) for x in signal_raw["voluntary_band_hz"]],
        filter_order=int(signal_raw["filter_order"]) if signal_raw.get("filter_order") is not None else None,
        window_length_s=float(signal_raw["window_length_s"]) if signal_raw.get("window_length_s") is not None else None,
        window_overlap_pct=float(signal_raw["window_overlap_pct"]),
        axis_strategy=str(signal_raw["axis_strategy"]) if signal_raw.get("axis_strategy") is not None else None,
    )

    ml_cfg = MlConfig(
        confidence_threshold=float(ml_raw["confidence_threshold"]) if ml_raw.get("confidence_threshold") is not None else None,
        random_seed=int(ml_raw["random_seed"]),
    )

    raw_pb = controller_raw.get("param_bounds")
    if raw_pb is not None:
        parsed_param_bounds: Optional[ParamBoundsConfig] = ParamBoundsConfig(
            amplitude=ParamBound(min=float(raw_pb["amplitude"]["min"]), max=float(raw_pb["amplitude"]["max"])),
            pulse_frequency_hz=ParamBound(min=float(raw_pb["pulse_frequency_hz"]["min"]), max=float(raw_pb["pulse_frequency_hz"]["max"])),
            pulse_width_us=ParamBound(min=float(raw_pb["pulse_width_us"]["min"]), max=float(raw_pb["pulse_width_us"]["max"])),
            duty_cycle=ParamBound(min=float(raw_pb["duty_cycle"]["min"]), max=float(raw_pb["duty_cycle"]["max"])),
            on_time_ms=ParamBound(min=float(raw_pb["on_time_ms"]["min"]), max=float(raw_pb["on_time_ms"]["max"])),
            off_time_ms=ParamBound(min=float(raw_pb["off_time_ms"]["min"]), max=float(raw_pb["off_time_ms"]["max"])),
        )
    else:
        parsed_param_bounds = None

    raw_mds = controller_raw.get("max_delta_per_step")
    if raw_mds is not None:
        parsed_max_delta: Optional[MaxDeltaConfig] = MaxDeltaConfig(
            amplitude=float(raw_mds["amplitude"]),
            pulse_frequency_hz=float(raw_mds["pulse_frequency_hz"]),
            pulse_width_us=float(raw_mds["pulse_width_us"]),
            duty_cycle=float(raw_mds["duty_cycle"]),
            on_time_ms=float(raw_mds["on_time_ms"]),
            off_time_ms=float(raw_mds["off_time_ms"]),
        )
    else:
        parsed_max_delta = None

    controller_cfg = ControllerConfig(
        severity_threshold=float(controller_raw["severity_threshold"]) if controller_raw.get("severity_threshold") is not None else None,
        target_suppression_pct=float(controller_raw["target_suppression_pct"]),
        suppression_tolerance_pct=float(controller_raw["suppression_tolerance_pct"]) if controller_raw.get("suppression_tolerance_pct") is not None else None,
        hysteresis_pct=float(controller_raw["hysteresis_pct"]) if controller_raw.get("hysteresis_pct") is not None else None,
        max_delta_per_step=parsed_max_delta,
        param_bounds=parsed_param_bounds,
    )

    simulation_cfg = SimulationConfig(
        timestep_s=float(simulation_raw["timestep_s"]) if simulation_raw.get("timestep_s") is not None else None,
        latency_ms=float(simulation_raw["latency_ms"]) if simulation_raw.get("latency_ms") is not None else None,
    )

    estimation_cfg = EstimationConfig(
        phase_enabled=bool(estimation_raw["phase_enabled"]),
    )

    return Config(
        sensor=sensor_cfg,
        signal=signal_cfg,
        ml=ml_cfg,
        controller=controller_cfg,
        simulation=simulation_cfg,
        estimation=estimation_cfg,
    )
