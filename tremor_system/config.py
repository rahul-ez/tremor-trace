"""Configuration loader and typed dataclasses for system_config.yaml."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
class ControllerConfig:
    severity_threshold: Optional[float]
    target_suppression_pct: float
    hysteresis_pct: Optional[float]
    max_delta_per_step: Optional[float]
    param_bounds: Optional[dict[str, Any]]


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

    controller_cfg = ControllerConfig(
        severity_threshold=float(controller_raw["severity_threshold"]) if controller_raw.get("severity_threshold") is not None else None,
        target_suppression_pct=float(controller_raw["target_suppression_pct"]),
        hysteresis_pct=float(controller_raw["hysteresis_pct"]) if controller_raw.get("hysteresis_pct") is not None else None,
        max_delta_per_step=float(controller_raw["max_delta_per_step"]) if controller_raw.get("max_delta_per_step") is not None else None,
        param_bounds=controller_raw.get("param_bounds"),
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
