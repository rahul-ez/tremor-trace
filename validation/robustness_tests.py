"""Robustness testing: noise, frequency/amplitude sweep, filtering delay, gating.

Feature 51 of the build plan. Directly verifies architecture.md Success
Criterion #9: the controller never exceeds parameter bounds, never
oscillates outside the hysteresis band, and correctly withholds mitigation
on low-confidence input.

Usage:
    python -m validation.robustness_tests
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, correlation_lags

from controller.controller_state import ControllerState
from controller.decision_logic import decide_mitigation
from scripts.run_closed_loop_simulation import slice_into_windows
from signal_processing.filtering import bandpass_filter, remove_baseline
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import Config, load_config
from tremor_system.types import InferenceResult
from validation.experiments.common import run_experiment_cycle, sim_result_to_record
from validation.metrics import controller_stability_oscillation

logger = logging.getLogger(__name__)

DEFAULT_FREQUENCIES_HZ = [4.0, 6.0, 12.0]
DEFAULT_AMPLITUDES_G = [0.5, 1.5, 3.0]
DEFAULT_NOISE_STDS = [0.0, 0.05, 0.2]
DEFAULT_N_CYCLES_PER_COMBO = 8


def sweep_closed_loop_robustness(
    frequencies_hz: list[float],
    amplitudes_g: list[float],
    noise_stds: list[float],
    model_path: Path,
    scaler_path: Path,
    config: Config,
    n_cycles_per_combo: int = DEFAULT_N_CYCLES_PER_COMBO,
) -> list[dict]:
    """Run the adaptive closed loop across a frequency/amplitude/noise grid.

    Args:
        frequencies_hz: Tremor frequencies to test.
        amplitudes_g: Peak sine amplitudes to test.
        noise_stds: Additive Gaussian noise standard deviations to test.
        model_path: Path to the trained classifier.
        scaler_path: Path to the paired StandardScaler.
        config: Loaded system configuration.
        n_cycles_per_combo: Cycles to run per (frequency, amplitude, noise)
            combination.

    Returns:
        One record per combination, with within_bounds (bool -- amplitude
        never left config.controller.param_bounds.amplitude) and
        oscillation (from validation/metrics.py::controller_stability_oscillation).
    """
    bounds = config.controller.param_bounds.amplitude
    results = []

    for frequency_hz in frequencies_hz:
        for amplitude_g in amplitudes_g:
            for noise_std in noise_stds:
                signal = generate_synthetic_tremor(
                    frequency_hz=frequency_hz,
                    amplitude=amplitude_g,
                    duration_s=n_cycles_per_combo * config.signal.window_length_s,
                    sample_rate_hz=config.sensor.sample_rate_hz,
                    noise_std=noise_std,
                    seed=42,
                )
                chunks = slice_into_windows(signal, config.signal.window_length_s, config.sensor.sample_rate_hz)

                state = ControllerState().reset()
                records = []
                for cycle_index, (analysis_chunk, _accel, _gyro) in enumerate(chunks):
                    sim_result, state, ml_output = run_experiment_cycle(
                        analysis_chunk, state, model_path, scaler_path, config, mode="adaptive"
                    )
                    records.append(sim_result_to_record(cycle_index, sim_result, state, ml_output))

                amplitudes_used = [r["amplitude"] for r in records if r["amplitude"] is not None]
                within_bounds = all(bounds.min <= a <= bounds.max for a in amplitudes_used)
                oscillation = controller_stability_oscillation(records)

                results.append(
                    {
                        "frequency_hz": frequency_hz,
                        "amplitude_g": amplitude_g,
                        "noise_std": noise_std,
                        "n_cycles": len(records),
                        "within_bounds": within_bounds,
                        "oscillation": oscillation,
                    }
                )
                if not within_bounds:
                    logger.warning(
                        "Robustness sweep: amplitude left param_bounds at frequency=%.1fHz "
                        "amplitude=%.1fg noise_std=%.2f", frequency_hz, amplitude_g, noise_std,
                    )
    return results


def measure_filtering_delay(config: Config) -> dict:
    """Measure the lag introduced by baseline removal + band-pass filtering.

    bandpass_filter() uses scipy.signal.filtfilt (zero-phase, non-causal),
    so the expected delay is ~0 samples; this check confirms that
    empirically rather than assuming it from reading the implementation.

    Args:
        config: Loaded system configuration.

    Returns:
        Dict with delay_samples (best-correlation lag) and delay_s.
    """
    signal = generate_synthetic_tremor(
        frequency_hz=6.0, amplitude=1.0, duration_s=config.signal.window_length_s,
        sample_rate_hz=config.sensor.sample_rate_hz,
    )
    filtered = bandpass_filter(
        remove_baseline(signal), sample_rate_hz=config.sensor.sample_rate_hz,
        band_hz=tuple(config.signal.tremor_band_hz), order=config.signal.filter_order,
    )

    correlation = correlate(filtered, signal, mode="full")
    lags = correlation_lags(filtered.size, signal.size, mode="full")
    best_lag_samples = int(lags[np.argmax(correlation)])

    return {
        "delay_samples": best_lag_samples,
        "delay_s": best_lag_samples / config.sensor.sample_rate_hz,
    }


def check_low_confidence_gating(config: Config) -> dict:
    """Confirm decide_mitigation() withholds mitigation on low-confidence input.

    Feeds a hand-built, deliberately low-confidence, high-severity
    InferenceResult directly into decide_mitigation() (Feature 36),
    reusing that production function rather than reimplementing gating
    logic here.

    Args:
        config: Loaded system configuration.

    Returns:
        Dict recording the input and whether mitigation was correctly
        withheld.
    """
    confidence_threshold = config.ml.confidence_threshold
    low_confidence_result = InferenceResult(
        label=True,
        severity=0.95,  # deliberately high, to confirm confidence gates BEFORE severity
        confidence=max(0.0, confidence_threshold - 0.1),
        dominant_frequency_hz=6.0,
        amplitude=1.0,
        phase=None,
    )
    state = ControllerState().reset()
    mitigate, updated_state = decide_mitigation(low_confidence_result, state, config)

    return {
        "input_confidence": low_confidence_result.confidence,
        "confidence_threshold": confidence_threshold,
        "input_severity": low_confidence_result.severity,
        "mitigate_decision": mitigate,
        "correctly_withheld": mitigate is False,
        "state_unchanged": updated_state.hysteresis_active == state.hysteresis_active,
    }


def run_robustness_suite(
    model_path: Path,
    scaler_path: Path,
    run_id: str | None = None,
) -> Path:
    """Run all four robustness checks and write a single consolidated report.

    Returns:
        Path to the written report.json.
    """
    import json

    config = load_config()

    sweep_results = sweep_closed_loop_robustness(
        DEFAULT_FREQUENCIES_HZ, DEFAULT_AMPLITUDES_G, DEFAULT_NOISE_STDS, model_path, scaler_path, config
    )
    delay_result = measure_filtering_delay(config)
    gating_result = check_low_confidence_gating(config)

    all_within_bounds = all(r["within_bounds"] for r in sweep_results)
    max_oscillation = max(r["oscillation"] for r in sweep_results) if sweep_results else 0.0

    summary = {
        "n_combinations_swept": len(sweep_results),
        "all_within_bounds": all_within_bounds,
        "max_oscillation": max_oscillation,
        "filtering_delay_s": delay_result["delay_s"],
        "low_confidence_correctly_withheld": gating_result["correctly_withheld"],
    }

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"robustness_{resolved_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_type": "robustness",
        "metrics": summary,
        "frequency_amplitude_noise_sweep": sweep_results,
        "filtering_delay": delay_result,
        "low_confidence_gating": gating_result,
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(
        "Robustness suite: all_within_bounds=%s max_oscillation=%.4f filtering_delay_s=%.4f gating_ok=%s",
        all_within_bounds, max_oscillation, delay_result["delay_s"], gating_result["correctly_withheld"],
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the robustness test suite")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "data/models/model_logistic_regression_v1.pkl")
    parser.add_argument("--scaler-path", type=Path, default=PROJECT_ROOT / "data/models/scaler_v1.pkl")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_robustness_suite(args.model_path, args.scaler_path, args.run_id)


if __name__ == "__main__":
    main()
