"""Shared per-cycle experiment runner for Features 44-46.

Per architecture.md -> Validation Architecture: "Validation experiments...
each call the SAME production controller/ and simulation/ code with
different configurations -- they do not fork or duplicate controller
logic." This module is the single place that difference lives: which
production stages get called, not a reimplementation of any of them.
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from controller.controller_state import ControllerState
from controller.decision_logic import decide_mitigation
from simulation import apply as apply_simulation
from simulation.closed_loop_runner import (
    SIMULATION_SESSION_ID,
    SIMULATION_SUBJECT_ID,
    build_single_window,
    no_mitigation_result,
    run_closed_loop_cycle,
)
from signal_processing.feature_extraction import extract_features
from ml.inference import predict
from tremor_system.config import Config
from tremor_system.types import InferenceResult, SimResult, StimParams

logger = logging.getLogger(__name__)

ExperimentMode = Literal["no_mitigation", "fixed", "adaptive"]

# Per-cycle record schema shared by all three experiments and consumed by
# validation/metrics.py. `confidence`/`label` come from InferenceResult so
# false_activation_rate-style metrics can be computed without re-running
# inference.
EXPERIMENT_RECORD_COLUMNS = [
    "cycle",
    "mitigate",
    "hysteresis_active",
    "achieved_suppression_pct",
    "residual_amplitude",
    "stability_warning",
    "latency_ms",
    "amplitude",
    "pulse_frequency_hz",
    "duty_cycle",
    "confidence",
    "label",
]


def _extract_features_and_infer(
    analysis_chunk: NDArray[np.float64],
    accel_chunk: Optional[NDArray[np.float64]],
    gyro_chunk: Optional[NDArray[np.float64]],
    model_path: Path,
    scaler_path: Path,
    config: Config,
) -> tuple[InferenceResult, NDArray[np.float64], float, int]:
    """Run the shared Detect + Analyze stages (Features 20, 27/30).

    Returns:
        (ml_output, analysis_window, sample_rate_hz, nperseg).
    """
    analysis_window, accel_window, gyro_window, sample_rate_hz = build_single_window(
        analysis_chunk, config, accel_signal=accel_chunk, gyro_signal=gyro_chunk
    )
    nperseg = analysis_window.size
    feature_vector = extract_features(
        analysis_window,
        sample_rate_hz,
        accel_window,
        gyro_window,
        subject_id=SIMULATION_SUBJECT_ID,
        session_id=SIMULATION_SESSION_ID,
        window_id=0,
        nperseg=nperseg,
    )
    ml_output = predict(
        feature_vector.to_dict(),
        analysis_window,
        sample_rate_hz,
        model_path,
        scaler_path,
        nperseg=nperseg,
        config=config,
    )
    return ml_output, analysis_window, sample_rate_hz, nperseg


def run_experiment_cycle(
    analysis_chunk: NDArray[np.float64],
    state: ControllerState,
    model_path: Path,
    scaler_path: Path,
    config: Config,
    mode: ExperimentMode,
    fixed_params: Optional[StimParams] = None,
    accel_chunk: Optional[NDArray[np.float64]] = None,
    gyro_chunk: Optional[NDArray[np.float64]] = None,
) -> tuple[SimResult, ControllerState, InferenceResult]:
    """Run one cycle under one of the three experiment conditions.

    Args:
        analysis_chunk: shape (n_samples,), one window's worth of the
            dominant motion axis.
        state: Current ControllerState, threaded explicitly.
        model_path: Path to the trained classifier.
        scaler_path: Path to the paired StandardScaler.
        config: Loaded system configuration.
        mode: "no_mitigation" (Feature 44) -- decide_mitigation() is never
            called; every cycle is forced to no-mitigation, matching the
            build plan's literal "forced to False for every cycle".
            "fixed" (Feature 45) -- decide_mitigation() runs normally
            (real confidence/severity gating), but when it decides to
            mitigate, `fixed_params` is applied via simulation.apply()
            directly; adapt_params() (Feature 38) is never called.
            "adaptive" (Feature 46) -- delegates entirely to
            run_closed_loop_cycle() (Feature 41) unmodified.
        fixed_params: Required when mode == "fixed"; the static StimParams
            applied on every mitigating cycle.
        accel_chunk: Real calibrated accel data, or None for synthetic
            input (see simulation/closed_loop_runner.py).
        gyro_chunk: Real calibrated gyro data, or None for synthetic input.

    Returns:
        (sim_result, updated_state, ml_output).

    Raises:
        ValueError: If mode == "fixed" and fixed_params is None, or mode
            is not one of the three recognized values.
    """
    if mode == "adaptive":
        # Reuse Feature 42's closed-loop runner unmodified, per build plan.
        sim_result, updated_state = run_closed_loop_cycle(
            analysis_chunk, state, model_path, scaler_path, config=config,
            accel_signal=accel_chunk, gyro_signal=gyro_chunk,
        )
        # ml_output is not returned by run_closed_loop_cycle(); re-derive it
        # for the caller's record (feature extraction + inference are cheap
        # relative to the ODE solve, and this keeps run_closed_loop_cycle's
        # own contract untouched rather than widening it for this caller).
        ml_output, _window, _sr, _nperseg = _extract_features_and_infer(
            analysis_chunk, accel_chunk, gyro_chunk, model_path, scaler_path, config
        )
        return sim_result, updated_state, ml_output

    ml_output, analysis_window, sample_rate_hz, nperseg = _extract_features_and_infer(
        analysis_chunk, accel_chunk, gyro_chunk, model_path, scaler_path, config
    )
    latency_ms = config.simulation.latency_ms or 0.0

    if mode == "no_mitigation":
        logger.debug("run_experiment_cycle: mode=no_mitigation -- decide_mitigation() not called")
        return no_mitigation_result(analysis_window, latency_ms), state, ml_output

    if mode == "fixed":
        if fixed_params is None:
            raise ValueError("fixed_params is required when mode='fixed'")

        mitigate, state_after_decision = decide_mitigation(ml_output, state, config)
        if not mitigate:
            return no_mitigation_result(analysis_window, latency_ms), state_after_decision, ml_output

        tremor_state = {
            "y0": np.array([analysis_window[0], 0.0], dtype=np.float64),
            "duration_s": nperseg / sample_rate_hz,
            "timestep_s": config.simulation.timestep_s,
            "signal": analysis_window,
            "sample_rate_hz": sample_rate_hz,
        }
        sim_result = apply_simulation(fixed_params, tremor_state, config)

        # adapt_params() (Feature 38) is deliberately never called here --
        # that is the entire point of the fixed-parameter experiment.
        updated_state = ControllerState(
            hysteresis_active=state_after_decision.hysteresis_active,
            current_params=fixed_params,
            last_decision_timestamp=state_after_decision.last_decision_timestamp,
            adaptation_history=state_after_decision.adaptation_history,
        )
        return sim_result, updated_state, ml_output

    raise ValueError(f"Unknown experiment mode: {mode!r}")


def sim_result_to_record(
    cycle_index: int,
    sim_result: SimResult,
    state: ControllerState,
    ml_output: InferenceResult,
) -> dict:
    """Assemble one EXPERIMENT_RECORD_COLUMNS-shaped row from a cycle's outputs."""
    return {
        "cycle": cycle_index,
        "mitigate": state.hysteresis_active,
        "hysteresis_active": state.hysteresis_active,
        "achieved_suppression_pct": sim_result.achieved_suppression_pct,
        "residual_amplitude": sim_result.residual_amplitude,
        "stability_warning": sim_result.stability_warning,
        "latency_ms": sim_result.latency_ms,
        "amplitude": state.current_params.amplitude if state.current_params else None,
        "pulse_frequency_hz": state.current_params.pulse_frequency_hz if state.current_params else None,
        "duty_cycle": state.current_params.duty_cycle if state.current_params else None,
        "confidence": ml_output.confidence,
        "label": ml_output.label,
    }


def write_validation_report(
    output_dir: Path,
    experiment_type: str,
    records: list[dict],
    metrics: dict,
) -> Path:
    """Write report.json + cycle_log.csv, per architecture.md -> Validation Architecture.

    Outputs land in data/validation/<experiment_id>/report.json (JSON
    summary + metrics) and cycle_log.csv (raw per-cycle records) --
    "never used to retroactively edit the underlying pipeline", per
    architecture.md.

    Args:
        output_dir: data/validation/<experiment_id>/ directory to write into.
        experiment_type: One of "no_mitigation", "fixed", "adaptive" (or a
            caller-chosen label), recorded in report.json for traceability.
        records: Per-cycle records (EXPERIMENT_RECORD_COLUMNS-shaped).
        metrics: Flat dict of metric_name -> value (validation/metrics.py
            outputs), embedded directly in report.json.

    Returns:
        Path to the written report.json.
    """
    import csv
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_type": experiment_type,
        "n_cycles": len(records),
        "metrics": metrics,
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log_path = output_dir / "cycle_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_RECORD_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    logger.info("Wrote validation report to %s (%d cycles)", report_path, len(records))
    return report_path
