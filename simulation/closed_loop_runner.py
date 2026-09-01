"""Closed-loop runner: wires controller (Phase 7) to simulation (Phase 6).

Feature 41 of the build plan. Runs one full Detect -> Analyze -> Decide ->
Mitigate -> Measure -> Adapt cycle on a single tremor-signal window, per
architecture.md's documented "Full closed-loop data flow (single window
cycle)". Each stage receives only the documented interface output of the
previous stage -- no stage reaches past its neighbor's contract.

This module accepts a raw tremor-signal window (synthetic, from Feature 32,
or a slice of a real recorded session, from Feature 09/21) and internally
runs Features 12-20 (axis handling through feature extraction), 27/30 (ML
inference), 40 (controller cycle), 35 (simulation.apply), and 38
(adaptation) in sequence.

Recorded-session callers should also pass the real accel_signal/gyro_signal
windows (see run_closed_loop_cycle()'s optional parameters) so
accel_magnitude/gyro_magnitude features reflect real sensor data. When
omitted -- the synthetic-input case, which has no 3-axis motion to offer --
tremor_signal is embedded into a synthetic 3-axis shape as a fallback; see
build_single_window()'s docstring.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from controller.adaptation import adapt_params
from controller.controller_state import ControllerState
from controller.cycle import run_controller_cycle
from estimation.amplitude_estimation import estimate_amplitude
from ml.inference import predict
from signal_processing.feature_extraction import extract_features
from signal_processing.filtering import bandpass_filter, remove_baseline
from signal_processing.windowing import segment_windows
from simulation import apply as apply_simulation
from tremor_system.config import Config, load_config
from tremor_system.types import SimResult

logger = logging.getLogger(__name__)

# Placeholder identity for feature-vector metadata when the input is
# synthetic or otherwise not tied to a real subject/session on disk.
SIMULATION_SUBJECT_ID = "simulation"
SIMULATION_SESSION_ID = "closed_loop"


def build_single_window(
    tremor_signal: NDArray[np.float64],
    config: Config,
    accel_signal: NDArray[np.float64] | None = None,
    gyro_signal: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Run baseline removal, band-pass filtering, and windowing for one cycle.

    tremor_signal (from generate_synthetic_tremor() or a recorded-session
    slice) is the already axis-selected dominant motion axis -- for
    synthetic input this is true by construction; for recorded input the
    caller has already run calibration/select_strongest_axis (Feature 43).

    accel_signal/gyro_signal, when provided, are the REAL calibrated 3-axis
    windows (from a recorded session) used for the accel_magnitude/
    gyro_magnitude features. When omitted (the synthetic-input case, which
    has no real 3-axis motion to offer), tremor_signal is embedded into a
    synthetic 3-axis accelerometer window (x = tremor_signal, z ~ 1g static
    baseline, y = 0) and a zero gyroscope window purely so
    extract_features()'s documented accel_window/gyro_window parameters
    have something to consume; this embedding has no physical meaning.

    Args:
        tremor_signal: shape (n_samples,), units g. Must contain at least
            one full window's worth of samples (config.signal.window_length_s
            at config.sensor.sample_rate_hz).
        config: Loaded system configuration.
        accel_signal: shape (n_samples, 3), units g -- real calibrated
            accelerometer data aligned sample-for-sample with tremor_signal.
            None for synthetic input.
        gyro_signal: shape (n_samples, 3), units deg/s -- real calibrated
            gyroscope data aligned sample-for-sample with tremor_signal.
            None for synthetic input.

    Returns:
        (analysis_window, accel_window, gyro_window, sample_rate_hz) for
        the first (and only) complete window.

    Raises:
        ValueError: If tremor_signal is too short to produce one complete
            window, or if a provided accel_signal/gyro_signal does not
            match tremor_signal's sample count.
    """
    sample_rate_hz = config.sensor.sample_rate_hz
    n_samples = tremor_signal.size

    if accel_signal is None:
        accel_signal = np.zeros((n_samples, 3), dtype=np.float64)
        accel_signal[:, 0] = tremor_signal
        accel_signal[:, 2] = 1.0  # static ~1g baseline on z, for a plausible accel_magnitude
    elif accel_signal.shape[0] != n_samples:
        raise ValueError(
            f"accel_signal has {accel_signal.shape[0]} samples, expected {n_samples} "
            "to match tremor_signal"
        )

    if gyro_signal is None:
        gyro_signal = np.zeros((n_samples, 3), dtype=np.float64)
    elif gyro_signal.shape[0] != n_samples:
        raise ValueError(
            f"gyro_signal has {gyro_signal.shape[0]} samples, expected {n_samples} "
            "to match tremor_signal"
        )

    analysis_signal = tremor_signal
    analysis_filtered = bandpass_filter(
        remove_baseline(analysis_signal),
        sample_rate_hz=sample_rate_hz,
        band_hz=tuple(config.signal.tremor_band_hz),
        order=config.signal.filter_order,
    )

    window_length_s = config.signal.window_length_s
    overlap_pct = config.signal.window_overlap_pct

    analysis_windows = segment_windows(analysis_filtered, sample_rate_hz, window_length_s, overlap_pct)
    accel_windows = segment_windows(accel_signal, sample_rate_hz, window_length_s, overlap_pct)
    gyro_windows = segment_windows(gyro_signal, sample_rate_hz, window_length_s, overlap_pct)

    if analysis_windows.shape[0] == 0:
        raise ValueError(
            f"tremor_signal has {n_samples} samples at {sample_rate_hz} Hz "
            f"({n_samples / sample_rate_hz:.2f}s), too short for one complete "
            f"{window_length_s}s window"
        )

    return analysis_windows[0], accel_windows[0], gyro_windows[0], sample_rate_hz


def no_mitigation_result(window: NDArray[np.float64], latency_ms: float) -> SimResult:
    """SimResult sentinel for a mitigate=False cycle -- no simulation ran.

    Args:
        window: The analysis window for this cycle, unchanged (tremor
            persists since no stimulation was applied).
        latency_ms: config.simulation.latency_ms, reported for consistency
            even though no actuation occurred.

    Returns:
        SimResult with achieved_suppression_pct=0.0, post_mitigation_signal
        equal to the input window, and stability_warning=False.
    """
    return SimResult(
        post_mitigation_signal=window.copy(),
        achieved_suppression_pct=0.0,
        residual_amplitude=estimate_amplitude(window),
        latency_ms=latency_ms,
        stability_warning=False,
    )


def run_closed_loop_cycle(
    tremor_signal: NDArray[np.float64],
    state: ControllerState,
    model_path: Path,
    scaler_path: Path,
    config: Config | None = None,
    accel_signal: NDArray[np.float64] | None = None,
    gyro_signal: NDArray[np.float64] | None = None,
) -> tuple[SimResult, ControllerState]:
    """Run one full Detect -> Decide -> Mitigate -> Measure -> Adapt cycle.

    Args:
        tremor_signal: shape (n_samples,), units g. One window's worth (or
            more, in which case only the first window is used) of the
            already axis-selected dominant motion axis -- synthetic
            (Feature 32) or a recorded-session slice (Feature 43).
        state: Current ControllerState, threaded explicitly.
        model_path: Path to the trained classifier (ml/train.py output).
        scaler_path: Path to the paired StandardScaler.
        config: Loaded system configuration; defaults to load_config().
        accel_signal: shape (n_samples, 3), units g -- real calibrated
            accelerometer data aligned with tremor_signal, for accurate
            accel_magnitude features. Omit for synthetic input (there is no
            real 3-axis data to offer); see build_single_window().
        gyro_signal: shape (n_samples, 3), units deg/s -- real calibrated
            gyroscope data aligned with tremor_signal, for accurate
            gyro_magnitude features. Omit for synthetic input.

    Returns:
        (sim_result, updated_state). sim_result reflects a real
        simulation.apply() call when the controller decided to mitigate;
        otherwise a "no mitigation" sentinel (see no_mitigation_result)
        with achieved_suppression_pct=0.0. updated_state reflects
        decide_mitigation()'s hysteresis update and, when mitigation
        occurred, adapt_params()'s parameter adjustment.

    Raises:
        ValueError: If tremor_signal is too short for one window, or if any
            required config field (confidence_threshold, severity_threshold,
            hysteresis_pct, param_bounds, max_delta_per_step,
            suppression_tolerance_pct, simulation.timestep_s,
            simulation.latency_ms) is not configured -- propagated from the
            underlying controller/simulation calls (fail-fast).
    """
    resolved_config = config or load_config()

    analysis_window, accel_window, gyro_window, sample_rate_hz = build_single_window(
        tremor_signal, resolved_config, accel_signal=accel_signal, gyro_signal=gyro_signal
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
        config=resolved_config,
    )

    params, state_after_decision = run_controller_cycle(ml_output, state, resolved_config)

    latency_ms = resolved_config.simulation.latency_ms or 0.0

    if params is None:
        logger.debug("run_closed_loop_cycle: mitigate=False -- no simulation this cycle")
        return no_mitigation_result(analysis_window, latency_ms), state_after_decision

    y0 = np.array([analysis_window[0], 0.0], dtype=np.float64)
    tremor_state = {
        "y0": y0,
        "duration_s": nperseg / sample_rate_hz,
        "timestep_s": resolved_config.simulation.timestep_s,
        "signal": analysis_window,
        "sample_rate_hz": sample_rate_hz,
    }

    sim_result = apply_simulation(params, tremor_state, resolved_config)

    new_params, state_after_adaptation = adapt_params(
        params,
        sim_result.achieved_suppression_pct,
        resolved_config,
        state_after_decision,
        sim_result.stability_warning,
    )

    logger.info(
        "run_closed_loop_cycle: mitigate=True, achieved_suppression_pct=%.1f%%, "
        "next amplitude=%.4f",
        sim_result.achieved_suppression_pct,
        new_params.amplitude,
    )

    return sim_result, state_after_adaptation
