"""Simulation-only tremor response model.

This is not a clinically validated physiological model. It exists only for
controller development and validation, and it has no physical stimulation
hardware path.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from tremor_system.config import Config, load_config
from tremor_system.types import StimParams

# Converts StimParams.amplitude (units abstract, bounded by
# config.controller.param_bounds.amplitude, v1 range [0, 5]) into the ODE's
# damping coefficient, in rad/s. Fix for a discovered scale mismatch: with
# no gain (damping = amplitude directly), the maximum achievable damping
# (5 rad/s) was two orders of magnitude below tremor angular frequencies
# (2*pi*4Hz..2*pi*12Hz = 25..75 rad/s), capping achievable suppression at
# under 4% regardless of amplitude -- see memory.md Phase 8 session update.
#
# Calibrated so that amplitude = half of the v1 max bound (2.5) yields
# damping = 2*pi*6Hz (37.7 rad/s), i.e. ~50% tremor-band power suppression
# at the band's midpoint frequency at half of the available amplitude
# range -- leaving headroom for adapt_params to increase toward stronger
# suppression or decrease toward the target from above. At the v1 max
# amplitude (5.0), suppression reaches ~50% even at the tremor band's
# hardest-to-damp edge (12 Hz) and up to ~90% at the easiest (4 Hz).
# Not a clinically validated relationship -- a calibration choice for
# controller development, not a hardware fact. Revisit if
# config.controller.param_bounds.amplitude.max or target_suppression_pct
# change.
DAMPING_GAIN_RAD_S_PER_UNIT_AMPLITUDE = 15.0

REQUIRED_TREMOR_STATE_KEYS = {
    "y0",
    "duration_s",
    "timestep_s",
    "signal",
    "sample_rate_hz",
}


def _validate_tremor_state(tremor_state: Mapping[str, object]) -> tuple[
    NDArray[np.float64], float, float, NDArray[np.float64], float
]:
    missing = REQUIRED_TREMOR_STATE_KEYS - set(tremor_state)
    if missing:
        raise ValueError(f"tremor_state is missing required keys: {sorted(missing)}")

    y0 = np.asarray(tremor_state["y0"], dtype=np.float64)
    signal = np.asarray(tremor_state["signal"], dtype=np.float64)
    duration_s = float(tremor_state["duration_s"])
    timestep_s = float(tremor_state["timestep_s"])
    sample_rate_hz = float(tremor_state["sample_rate_hz"])

    if y0.shape != (2,) or not np.all(np.isfinite(y0)):
        raise ValueError("tremor_state['y0'] must be a finite shape (2,) array")
    if signal.ndim != 1 or signal.size < 2 or not np.all(np.isfinite(signal)):
        raise ValueError(
            "tremor_state['signal'] must be a finite 1D array with at least 2 samples"
        )
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("tremor_state['duration_s'] must be positive and finite")
    if not np.isfinite(timestep_s) or timestep_s <= 0.0:
        raise ValueError("tremor_state['timestep_s'] must be positive and finite")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("tremor_state['sample_rate_hz'] must be positive and finite")

    expected_duration_s = signal.size / sample_rate_hz
    if not np.isclose(duration_s, expected_duration_s, rtol=0.0, atol=1.0 / sample_rate_hz):
        raise ValueError(
            "tremor_state duration_s must match signal length and sample_rate_hz"
        )

    return y0, duration_s, timestep_s, signal, sample_rate_hz


def simulate_tremor_response(
    params: StimParams,
    tremor_state: Mapping[str, object],
    config: Config | None = None,
) -> tuple[NDArray[np.float64], bool]:
    """Simulate a delayed, amplitude-dependent tremor response.

    This is not a clinically validated physiological model. The full input
    signal drives a two-state ODE whose states are residual position and
    residual velocity. Stimulation adds damping after the configured latency.

    Args:
        params: Simulation-only stimulation parameters. ``amplitude`` controls
            damping strength; other fields remain part of the shared contract.
        tremor_state: Dict containing ``y0`` shape (2,), ``duration_s`` in s,
            ``timestep_s`` in s, ``signal`` shape (n_samples,) in g, and
            ``sample_rate_hz`` in Hz.
        config: Loaded system configuration; defaults to ``load_config()``.

    Returns:
        A tuple of post-mitigation signal shape (n_samples,), units g, and a
        stability warning flag. Output samples align with the input signal.

    Raises:
        ValueError: If the tremor state, timing, or stimulation amplitude is
            invalid.
    """
    if not np.isfinite(params.amplitude) or params.amplitude < 0.0:
        raise ValueError("params.amplitude must be finite and non-negative")

    y0, duration_s, timestep_s, signal, sample_rate_hz = _validate_tremor_state(
        tremor_state
    )
    resolved_config = config or load_config()
    latency_ms = resolved_config.simulation.latency_ms
    if latency_ms is None or not np.isfinite(latency_ms) or latency_ms < 0.0:
        raise ValueError("simulation.latency_ms must be configured and non-negative")

    time_s = np.arange(signal.size, dtype=np.float64) / sample_rate_hz
    forcing_velocity = np.gradient(signal, time_s)
    forcing_acceleration = np.gradient(forcing_velocity, time_s)
    latency_s = latency_ms / 1000.0

    def rhs(current_time_s: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        forcing_velocity_at_time = float(np.interp(current_time_s, time_s, forcing_velocity))
        forcing_acceleration_at_time = float(np.interp(current_time_s, time_s, forcing_acceleration))
        damping = (
            DAMPING_GAIN_RAD_S_PER_UNIT_AMPLITUDE * params.amplitude
            if current_time_s >= latency_s
            else 0.0
        )
        return np.asarray(
            [forcing_velocity_at_time - damping * state[0],
             forcing_acceleration_at_time - damping * state[1]],
            dtype=np.float64,
        )

    result = solve_ivp(
        rhs,
        (0.0, duration_s),
        y0,
        t_eval=time_s,
        max_step=timestep_s,
    )

    input_peak = float(np.max(np.abs(signal)))
    if input_peak == 0.0:
        state_limit = 0.0
    else:
        state_limit = input_peak
    stability_warning = not result.success or result.y.shape[1] != signal.size
    if result.y.shape[0] != 2 or result.y.shape[1] != signal.size:
        return np.asarray(signal, dtype=np.float64), True

    post_signal = np.asarray(result.y[0], dtype=np.float64)
    latency_samples = int(np.searchsorted(time_s, latency_s, side="left"))
    post_signal[:latency_samples] = signal[:latency_samples]
    if params.amplitude == 0.0:
        post_signal = np.asarray(signal, dtype=np.float64)
    tolerance = np.finfo(np.float64).eps * max(1.0, state_limit)
    if not np.all(np.isfinite(post_signal)):
        stability_warning = True
        post_signal = np.nan_to_num(post_signal, nan=0.0, posinf=state_limit, neginf=-state_limit)
    if np.any(np.abs(post_signal) > state_limit + tolerance):
        stability_warning = True
    post_signal = np.clip(post_signal, -state_limit, state_limit)

    return post_signal, stability_warning
