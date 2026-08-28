"""Public entry point for the simulation package (Feature 35).

The single documented interface for external callers is `apply()`.
No code outside `simulation/` should call `simulate_tremor_response`
or `compute_suppression_pct` directly.

This module is not a clinically validated physiological model.  It exists
only for controller development and validation, and it has no physical
stimulation hardware path.
"""

import logging
from collections.abc import Mapping

import numpy as np

from signal_processing.time_domain import compute_rms
from simulation.stimulation_model import simulate_tremor_response
from simulation.suppression import compute_suppression_pct
from tremor_system.config import Config, load_config
from tremor_system.types import SimResult, StimParams

logger = logging.getLogger(__name__)


def _validate_stim_params(params: StimParams) -> None:
    """Structural validation of StimParams before passing to the simulation.

    Enforces type correctness and physical feasibility constraints that can be
    checked without knowing the configured numeric bounds.  Full numeric
    min/max range enforcement is deferred to when config.controller.param_bounds
    is defined -- see architecture.md Open Question #7.

    Raises:
        ValueError: If any field is non-finite, has the wrong sign, or violates
            a structural invariant (e.g. duty_cycle outside [0, 1]).
    """
    if not np.isfinite(params.amplitude) or params.amplitude < 0.0:
        raise ValueError(
            f"StimParams.amplitude must be finite and non-negative, got {params.amplitude}"
        )
    if not np.isfinite(params.pulse_frequency_hz) or params.pulse_frequency_hz < 0.0:
        raise ValueError(
            f"StimParams.pulse_frequency_hz must be finite and non-negative, "
            f"got {params.pulse_frequency_hz}"
        )
    if not np.isfinite(params.pulse_width_us) or params.pulse_width_us < 0.0:
        raise ValueError(
            f"StimParams.pulse_width_us must be finite and non-negative, "
            f"got {params.pulse_width_us}"
        )
    if not np.isfinite(params.duty_cycle) or not (0.0 <= params.duty_cycle <= 1.0):
        raise ValueError(
            f"StimParams.duty_cycle must be finite and in [0.0, 1.0], "
            f"got {params.duty_cycle}"
        )
    on_ms, off_ms = params.on_off_timing
    if not np.isfinite(on_ms) or on_ms < 0.0:
        raise ValueError(
            f"StimParams.on_off_timing[0] (on_ms) must be finite and non-negative, "
            f"got {on_ms}"
        )
    if not np.isfinite(off_ms) or off_ms < 0.0:
        raise ValueError(
            f"StimParams.on_off_timing[1] (off_ms) must be finite and non-negative, "
            f"got {off_ms}"
        )
    if params.phase is not None and not np.isfinite(params.phase):
        raise ValueError(
            f"StimParams.phase must be finite when provided, got {params.phase}"
        )


def apply(
    params: StimParams,
    tremor_state: Mapping[str, object],
    config: Config | None = None,
) -> SimResult:
    """Run one stimulation simulation cycle and return the full result.

    This is the sole public entry point for `simulation/`.  Validates
    stimulation parameters, drives the ODE-based tremor-response model,
    measures tremor-band suppression, and returns a complete `SimResult`.

    This function is not a clinically validated physiological model.  It exists
    only for controller development and validation, and it has no physical
    stimulation hardware path.

    Args:
        params: Stimulation parameters from the controller.  Structurally
            validated here (non-negative, finite); numeric bounds are enforced
            once `config.controller.param_bounds` is configured (Open Question #7).
        tremor_state: Dict with keys `y0` (shape (2,)), `duration_s`,
            `timestep_s`, `signal` (shape (n_samples,), units g), and
            `sample_rate_hz`.
        config: Loaded system configuration; defaults to `load_config()`.

    Returns:
        `SimResult` with `post_mitigation_signal`, `achieved_suppression_pct`,
        `residual_amplitude`, `latency_ms`, and `stability_warning`.

    Raises:
        ValueError: If `params` fails structural validation or if
            `tremor_state` is malformed.
    """
    _validate_stim_params(params)

    resolved_config = config or load_config()
    tremor_band_hz = tuple(resolved_config.signal.tremor_band_hz)
    sample_rate_hz = float(tremor_state["sample_rate_hz"])
    latency_ms = resolved_config.simulation.latency_ms or 0.0

    pre_signal = np.asarray(tremor_state["signal"], dtype=np.float64)

    post_signal, stability_warning = simulate_tremor_response(
        params, tremor_state, resolved_config
    )

    if stability_warning:
        logger.warning(
            "Simulation ODE reported a stability warning; "
            "suppression measurement may be unreliable."
        )

    achieved_suppression_pct = compute_suppression_pct(
        pre_signal, post_signal, sample_rate_hz, tremor_band_hz
    )
    residual_amplitude = compute_rms(post_signal)

    return SimResult(
        post_mitigation_signal=post_signal,
        achieved_suppression_pct=achieved_suppression_pct,
        residual_amplitude=residual_amplitude,
        latency_ms=latency_ms,
        stability_warning=stability_warning,
    )
