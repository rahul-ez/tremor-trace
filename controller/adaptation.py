"""Adaptation logic — increase, decrease, or maintain stimulation parameters.

Feature 38 of the build plan.

adapt_params() is called after simulation.apply() returns a SimResult for the
current cycle. It compares achieved_suppression_pct against target_suppression_pct
and updates StimParams according to three mutually exclusive cases:

  INCREASE : achieved_suppression_pct < target - suppression_tolerance_pct
             Stimulation is insufficient; increase amplitude by max_delta.amplitude
             toward param_bounds.amplitude.max, clamped to that bound.

  DECREASE : achieved_suppression_pct > target + suppression_tolerance_pct
             Suppression objective is met with margin; decrease amplitude toward
             param_bounds.amplitude.min to minimise exposure, per the objective
             stated in architecture.md -> Adaptive Controller -> Objective.

  MAINTAIN : |achieved - target| <= suppression_tolerance_pct
             Within the tolerance band; hold current params unchanged.

Amplitude is the primary adaptation lever in this implementation. Other fields
(pulse_width_us, duty_cycle, on_off_timing) are held at their current values
across all three cases; future phases may add multi-field adaptation.

INVALID INPUT -> MAINTAIN:
  - achieved_suppression_pct is None (SimResult not available or estimation
    failed upstream).
  - stability_warning is True (SimResult.stability_warning signalled an
    integration instability — ODE diverged or constraint violated in
    stimulation_model.py). Never increase parameters when the simulation
    flagged instability.

Per architecture.md -> Adaptive Controller and code-standards.md:
- Controller input must be the documented StimParams + SimResult + config
  contracts; no raw signal, PSD, or feature vector is accepted here.
- All emitted StimParams must be clamped to config.controller.param_bounds.*;
  bound_clamped=True is logged at WARNING level on any clamp.
- Adaptation per cycle must not exceed config.controller.max_delta_per_step
  per field.
- Decisions are deterministic given identical inputs and state.
- ControllerState is passed in and returned explicitly; never a global.
"""

import logging
from typing import Optional

from controller.controller_state import ControllerState
from tremor_system.config import Config, MaxDeltaConfig, ParamBoundsConfig, load_config
from tremor_system.types import StimParams

logger = logging.getLogger(__name__)

# Human-readable strings for the three adaptation decisions. Matched by
# validation metrics and controller dashboard (Feature 54).
ADAPTATION_INCREASE = "INCREASE"
ADAPTATION_DECREASE = "DECREASE"
ADAPTATION_MAINTAIN = "MAINTAIN"


def _require_suppression_tolerance(config: Config) -> float:
    """Return suppression_tolerance_pct, raising if not configured.

    Args:
        config: Loaded system configuration.

    Returns:
        Suppression tolerance as a non-negative float (percentage points).

    Raises:
        ValueError: If config.controller.suppression_tolerance_pct is None.
    """
    tol = config.controller.suppression_tolerance_pct
    if tol is None:
        raise ValueError(
            "config.controller.suppression_tolerance_pct is None. "
            "Set a value in config/system_config.yaml before running the controller."
        )
    return tol


def _require_max_delta(config: Config) -> MaxDeltaConfig:
    """Return max_delta_per_step, raising if not configured.

    Args:
        config: Loaded system configuration.

    Returns:
        Typed MaxDeltaConfig with per-field caps.

    Raises:
        ValueError: If config.controller.max_delta_per_step is None.
    """
    mds = config.controller.max_delta_per_step
    if mds is None:
        raise ValueError(
            "config.controller.max_delta_per_step is None. "
            "Set all per-field delta caps in config/system_config.yaml before "
            "running the controller."
        )
    return mds


def _require_param_bounds(config: Config) -> ParamBoundsConfig:
    """Return param_bounds, raising if not configured.

    Args:
        config: Loaded system configuration.

    Returns:
        Typed ParamBoundsConfig.

    Raises:
        ValueError: If config.controller.param_bounds is None.
    """
    bounds = config.controller.param_bounds
    if bounds is None:
        raise ValueError(
            "config.controller.param_bounds is None. "
            "Set all parameter bounds in config/system_config.yaml before "
            "running the controller."
        )
    return bounds


def _clamp_with_log(value: float, lo: float, hi: float, field: str) -> float:
    """Clamp value to [lo, hi], logging a WARNING if clamping occurs.

    Per code-standards.md -> Adaptive Controller: "a parameter selection or
    adaptation step that would exceed a bound must clamp to the bound and log
    a bound_clamped=True event — never silently exceed or silently clamp."

    Args:
        value: Candidate parameter value after delta application.
        lo: Lower bound (from param_bounds).
        hi: Upper bound (from param_bounds).
        field: StimParams field name, used in the log message.

    Returns:
        Value clamped to [lo, hi].
    """
    if value < lo:
        logger.warning(
            "bound_clamped=True: %s=%.4f clamped to min=%.4f",
            field,
            value,
            lo,
        )
        return lo
    if value > hi:
        logger.warning(
            "bound_clamped=True: %s=%.4f clamped to max=%.4f",
            field,
            value,
            hi,
        )
        return hi
    return value


def adapt_params(
    current_params: StimParams,
    achieved_suppression_pct: Optional[float],
    config: Optional[Config] = None,
    state: Optional[ControllerState] = None,
    stability_warning: bool = False,
) -> tuple[StimParams, ControllerState]:
    """Adjust stimulation parameters based on measured suppression.

    Called once per adaptation cycle, after simulation.apply() returns a
    SimResult. Uses the suppression gap to determine whether to increase,
    decrease, or maintain current parameters.

    Amplitude is the primary adaptation lever:
    - INCREASE: amplitude += min(max_delta.amplitude, distance_to_max_bound)
    - DECREASE: amplitude -= min(max_delta.amplitude, distance_to_min_bound)
    - MAINTAIN: amplitude unchanged

    All other StimParams fields are unchanged in this implementation.

    Args:
        current_params: StimParams currently being applied. The returned
            StimParams will differ only in the adapted field(s).
        achieved_suppression_pct: Suppression measured by the simulation for
            the most recent cycle (from SimResult.achieved_suppression_pct).
            Pass None if the result is unavailable or invalid — triggers
            MAINTAIN.
        config: Loaded system configuration. Defaults to load_config().
        state: Current ControllerState. If None, a fresh ControllerState is
            used (useful for isolated unit tests). The adaptation event is
            appended to adaptation_history in the returned state.
        stability_warning: True if SimResult.stability_warning was set for
            this cycle. Triggers MAINTAIN regardless of suppression value,
            per code-standards.md -> Adaptive Controller (invalid input ->
            maintain).

    Returns:
        A tuple of (new_params: StimParams, updated_state: ControllerState).
        new_params has amplitude adjusted per the adaptation decision.
        updated_state has the new event appended to adaptation_history and
        current_params updated to new_params.

    Raises:
        ValueError: If any required config field (suppression_tolerance_pct,
            max_delta_per_step, param_bounds) is None (fail-fast).
    """
    resolved_config = config or load_config()
    resolved_state = state if state is not None else ControllerState()

    tolerance = _require_suppression_tolerance(resolved_config)
    max_delta = _require_max_delta(resolved_config)
    bounds = _require_param_bounds(resolved_config)

    target = resolved_config.controller.target_suppression_pct

    # --- Determine adaptation decision ---
    if stability_warning:
        decision = ADAPTATION_MAINTAIN
        logger.warning(
            "stability_warning=True — defaulting to MAINTAIN "
            "(suppression=%.1f%%, target=%.1f%%).",
            achieved_suppression_pct if achieved_suppression_pct is not None else float("nan"),
            target,
        )
    elif achieved_suppression_pct is None:
        decision = ADAPTATION_MAINTAIN
        logger.warning(
            "achieved_suppression_pct=None — defaulting to MAINTAIN "
            "(invalid/missing SimResult; target=%.1f%%).",
            target,
        )
    elif achieved_suppression_pct < target - tolerance:
        decision = ADAPTATION_INCREASE
    elif achieved_suppression_pct > target + tolerance:
        decision = ADAPTATION_DECREASE
    else:
        decision = ADAPTATION_MAINTAIN

    # --- Apply adaptation to amplitude (primary lever) ---
    current_amplitude = current_params.amplitude

    if decision == ADAPTATION_INCREASE:
        raw_new_amplitude = current_amplitude + max_delta.amplitude
        new_amplitude = _clamp_with_log(
            raw_new_amplitude,
            bounds.amplitude.min,
            bounds.amplitude.max,
            "amplitude",
        )
        logger.info(
            "INCREASE: amplitude %.4f -> %.4f "
            "(achieved=%.1f%% < target-tol=%.1f%%).",
            current_amplitude,
            new_amplitude,
            achieved_suppression_pct,
            target - tolerance,
        )
    elif decision == ADAPTATION_DECREASE:
        raw_new_amplitude = current_amplitude - max_delta.amplitude
        new_amplitude = _clamp_with_log(
            raw_new_amplitude,
            bounds.amplitude.min,
            bounds.amplitude.max,
            "amplitude",
        )
        logger.info(
            "DECREASE: amplitude %.4f -> %.4f "
            "(achieved=%.1f%% > target+tol=%.1f%%).",
            current_amplitude,
            new_amplitude,
            achieved_suppression_pct,
            target + tolerance,
        )
    else:
        new_amplitude = current_amplitude
        logger.debug(
            "MAINTAIN: amplitude=%.4f unchanged "
            "(achieved=%s%%, target±tol=[%.1f%%, %.1f%%]).",
            current_amplitude,
            f"{achieved_suppression_pct:.1f}" if achieved_suppression_pct is not None else "N/A",
            target - tolerance,
            target + tolerance,
        )

    # All other StimParams fields are unchanged.
    new_params = StimParams(
        amplitude=new_amplitude,
        pulse_frequency_hz=current_params.pulse_frequency_hz,
        pulse_width_us=current_params.pulse_width_us,
        duty_cycle=current_params.duty_cycle,
        on_off_timing=current_params.on_off_timing,
        phase=current_params.phase,
    )

    # --- Update ControllerState ---
    history_entry: dict = {
        "decision": decision,
        "achieved_suppression_pct": achieved_suppression_pct,
        "target_suppression_pct": target,
        "stability_warning": stability_warning,
        "amplitude_before": current_amplitude,
        "amplitude_after": new_amplitude,
    }
    updated_state = ControllerState(
        hysteresis_active=resolved_state.hysteresis_active,
        current_params=new_params,
        last_decision_timestamp=resolved_state.last_decision_timestamp,
        adaptation_history=resolved_state.adaptation_history + [history_entry],
    )

    return new_params, updated_state
