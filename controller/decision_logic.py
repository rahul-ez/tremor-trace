"""Confidence gating and mitigation decision logic.

Feature 36 of the build plan.

Owns the single decision question: should the controller mitigate this cycle?

Decision flow (applied in strict order):
1. Confidence gate — if confidence < config.ml.confidence_threshold, the
   answer is always False, logged as LOW_CONFIDENCE_NO_ACTION.  This is a
   distinct state, not identical to a genuine no-tremor result.
2. Label gate — if InferenceResult.label is False (ML says no tremor),
   the answer is False.
3. Hysteresis — severity is compared against the enter OR exit threshold
   depending on whether mitigation is currently active:
     - Not active: enter when severity >= severity_threshold (enter threshold).
     - Active:     exit  when severity <  exit_threshold
                         (= severity_threshold - hysteresis_pct / 100).
   This prevents rapid ON/OFF chatter across the boundary.

Per architecture.md -> Adaptive Controller -> Confidence gating and
code-standards.md -> Adaptive Controller:
- Controller must NEVER accept raw signal, PSD, or feature vector as input.
- Confidence gate is mandatory and checked FIRST.
- Hysteresis uses distinct enter/exit thresholds.
- All decisions are deterministic — no randomness here.
- ControllerState is always passed in and returned (never a global).
"""

import logging
from typing import Optional

from controller.controller_state import ControllerState
from tremor_system.config import Config, load_config
from tremor_system.types import InferenceResult

logger = logging.getLogger(__name__)

# Log-event string used for the distinct low-confidence non-action state.
# Consumers (dashboard, validation) should match this literal, not infer it.
LOW_CONFIDENCE_NO_ACTION = "LOW_CONFIDENCE_NO_ACTION"


def _require_confidence_threshold(config: Config) -> float:
    """Return the confidence threshold, raising if it is not configured.

    Args:
        config: Loaded system configuration.

    Returns:
        Confidence threshold as a float in [0, 1].

    Raises:
        ValueError: If config.ml.confidence_threshold is None.
    """
    threshold = config.ml.confidence_threshold
    if threshold is None:
        raise ValueError(
            "config.ml.confidence_threshold is None. "
            "Set a value in config/system_config.yaml before running the controller."
        )
    return threshold


def _require_severity_threshold(config: Config) -> float:
    """Return the severity threshold, raising if it is not configured.

    Args:
        config: Loaded system configuration.

    Returns:
        Severity threshold as a float in [0, 1].

    Raises:
        ValueError: If config.controller.severity_threshold is None.
    """
    threshold = config.controller.severity_threshold
    if threshold is None:
        raise ValueError(
            "config.controller.severity_threshold is None. "
            "Set a value in config/system_config.yaml before running the controller."
        )
    return threshold


def _require_hysteresis_pct(config: Config) -> float:
    """Return the hysteresis band, raising if it is not configured.

    Args:
        config: Loaded system configuration.

    Returns:
        Hysteresis percentage as a non-negative float.

    Raises:
        ValueError: If config.controller.hysteresis_pct is None.
    """
    hysteresis_pct = config.controller.hysteresis_pct
    if hysteresis_pct is None:
        raise ValueError(
            "config.controller.hysteresis_pct is None. "
            "Set a value in config/system_config.yaml before running the controller."
        )
    return hysteresis_pct


def decide_mitigation(
    ml_output: InferenceResult,
    state: ControllerState,
    config: Optional[Config] = None,
) -> tuple[bool, ControllerState]:
    """Decide whether to activate mitigation for this cycle.

    Applies the three-stage decision in strict order:
    1. Confidence gate (always checked first).
    2. Label gate (ML classification).
    3. Hysteresis-gated severity threshold.

    Args:
        ml_output: Output from ml/inference.py for the current window. The
            controller consumes ONLY this documented contract — never raw
            signal, PSD, or feature vectors directly (architecture.md
            invariant 8).
        state: Current ControllerState. hysteresis_active is read to select
            enter vs exit threshold.
        config: Loaded system configuration. Defaults to load_config().

    Returns:
        A tuple of (mitigate: bool, updated_state: ControllerState).
        updated_state has hysteresis_active set to match the new decision.

    Raises:
        ValueError: If any required config threshold is None (fail-fast —
            the controller must not silently fall back to an arbitrary value).
    """
    resolved_config = config or load_config()

    confidence_threshold = _require_confidence_threshold(resolved_config)
    severity_threshold = _require_severity_threshold(resolved_config)
    hysteresis_pct = _require_hysteresis_pct(resolved_config)

    # Exit threshold is always strictly below the enter threshold.
    # hysteresis_pct is a percentage in [0, 100] that maps to a severity-unit band.
    exit_threshold = severity_threshold - (hysteresis_pct / 100.0)

    # --- Stage 1: Confidence gate (mandatory first check) ---
    if ml_output.confidence < confidence_threshold:
        logger.warning(
            "%s: confidence=%.3f below threshold=%.3f — no mitigation change, "
            "severity=%.3f, label=%s",
            LOW_CONFIDENCE_NO_ACTION,
            ml_output.confidence,
            confidence_threshold,
            ml_output.severity,
            ml_output.label,
        )
        # State is unchanged: hysteresis position is preserved.
        return False, state

    # --- Stage 2: Label gate (ML classification) ---
    if not ml_output.label:
        if state.hysteresis_active:
            logger.info(
                "ML label=False (no tremor) with confidence=%.3f — deactivating mitigation.",
                ml_output.confidence,
            )
            new_state = ControllerState(
                hysteresis_active=False,
                current_params=state.current_params,
                last_decision_timestamp=state.last_decision_timestamp,
                adaptation_history=state.adaptation_history,
            )
            return False, new_state
        return False, state

    # --- Stage 3: Hysteresis-gated severity threshold ---
    severity = ml_output.severity

    if state.hysteresis_active:
        # Currently mitigating: only exit if severity drops below exit threshold.
        if severity < exit_threshold:
            logger.info(
                "Severity=%.3f below exit_threshold=%.3f — deactivating mitigation.",
                severity,
                exit_threshold,
            )
            new_state = ControllerState(
                hysteresis_active=False,
                current_params=state.current_params,
                last_decision_timestamp=state.last_decision_timestamp,
                adaptation_history=state.adaptation_history,
            )
            return False, new_state
        else:
            # Remain in mitigation (hysteresis hold).
            logger.debug(
                "Hysteresis hold: severity=%.3f >= exit_threshold=%.3f — maintaining mitigation.",
                severity,
                exit_threshold,
            )
            return True, state
    else:
        # Not currently mitigating: only enter if severity reaches enter threshold.
        if severity >= severity_threshold:
            logger.info(
                "Severity=%.3f >= enter_threshold=%.3f — activating mitigation.",
                severity,
                severity_threshold,
            )
            new_state = ControllerState(
                hysteresis_active=True,
                current_params=state.current_params,
                last_decision_timestamp=state.last_decision_timestamp,
                adaptation_history=state.adaptation_history,
            )
            return True, new_state
        else:
            logger.debug(
                "Severity=%.3f < enter_threshold=%.3f — no mitigation.",
                severity,
                severity_threshold,
            )
            return False, state
