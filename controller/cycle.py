"""Full single-cycle controller facade.

Feature 40 of the build plan.

Assembles Features 36-39 into one function representing a single controller
cycle.  This function does NOT call the simulation or adapt_params() —
adaptation happens only after the simulation has produced a real
achieved_suppression_pct.  The full closed loop is:

  run_controller_cycle() [Feature 40]
    → simulation.apply()  [Feature 35]
    → suppression_measurement.compute_suppression_pct()  [Feature 34]
    → adaptation.adapt_params()  [Feature 38]

These four stages are wired together in Feature 41 (closed_loop_runner.py).
This module owns only the first stage.

Decision summary (deliberate, not default):
  mitigate=False : current_params is PRESERVED in the returned state.
    Rationale: hysteresis filtering in decide_mitigation ensures a
    mitigate=False result reflects a genuine severity drop, not noise.  The
    most common cause is a brief intra-session dip, not "tremor resolved."
    Preserving the converged amplitude means re-activation resumes from a
    near-optimal starting point (one adapt step to correct) rather than
    restarting cold and requiring N under-suppressed cycles to re-converge.
    The tradeoff (stale params on a genuine long-off re-entry) is bounded:
    adapt_params corrects after the first active cycle's simulation result.

  mitigate=True, first cycle (current_params is None):
    Calls select_initial_params() to produce a severity/frequency-informed
    initial StimParams.  Returns those params; sets state.current_params.

  mitigate=True, later cycle (current_params is not None):
    Returns state.current_params unchanged.  These params already reflect
    whatever adapt_params applied at the end of the prior cycle in the
    closed-loop runner (Feature 41).  No re-selection, no adaptation here.

Per architecture.md -> Adaptive Controller and code-standards.md:
- Controller input is ONLY the documented InferenceResult contract.
- No raw signal, PSD, feature vector, or simulation call is made here.
- All state is passed in and returned explicitly.
"""

import logging
from typing import Optional

from controller.controller_state import ControllerState
from controller.decision_logic import decide_mitigation
from controller.parameter_selection import select_initial_params
from tremor_system.config import Config, load_config
from tremor_system.types import InferenceResult, StimParams

logger = logging.getLogger(__name__)


def run_controller_cycle(
    ml_output: InferenceResult,
    state: ControllerState,
    config: Optional[Config] = None,
) -> tuple[Optional[StimParams], ControllerState]:
    """Execute one complete controller cycle (decide + select/hold, no simulation).

    Applies the full mitigation decision pipeline for a single window of ML
    output.  Does NOT call adapt_params() — that is Feature 41's
    responsibility, called after simulation.apply() returns a real
    achieved_suppression_pct.

    Stage 1 — decide_mitigation():
        Applies confidence gate, label gate, and hysteresis-gated severity
        threshold.  Returns mitigate: bool and an updated ControllerState
        (hysteresis_active, current_params preserved).

    Stage 2 — (if mitigate=True):
        First cycle  : select_initial_params() → severity/frequency-informed
                       bounded StimParams.  Sets state.current_params.
        Later cycles : returns state.current_params unchanged (Feature 41
                       will have adapted them at the end of the prior cycle).

    Args:
        ml_output: Output from ml/inference.py for the current window.  The
            controller consumes ONLY the documented InferenceResult contract —
            never raw signal, PSD, or feature vectors (architecture.md
            invariant 8).
        state: Current ControllerState, threaded explicitly (never global).
        config: Loaded system configuration.  Defaults to load_config().

    Returns:
        A tuple of (params, updated_state):
            params        : StimParams to apply this cycle, or None if the
                            decision is not to mitigate.
            updated_state : ControllerState with hysteresis_active updated and
                            current_params set (mitigate=True, first cycle) or
                            preserved (all other cases).

    Raises:
        ValueError: If any required config threshold is None (propagated from
            decide_mitigation or select_initial_params — fail-fast).
    """
    resolved_config = config or load_config()

    # --- Stage 1: Mitigation decision ---
    mitigate, state_after_decision = decide_mitigation(ml_output, state, resolved_config)

    if not mitigate:
        # current_params is preserved inside state_after_decision (decide_mitigation
        # always copies current_params into the returned ControllerState so that
        # re-activation can resume from the last converged amplitude — see module
        # docstring for the design rationale).
        logger.debug(
            "run_controller_cycle: mitigate=False — no stimulation this cycle "
            "(current_params preserved in state for potential re-activation)."
        )
        return None, state_after_decision

    # --- Stage 2: Parameter selection or hold ---
    if state_after_decision.current_params is None:
        # First mitigate cycle: no prior params exist — select an initial set.
        initial_params = select_initial_params(ml_output, resolved_config)
        logger.info(
            "run_controller_cycle: first mitigate cycle — "
            "select_initial_params() -> amplitude=%.4f, pulse_frequency_hz=%.2f Hz.",
            initial_params.amplitude,
            initial_params.pulse_frequency_hz,
        )
        # Update current_params in state.
        updated_state = ControllerState(
            hysteresis_active=state_after_decision.hysteresis_active,
            current_params=initial_params,
            last_decision_timestamp=state_after_decision.last_decision_timestamp,
            adaptation_history=state_after_decision.adaptation_history,
        )
        return initial_params, updated_state

    else:
        # Later cycle: params already reflect prior adaptation — hold unchanged.
        # adapt_params() will be called by the closed-loop runner (Feature 41)
        # after simulation.apply() returns the real achieved_suppression_pct.
        current = state_after_decision.current_params
        logger.debug(
            "run_controller_cycle: later mitigate cycle — "
            "holding current params (amplitude=%.4f). "
            "Feature 41 runner will adapt after simulation.",
            current.amplitude,
        )
        return current, state_after_decision
