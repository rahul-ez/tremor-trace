"""Controller state management.

Defines ControllerState — the explicit state object threaded through every
controller function call. No module-level mutable globals are used; state is
always passed in and returned explicitly.

This module is introduced in Feature 36 (hysteresis tracking required by
decision_logic.py) and formally verified as Feature 39.

Per code-standards.md -> Adaptive Controller:
- ControllerState is the sole carrier of inter-cycle mutable state.
- All controller functions are deterministic given the same inputs and state.
- No randomness lives inside decision_logic.py, adaptation.py, or here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from tremor_system.types import StimParams

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """Mutable state threaded explicitly through every controller cycle.

    Fields:
        hysteresis_active: True when the controller is currently in the
            mitigation-ON state. Used by decision_logic.py to apply the
            distinct exit threshold rather than the enter threshold, preventing
            rapid ON/OFF chatter at the boundary (per architecture.md ->
            Adaptive Controller -> Hysteresis).
        current_params: The StimParams most recently emitted by the controller.
            None when no mitigation is active (first cycle, or after a
            mitigate=False decision). Consumed by adaptation.py (Feature 38)
            as the starting point for parameter adjustment.
        last_decision_timestamp: Monotonic timestamp (seconds, e.g. time.monotonic())
            of the most recent decide_mitigation() call. None before the first
            cycle. Exposed for logging and dashboard display (Feature 54).
        adaptation_history: Ordered log of per-cycle adaptation events, each a
            plain dict with at minimum keys ``cycle``, ``achieved_suppression_pct``,
            ``decision``, and ``params``. Used by Feature 38 (adaptation) and
            Feature 54 (controller dashboard). Appended to, never mutated.
    """

    hysteresis_active: bool = False
    current_params: Optional[StimParams] = None
    last_decision_timestamp: Optional[float] = None
    adaptation_history: list[dict] = field(default_factory=list)

    def reset(self) -> "ControllerState":
        """Return a fresh ControllerState with all fields at defaults.

        Does not mutate self. Useful in validation experiments that require a
        clean instance per run, per code-standards.md -> Validation and
        Experiments.
        """
        return ControllerState()
