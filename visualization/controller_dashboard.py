"""Controller dashboard: state, decision, and StimParams over time.

Feature 54 of the build plan. Read-only: consumes a written cycle_log.csv
(architecture.md -> Dashboard / Visualization: display only, no
controller/simulation calls).
"""

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive, file-only backend -- avoids requiring
# a working Tcl/Tk installation (this project never displays figures
# interactively; every function here only ever saves to output_path).
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _read_cycle_log(cycle_log_path: Path) -> list[dict]:
    with open(cycle_log_path) as f:
        return list(csv.DictReader(f))


def _parse_optional_float(value: str) -> float | None:
    return None if value in ("", "None") else float(value)


def plot_controller_dashboard(
    cycle_log_path: Path,
    target_suppression_pct: float,
    output_path: Path | None = None,
):
    """Plot mitigation decision, suppression vs. target, and StimParams over cycles.

    Consumes the per-cycle CSV log produced by
    scripts/run_closed_loop_simulation.py (Feature 42/43,
    CYCLE_LOG_COLUMNS) or validation/experiments/*.py (Features 44-46,
    the richer EXPERIMENT_RECORD_COLUMNS) -- both share the columns this
    function reads (cycle, mitigate, achieved_suppression_pct, amplitude,
    pulse_frequency_hz); either log works without modification.

    duty_cycle is intentionally not given its own panel: per this
    project's current controller design (see memory.md's Phase 9 session
    update), controller/parameter_selection.py always pins duty_cycle to
    its configured minimum bound and adapt_params() never adjusts it, so
    a duty_cycle-vs-cycle panel would currently just be a flat line at
    that constant value.

    Args:
        cycle_log_path: Path to a cycle_log.csv.
        target_suppression_pct: config.controller.target_suppression_pct,
            drawn as a reference line.
        output_path: If given, saves the figure (PNG) to this path.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If cycle_log_path has no rows.
    """
    rows = _read_cycle_log(cycle_log_path)
    if not rows:
        raise ValueError(f"{cycle_log_path} has no cycle rows")

    cycles = [int(r["cycle"]) for r in rows]
    mitigate = [1 if r["mitigate"] == "True" else 0 for r in rows]
    suppression = [float(r["achieved_suppression_pct"]) for r in rows]
    amplitude = [_parse_optional_float(r["amplitude"]) for r in rows]
    pulse_frequency_hz = [_parse_optional_float(r["pulse_frequency_hz"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].step(cycles, mitigate, where="post")
    axes[0].set_ylabel("Mitigate")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0, 1])
    axes[0].set_title("Mitigation decision")

    axes[1].plot(cycles, suppression, marker="o", markersize=3, label="achieved")
    axes[1].axhline(target_suppression_pct, color="red", linestyle="--", label="target")
    axes[1].set_ylabel("Suppression (%)")
    axes[1].legend()
    axes[1].set_title("Achieved vs. target suppression")

    axes[2].plot(cycles, amplitude, label="amplitude", marker=".", color="tab:blue")
    axes[2].set_ylabel("Amplitude", color="tab:blue")
    axes[2].set_xlabel("Cycle")
    ax_freq = axes[2].twinx()
    ax_freq.plot(cycles, pulse_frequency_hz, label="pulse_frequency_hz", color="tab:orange", alpha=0.6)
    ax_freq.set_ylabel("Pulse frequency (Hz)", color="tab:orange")
    axes[2].set_title("StimParams over cycles")

    fig.suptitle("Controller dashboard")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Wrote controller dashboard plot to %s", output_path)
    return fig
