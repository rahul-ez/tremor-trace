"""Validation report visualization: no-mitigation vs. fixed vs. adaptive.

Feature 55 of the build plan. Read-only: consumes a written comparison
report.json (validation/comparison.py), never calls validation/
experiment-running code itself (architecture.md -> Dashboard /
Visualization: display only).
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive, file-only backend -- avoids requiring
# a working Tcl/Tk installation (this project never displays figures
# interactively; every function here only ever saves to output_path).
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

STRATEGY_ORDER = ["no_mitigation", "fixed_mitigation", "adaptive_mitigation"]
STRATEGY_LABELS = {"no_mitigation": "No mitigation", "fixed_mitigation": "Fixed", "adaptive_mitigation": "Adaptive"}


def plot_validation_comparison(comparison_report_path: Path, output_path: Path | None = None):
    """Plot suppression and exposure side by side across the three strategies.

    Args:
        comparison_report_path: Path to a validation/comparison.py report.json.
        output_path: If given, saves the figure (PNG) to this path.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If the report is missing any of the three strategies.
    """
    with open(comparison_report_path) as f:
        report = json.load(f)

    strategies = report["strategies"]
    missing = set(STRATEGY_ORDER) - set(strategies)
    if missing:
        raise ValueError(f"comparison report is missing strategies: {missing}")

    suppression_values = [strategies[name].get("mean_suppression_pct", 0.0) for name in STRATEGY_ORDER]
    exposure_values = [strategies[name].get("total_simulated_exposure", 0.0) for name in STRATEGY_ORDER]
    labels = [STRATEGY_LABELS[name] for name in STRATEGY_ORDER]

    fig, (ax_suppression, ax_exposure) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_suppression.bar(labels, suppression_values, color=["gray", "tab:orange", "tab:green"])
    ax_suppression.set_ylabel("Mean suppression (%)")
    ax_suppression.set_title("Suppression by strategy")

    ax_exposure.bar(labels, exposure_values, color=["gray", "tab:orange", "tab:green"])
    ax_exposure.set_ylabel("Total simulated exposure")
    ax_exposure.set_title("Exposure by strategy")

    fig.suptitle("No-mitigation vs. fixed vs. adaptive comparison")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Wrote validation comparison plot to %s", output_path)
    return fig
