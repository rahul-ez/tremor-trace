"""Signal, PSD, and detection-history visualizations.

Features 52-53 of the build plan. Read-only: consumes arrays/objects
already computed elsewhere, never calls signal_processing/ml/controller/
simulation functions that mutate state (architecture.md -> Dashboard /
Visualization: "No visualization module may call controller/ or
simulation/ functions that mutate state — display only.").
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive, file-only backend -- avoids requiring
# a working Tcl/Tk installation (this project never displays figures
# interactively; every function here only ever saves to output_path).
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from tremor_system.types import InferenceResult

logger = logging.getLogger(__name__)


def plot_raw_vs_filtered(
    raw_signal: NDArray[np.float64],
    filtered_signal: NDArray[np.float64],
    sample_rate_hz: float,
    output_path: Path | None = None,
):
    """Plot raw vs. band-pass-filtered signal, time domain.

    Args:
        raw_signal: shape (n_samples,), units g.
        filtered_signal: shape (n_samples,), units g -- same length as raw_signal.
        sample_rate_hz: Sampling rate in Hz.
        output_path: If given, saves the figure (PNG) to this path.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If raw_signal and filtered_signal have different shapes.
    """
    if raw_signal.shape != filtered_signal.shape:
        raise ValueError(
            f"raw_signal shape {raw_signal.shape} != filtered_signal shape {filtered_signal.shape}"
        )
    time_s = np.arange(raw_signal.size) / sample_rate_hz

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_s, raw_signal, label="Raw", alpha=0.6, linewidth=1)
    ax.plot(time_s, filtered_signal, label="Filtered (tremor-band)", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title("Raw vs. filtered signal")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Wrote raw-vs-filtered plot to %s", output_path)
    return fig


def plot_psd(
    freqs_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
    tremor_band_hz: tuple[float, float],
    output_path: Path | None = None,
):
    """Plot power spectral density with the tremor band shaded.

    Args:
        freqs_hz: shape (n_bins,), from compute_welch_psd().
        psd: shape (n_bins,), from compute_welch_psd().
        tremor_band_hz: (low, high) in Hz, shaded on the plot.
        output_path: If given, saves the figure (PNG) to this path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs_hz, psd)
    ax.axvspan(tremor_band_hz[0], tremor_band_hz[1], color="red", alpha=0.15, label="Tremor band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (g^2/Hz)")
    ax.set_title("Power spectral density")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Wrote PSD plot to %s", output_path)
    return fig


def plot_detection_dashboard(
    inference_results: list[InferenceResult],
    cycle_duration_s: float,
    output_path: Path | None = None,
):
    """Plot label/severity/confidence over time for a sequence of InferenceResults.

    Feature 53. Module choice: extended signal_plots.py rather than adding
    a new file, per the build plan's "module TBD — extend signal_plots.py
    or a new file" allowance.

    Args:
        inference_results: Sequence of InferenceResult, in cycle/window order.
        cycle_duration_s: Duration of one cycle/window in seconds, for the
            time axis.
        output_path: If given, saves the figure (PNG) to this path.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If inference_results is empty.
    """
    if not inference_results:
        raise ValueError("inference_results must be non-empty")

    time_s = np.arange(len(inference_results)) * cycle_duration_s
    labels = [1 if r.label else 0 for r in inference_results]
    severities = [r.severity for r in inference_results]
    confidences = [r.confidence for r in inference_results]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].step(time_s, labels, where="post")
    axes[0].set_ylabel("Label (tremor)")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0, 1])

    axes[1].plot(time_s, severities)
    axes[1].set_ylabel("Severity")
    axes[1].set_ylim(0, 1)

    axes[2].plot(time_s, confidences)
    axes[2].set_ylabel("Confidence")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Time (s)")

    fig.suptitle("Detection history: label / severity / confidence")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info("Wrote detection dashboard plot to %s", output_path)
    return fig
