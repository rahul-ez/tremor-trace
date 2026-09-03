"""Tests for visualization/signal_plots.py (Features 52-53)."""

from pathlib import Path

import numpy as np
import pytest

from tremor_system.types import InferenceResult
from visualization.signal_plots import plot_detection_dashboard, plot_psd, plot_raw_vs_filtered


def _sine(frequency_hz: float = 6.0, n_samples: int = 200, sample_rate_hz: float = 100.0) -> np.ndarray:
    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    return np.sin(2.0 * np.pi * frequency_hz * time_s)


def test_plot_raw_vs_filtered_renders_and_saves(tmp_path: Path) -> None:
    raw = _sine()
    filtered = _sine() * 0.9
    output_path = tmp_path / "raw_vs_filtered.png"

    fig = plot_raw_vs_filtered(raw, filtered, 100.0, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "Time (s)"


def test_plot_raw_vs_filtered_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        plot_raw_vs_filtered(np.zeros(100), np.zeros(50), 100.0)


def test_plot_psd_renders_and_saves(tmp_path: Path) -> None:
    freqs_hz = np.linspace(0, 50, 101)
    psd = np.exp(-((freqs_hz - 6.0) ** 2))
    output_path = tmp_path / "psd.png"

    fig = plot_psd(freqs_hz, psd, (4.0, 12.0), output_path=output_path)

    assert output_path.exists()
    assert len(fig.axes) == 1


def test_plot_detection_dashboard_renders_and_saves(tmp_path: Path) -> None:
    results = [
        InferenceResult(label=True, severity=0.8, confidence=0.9, dominant_frequency_hz=6.0, amplitude=0.3, phase=None)
        for _ in range(5)
    ]
    output_path = tmp_path / "detection.png"

    fig = plot_detection_dashboard(results, cycle_duration_s=2.0, output_path=output_path)

    assert output_path.exists()
    assert len(fig.axes) == 3


def test_plot_detection_dashboard_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_detection_dashboard([], cycle_duration_s=2.0)
