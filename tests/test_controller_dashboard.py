"""Tests for visualization/controller_dashboard.py (Feature 54)."""

import csv
from pathlib import Path

import pytest

from visualization.controller_dashboard import plot_controller_dashboard


def _write_cycle_log(path: Path, rows: list[dict]) -> None:
    fieldnames = ["cycle", "mitigate", "hysteresis_active", "achieved_suppression_pct", "stability_warning", "amplitude", "pulse_frequency_hz", "duty_cycle"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_plot_controller_dashboard_renders_and_saves(tmp_path: Path) -> None:
    cycle_log_path = tmp_path / "cycle_log.csv"
    rows = [
        {"cycle": i, "mitigate": "True", "hysteresis_active": "True", "achieved_suppression_pct": 80.0 - i * 5,
         "stability_warning": "False", "amplitude": 5.0 - i * 0.3, "pulse_frequency_hz": 20.0, "duty_cycle": 0.0}
        for i in range(6)
    ]
    _write_cycle_log(cycle_log_path, rows)
    output_path = tmp_path / "dashboard.png"

    fig = plot_controller_dashboard(cycle_log_path, target_suppression_pct=50.0, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert len(fig.axes) >= 3  # 3 panels + the twinx axis


def test_plot_controller_dashboard_rejects_empty_log(tmp_path: Path) -> None:
    cycle_log_path = tmp_path / "empty_log.csv"
    _write_cycle_log(cycle_log_path, [])

    with pytest.raises(ValueError, match="no cycle rows"):
        plot_controller_dashboard(cycle_log_path, target_suppression_pct=50.0)
