"""Tests for validation/comparison.py and visualization/validation_report.py (Feature 55)."""

import json
from pathlib import Path

import pytest

from validation.comparison import build_comparison_report, write_comparison_report
from visualization.validation_report import plot_validation_comparison


def _write_report(path: Path, mean_suppression_pct: float, total_simulated_exposure: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_type": "test",
        "metrics": {
            "mean_suppression_pct": mean_suppression_pct,
            "total_simulated_exposure": total_simulated_exposure,
        },
    }
    with open(path, "w") as f:
        json.dump(report, f)


def test_build_comparison_report_merges_all_three(tmp_path: Path) -> None:
    no_mit = tmp_path / "no_mitigation_x" / "report.json"
    fixed = tmp_path / "fixed_mitigation_x" / "report.json"
    adaptive = tmp_path / "adaptive_mitigation_x" / "report.json"
    _write_report(no_mit, 0.0, 0.0)
    _write_report(fixed, 73.6, 160.0)
    _write_report(adaptive, 58.0, 110.0)

    comparison = build_comparison_report(no_mit, fixed, adaptive)

    assert comparison["strategies"]["no_mitigation"]["mean_suppression_pct"] == pytest.approx(0.0)
    assert comparison["strategies"]["fixed_mitigation"]["mean_suppression_pct"] == pytest.approx(73.6)
    assert comparison["strategies"]["adaptive_mitigation"]["total_simulated_exposure"] == pytest.approx(110.0)


def test_build_comparison_report_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_comparison_report(tmp_path / "missing.json", tmp_path / "missing2.json", tmp_path / "missing3.json")


def test_write_comparison_report_writes_expected_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    no_mit = tmp_path / "no_mitigation_x" / "report.json"
    fixed = tmp_path / "fixed_mitigation_x" / "report.json"
    adaptive = tmp_path / "adaptive_mitigation_x" / "report.json"
    _write_report(no_mit, 0.0, 0.0)
    _write_report(fixed, 73.6, 160.0)
    _write_report(adaptive, 58.0, 110.0)

    import validation.comparison as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    report_path = write_comparison_report(no_mit, fixed, adaptive, run_id="test_run")

    assert report_path == tmp_path / "data" / "validation" / "comparison_test_run" / "report.json"
    assert report_path.exists()


def test_plot_validation_comparison_renders_and_saves(tmp_path: Path) -> None:
    no_mit = tmp_path / "no_mitigation_x" / "report.json"
    fixed = tmp_path / "fixed_mitigation_x" / "report.json"
    adaptive = tmp_path / "adaptive_mitigation_x" / "report.json"
    _write_report(no_mit, 0.0, 0.0)
    _write_report(fixed, 73.6, 160.0)
    _write_report(adaptive, 58.0, 110.0)
    comparison_path = write_comparison_report(no_mit, fixed, adaptive, run_id="test_plot")
    output_path = tmp_path / "comparison.png"

    fig = plot_validation_comparison(comparison_path, output_path=output_path)

    assert output_path.exists()
    assert len(fig.axes) == 2


def test_plot_validation_comparison_rejects_missing_strategy(tmp_path: Path) -> None:
    incomplete_report = tmp_path / "report.json"
    with open(incomplete_report, "w") as f:
        json.dump({"strategies": {"no_mitigation": {}, "fixed_mitigation": {}}}, f)

    with pytest.raises(ValueError, match="missing strategies"):
        plot_validation_comparison(incomplete_report)
