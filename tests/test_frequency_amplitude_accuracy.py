"""Tests for the frequency/amplitude estimation accuracy sweep (Feature 49)."""

from pathlib import Path

import pytest

from tremor_system.config import load_config
from validation.frequency_amplitude_accuracy import (
    run_frequency_amplitude_accuracy_report,
    run_frequency_amplitude_sweep,
    summarize_accuracy,
)


def test_sweep_produces_one_record_per_combination() -> None:
    config = load_config()

    records = run_frequency_amplitude_sweep([4.0, 6.0], [0.5, 1.0], config)

    assert len(records) == 4  # 2 frequencies x 2 amplitudes
    for record in records:
        assert record["true_frequency_hz"] in (4.0, 6.0)
        assert record["amplitude_error_g"] >= 0.0


def test_sweep_on_bin_aligned_frequency_has_near_zero_error() -> None:
    config = load_config()

    records = run_frequency_amplitude_sweep([6.0], [1.0], config)

    assert records[0]["frequency_error_hz"] == pytest.approx(0.0, abs=1e-6)
    assert records[0]["amplitude_error_g"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_accuracy_hand_computed() -> None:
    records = [
        {"frequency_error_hz": 0.2, "amplitude_error_g": 0.01},
        {"frequency_error_hz": 0.4, "amplitude_error_g": 0.03},
        {"frequency_error_hz": None, "amplitude_error_g": 0.05},  # missed peak
    ]

    summary = summarize_accuracy(records)

    assert summary["n_combinations"] == 3
    assert summary["n_missed_peak"] == 1
    assert summary["mean_frequency_error_hz"] == pytest.approx(0.3)  # (0.2+0.4)/2, excludes the miss
    assert summary["max_frequency_error_hz"] == pytest.approx(0.4)
    assert summary["mean_amplitude_error_g"] == pytest.approx(0.03)  # (0.01+0.03+0.05)/3
    assert summary["max_amplitude_error_g"] == pytest.approx(0.05)


def test_summarize_accuracy_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_accuracy([])


def test_run_frequency_amplitude_accuracy_report_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.chdir(tmp_path)

    import validation.frequency_amplitude_accuracy as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "DEFAULT_FREQUENCY_SWEEP_HZ", [6.0])
    monkeypatch.setattr(module, "DEFAULT_AMPLITUDE_SWEEP_G", [1.0])

    report_path = run_frequency_amplitude_accuracy_report(run_id="test_report")

    assert report_path.exists()
    with open(report_path) as f:
        report = json.load(f)
    assert report["experiment_type"] == "frequency_amplitude_accuracy"
    assert report["metrics"]["n_combinations"] == 1
    assert len(report["records"]) == 1
