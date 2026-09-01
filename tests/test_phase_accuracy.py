"""Tests for the phase-estimation accuracy sweep (Feature 50, conditional)."""

import dataclasses

import pytest

from tremor_system.config import load_config
from validation.phase_accuracy import run_phase_accuracy_report, run_phase_sweep, summarize_phase_accuracy


def test_phase_accuracy_report_raises_when_feature_31_disabled() -> None:
    """Confirms the build plan's requirement: skip entirely while disabled."""
    config = load_config()
    assert config.estimation.phase_enabled is False  # this project's actual current default

    with pytest.raises(RuntimeError, match="phase_enabled is False"):
        run_phase_accuracy_report()


def test_phase_sweep_accurate_when_enabled_via_config_override() -> None:
    """The sweep itself works correctly -- exercised via an explicit override,
    without needing config/system_config.yaml's real phase_enabled flag flipped.
    """
    base_config = load_config()
    enabled_config = dataclasses.replace(
        base_config, estimation=dataclasses.replace(base_config.estimation, phase_enabled=True)
    )

    records = run_phase_sweep([6.0, 8.5], enabled_config)

    assert len(records) == 2
    for record in records:
        assert record["estimated_phase_rad"] is not None
        assert record["phase_error_rad"] < 0.05  # clean cosine, should be near-exact


def test_summarize_phase_accuracy_hand_computed() -> None:
    records = [
        {"phase_error_rad": 0.1},
        {"phase_error_rad": 0.3},
        {"phase_error_rad": None},
    ]

    summary = summarize_phase_accuracy(records)

    assert summary["n_frequencies"] == 3
    assert summary["n_undefined"] == 1
    assert summary["mean_phase_error_rad"] == pytest.approx(0.2)
    assert summary["max_phase_error_rad"] == pytest.approx(0.3)


def test_summarize_phase_accuracy_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        summarize_phase_accuracy([])
