"""Tests for the fixed-parameter mitigation experiment (Feature 45)."""

import argparse
from pathlib import Path

import pytest

from validation.experiments.fixed_mitigation import run_fixed_mitigation_experiment

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl").resolve()
SCALER_PATH = Path("data/models/scaler_v1.pkl").resolve()

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists()),
    reason="Trained model/scaler not found locally; run scripts/build_features.py "
    "and scripts/run_training.py first to exercise this integration test.",
)


def _args(run_id: str, fixed_amplitude: float = 3.0) -> argparse.Namespace:
    return argparse.Namespace(
        source="synthetic",
        path=None,
        frequency_hz=6.0,
        amplitude=1.5,
        n_cycles=15,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        run_id=run_id,
        fixed_amplitude=fixed_amplitude,
        fixed_pulse_frequency_hz=100.0,
        fixed_pulse_width_us=200.0,
        fixed_duty_cycle=0.5,
        fixed_on_time_ms=10.0,
        fixed_off_time_ms=10.0,
    )


def test_fixed_mitigation_suppresses_consistently_without_adapting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm suppression > 0 consistently but the parameter trace is flat (no adaptation)."""
    import csv
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    report_path = run_fixed_mitigation_experiment(_args("test_fixed"))

    with open(report_path) as f:
        report = json.load(f)
    assert report["experiment_type"] == "fixed_mitigation"
    assert report["metrics"]["fixed_params"]["amplitude"] == pytest.approx(3.0)

    with open(report_path.parent / "cycle_log.csv") as f:
        rows = list(csv.DictReader(f))

    mitigating_rows = [r for r in rows if r["mitigate"] == "True"]
    assert mitigating_rows, "expected at least some cycles to mitigate on a strong tremor signal"
    assert all(float(r["achieved_suppression_pct"]) > 0.0 for r in mitigating_rows)

    # Flat parameter trace by construction -- adapt_params() was never called.
    amplitudes = {float(r["amplitude"]) for r in mitigating_rows}
    assert amplitudes == {3.0}
