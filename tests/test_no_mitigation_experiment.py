"""Tests for the no-mitigation baseline experiment (Feature 44)."""

import argparse
from pathlib import Path

import pytest

from validation.experiments.no_mitigation import run_no_mitigation_experiment

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl").resolve()
SCALER_PATH = Path("data/models/scaler_v1.pkl").resolve()

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists()),
    reason="Trained model/scaler not found locally; run scripts/build_features.py "
    "and scripts/run_training.py first to exercise this integration test.",
)


def _args(run_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        source="synthetic",
        path=None,
        frequency_hz=6.0,
        amplitude=1.5,
        n_cycles=15,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        run_id=run_id,
    )


def test_no_mitigation_tremor_power_remains_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm tremor power stays statistically unchanged -- no suppression trend."""
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    report_path = run_no_mitigation_experiment(_args("test_baseline"))

    with open(report_path) as f:
        report = json.load(f)

    assert report["experiment_type"] == "no_mitigation"
    assert report["n_cycles"] == 15
    # Every cycle should show exactly 0% suppression -- no intervention occurred.
    assert report["metrics"]["mean_suppression_pct"] == pytest.approx(0.0)

    with open(report_path.parent / "cycle_log.csv") as f:
        import csv

        rows = list(csv.DictReader(f))
    suppression_values = [float(r["achieved_suppression_pct"]) for r in rows]
    assert all(v == pytest.approx(0.0) for v in suppression_values)
    assert all(r["mitigate"] == "False" for r in rows)
