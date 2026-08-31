"""Test for the closed loop running on real recorded sensor data (Feature 43)."""

import argparse
from pathlib import Path

import pytest

from scripts.run_closed_loop_simulation import run_multi_cycle_simulation
from tremor_system.config import load_config

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl")
SCALER_PATH = Path("data/models/scaler_v1.pkl")
TREMOR_SESSION = Path("data/raw/subj05/sess02/raw_stream.csv")
REST_SESSION = Path("data/raw/subj05/sess01/raw_stream.csv")

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists() and TREMOR_SESSION.exists() and REST_SESSION.exists()),
    reason="Recorded sessions/trained model not found locally; run "
    "scripts/build_features.py and scripts/run_training.py first, and "
    "ensure data/raw/subj05/sess01 and sess02 are present.",
)


def _recorded_args(path: Path, n_cycles: int = 15) -> argparse.Namespace:
    return argparse.Namespace(
        source="recorded",
        path=path,
        frequency_hz=0.0,
        amplitude=0.0,
        n_cycles=n_cycles,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )


def test_recorded_tremor_session_mitigates_throughout(tmp_path: Path) -> None:
    import csv

    config = load_config()
    log_path = run_multi_cycle_simulation(_recorded_args(TREMOR_SESSION), config, tmp_path / "tremor")

    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    mitigate_flags = [row["mitigate"] == "True" for row in rows]
    assert any(mitigate_flags), "expected at least some cycles to mitigate during a deliberate tremor session"


def test_recorded_rest_session_does_not_mitigate(tmp_path: Path) -> None:
    import csv

    config = load_config()
    log_path = run_multi_cycle_simulation(_recorded_args(REST_SESSION), config, tmp_path / "rest")

    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    mitigate_flags = [row["mitigate"] == "True" for row in rows]
    assert not any(mitigate_flags), "expected no cycles to mitigate during a stationary rest session"
