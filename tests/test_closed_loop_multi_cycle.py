"""Tests for the multi-cycle closed-loop simulation script (Feature 42)."""

import argparse
from pathlib import Path

import pytest

from scripts.run_closed_loop_simulation import run_multi_cycle_simulation
from tremor_system.config import load_config

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl")
SCALER_PATH = Path("data/models/scaler_v1.pkl")

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists()),
    reason="Trained model/scaler not found locally; run scripts/build_features.py "
    "and scripts/run_training.py first to exercise this integration test.",
)


def _synthetic_args(n_cycles: int = 10) -> argparse.Namespace:
    return argparse.Namespace(
        source="synthetic",
        path=None,
        frequency_hz=6.0,
        amplitude=1.5,
        n_cycles=n_cycles,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )


def test_multi_cycle_simulation_writes_log_and_plot(tmp_path: Path) -> None:
    config = load_config()
    output_dir = tmp_path / "experiment"

    log_path = run_multi_cycle_simulation(_synthetic_args(n_cycles=10), config, output_dir)

    assert log_path.exists()
    assert (output_dir / "suppression_plot.png").exists()

    with open(log_path) as f:
        rows = f.readlines()
    assert len(rows) == 11  # header + 10 cycles


def test_multi_cycle_simulation_does_not_oscillate_unboundedly(tmp_path: Path) -> None:
    """Confirm suppression stabilizes (does not oscillate without bound).

    Does not assert convergence to config.controller.target_suppression_pct
    -- with the current v1 provisional stimulation-model/config values, this
    signal does not reach target within a 2s window (see memory.md Phase 8
    session update). What this test does confirm, per Feature 42's
    verification requirement, is that the loop reaches a *stable* value
    rather than oscillating.
    """
    import csv

    config = load_config()
    output_dir = tmp_path / "experiment"

    log_path = run_multi_cycle_simulation(_synthetic_args(n_cycles=15), config, output_dir)

    with open(log_path) as f:
        rows = list(csv.DictReader(f))

    last_five = [float(row["achieved_suppression_pct"]) for row in rows[-5:]]
    spread = max(last_five) - min(last_five)
    assert spread < 1.0  # stabilized, not oscillating

    amplitudes = [float(row["amplitude"]) for row in rows if row["amplitude"]]
    bounds = config.controller.param_bounds.amplitude
    assert all(bounds.min <= a <= bounds.max for a in amplitudes)
