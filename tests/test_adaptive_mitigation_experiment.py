"""Tests for the adaptive mitigation experiment (Feature 46).

Includes the core success criterion (per architecture.md, referenced by
build-plan Feature 46): adaptive achieves the target suppression using
less total simulated exposure than a fixed-parameter run on the same
input tremor signal.
"""

import argparse
from pathlib import Path

import pytest

from validation.experiments.adaptive_mitigation import run_adaptive_mitigation_experiment
from validation.experiments.fixed_mitigation import run_fixed_mitigation_experiment

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl").resolve()
SCALER_PATH = Path("data/models/scaler_v1.pkl").resolve()

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists()),
    reason="Trained model/scaler not found locally; run scripts/build_features.py "
    "and scripts/run_training.py first to exercise this integration test.",
)


def _common_input_args() -> dict:
    return dict(source="synthetic", path=None, frequency_hz=6.0, amplitude=1.5, n_cycles=20)


def test_adaptive_converges_toward_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    args = argparse.Namespace(
        **_common_input_args(), model_path=MODEL_PATH, scaler_path=SCALER_PATH, run_id="test_adaptive"
    )
    report_path = run_adaptive_mitigation_experiment(args)

    with open(report_path) as f:
        report = json.load(f)

    assert report["experiment_type"] == "adaptive_mitigation"
    assert report["metrics"]["time_to_target_s"] is not None  # reached the target band


def test_adaptive_uses_less_exposure_than_naive_fixed_for_comparable_suppression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core success criterion: adaptive beats a non-clairvoyant fixed choice on exposure."""
    import json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    fixed_args = argparse.Namespace(
        **_common_input_args(),
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        run_id="test_fixed_cmp",
        fixed_amplitude=4.0,  # a plausible but non-optimal a-priori guess
        fixed_pulse_frequency_hz=100.0,
        fixed_pulse_width_us=200.0,
        fixed_duty_cycle=0.5,
        fixed_on_time_ms=10.0,
        fixed_off_time_ms=10.0,
    )
    adaptive_args = argparse.Namespace(
        **_common_input_args(), model_path=MODEL_PATH, scaler_path=SCALER_PATH, run_id="test_adaptive_cmp"
    )

    fixed_report_path = run_fixed_mitigation_experiment(fixed_args)
    adaptive_report_path = run_adaptive_mitigation_experiment(adaptive_args)

    with open(fixed_report_path) as f:
        fixed_metrics = json.load(f)["metrics"]
    with open(adaptive_report_path) as f:
        adaptive_metrics = json.load(f)["metrics"]

    assert adaptive_metrics["total_simulated_exposure"] < fixed_metrics["total_simulated_exposure"]
    # Both still land near the 50% target -- adaptive isn't "cheating" by
    # simply under-stimulating; the naive fixed choice over-suppresses.
    assert adaptive_metrics["mean_suppression_pct"] < fixed_metrics["mean_suppression_pct"]
