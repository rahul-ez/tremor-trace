"""Tests for validation/robustness_tests.py (Feature 51).

Directly verifies architecture.md Success Criterion #9: the controller
never exceeds parameter bounds, never oscillates outside the hysteresis
band, and correctly withholds mitigation on low-confidence input.
"""

from pathlib import Path

import pytest

from tremor_system.config import load_config
from validation.robustness_tests import (
    check_low_confidence_gating,
    measure_filtering_delay,
    run_robustness_suite,
    sweep_closed_loop_robustness,
)

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl").resolve()
SCALER_PATH = Path("data/models/scaler_v1.pkl").resolve()

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists()),
    reason="Trained model/scaler not found locally; run scripts/build_features.py "
    "and scripts/run_training.py first to exercise this integration test.",
)


def test_sweep_never_exceeds_param_bounds_or_oscillates_unboundedly() -> None:
    config = load_config()

    results = sweep_closed_loop_robustness(
        frequencies_hz=[6.0],
        amplitudes_g=[1.5, 3.0],
        noise_stds=[0.0, 0.1],
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        config=config,
        n_cycles_per_combo=6,
    )

    assert len(results) == 4  # 1 freq x 2 amplitudes x 2 noise levels
    for result in results:
        assert result["within_bounds"] is True
        assert result["oscillation"] < 2.0  # not oscillating without bound


def test_filtering_delay_is_near_zero_zero_phase_filter() -> None:
    config = load_config()

    result = measure_filtering_delay(config)

    # filtfilt is zero-phase by construction; a few samples of tolerance
    # for numerical/discretization effects.
    assert abs(result["delay_samples"]) <= 2


def test_low_confidence_input_withholds_mitigation() -> None:
    config = load_config()

    result = check_low_confidence_gating(config)

    assert result["input_confidence"] < result["confidence_threshold"]
    assert result["input_severity"] > 0.5  # high severity, to prove confidence gates first
    assert result["mitigate_decision"] is False
    assert result["correctly_withheld"] is True
    assert result["state_unchanged"] is True


def test_run_robustness_suite_writes_consolidated_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.chdir(tmp_path)

    import validation.robustness_tests as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "DEFAULT_FREQUENCIES_HZ", [6.0])
    monkeypatch.setattr(module, "DEFAULT_AMPLITUDES_G", [1.5])
    monkeypatch.setattr(module, "DEFAULT_NOISE_STDS", [0.0])
    monkeypatch.setattr(module, "DEFAULT_N_CYCLES_PER_COMBO", 4)

    report_path = run_robustness_suite(MODEL_PATH, SCALER_PATH, run_id="test_report")

    with open(report_path) as f:
        report = json.load(f)

    assert report["experiment_type"] == "robustness"
    assert report["metrics"]["n_combinations_swept"] == 1
    assert "all_within_bounds" in report["metrics"]
    assert "filtering_delay_s" in report["metrics"]
    assert "low_confidence_correctly_withheld" in report["metrics"]
