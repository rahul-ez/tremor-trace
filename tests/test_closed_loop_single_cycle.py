"""Test for the closed-loop runner's single-cycle execution (Feature 41)."""

from pathlib import Path

import pytest

from controller.controller_state import ControllerState
from simulation.closed_loop_runner import run_closed_loop_cycle
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import load_config

MODEL_PATH = Path("data/models/model_logistic_regression_v1.pkl")
SCALER_PATH = Path("data/models/scaler_v1.pkl")

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SCALER_PATH.exists()),
    reason="Trained model/scaler not found locally; run scripts/build_features.py "
    "and scripts/run_training.py first to exercise this integration test.",
)


def test_closed_loop_single_cycle_mitigates_and_suppresses() -> None:
    config = load_config()
    tremor_signal = generate_synthetic_tremor(
        frequency_hz=6.0,
        amplitude=2.0,  # strong, unambiguous tremor -> high severity
        duration_s=config.signal.window_length_s,
        sample_rate_hz=config.sensor.sample_rate_hz,
    )
    state = ControllerState()

    result, updated_state = run_closed_loop_cycle(
        tremor_signal, state, MODEL_PATH, SCALER_PATH, config=config
    )

    assert updated_state.hysteresis_active is True
    assert updated_state.current_params is not None
    assert result.achieved_suppression_pct > 0.0
    assert result.post_mitigation_signal.shape == result.post_mitigation_signal.shape
    assert isinstance(result.stability_warning, bool)


def test_closed_loop_single_cycle_no_mitigation_for_quiet_signal() -> None:
    config = load_config()
    quiet_signal = generate_synthetic_tremor(
        frequency_hz=6.0,
        amplitude=0.001,  # negligible motion -> low severity
        duration_s=config.signal.window_length_s,
        sample_rate_hz=config.sensor.sample_rate_hz,
    )
    state = ControllerState()

    result, updated_state = run_closed_loop_cycle(
        quiet_signal, state, MODEL_PATH, SCALER_PATH, config=config
    )

    assert updated_state.hysteresis_active is False
    assert result.achieved_suppression_pct == pytest.approx(0.0)
    assert result.stability_warning is False


def test_closed_loop_single_cycle_rejects_too_short_signal() -> None:
    config = load_config()
    state = ControllerState()
    # Long enough to pass the band-pass filter's padlen requirement, but
    # shorter than one full window (config.signal.window_length_s).
    too_short_signal = generate_synthetic_tremor(
        frequency_hz=6.0,
        amplitude=1.0,
        duration_s=1.0,
        sample_rate_hz=config.sensor.sample_rate_hz,
    )

    with pytest.raises(ValueError, match="too short"):
        run_closed_loop_cycle(too_short_signal, state, MODEL_PATH, SCALER_PATH, config=config)
