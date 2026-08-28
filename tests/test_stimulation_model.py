"""Tests for the simulated stimulation-response model (Feature 33)."""

import numpy as np
import pytest

from simulation.stimulation_model import simulate_tremor_response
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import load_config
from tremor_system.types import StimParams


def _tremor_state(duration_s: float = 2.0) -> dict:
    sample_rate_hz = 100.0
    signal = generate_synthetic_tremor(6.0, 0.4, duration_s, sample_rate_hz)
    return {
        "y0": np.array(
            [signal[0], (signal[1] - signal[0]) * sample_rate_hz],
            dtype=np.float64,
        ),
        "duration_s": duration_s,
        "timestep_s": load_config().simulation.timestep_s,
        "signal": signal,
        "sample_rate_hz": sample_rate_hz,
    }


def _params(amplitude: float) -> StimParams:
    return StimParams(
        amplitude=amplitude,
        pulse_frequency_hz=50.0,
        pulse_width_us=200.0,
        duty_cycle=0.2,
        on_off_timing=(100.0, 50.0),
    )


def test_simulate_tremor_response_is_aligned_and_stable() -> None:
    tremor_state = _tremor_state()

    post_signal, stability_warning = simulate_tremor_response(_params(0.5), tremor_state)

    assert post_signal.shape == tremor_state["signal"].shape
    assert post_signal.dtype == np.float64
    assert stability_warning is False
    assert np.all(np.isfinite(post_signal))


def test_zero_stimulation_reproduces_input_signal() -> None:
    tremor_state = _tremor_state()

    post_signal, stability_warning = simulate_tremor_response(_params(0.0), tremor_state)

    assert stability_warning is False
    np.testing.assert_allclose(post_signal, tremor_state["signal"], atol=0.01)


def test_increasing_amplitude_increases_suppression() -> None:
    tremor_state = _tremor_state()
    low, low_warning = simulate_tremor_response(_params(0.5), tremor_state)
    high, high_warning = simulate_tremor_response(_params(2.0), tremor_state)

    assert low_warning is False
    assert high_warning is False
    assert np.sqrt(np.mean(high**2)) < np.sqrt(np.mean(low**2))


def test_latency_delays_stimulation_effect() -> None:
    tremor_state = _tremor_state()
    no_stimulation, _ = simulate_tremor_response(_params(0.0), tremor_state)
    with_stimulation, _ = simulate_tremor_response(_params(2.0), tremor_state)
    latency_samples = int(round(load_config().simulation.latency_ms * 100.0 / 1000.0))

    np.testing.assert_allclose(
        with_stimulation[:latency_samples], no_stimulation[:latency_samples], atol=1e-6
    )
    assert not np.allclose(with_stimulation[latency_samples:], no_stimulation[latency_samples:])


@pytest.mark.parametrize(
    "tremor_state_update",
    [
        {"signal": np.zeros(1, dtype=np.float64)},
        {"timestep_s": 0.0},
        {"sample_rate_hz": 0.0},
    ],
)
def test_simulate_tremor_response_rejects_invalid_state(tremor_state_update: dict) -> None:
    tremor_state = _tremor_state()
    tremor_state.update(tremor_state_update)

    with pytest.raises(ValueError):
        simulate_tremor_response(_params(1.0), tremor_state)
