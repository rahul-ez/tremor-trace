"""Tests for the simulation.apply() entry point (Feature 35)."""

import numpy as np
import pytest

from simulation import apply
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import load_config
from tremor_system.types import SimResult, StimParams

_CFG = load_config()


def _tremor_state(duration_s: float = 2.0, amplitude: float = 0.4) -> dict:
    sample_rate_hz = 100.0
    signal = generate_synthetic_tremor(6.0, amplitude, duration_s, sample_rate_hz)
    return {
        "y0": np.array(
            [signal[0], (signal[1] - signal[0]) * sample_rate_hz],
            dtype=np.float64,
        ),
        "duration_s": duration_s,
        "timestep_s": _CFG.simulation.timestep_s,
        "signal": signal,
        "sample_rate_hz": sample_rate_hz,
    }


def _valid_params(amplitude: float = 1.0) -> StimParams:
    return StimParams(
        amplitude=amplitude,
        pulse_frequency_hz=50.0,
        pulse_width_us=200.0,
        duty_cycle=0.2,
        on_off_timing=(100.0, 50.0),
    )


# ---------------------------------------------------------------------------
# Happy-path: well-formed SimResult returned
# ---------------------------------------------------------------------------


def test_apply_returns_well_formed_sim_result() -> None:
    result = apply(_valid_params(), _tremor_state())

    assert isinstance(result, SimResult)
    assert isinstance(result.post_mitigation_signal, np.ndarray)
    assert result.post_mitigation_signal.dtype == np.float64
    assert result.post_mitigation_signal.ndim == 1
    assert np.all(np.isfinite(result.post_mitigation_signal))
    assert np.isfinite(result.achieved_suppression_pct)
    assert 0.0 <= result.achieved_suppression_pct <= 100.0
    assert np.isfinite(result.residual_amplitude)
    assert result.residual_amplitude >= 0.0
    assert np.isfinite(result.latency_ms)
    assert result.latency_ms >= 0.0
    assert isinstance(result.stability_warning, bool)


def test_apply_zero_stimulation_achieves_zero_suppression() -> None:
    """Zero amplitude means no intervention: suppression must be ~0%."""
    result = apply(_valid_params(amplitude=0.0), _tremor_state())

    assert result.achieved_suppression_pct == pytest.approx(0.0, abs=1.0)
    assert result.stability_warning is False


def test_apply_positive_stimulation_achieves_positive_suppression() -> None:
    result = apply(_valid_params(amplitude=1.5), _tremor_state())

    assert result.achieved_suppression_pct > 0.0


def test_apply_latency_ms_matches_config() -> None:
    result = apply(_valid_params(), _tremor_state())

    assert result.latency_ms == pytest.approx(_CFG.simulation.latency_ms)


def test_apply_post_signal_length_matches_pre_signal() -> None:
    state = _tremor_state()
    result = apply(_valid_params(), state)

    assert result.post_mitigation_signal.shape == state["signal"].shape


def test_apply_residual_amplitude_is_rms_of_post_signal() -> None:
    """residual_amplitude must equal the RMS of post_mitigation_signal."""
    result = apply(_valid_params(amplitude=1.0), _tremor_state())
    expected_rms = float(np.sqrt(np.mean(np.square(result.post_mitigation_signal))))

    assert result.residual_amplitude == pytest.approx(expected_rms, rel=1e-6)


# ---------------------------------------------------------------------------
# Structural validation: invalid StimParams must raise ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_params",
    [
        # negative amplitude
        StimParams(amplitude=-1.0, pulse_frequency_hz=50.0, pulse_width_us=200.0,
                   duty_cycle=0.2, on_off_timing=(100.0, 50.0)),
        # non-finite amplitude
        StimParams(amplitude=float("inf"), pulse_frequency_hz=50.0, pulse_width_us=200.0,
                   duty_cycle=0.2, on_off_timing=(100.0, 50.0)),
        # duty_cycle > 1.0
        StimParams(amplitude=1.0, pulse_frequency_hz=50.0, pulse_width_us=200.0,
                   duty_cycle=1.5, on_off_timing=(100.0, 50.0)),
        # duty_cycle < 0.0
        StimParams(amplitude=1.0, pulse_frequency_hz=50.0, pulse_width_us=200.0,
                   duty_cycle=-0.1, on_off_timing=(100.0, 50.0)),
        # negative pulse_frequency_hz
        StimParams(amplitude=1.0, pulse_frequency_hz=-10.0, pulse_width_us=200.0,
                   duty_cycle=0.2, on_off_timing=(100.0, 50.0)),
        # negative pulse_width_us
        StimParams(amplitude=1.0, pulse_frequency_hz=50.0, pulse_width_us=-1.0,
                   duty_cycle=0.2, on_off_timing=(100.0, 50.0)),
        # negative on_ms
        StimParams(amplitude=1.0, pulse_frequency_hz=50.0, pulse_width_us=200.0,
                   duty_cycle=0.2, on_off_timing=(-1.0, 50.0)),
        # non-finite phase
        StimParams(amplitude=1.0, pulse_frequency_hz=50.0, pulse_width_us=200.0,
                   duty_cycle=0.2, on_off_timing=(100.0, 50.0), phase=float("nan")),
    ],
)
def test_apply_raises_on_invalid_stim_params(bad_params: StimParams) -> None:
    with pytest.raises(ValueError):
        apply(bad_params, _tremor_state())


# ---------------------------------------------------------------------------
# stability_warning propagation
# ---------------------------------------------------------------------------


def test_apply_propagates_stability_warning_field() -> None:
    """SimResult.stability_warning must be a bool (True or False) always."""
    result = apply(_valid_params(), _tremor_state())
    assert isinstance(result.stability_warning, bool)
