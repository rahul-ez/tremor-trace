"""Tests for suppression measurement (Feature 34)."""

import numpy as np
import pytest

from simulation.suppression import compute_suppression_pct
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import load_config

_CFG = load_config()
_TREMOR_BAND = tuple(_CFG.signal.tremor_band_hz)
_SAMPLE_RATE = _CFG.sensor.sample_rate_hz


def _in_band_signal(amplitude: float = 0.4, duration_s: float = 2.0) -> np.ndarray:
    """Pure 6 Hz tremor-band sine -- maximises tremor-band power."""
    return generate_synthetic_tremor(6.0, amplitude, duration_s, _SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_known_power_reduction_within_tolerance() -> None:
    """Full suppression (zero post-signal) must return 100 %."""
    pre = _in_band_signal(amplitude=0.4)
    post = np.zeros_like(pre)

    result = compute_suppression_pct(pre, post, _SAMPLE_RATE, _TREMOR_BAND)

    assert result == pytest.approx(100.0, abs=0.1)


def test_no_suppression_returns_zero() -> None:
    """Identical pre and post must return 0 % suppression."""
    signal = _in_band_signal(amplitude=0.4)

    result = compute_suppression_pct(signal, signal.copy(), _SAMPLE_RATE, _TREMOR_BAND)

    assert result == pytest.approx(0.0, abs=0.1)


def test_partial_suppression_is_between_zero_and_hundred() -> None:
    """Smaller-amplitude post-signal must produce suppression in (0, 100)."""
    pre = _in_band_signal(amplitude=0.4)
    # Halved amplitude => power reduced to 1/4 => ~75 % suppression
    post = _in_band_signal(amplitude=0.2)

    result = compute_suppression_pct(pre, post, _SAMPLE_RATE, _TREMOR_BAND)

    assert 0.0 < result < 100.0


def test_post_greater_than_pre_is_clamped_to_zero() -> None:
    """When post has more power than pre the result must be 0.0, not negative."""
    pre = _in_band_signal(amplitude=0.2)
    post = _in_band_signal(amplitude=0.4)  # more power than pre

    result = compute_suppression_pct(pre, post, _SAMPLE_RATE, _TREMOR_BAND)

    assert result == 0.0


def test_near_zero_pre_signal_returns_zero_without_raising() -> None:
    """Near-silent pre-signal must return 0.0 gracefully (warning logged, no exception)."""
    pre = np.zeros(200, dtype=np.float64)
    post = np.zeros(200, dtype=np.float64)

    result = compute_suppression_pct(pre, post, _SAMPLE_RATE, _TREMOR_BAND)

    assert result == 0.0


def test_result_is_always_within_bounds() -> None:
    """Result must always be in [0.0, 100.0] for arbitrary in-band signals."""
    for amp_pre, amp_post in [(0.1, 0.4), (0.4, 0.1), (0.3, 0.3)]:
        pre = _in_band_signal(amplitude=amp_pre)
        post = _in_band_signal(amplitude=amp_post)
        result = compute_suppression_pct(pre, post, _SAMPLE_RATE, _TREMOR_BAND)
        assert 0.0 <= result <= 100.0, f"Out of bounds for amp_pre={amp_pre}, amp_post={amp_post}"


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pre, post, sample_rate_hz",
    [
        (np.zeros((2, 2), dtype=np.float64), np.zeros(4, dtype=np.float64), 100.0),  # 2D pre
        (np.zeros(4, dtype=np.float64), np.zeros((2, 2), dtype=np.float64), 100.0),  # 2D post
        (np.zeros(0, dtype=np.float64), np.zeros(4, dtype=np.float64), 100.0),       # empty pre
        (np.zeros(4, dtype=np.float64), np.zeros(3, dtype=np.float64), 100.0),       # length mismatch
        (np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64), 0.0),         # zero sample rate
        (np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64), -1.0),        # negative sample rate
    ],
)
def test_compute_suppression_pct_raises_on_invalid_input(
    pre: np.ndarray, post: np.ndarray, sample_rate_hz: float
) -> None:
    with pytest.raises(ValueError):
        compute_suppression_pct(pre, post, sample_rate_hz, _TREMOR_BAND)
