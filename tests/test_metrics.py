"""Tests for validation/metrics.py, with hand-computed expected values."""

import pytest

from validation.metrics import (
    controller_stability_oscillation,
    false_activation_rate,
    mean_detection_to_actuation_latency_ms,
    mean_residual_amplitude,
    mean_suppression_pct,
    mitigation_duty_cycle_pct,
    time_to_target,
    total_simulated_exposure,
)


def _record(
    cycle: int,
    mitigate: bool,
    suppression: float = 0.0,
    residual: float = 0.0,
    amplitude: float | None = None,
    duty_cycle: float | None = None,
    latency_ms: float = 20.0,
) -> dict:
    return {
        "cycle": cycle,
        "mitigate": mitigate,
        "hysteresis_active": mitigate,
        "achieved_suppression_pct": suppression,
        "residual_amplitude": residual,
        "stability_warning": False,
        "latency_ms": latency_ms,
        "amplitude": amplitude,
        "pulse_frequency_hz": 100.0 if amplitude is not None else None,
        "duty_cycle": duty_cycle,
        "confidence": 0.9,
        "label": mitigate,
    }


def test_mean_suppression_pct_hand_computed() -> None:
    records = [_record(0, True, suppression=20.0), _record(1, True, suppression=40.0), _record(2, False, suppression=0.0)]

    assert mean_suppression_pct(records) == pytest.approx(20.0)  # (20+40+0)/3


def test_mean_suppression_pct_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mean_suppression_pct([])


def test_mean_residual_amplitude_hand_computed() -> None:
    records = [_record(0, True, residual=0.1), _record(1, True, residual=0.3)]

    assert mean_residual_amplitude(records) == pytest.approx(0.2)  # (0.1+0.3)/2


def test_time_to_target_finds_first_entry_and_converts_to_seconds() -> None:
    records = [
        _record(0, True, suppression=80.0),
        _record(1, True, suppression=60.0),
        _record(2, True, suppression=52.0),  # first cycle within [45, 55]
        _record(3, True, suppression=51.0),
    ]

    result = time_to_target(records, target_suppression_pct=50.0, tolerance_pct=5.0, cycle_duration_s=2.0)

    assert result == pytest.approx(4.0)  # cycle 2 * 2.0s/cycle


def test_time_to_target_returns_none_when_never_reached() -> None:
    records = [_record(0, True, suppression=90.0), _record(1, True, suppression=85.0)]

    result = time_to_target(records, target_suppression_pct=50.0, tolerance_pct=5.0, cycle_duration_s=2.0)

    assert result is None


def test_mitigation_duty_cycle_pct_hand_computed() -> None:
    records = [_record(0, True), _record(1, True), _record(2, False), _record(3, False)]

    assert mitigation_duty_cycle_pct(records) == pytest.approx(50.0)  # 2/4


def test_total_simulated_exposure_hand_computed() -> None:
    records = [
        _record(0, True, amplitude=2.0, duty_cycle=0.5),
        _record(1, True, amplitude=4.0, duty_cycle=0.25),
        _record(2, False),  # amplitude None -> contributes 0
    ]

    # (2.0 + 4.0) * 2.0s = 12.0 -- amplitude only, not amplitude*duty_cycle
    # (see total_simulated_exposure's docstring for why duty_cycle is excluded)
    assert total_simulated_exposure(records, cycle_duration_s=2.0) == pytest.approx(12.0)


def test_false_activation_rate_hand_computed() -> None:
    records = [_record(0, True), _record(1, False), _record(2, True), _record(3, False)]
    ground_truth = [False, False, True, False]
    # no-tremor cycles (ground_truth=False): indices 0,1,3 -> mitigate=[True, False, False]
    # false activations: 1 out of 3 no-tremor cycles

    rate = false_activation_rate(records, ground_truth)

    assert rate == pytest.approx(100.0 / 3.0)


def test_false_activation_rate_zero_when_no_negative_cycles() -> None:
    records = [_record(0, True)]
    ground_truth = [True]

    assert false_activation_rate(records, ground_truth) == pytest.approx(0.0)


def test_false_activation_rate_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="aligned"):
        false_activation_rate([_record(0, True)], [True, False])


def test_mean_detection_to_actuation_latency_hand_computed() -> None:
    records = [_record(0, True, latency_ms=20.0), _record(1, True, latency_ms=30.0), _record(2, False, latency_ms=999.0)]

    # only mitigating cycles counted: (20+30)/2 = 25
    assert mean_detection_to_actuation_latency_ms(records) == pytest.approx(25.0)


def test_mean_detection_to_actuation_latency_zero_when_never_mitigated() -> None:
    records = [_record(0, False), _record(1, False)]

    assert mean_detection_to_actuation_latency_ms(records) == pytest.approx(0.0)


def test_controller_stability_oscillation_hand_computed() -> None:
    records = [
        _record(0, True, amplitude=1.0),
        _record(1, True, amplitude=3.0),
        _record(2, True, amplitude=2.0),
    ]

    # |3-1| + |2-3| = 2 + 1 = 3, mean over 2 deltas = 1.5
    assert controller_stability_oscillation(records) == pytest.approx(1.5)


def test_controller_stability_oscillation_zero_for_single_cycle() -> None:
    records = [_record(0, True, amplitude=1.0)]

    assert controller_stability_oscillation(records) == pytest.approx(0.0)
