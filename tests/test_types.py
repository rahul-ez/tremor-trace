"""Tests for shared data-contract types."""

import numpy as np

from tremor_system.types import (
    FeatureVector,
    InferenceResult,
    SimResult,
    StimParams,
    WindowResult,
)


def test_stim_params_instantiation() -> None:
    params = StimParams(
        amplitude=1.5,
        pulse_frequency_hz=50.0,
        pulse_width_us=200.0,
        duty_cycle=0.2,
        on_off_timing=(100.0, 50.0),
        phase=None,
    )
    assert params.amplitude == 1.5
    assert params.pulse_frequency_hz == 50.0
    assert params.pulse_width_us == 200.0
    assert params.duty_cycle == 0.2
    assert params.on_off_timing == (100.0, 50.0)
    assert params.phase is None


def test_sim_result_instantiation() -> None:
    signal = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    res = SimResult(
        post_mitigation_signal=signal,
        achieved_suppression_pct=65.0,
        residual_amplitude=0.05,
        latency_ms=15.0,
        stability_warning=False,
    )
    assert np.array_equal(res.post_mitigation_signal, signal)
    assert res.achieved_suppression_pct == 65.0
    assert res.residual_amplitude == 0.05
    assert res.latency_ms == 15.0
    assert res.stability_warning is False


def test_inference_result_instantiation() -> None:
    inf = InferenceResult(
        label=True,
        severity=0.7,
        confidence=0.92,
        dominant_frequency_hz=5.5,
        amplitude=0.45,
        phase=None,
    )
    assert inf.label is True
    assert inf.severity == 0.7
    assert inf.confidence == 0.92
    assert inf.dominant_frequency_hz == 5.5
    assert inf.amplitude == 0.45
    assert inf.phase is None


def test_feature_vector_instantiation_and_dict() -> None:
    fv = FeatureVector(
        subject_id="subj01",
        session_id="sess01",
        window_id=0,
        tremor_band_power=0.8,
        total_power=1.0,
        power_ratio=0.8,
        dominant_frequency_hz=6.0,
        rms_amplitude=0.25,
        variance=0.0625,
        spectral_entropy=0.4,
        accel_magnitude=0.3,
        gyro_magnitude=12.5,
    )
    assert fv.subject_id == "subj01"
    assert fv.session_id == "sess01"
    assert fv.window_id == 0

    d = fv.to_dict()
    expected_keys = [
        "tremor_band_power",
        "total_power",
        "power_ratio",
        "dominant_frequency_hz",
        "rms_amplitude",
        "variance",
        "spectral_entropy",
        "accel_magnitude",
        "gyro_magnitude",
    ]
    assert list(d.keys()) == expected_keys
    assert d["tremor_band_power"] == 0.8
    assert d["gyro_magnitude"] == 12.5


def test_window_result_instantiation() -> None:
    sig = np.zeros((100, 3), dtype=np.float64)
    wr = WindowResult(
        subject_id="subj01",
        session_id="sess01",
        window_id=1,
        start_timestamp_us=1000000,
        end_timestamp_us=2000000,
        signal=sig,
    )
    assert wr.subject_id == "subj01"
    assert wr.start_timestamp_us == 1000000
    assert wr.end_timestamp_us == 2000000
    assert wr.signal.shape == (100, 3)
