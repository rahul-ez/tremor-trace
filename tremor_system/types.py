"""Shared data-contract types for tremor system.

Defines the exact dataclasses for interfaces between firmware, signal processing,
ML, estimation, controller, and simulation modules.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class StimParams:
    """Stimulation parameters emitted by the controller and consumed by simulation.

    Fields:
        amplitude: Stimulation amplitude/current.
        pulse_frequency_hz: Pulse frequency in Hz.
        pulse_width_us: Pulse width in microseconds.
        duty_cycle: Duty cycle fraction (0.0 to 1.0).
        on_off_timing: (on_ms, off_ms) timing tuple.
        phase: Optional phase offset in radians/degrees (advanced stage, default None).
    """

    amplitude: float
    pulse_frequency_hz: float
    pulse_width_us: float
    duty_cycle: float
    on_off_timing: Tuple[float, float]
    phase: Optional[float] = None


@dataclass
class SimResult:
    """Result returned by stimulation simulation.

    Fields:
        post_mitigation_signal: Time-series array after simulated mitigation.
        achieved_suppression_pct: Measured tremor suppression percentage.
        residual_amplitude: Remaining tremor amplitude post-mitigation.
        latency_ms: Modeled detection-to-actuation latency in milliseconds.
        stability_warning: True if numerical ODE solver encountered recoverable instability.
    """

    post_mitigation_signal: NDArray[np.float64]
    achieved_suppression_pct: float
    residual_amplitude: float
    latency_ms: float
    stability_warning: bool = False


@dataclass
class InferenceResult:
    """Output contract produced by ML and estimation pipelines for controller input.

    Fields:
        label: True if tremor detected, False otherwise.
        severity: Tremor severity estimate (0.0 to 1.0 scale).
        confidence: Model detection confidence probability (0.0 to 1.0).
        dominant_frequency_hz: Dominant peak frequency in tremor band.
        amplitude: Tremor RMS amplitude.
        phase: Optional estimated phase (advanced stage, default None).
    """

    label: bool
    severity: float
    confidence: float
    dominant_frequency_hz: Optional[float]
    amplitude: Optional[float]
    phase: Optional[float] = None


@dataclass
class FeatureVector:
    """Extracted feature vector for a single window.

    Fields:
        subject_id: Identifier of subject.
        session_id: Identifier of recording session.
        window_id: Index of window within session.
        tremor_band_power: Integrated power in 4-12 Hz tremor band.
        total_power: Total spectral power across frequencies.
        power_ratio: Ratio of tremor-band power to total power.
        dominant_frequency_hz: Peak frequency bin in tremor band.
        rms_amplitude: Time-domain RMS amplitude.
        variance: Time-domain signal variance.
        spectral_entropy: Normalized spectral entropy of PSD.
        accel_magnitude: 3-axis accelerometer norm magnitude.
        gyro_magnitude: 3-axis gyroscope norm magnitude.
    """

    subject_id: str
    session_id: str
    window_id: int
    tremor_band_power: float
    total_power: float
    power_ratio: float
    dominant_frequency_hz: float
    rms_amplitude: float
    variance: float
    spectral_entropy: float
    accel_magnitude: float
    gyro_magnitude: float

    def to_dict(self) -> dict[str, float]:
        """Return numeric features dict in exact standard order for ML input."""
        return {
            "tremor_band_power": self.tremor_band_power,
            "total_power": self.total_power,
            "power_ratio": self.power_ratio,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "rms_amplitude": self.rms_amplitude,
            "variance": self.variance,
            "spectral_entropy": self.spectral_entropy,
            "accel_magnitude": self.accel_magnitude,
            "gyro_magnitude": self.gyro_magnitude,
        }


@dataclass
class WindowResult:
    """Segmented time-domain window record.

    Fields:
        subject_id: Identifier of subject.
        session_id: Identifier of recording session.
        window_id: Index of window within session.
        start_timestamp_us: Microsecond timestamp of first sample in window.
        end_timestamp_us: Microsecond timestamp of last sample in window.
        signal: Calibrated and filtered sensor array shape (samples, channels).
    """

    subject_id: str
    session_id: str
    window_id: int
    start_timestamp_us: int
    end_timestamp_us: int
    signal: NDArray[np.float64]
