"""Deterministic synthetic tremor signal generation for simulation and validation."""

import numpy as np
from numpy.typing import NDArray


def generate_synthetic_tremor(
    frequency_hz: float,
    amplitude: float,
    duration_s: float,
    sample_rate_hz: float,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate a sinusoidal tremor trajectory with optional seeded noise.

    Args:
        frequency_hz: Tremor frequency in Hz.
        amplitude: Sinusoidal peak amplitude in g.
        duration_s: Requested signal duration in seconds.
        sample_rate_hz: Signal sampling rate in Hz.
        noise_std: Standard deviation of additive Gaussian noise in g.
        seed: Optional seed for reproducible noise.

    Returns:
        Float64 signal of shape ``(n_samples,)`` in g. The sample count is
        ``round(duration_s * sample_rate_hz)``.

    Raises:
        ValueError: If frequency, amplitude, duration, sample rate, or noise
            standard deviation is invalid.
    """
    if frequency_hz < 0.0:
        raise ValueError(f"frequency_hz must be non-negative, got {frequency_hz}")
    if amplitude < 0.0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}")
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if sample_rate_hz <= 0.0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    n_samples = int(round(duration_s * sample_rate_hz))
    if n_samples < 2:
        raise ValueError("duration_s * sample_rate_hz must produce at least 2 samples")

    time_s = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    signal = amplitude * np.sin(2.0 * np.pi * frequency_hz * time_s)
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        signal = signal + rng.normal(0.0, noise_std, size=n_samples)

    return np.asarray(signal, dtype=np.float64)
