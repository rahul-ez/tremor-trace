"""Welch power spectral density and tremor-band spectral features."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

from tremor_system.config import load_config


def _validate_spectral_arrays(
    freqs_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
) -> None:
    if freqs_hz.ndim != 1 or psd.ndim != 1:
        raise ValueError("Expected 1D frequency and PSD arrays")
    if freqs_hz.shape != psd.shape:
        raise ValueError(
            f"Frequency and PSD shapes must match, got {freqs_hz.shape} and {psd.shape}"
        )
    if freqs_hz.size == 0:
        raise ValueError("Expected non-empty frequency and PSD arrays")


def _band_mask(
    freqs_hz: NDArray[np.float64],
    band_hz: tuple[float, float],
) -> NDArray[np.bool_]:
    if len(band_hz) != 2 or not 0 <= band_hz[0] < band_hz[1]:
        raise ValueError(f"Expected band_hz=(low, high), got {band_hz}")
    mask = (freqs_hz >= band_hz[0]) & (freqs_hz <= band_hz[1])
    if not np.any(mask):
        raise ValueError(f"No frequency bins found in band {band_hz}")
    return mask


def compute_welch_psd(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    nperseg: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Welch PSD for one complete signal window.

    Args:
        signal: shape (n_samples,), units: g or deg/s.
        sample_rate_hz: Sampling rate in Hz.
        nperseg: Samples per Welch segment; v1 uses 200.

    Returns:
        Frequencies in Hz and PSD values, both shape (n_frequency_bins,).
        This is an offline/non-causal windowed analysis.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal array, got shape {signal.shape}")
    if signal.size == 0:
        raise ValueError("Expected non-empty signal")
    if sample_rate_hz <= 0:
        raise ValueError(f"Expected positive sample_rate_hz, got {sample_rate_hz}")
    if nperseg <= 0 or nperseg > signal.size:
        raise ValueError(
            f"Expected nperseg in [1, {signal.size}], got {nperseg}"
        )
    freqs_hz, psd = welch(signal, fs=sample_rate_hz, nperseg=nperseg)
    return np.asarray(freqs_hz, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def tremor_band_power(
    freqs_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
    tremor_band_hz: tuple[float, float] | None = None,
) -> float:
    """Integrate PSD power over the configured tremor band.

    Args:
        freqs_hz: shape (n_frequency_bins,), units: Hz.
        psd: shape (n_frequency_bins,), power spectral density units.
        tremor_band_hz: Inclusive band in Hz; defaults to configuration.

    Returns:
        Numerically integrated tremor-band power.
    """
    _validate_spectral_arrays(freqs_hz, psd)
    band = (
        tuple(load_config().signal.tremor_band_hz)
        if tremor_band_hz is None
        else tremor_band_hz
    )
    mask = _band_mask(freqs_hz, band)
    return float(np.trapezoid(psd[mask], freqs_hz[mask]))


def total_power(
    freqs_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
) -> float:
    """Integrate the PSD across its complete supplied frequency range."""
    _validate_spectral_arrays(freqs_hz, psd)
    return float(np.trapezoid(psd, freqs_hz))


def spectral_entropy(psd: NDArray[np.float64]) -> float:
    """Compute normalized Shannon entropy of a non-negative PSD.

    Args:
        psd: shape (n_frequency_bins,), non-negative power spectral density.

    Returns:
        Normalized entropy in the range [0, 1]. Zero PSD bins contribute zero.
    """
    if psd.ndim != 1:
        raise ValueError(f"Expected 1D PSD array, got shape {psd.shape}")
    if psd.size == 0:
        raise ValueError("Expected non-empty PSD")
    if np.any(psd < 0):
        raise ValueError("PSD values must be non-negative")

    psd_sum = float(np.sum(psd))
    if psd_sum == 0.0 or psd.size == 1:
        return 0.0

    probabilities = psd / psd_sum
    nonzero_probabilities = probabilities[probabilities > 0.0]
    entropy = -float(
        np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))
    )
    normalized_entropy = entropy / float(np.log2(psd.size))
    return float(np.clip(normalized_entropy, 0.0, 1.0))


def power_ratio(tremor_power: float, total_power_value: float) -> float:
    """Return tremor-to-total power using the required numerical guard."""
    return float(tremor_power / (total_power_value + 1e-12))


def dominant_frequency(
    freqs_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
    tremor_band_hz: tuple[float, float] | None = None,
) -> float:
    """Return the maximum PSD frequency bin within the tremor band."""
    _validate_spectral_arrays(freqs_hz, psd)
    band = (
        tuple(load_config().signal.tremor_band_hz)
        if tremor_band_hz is None
        else tremor_band_hz
    )
    mask = _band_mask(freqs_hz, band)
    band_indices = np.flatnonzero(mask)
    strongest_index = band_indices[int(np.argmax(psd[mask]))]
    return float(freqs_hz[strongest_index])