"""Dominant tremor-frequency estimation from a windowed Welch PSD.

Feature 28 of the build plan. Thin wrapper around
signal_processing/spectral_analysis.py's dominant_frequency(), adding a
flat-spectrum guard so callers get None instead of an arbitrary bin choice
when there is no real spectral peak (e.g. noise-only input).
"""

import numpy as np
from numpy.typing import NDArray

from signal_processing.spectral_analysis import dominant_frequency
from tremor_system.config import Config, load_config

# Minimum ratio of peak in-band PSD value to mean in-band PSD value required
# to call a peak "real" rather than noise. Experimental heuristic --
# architecture.md's Open Questions has no assigned value for this; tune
# once labeled tremor/no-tremor recordings are available to validate it
# against, and record the chosen value in memory.md when it changes.
DEFAULT_MIN_PEAK_TO_MEAN_RATIO = 1.5


def _validate_spectral_arrays(freqs_hz: NDArray[np.float64], psd: NDArray[np.float64]) -> None:
    if freqs_hz.ndim != 1 or psd.ndim != 1 or freqs_hz.shape != psd.shape or freqs_hz.size == 0:
        raise ValueError("freqs_hz and psd must be non-empty, matching, 1D arrays")


def estimate_frequency(
    freqs_hz: NDArray[np.float64],
    psd: NDArray[np.float64],
    tremor_band_hz: tuple[float, float] | None = None,
    min_peak_to_mean_ratio: float = DEFAULT_MIN_PEAK_TO_MEAN_RATIO,
    config: Config | None = None,
) -> float | None:
    """Return the dominant in-band frequency, or None if the spectrum is flat.

    Args:
        freqs_hz: shape (n_frequency_bins,), from compute_welch_psd().
        psd: shape (n_frequency_bins,), from compute_welch_psd().
        tremor_band_hz: Inclusive band in Hz; defaults to configuration.
        min_peak_to_mean_ratio: Peak/mean in-band PSD ratio required to
            treat the peak as real rather than noise.
        config: Loaded system configuration; defaults to load_config().
            Exposed as a parameter so callers doing per-window inference in
            a loop can load it once and pass it through.

    Returns:
        Dominant frequency in Hz, or None if no clear peak exists.
    """
    _validate_spectral_arrays(freqs_hz, psd)
    resolved_config = config or load_config()
    band = tuple(resolved_config.signal.tremor_band_hz) if tremor_band_hz is None else tremor_band_hz
    mask = (freqs_hz >= band[0]) & (freqs_hz <= band[1])
    if not np.any(mask):
        raise ValueError(f"No frequency bins found in band {band}")

    band_psd = psd[mask]
    mean_power = float(np.mean(band_psd))
    peak_power = float(np.max(band_psd))
    if mean_power <= 0.0 or peak_power < min_peak_to_mean_ratio * mean_power:
        return None

    return dominant_frequency(freqs_hz, psd, tremor_band_hz=band)
