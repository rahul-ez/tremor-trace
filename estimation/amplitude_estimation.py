"""Tremor amplitude estimation from a filtered time-domain window.

Feature 29 of the build plan. Thin wrapper around
signal_processing/time_domain.py's compute_rms(), behind the estimation/
module boundary.
"""

import numpy as np
from numpy.typing import NDArray

from signal_processing.time_domain import compute_rms


def estimate_amplitude(filtered_signal: NDArray[np.float64]) -> float:
    """Estimate tremor amplitude as the RMS of the filtered tremor-band signal.

    Args:
        filtered_signal: shape (n_samples,), band-pass filtered tremor-band
            signal for one window, units: g.

    Returns:
        RMS amplitude in g.
    """
    return compute_rms(filtered_signal)
