"""Frequency/amplitude estimation accuracy sweep against known synthetic ground truth.

Feature 49 of the build plan.

Usage:
    python -m validation.frequency_amplitude_accuracy
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from estimation.amplitude_estimation import estimate_amplitude
from estimation.frequency_estimation import estimate_frequency
from signal_processing.spectral_analysis import compute_welch_psd
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import Config, load_config

logger = logging.getLogger(__name__)

DEFAULT_FREQUENCY_SWEEP_HZ = [4.0, 4.7, 5.0, 6.0, 6.3, 7.0, 8.5, 9.0, 10.0, 11.2, 12.0]
DEFAULT_AMPLITUDE_SWEEP_G = [0.2, 0.5, 1.0, 2.0]


def run_frequency_amplitude_sweep(
    frequencies_hz: list[float],
    amplitudes_g: list[float],
    config: Config,
) -> list[dict]:
    """Run Features 28-29's estimators across a (frequency, amplitude) sweep.

    Args:
        frequencies_hz: Known tremor frequencies to test.
        amplitudes_g: Known peak sine amplitudes to test (see
            generate_synthetic_tremor -- true RMS is amplitude/sqrt(2)).
        config: Loaded system configuration.

    Returns:
        One record per (frequency, amplitude) combination, with true and
        estimated values plus per-combination error.
    """
    records = []
    for frequency_hz in frequencies_hz:
        for amplitude_g in amplitudes_g:
            signal = generate_synthetic_tremor(
                frequency_hz=frequency_hz,
                amplitude=amplitude_g,
                duration_s=config.signal.window_length_s,
                sample_rate_hz=config.sensor.sample_rate_hz,
            )
            freqs_hz, psd = compute_welch_psd(signal, config.sensor.sample_rate_hz, nperseg=signal.size)
            estimated_frequency_hz = estimate_frequency(
                freqs_hz, psd, tuple(config.signal.tremor_band_hz), config=config
            )
            estimated_amplitude_rms_g = estimate_amplitude(signal)
            true_amplitude_rms_g = amplitude_g / np.sqrt(2.0)

            records.append(
                {
                    "true_frequency_hz": frequency_hz,
                    "true_amplitude_rms_g": true_amplitude_rms_g,
                    "estimated_frequency_hz": estimated_frequency_hz,
                    "estimated_amplitude_rms_g": estimated_amplitude_rms_g,
                    "frequency_error_hz": (
                        None if estimated_frequency_hz is None else abs(estimated_frequency_hz - frequency_hz)
                    ),
                    "amplitude_error_g": abs(estimated_amplitude_rms_g - true_amplitude_rms_g),
                }
            )
    return records


def summarize_accuracy(records: list[dict]) -> dict:
    """Aggregate mean/max error across a sweep's records.

    Args:
        records: Output of run_frequency_amplitude_sweep().

    Returns:
        Dict with n_combinations, n_missed_peak (frequency_error_hz was
        None -- estimate_frequency found no clear peak), and
        mean/max_frequency_error_hz (excluding missed peaks) and
        mean/max_amplitude_error_g.

    Raises:
        ValueError: If records is empty.
    """
    if not records:
        raise ValueError("records must be non-empty")

    frequency_errors = [r["frequency_error_hz"] for r in records if r["frequency_error_hz"] is not None]
    amplitude_errors = [r["amplitude_error_g"] for r in records]

    return {
        "n_combinations": len(records),
        "n_missed_peak": sum(1 for r in records if r["frequency_error_hz"] is None),
        "mean_frequency_error_hz": (
            sum(frequency_errors) / len(frequency_errors) if frequency_errors else None
        ),
        "max_frequency_error_hz": max(frequency_errors) if frequency_errors else None,
        "mean_amplitude_error_g": sum(amplitude_errors) / len(amplitude_errors),
        "max_amplitude_error_g": max(amplitude_errors),
    }


def run_frequency_amplitude_accuracy_report(run_id: str | None = None) -> Path:
    """Run the full sweep and write data/validation/frequency_amplitude_accuracy_<run_id>/report.json.

    Note: architecture.md Success Criteria #4's "experimentally defined
    acceptable error" threshold is currently TBD (not yet set anywhere in
    this project's config or docs). This function records mean/max error
    for review, per Feature 49's verification requirement, and does not
    assert or compare against a threshold that does not exist yet.

    Returns:
        Path to the written report.json.
    """
    import json

    config = load_config()
    records = run_frequency_amplitude_sweep(DEFAULT_FREQUENCY_SWEEP_HZ, DEFAULT_AMPLITUDE_SWEEP_G, config)
    summary = summarize_accuracy(records)

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"frequency_amplitude_accuracy_{resolved_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"experiment_type": "frequency_amplitude_accuracy", "metrics": summary, "records": records}
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    mean_freq_error = summary["mean_frequency_error_hz"]
    logger.info(
        "Frequency/amplitude accuracy: mean_freq_error=%s Hz, mean_amp_error=%.4f g, missed_peak=%d/%d",
        f"{mean_freq_error:.3f}" if mean_freq_error is not None else "N/A",
        summary["mean_amplitude_error_g"],
        summary["n_missed_peak"],
        summary["n_combinations"],
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frequency/amplitude estimation accuracy sweep")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_frequency_amplitude_accuracy_report(args.run_id)


if __name__ == "__main__":
    main()
