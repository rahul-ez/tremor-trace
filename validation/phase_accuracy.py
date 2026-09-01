"""Phase-estimation accuracy sweep against known synthetic ground truth.

Feature 50 of the build plan. Conditional -- only meaningful if Feature 31
(estimation/phase_estimation.py) is enabled via
config.estimation.phase_enabled. Per the build plan's Verification: "If
Feature 31 remains disabled, skip this feature entirely -- do not
fabricate phase-aware controller behavior without this verification
passing first." run_phase_accuracy_report() raises rather than silently
returning an empty/fake report when the flag is off, so nothing downstream
can mistake "skipped" for "passed".

Same sweep methodology as Feature 49 (validation/frequency_amplitude_accuracy.py).

Usage:
    python -m validation.phase_accuracy
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

from estimation.phase_estimation import estimate_phase
from simulation.tremor_model import generate_synthetic_tremor
from tremor_system.config import Config, load_config

logger = logging.getLogger(__name__)

DEFAULT_FREQUENCY_SWEEP_HZ = [4.0, 6.0, 8.5, 10.0, 12.0]


def run_phase_sweep(frequencies_hz: list[float], config: Config) -> list[dict]:
    """Run estimate_phase() across a sweep of known frequencies.

    Ground truth phase at the window's final sample is computed directly
    from the same cosine convention estimate_phase() uses (see
    estimation/phase_estimation.py's tests for the derivation): for
    signal = amplitude * cos(2*pi*f*t), the analytic-signal phase at time
    t equals (2*pi*f*t) wrapped to (-pi, pi].

    Args:
        frequencies_hz: Known tremor frequencies to test.
        config: Loaded system configuration.

    Returns:
        One record per frequency, with true and estimated phase (radians)
        plus absolute error.
    """
    records = []
    for frequency_hz in frequencies_hz:
        n_samples = int(round(config.signal.window_length_s * config.sensor.sample_rate_hz))
        time_s = np.arange(n_samples, dtype=np.float64) / config.sensor.sample_rate_hz
        signal = np.cos(2.0 * np.pi * frequency_hz * time_s)  # cosine, not generate_synthetic_tremor's sine

        estimated_phase = estimate_phase(signal)
        true_phase = (2.0 * np.pi * frequency_hz * time_s[-1] + np.pi) % (2.0 * np.pi) - np.pi

        records.append(
            {
                "frequency_hz": frequency_hz,
                "true_phase_rad": float(true_phase),
                "estimated_phase_rad": estimated_phase,
                "phase_error_rad": None if estimated_phase is None else abs(estimated_phase - true_phase),
            }
        )
    return records


def summarize_phase_accuracy(records: list[dict]) -> dict:
    """Aggregate mean/max phase error across a sweep's records.

    Raises:
        ValueError: If records is empty.
    """
    if not records:
        raise ValueError("records must be non-empty")

    errors = [r["phase_error_rad"] for r in records if r["phase_error_rad"] is not None]
    return {
        "n_frequencies": len(records),
        "n_undefined": sum(1 for r in records if r["phase_error_rad"] is None),
        "mean_phase_error_rad": sum(errors) / len(errors) if errors else None,
        "max_phase_error_rad": max(errors) if errors else None,
    }


def run_phase_accuracy_report(run_id: str | None = None) -> Path:
    """Run the phase sweep and write data/validation/phase_accuracy_<run_id>/report.json.

    Returns:
        Path to the written report.json.

    Raises:
        RuntimeError: If config.estimation.phase_enabled is False -- per
            the build plan, this feature must be skipped entirely (not run
            with a fabricated/placeholder result) while Feature 31 remains
            disabled.
    """
    import json

    config = load_config()
    if not config.estimation.phase_enabled:
        raise RuntimeError(
            "config.estimation.phase_enabled is False; Feature 50 (phase accuracy) "
            "is skipped per the build plan -- enable Feature 31 first if you want "
            "this report."
        )

    records = run_phase_sweep(DEFAULT_FREQUENCY_SWEEP_HZ, config)
    summary = summarize_phase_accuracy(records)

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"phase_accuracy_{resolved_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"experiment_type": "phase_accuracy", "metrics": summary, "records": records}
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Phase accuracy report written to %s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase-estimation accuracy sweep")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_phase_accuracy_report(args.run_id)


if __name__ == "__main__":
    main()
