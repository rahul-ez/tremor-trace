"""Demo script: no-mitigation vs. fixed vs. adaptive comparison, end to end (Feature 55).

Not a build-plan feature itself -- runs Features 44-46 on the same input
signal, merges their reports (validation/comparison.py), and plots the
comparison (visualization/validation_report.py), in one command.

Usage:
    python scripts/demo_validation_comparison.py
    python scripts/demo_validation_comparison.py --frequency-hz 6.0 --amplitude 1.5 --n-cycles 20 --fixed-amplitude 4.0
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.comparison import write_comparison_report
from validation.experiments.adaptive_mitigation import run_adaptive_mitigation_experiment
from validation.experiments.fixed_mitigation import run_fixed_mitigation_experiment
from validation.experiments.no_mitigation import run_no_mitigation_experiment
from visualization.validation_report import plot_validation_comparison

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: no-mitigation vs. fixed vs. adaptive comparison")
    parser.add_argument("--source", choices=["synthetic", "recorded"], default="synthetic")
    parser.add_argument("--path", type=Path, help="Required when --source recorded")
    parser.add_argument("--frequency-hz", type=float, default=6.0)
    parser.add_argument("--amplitude", type=float, default=1.5)
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "data/models/model_logistic_regression_v1.pkl")
    parser.add_argument("--scaler-path", type=Path, default=PROJECT_ROOT / "data/models/scaler_v1.pkl")
    parser.add_argument("--fixed-amplitude", type=float, default=4.0, help="A non-optimal a-priori guess, on purpose")
    parser.add_argument("--run-id", type=str, default="demo")
    args = parser.parse_args()

    if args.source == "recorded" and args.path is None:
        parser.error("--path is required when --source recorded")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    common = dict(
        source=args.source, path=args.path, frequency_hz=args.frequency_hz, amplitude=args.amplitude,
        n_cycles=args.n_cycles, model_path=args.model_path, scaler_path=args.scaler_path, run_id=args.run_id,
    )

    no_mit_args = argparse.Namespace(**common)
    fixed_args = argparse.Namespace(
        **common, fixed_amplitude=args.fixed_amplitude, fixed_pulse_frequency_hz=100.0,
        fixed_pulse_width_us=200.0, fixed_duty_cycle=0.5, fixed_on_time_ms=10.0, fixed_off_time_ms=10.0,
    )
    adaptive_args = argparse.Namespace(**common)

    no_mit_report = run_no_mitigation_experiment(no_mit_args)
    fixed_report = run_fixed_mitigation_experiment(fixed_args)
    adaptive_report = run_adaptive_mitigation_experiment(adaptive_args)

    comparison_report = write_comparison_report(no_mit_report, fixed_report, adaptive_report, run_id=args.run_id)

    plot_path = PROJECT_ROOT / "data" / "validation" / "demo_plots" / "comparison.png"
    plot_validation_comparison(comparison_report, output_path=plot_path)

    logger.info("Done. Open %s", plot_path)


if __name__ == "__main__":
    main()
