"""No-mitigation baseline experiment.

Feature 44 of the build plan. Runs the closed-loop pipeline with the
controller's mitigation decision forced to False for every cycle (Feature
20/27/30's Detect+Analyze stages still run for real; decide_mitigation()
is simply never called -- see validation/experiments/common.py), recording
tremor power over time with no intervention. Establishes the baseline
Feature 45/46 are compared against.

Usage:
    python -m validation.experiments.no_mitigation \\
        --source synthetic --frequency-hz 6.0 --amplitude 1.5 --n-cycles 30
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.controller_state import ControllerState
from tremor_system.config import load_config
from validation.experiments.common import run_experiment_cycle, sim_result_to_record, write_validation_report
from validation.experiments.io_utils import add_common_args, load_input_signal
from validation.metrics import mean_residual_amplitude, mean_suppression_pct

logger = logging.getLogger(__name__)


def run_no_mitigation_experiment(args: argparse.Namespace) -> Path:
    """Run the no-mitigation baseline and write a validation report.

    Returns:
        Path to the written report.json.
    """
    config = load_config()
    chunks = load_input_signal(args, config)

    state = ControllerState().reset()
    records = []
    for cycle_index, (analysis_chunk, accel_chunk, gyro_chunk) in enumerate(chunks):
        sim_result, state, ml_output = run_experiment_cycle(
            analysis_chunk, state, args.model_path, args.scaler_path, config,
            mode="no_mitigation", accel_chunk=accel_chunk, gyro_chunk=gyro_chunk,
        )
        records.append(sim_result_to_record(cycle_index, sim_result, state, ml_output))
        logger.info("cycle=%d achieved_suppression_pct=%.1f%%", cycle_index, sim_result.achieved_suppression_pct)

    metrics = {
        "mean_suppression_pct": mean_suppression_pct(records),
        "mean_residual_amplitude": mean_residual_amplitude(records),
    }

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"no_mitigation_{run_id}"
    return write_validation_report(output_dir, "no_mitigation", records, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-mitigation baseline experiment")
    add_common_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_no_mitigation_experiment(args)


if __name__ == "__main__":
    main()
