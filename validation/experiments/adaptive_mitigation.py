"""Adaptive mitigation experiment.

Feature 46 of the build plan. Reuses Feature 42's multi-cycle closed-loop
runner unmodified (via validation/experiments/common.py::run_experiment_cycle
mode="adaptive", which delegates directly to
simulation/closed_loop_runner.py::run_closed_loop_cycle()), writing results
to the validation report format instead of scripts/run_closed_loop_simulation.py's
plain CSV+plot output.

Usage:
    python -m validation.experiments.adaptive_mitigation \\
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
from validation.metrics import (
    controller_stability_oscillation,
    mean_residual_amplitude,
    mean_suppression_pct,
    mitigation_duty_cycle_pct,
    time_to_target,
    total_simulated_exposure,
)

logger = logging.getLogger(__name__)


def run_adaptive_mitigation_experiment(args: argparse.Namespace) -> Path:
    """Run the full adaptive closed loop and write a validation report.

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
            mode="adaptive", accel_chunk=accel_chunk, gyro_chunk=gyro_chunk,
        )
        records.append(sim_result_to_record(cycle_index, sim_result, state, ml_output))
        logger.info(
            "cycle=%d mitigate=%s achieved_suppression_pct=%.1f%% amplitude=%s",
            cycle_index, state.hysteresis_active, sim_result.achieved_suppression_pct,
            f"{state.current_params.amplitude:.3f}" if state.current_params else "N/A",
        )

    metrics = {
        "mean_suppression_pct": mean_suppression_pct(records),
        "mean_residual_amplitude": mean_residual_amplitude(records),
        "mitigation_duty_cycle_pct": mitigation_duty_cycle_pct(records),
        "total_simulated_exposure": total_simulated_exposure(records, config.signal.window_length_s),
        "time_to_target_s": time_to_target(
            records,
            config.controller.target_suppression_pct,
            config.controller.suppression_tolerance_pct,
            config.signal.window_length_s,
        ),
        "controller_stability_oscillation": controller_stability_oscillation(records),
    }

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"adaptive_mitigation_{run_id}"
    return write_validation_report(output_dir, "adaptive_mitigation", records, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adaptive mitigation experiment")
    add_common_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_adaptive_mitigation_experiment(args)


if __name__ == "__main__":
    main()
