"""Fixed-parameter mitigation experiment.

Feature 45 of the build plan. Runs the closed-loop pipeline with a static,
non-adapting StimParams (Feature 38's adapt_params is deliberately never
called -- see validation/experiments/common.py), recording suppression
achieved with a fixed strategy. Compared against Feature 46 (adaptive) to
demonstrate the value of adaptation.

Usage:
    python -m validation.experiments.fixed_mitigation \\
        --source synthetic --frequency-hz 6.0 --amplitude 1.5 --n-cycles 30 \\
        --fixed-amplitude 2.5 --fixed-pulse-frequency-hz 100.0 \\
        --fixed-pulse-width-us 200.0 --fixed-duty-cycle 0.5
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
from tremor_system.types import StimParams
from validation.experiments.common import run_experiment_cycle, sim_result_to_record, write_validation_report
from validation.experiments.io_utils import add_common_args, load_input_signal
from validation.metrics import (
    mean_residual_amplitude,
    mean_suppression_pct,
    mitigation_duty_cycle_pct,
    total_simulated_exposure,
)

logger = logging.getLogger(__name__)


def run_fixed_mitigation_experiment(args: argparse.Namespace) -> Path:
    """Run the fixed-parameter mitigation experiment and write a validation report.

    Returns:
        Path to the written report.json.
    """
    config = load_config()
    chunks = load_input_signal(args, config)

    fixed_params = StimParams(
        amplitude=args.fixed_amplitude,
        pulse_frequency_hz=args.fixed_pulse_frequency_hz,
        pulse_width_us=args.fixed_pulse_width_us,
        duty_cycle=args.fixed_duty_cycle,
        on_off_timing=(args.fixed_on_time_ms, args.fixed_off_time_ms),
    )

    state = ControllerState().reset()
    records = []
    for cycle_index, (analysis_chunk, accel_chunk, gyro_chunk) in enumerate(chunks):
        sim_result, state, ml_output = run_experiment_cycle(
            analysis_chunk, state, args.model_path, args.scaler_path, config,
            mode="fixed", fixed_params=fixed_params, accel_chunk=accel_chunk, gyro_chunk=gyro_chunk,
        )
        records.append(sim_result_to_record(cycle_index, sim_result, state, ml_output))
        logger.info(
            "cycle=%d mitigate=%s achieved_suppression_pct=%.1f%%",
            cycle_index, state.hysteresis_active, sim_result.achieved_suppression_pct,
        )

    metrics = {
        "mean_suppression_pct": mean_suppression_pct(records),
        "mean_residual_amplitude": mean_residual_amplitude(records),
        "mitigation_duty_cycle_pct": mitigation_duty_cycle_pct(records),
        "total_simulated_exposure": total_simulated_exposure(records, config.signal.window_length_s),
        "fixed_params": {
            "amplitude": fixed_params.amplitude,
            "pulse_frequency_hz": fixed_params.pulse_frequency_hz,
            "pulse_width_us": fixed_params.pulse_width_us,
            "duty_cycle": fixed_params.duty_cycle,
        },
    }

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"fixed_mitigation_{run_id}"
    return write_validation_report(output_dir, "fixed_mitigation", records, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-parameter mitigation experiment")
    add_common_args(parser)
    parser.add_argument("--fixed-amplitude", type=float, default=2.5)
    parser.add_argument("--fixed-pulse-frequency-hz", type=float, default=100.0)
    parser.add_argument("--fixed-pulse-width-us", type=float, default=200.0)
    parser.add_argument("--fixed-duty-cycle", type=float, default=0.5)
    parser.add_argument("--fixed-on-time-ms", type=float, default=10.0)
    parser.add_argument("--fixed-off-time-ms", type=float, default=10.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_fixed_mitigation_experiment(args)


if __name__ == "__main__":
    main()
