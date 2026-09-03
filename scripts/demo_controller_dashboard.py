"""Demo script: controller dashboard, end to end (Feature 54).

Not a build-plan feature itself -- runs scripts/run_closed_loop_simulation.py
(Feature 42/43) to produce a real cycle_log.csv, then plots it with
visualization/controller_dashboard.py, in one command.

Usage:
    python scripts/demo_controller_dashboard.py
    python scripts/demo_controller_dashboard.py --source recorded --path data/raw/subj02/sess02/raw_stream.csv
    python scripts/demo_controller_dashboard.py --source synthetic --frequency-hz 6.0 --amplitude 1.5
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_closed_loop_simulation import run_multi_cycle_simulation
from tremor_system.config import load_config
from visualization.controller_dashboard import plot_controller_dashboard

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: controller dashboard, end to end")
    parser.add_argument("--source", choices=["synthetic", "recorded"], default="synthetic")
    parser.add_argument("--path", type=Path, help="Required when --source recorded")
    parser.add_argument("--frequency-hz", type=float, default=6.0)
    parser.add_argument("--amplitude", type=float, default=1.5)
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "data/models/model_logistic_regression_v1.pkl")
    parser.add_argument("--scaler-path", type=Path, default=PROJECT_ROOT / "data/models/scaler_v1.pkl")
    args = parser.parse_args()

    if args.source == "recorded" and args.path is None:
        parser.error("--path is required when --source recorded")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config()
    output_dir = PROJECT_ROOT / "data" / "simulation" / "demo_controller_dashboard"
    log_path = run_multi_cycle_simulation(args, config, output_dir)

    plot_path = PROJECT_ROOT / "data" / "validation" / "demo_plots" / "controller_dashboard.png"
    plot_controller_dashboard(log_path, config.controller.target_suppression_pct, output_path=plot_path)

    logger.info("Done. Open %s", plot_path)


if __name__ == "__main__":
    main()
