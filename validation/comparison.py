"""Consolidates no_mitigation/fixed/adaptive reports into one comparison report.

Fills a gap left by Phase 9: build-plan Feature 55 expects
data/validation/comparison_<run_id>/report.json to exist, combining all
three experiment types' metrics side by side. validation/metrics.py
(Feature 47) computes the per-experiment metrics; Features 44-46 each
write their own separate report.json; nothing previously merged them.

Usage:
    python -m validation.comparison \\
        --no-mitigation-report data/validation/no_mitigation_X/report.json \\
        --fixed-report data/validation/fixed_mitigation_X/report.json \\
        --adaptive-report data/validation/adaptive_mitigation_X/report.json
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def build_comparison_report(
    no_mitigation_report_path: Path,
    fixed_report_path: Path,
    adaptive_report_path: Path,
) -> dict:
    """Merge three experiment reports' metrics into one comparison dict.

    Args:
        no_mitigation_report_path: Path to a Feature 44 report.json.
        fixed_report_path: Path to a Feature 45 report.json.
        adaptive_report_path: Path to a Feature 46 report.json.

    Returns:
        Dict with "strategies": {"no_mitigation": ..., "fixed_mitigation":
        ..., "adaptive_mitigation": ...}, each mapping to that report's
        "metrics" dict, plus "cycle_log_paths" pointing at each
        experiment's raw per-cycle CSV (for visualization/validation_report.py).

    Raises:
        FileNotFoundError: If any report path does not exist.
    """
    def load(path: Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")
        with open(path) as f:
            return json.load(f)

    no_mitigation = load(no_mitigation_report_path)
    fixed = load(fixed_report_path)
    adaptive = load(adaptive_report_path)

    return {
        "experiment_type": "comparison",
        "strategies": {
            "no_mitigation": no_mitigation["metrics"],
            "fixed_mitigation": fixed["metrics"],
            "adaptive_mitigation": adaptive["metrics"],
        },
        "cycle_log_paths": {
            "no_mitigation": str(Path(no_mitigation_report_path).parent / "cycle_log.csv"),
            "fixed_mitigation": str(Path(fixed_report_path).parent / "cycle_log.csv"),
            "adaptive_mitigation": str(Path(adaptive_report_path).parent / "cycle_log.csv"),
        },
    }


def write_comparison_report(
    no_mitigation_report_path: Path,
    fixed_report_path: Path,
    adaptive_report_path: Path,
    run_id: str | None = None,
) -> Path:
    """Build and write data/validation/comparison_<run_id>/report.json.

    Returns:
        Path to the written report.json.
    """
    comparison = build_comparison_report(no_mitigation_report_path, fixed_report_path, adaptive_report_path)

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "validation" / f"comparison_{resolved_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info("Wrote comparison report to %s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Feature 44-46 reports into one comparison report")
    parser.add_argument("--no-mitigation-report", type=Path, required=True)
    parser.add_argument("--fixed-report", type=Path, required=True)
    parser.add_argument("--adaptive-report", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    write_comparison_report(args.no_mitigation_report, args.fixed_report, args.adaptive_report, args.run_id)


if __name__ == "__main__":
    main()
