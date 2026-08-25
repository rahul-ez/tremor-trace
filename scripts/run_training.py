"""Train, evaluate, and select the Phase 4 tremor-detection model.

Ties together Features 22-26 of the build plan:
    dataset_builder (labels + subject-level folds)
        -> baseline_detector (non-ML reference point)
        -> train (LogReg/SVM/RandomForest per fold)
        -> evaluate (precision/recall/F1/sensitivity/FPR per fold)
        -> model_selection (smallest model meeting thresholds)

Saves the winning model + scaler under data/models/ with a versioned
filename, and writes a full evaluation report (including the baseline) to
data/models/evaluation_results.csv.

Usage:
    python scripts/run_training.py \\
        --features data/features/features.csv \\
        --min-recall 0.7 --min-precision 0.7
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.baseline_detector import DEFAULT_THRESHOLD_FEATURE, evaluate_baseline
from ml.dataset_builder import FEATURE_COLUMNS, assign_labels, get_subject_level_folds, load_dataset
from ml.evaluate import aggregate_fold_results, evaluate_candidates_across_folds
from ml.model_selection import select_model
from ml.train import save_artifacts, train_and_scale
from tremor_system.config import load_config

logger = logging.getLogger(__name__)


def run_training(
    features_path: Path,
    models_dir: Path,
    min_recall: float,
    min_precision: float,
    n_splits: int,
    baseline_threshold: float,
    version: str,
) -> dict:
    """Run the full Feature 22-26 pipeline and persist the selected model.

    Args:
        features_path: Consolidated features.csv (scripts/build_features.py output).
        models_dir: Output directory for saved model/scaler artifacts.
        min_recall: Minimum recall required for model_selection.select_model().
        min_precision: Minimum precision required for model_selection.select_model().
        n_splits: Number of subject-level GroupKFold folds.
        baseline_threshold: Decision threshold for the non-ML baseline detector.
        version: Version tag for saved artifact filenames.

    Returns:
        Dict with "evaluation_results" (DataFrame), "selected_model"
        (str or None), and "artifact_paths" (dict, only if a model was
        selected and saved).
    """
    config = load_config()
    df = load_dataset(features_path)
    labeled_df = assign_labels(df)

    X = labeled_df[FEATURE_COLUMNS].to_numpy()
    y = labeled_df["label"].astype(int).to_numpy()
    groups = labeled_df["subject_id"].to_numpy()

    baseline_metrics = evaluate_baseline(labeled_df, baseline_threshold, DEFAULT_THRESHOLD_FEATURE)
    logger.info("Baseline (non-ML) reference: %s", baseline_metrics)

    fitted_models_by_fold = []
    scalers_by_fold = []
    test_folds = []
    for fold_index, (train_idx, test_idx) in enumerate(
        get_subject_level_folds(X, y, groups, n_splits=n_splits)
    ):
        logger.info(
            "Fold %d: %d train windows / %d test windows, subjects_test=%s",
            fold_index,
            len(train_idx),
            len(test_idx),
            sorted(set(groups[test_idx])),
        )
        fitted_models, scaler = train_and_scale(X[train_idx], y[train_idx], config.ml.random_seed)
        fitted_models_by_fold.append(fitted_models)
        scalers_by_fold.append(scaler)
        test_folds.append((X[test_idx], y[test_idx]))

    fold_results = evaluate_candidates_across_folds(fitted_models_by_fold, scalers_by_fold, test_folds)
    aggregated = aggregate_fold_results(fold_results)

    models_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(models_dir / "evaluation_results_by_fold.csv", index=False)
    aggregated.to_csv(models_dir / "evaluation_results.csv", index=False)

    # Refit each candidate on the FULL dataset for the final saved artifact
    # (fold models above exist only to produce honest held-out metrics).
    final_models, final_scaler = train_and_scale(X, y, config.ml.random_seed)
    artifact_paths = save_artifacts(final_models, final_scaler, models_dir, version=version)

    sized_results = aggregated.copy()
    sized_results["artifact_size_bytes"] = sized_results["model_name"].map(
        lambda name: artifact_paths[name].stat().st_size
    )
    sized_results.to_csv(models_dir / "evaluation_results.csv", index=False)

    selected_model = select_model(sized_results, min_recall=min_recall, min_precision=min_precision)

    result = {
        "baseline_metrics": baseline_metrics,
        "evaluation_results": sized_results,
        "selected_model": selected_model,
        "artifact_paths": artifact_paths,
    }
    if selected_model is None:
        logger.warning(
            "No candidate met min_recall=%.3f / min_precision=%.3f. All fitted "
            "artifacts remain under %s for inspection; re-run model_selection "
            "with different thresholds once you decide on them (see "
            "architecture.md Open Questions).",
            min_recall,
            min_precision,
            models_dir,
        )
    else:
        logger.info("Selected model for ESP32 deployment: %s", selected_model)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Phase 4 tremor-detection models")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "features" / "features.csv")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "data" / "models")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.7,
        help="Placeholder default -- config.ml.confidence_threshold and this "
        "project's detection-requirement thresholds are still TBD "
        "(architecture.md Open Questions). Override once decided.",
    )
    parser.add_argument("--min-precision", type=float, default=0.7)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--baseline-threshold", type=float, default=0.3)
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_training(
        features_path=args.features,
        models_dir=args.models_dir,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        n_splits=args.n_splits,
        baseline_threshold=args.baseline_threshold,
        version=args.version,
    )


if __name__ == "__main__":
    main()
