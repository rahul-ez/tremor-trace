"""Fit LogisticRegression/SVM/RandomForest candidates with proper scaling.

Feature 24 of the build plan.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tremor_system.config import load_config

logger = logging.getLogger(__name__)


def get_candidate_models(random_seed: int) -> dict[str, ClassifierMixin]:
    """Return the three fresh (unfitted) candidate classifiers.

    Args:
        random_seed: Seed applied to every stochastic estimator.

    Returns:
        Dict of model_name -> unfitted sklearn estimator. SVM is wrapped in
        CalibratedClassifierCV so ml/inference.py can read predict_proba()
        -- SVC(probability=True) is deprecated as of scikit-learn 1.9 and
        scheduled for removal in 1.11 (see
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
    """
    return {
        "logistic_regression": LogisticRegression(random_state=random_seed, max_iter=1000),
        "svm": CalibratedClassifierCV(
            SVC(random_state=random_seed), method="sigmoid", ensemble=False
        ),
        "random_forest": RandomForestClassifier(random_state=random_seed, n_estimators=100),
    }


def train_and_scale(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: Optional[int] = None,
) -> tuple[dict[str, ClassifierMixin], StandardScaler]:
    """Fit a StandardScaler on training data only, then fit all candidates.

    Args:
        X_train: shape (n_windows, n_features), training feature matrix.
        y_train: shape (n_windows,), training labels.
        random_seed: Overrides config.ml.random_seed if provided.

    Returns:
        (fitted_models, scaler) -- fitted_models maps model_name to a
        fitted classifier; scaler is fit on X_train only (never re-fit
        at inference time, per code-standards.md -> Machine Learning).
    """
    seed = load_config().ml.random_seed if random_seed is None else random_seed

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    candidates = get_candidate_models(seed)
    fitted_models = {
        name: model.fit(X_train_scaled, y_train) for name, model in candidates.items()
    }
    logger.info("Fitted %d candidate models: %s", len(fitted_models), list(fitted_models))
    return fitted_models, scaler


def save_artifacts(
    fitted_models: dict[str, ClassifierMixin],
    scaler: StandardScaler,
    output_dir: Path,
    version: str = "v1",
) -> dict[str, Path]:
    """Persist fitted models and their paired scaler with versioned filenames.

    Follows code-standards.md -> File and Folder Naming:
    model_<algorithm>_<version>.pkl / scaler_<version>.pkl. Never silently
    overwrites -- raises if any target path already exists.

    Args:
        fitted_models: Output of train_and_scale().
        scaler: Output of train_and_scale().
        output_dir: Directory to write artifacts into (e.g. data/models/).
        version: Version tag shared by the model and scaler filenames.

    Returns:
        Dict mapping "scaler" and each model name to its saved Path.

    Raises:
        FileExistsError: If a target artifact path already exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    scaler_path = output_dir / f"scaler_{version}.pkl"
    if scaler_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {scaler_path}")
    joblib.dump(scaler, scaler_path)
    paths["scaler"] = scaler_path

    for name, model in fitted_models.items():
        model_path = output_dir / f"model_{name}_{version}.pkl"
        if model_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {model_path}")
        joblib.dump(model, model_path)
        paths[name] = model_path
        logger.info("Saved %s -> %s (%d bytes)", name, model_path, model_path.stat().st_size)

    return paths