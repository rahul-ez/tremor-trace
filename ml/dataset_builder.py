"""Feature-table assembly, label assignment, and subject-level splitting.

Owns Feature 22 of the build plan: turning per-window feature vectors
(produced by ``signal_processing/feature_extraction.py`` and consolidated by
``scripts/build_features.py``) into a labeled ``pandas.DataFrame`` plus
subject-level train/test index folds for model training and evaluation.
"""

import logging
from pathlib import Path
from typing import Callable, Generator, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

# Must match the documented Feature vector schema in architecture.md ->
# Interfaces & Data Contracts exactly.
REQUIRED_FEATURE_COLUMNS = [
    "subject_id",
    "session_id",
    "window_id",
    "tremor_band_power",
    "total_power",
    "power_ratio",
    "dominant_frequency_hz",
    "rms_amplitude",
    "variance",
    "spectral_entropy",
    "accel_magnitude",
    "gyro_magnitude",
]

# Numeric columns only, in the fixed order consumed by ml/train.py,
# ml/evaluate.py, and ml/inference.py.
FEATURE_COLUMNS = [
    "tremor_band_power",
    "total_power",
    "power_ratio",
    "dominant_frequency_hz",
    "rms_amplitude",
    "variance",
    "spectral_entropy",
    "accel_magnitude",
    "gyro_magnitude",
]


def build_feature_table(feature_records: Iterable[dict]) -> pd.DataFrame:
    """Assemble a feature table from per-window feature dicts.

    Args:
        feature_records: Iterable of dicts, each containing subject_id,
            session_id, window_id, and the documented feature columns
            (see architecture.md -> Interfaces & Data Contracts).

    Returns:
        DataFrame with subject_id/session_id as string dtype (never coerced
        to numeric, to keep GroupKFold grouping correct).

    Raises:
        ValueError: If any required column is missing.
    """
    df = pd.DataFrame.from_records(list(feature_records))
    if df.empty:
        raise ValueError("feature_records is empty; nothing to build a table from")

    missing = set(REQUIRED_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["session_id"] = df["session_id"].astype(str)
    return df[REQUIRED_FEATURE_COLUMNS]


def load_dataset(features_path: Path) -> pd.DataFrame:
    """Load a consolidated features.csv, preserving subject_id/session_id for grouping.

    Args:
        features_path: Path to a features.csv produced by
            ``scripts/build_features.py`` (or ``build_feature_table``).

    Returns:
        DataFrame with the documented feature columns.

    Raises:
        FileNotFoundError: If features_path does not exist.
        ValueError: If required columns are missing.
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}")

    df = pd.read_csv(features_path, dtype={"subject_id": str, "session_id": str})
    missing = set(REQUIRED_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{features_path} is missing required columns: {missing}")
    return df


def default_session_label(session_id: str) -> bool:
    """Confirmed label convention: infer tremor/no-tremor from session_id suffix.

    Each subject recorded exactly two sessions: a stationary "hand resting
    in the glove" session ("sess01") and a deliberate tremor-like motion
    session ("sess02"). Confirmed with the project owner:

        session_id ending in "01" -> label=False (no tremor / stationary rest)
        session_id ending in "02" -> label=True  (tremor-like motion)

    This is also the source recording used for per-subject sensor
    calibration in scripts/build_features.py::estimate_subject_offsets(),
    since "sess01" is the only genuinely stationary recording per subject.

    Args:
        session_id: Session identifier, e.g. "sess01".

    Returns:
        True if the session should be labeled tremor-positive.

    Raises:
        ValueError: If session_id does not match the expected convention.
    """
    if session_id.endswith("01"):
        return False
    if session_id.endswith("02"):
        return True
    raise ValueError(
        f"Cannot infer a tremor/no-tremor label for session_id={session_id!r} "
        "under the default '01'=rest / '02'=tremor convention. Pass an "
        "explicit label_fn to assign_labels() if your protocol differs."
    )


def assign_labels(
    df: pd.DataFrame,
    label_fn: Optional[Callable[[str], bool]] = None,
) -> pd.DataFrame:
    """Tag each row with a tremor/no-tremor label at the session level.

    Args:
        df: Feature table from build_feature_table()/load_dataset().
        label_fn: Maps session_id -> bool label. Defaults to
            default_session_label(); pass a custom function (e.g. reading
            a labels.csv) once the real recording protocol is confirmed.

    Returns:
        Copy of df with an added boolean "label" column.
    """
    resolved_label_fn = label_fn or default_session_label
    labeled = df.copy()
    labeled["label"] = labeled["session_id"].map(resolved_label_fn).astype(bool)
    return labeled


def get_subject_level_folds(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield subject-level train/test index pairs -- never window-level.

    Args:
        X: Feature array, shape (n_windows, n_features).
        y: Labels, shape (n_windows,).
        groups: subject_id per window, shape (n_windows,).
        n_splits: Number of folds; must not exceed the number of distinct
            subjects in groups.

    Yields:
        (train_idx, test_idx) index arrays into X/y, grouped so that no
        subject_id appears in both the train and test set of any fold.
    """
    n_groups = len(set(groups))
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of distinct subjects "
            f"({n_groups}); reduce n_splits or add more subjects."
        )
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(X, y, groups=groups)
