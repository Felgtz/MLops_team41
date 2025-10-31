"""
src.models.predict
==================

Utilities for loading a trained estimator from *artifacts/models/* and
running batch inference.

Why a separate module?
----------------------
• Keeps `scripts/predict.py` extremely thin—just CLI parsing plus a call to
  `run_prediction()`.  
• Hides path-handling & log-space corrections behind a single function.  
• Works for every algorithm that stores a ``.pkl`` in
  *artifacts/models/* (baseline, xgboost, …).

Typical usage
-------------
from src.models.predict import run_prediction

preds = run_prediction(
    X,                      # pd.DataFrame of features
    model_key="xgboost",    # matches the filename <model_key>.pkl
    artifacts_dir="artifacts",
)
preds.to_csv("predictions.csv", index=False)

Public API
----------
load_model(model_path)                  → estimator
run_prediction(X, model_key, …)         → pd.Series
list_available_models(artifacts_dir=".")→ list[str]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------------------------------------- #
# 1. Helpers                                                                  #
# --------------------------------------------------------------------------- #


def load_model(model_path: str | Path):
    """
    Load a fitted estimator saved by `joblib.dump`.

    Parameters
    ----------
    model_path : str | Path
        Path to a ``.pkl`` model artefact.

    Returns
    -------
    Any scikit-learn compatible estimator.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def list_available_models(artifacts_dir: str | Path = "artifacts") -> List[str]:
    """
    Return the list of ``<name>.pkl`` files under *artifacts/models/* without
    the ``.pkl`` suffix.

    Useful for auto-completion in CLI tools.
    """
    models_dir = Path(artifacts_dir) / "models"
    return sorted(p.stem for p in models_dir.glob("*.pkl"))


# --------------------------------------------------------------------------- #
# 2. Main inference routine                                                   #
# --------------------------------------------------------------------------- #


def run_prediction(
    X: pd.DataFrame,
    *,
    model_key: str,
    artifacts_dir: str | Path = "artifacts",
    log_target: bool = True,
    return_numpy: bool = False,
):
    """
    Make predictions with a stored estimator.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    model_key : str
        Filename stem of the pickled model, e.g. ``"baseline_linear_regression"``
        if the artefact is *artifacts/models/baseline_linear_regression.pkl*.
    artifacts_dir : str | Path, default "artifacts"
        Root directory that contains *models/*.
    log_target : bool, default True
        If True, assumes the estimator was trained on log-space targets and
        applies ``np.expm1`` to bring predictions back to the original scale.
    return_numpy : bool, default False
        If True, return a 1-D ``np.ndarray``; otherwise a ``pd.Series``.

    Returns
    -------
    pd.Series | np.ndarray
        Predictions in the original target scale.
    """
    model_path = Path(artifacts_dir) / "models" / f"{model_key}.pkl"
    model = load_model(model_path)

    logger.info("Running inference on %d rows …", len(X))
    preds = model.predict(X)

    if log_target:
        preds = np.expm1(preds)

    if return_numpy:
        return preds
    return pd.Series(preds, name="prediction")
