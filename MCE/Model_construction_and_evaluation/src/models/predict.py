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
    X: pd.DataFrame | None = None,          # old name
    *,                                      # keyword-only below
    df: pd.DataFrame | None = None,         # new alias
    model_key: str | None = None,
    model_path: str | Path | None = None,   # new alias
    artifacts_dir: str | Path = "artifacts",
    feature_list: list[str] | None = None,  # new
    transform: str = "log1p",               # "log1p" or "none"
    return_numpy: bool = False,
):
    """
    Make predictions with a stored estimator.

    You may supply either
      • model_key  + artifacts_dir
      • model_path (full path)
    and either
      • X          (old name)
      • df         (alias)
    """
    # ------------------------------------------------------------------
    # 0. Resolve aliases                                                |
    # ------------------------------------------------------------------
    if X is None and df is None:
        raise ValueError("Provide X= or df=")
    if X is not None and df is not None:
        raise ValueError("Provide only one of X=  or df=, not both")
    X = X if X is not None else df

   # ------------------------------------------------------------
    # figure out which .pkl file we must open
    # ------------------------------------------------------------
    if model_path is None and model_key is None:
        raise ValueError("Provide either model_path or model_key")

    if model_path is None:                      # user gave a key
        model_path = Path(artifacts_dir) / "models" / f"{model_key}.pkl"
    else:                                       # user gave a full/relative path
        model_path = Path(model_path)           # ← DO NOT prepend/append anything

    # reorder / subset columns if a list is supplied
    if feature_list is not None:
        X = X[feature_list]

    # ------------------------------------------------------------------
    # 1. Load model and predict                                         |
    # ------------------------------------------------------------------
    model = load_model(model_path)
    preds = model.predict(X)

    # inverse transform if necessary
    if transform == "log1p":
        preds = np.expm1(preds)
    elif transform != "none":
        raise ValueError("transform must be 'log1p' or 'none'")

    return preds if return_numpy else pd.Series(preds, name="prediction")

