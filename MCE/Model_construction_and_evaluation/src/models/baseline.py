"""
src.models.baseline
===================

Lightweight baseline models for tabular data.

Public API
----------

• BaselineConfig           – dataclass holding generic hyper-params  
• get_models(cfg)          – build a dict {name: sklearn-estimator}  
• train_and_evaluate(cfg,
                     X_tr, X_te,
                     y_tr, y_te)    – fit each model and report a score

Nothing in here is pipeline-specific; the higher-level code decides which
metrics to log, where to persist models, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from math import sqrt

# ---------------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    """
    Generic knobs for baseline models.

    Attributes
    ----------
    task : {"regression", "classification"}
        Determines which family of models / metrics to use.
    random_state : int
        Controls the RNG for models that support it.
    models : list[str]
        Subset of models to build; valid names depend on `task`.
        If empty, build all defaults.
    """

    task: str = "regression"
    random_state: int = 42
    models: List[str] = field(default_factory=list)

    # internal helper
    def _want(self, name: str) -> bool:
        return not self.models or name in self.models


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def get_models(cfg: BaselineConfig) -> Dict[str, BaseEstimator]:
    """
    Build a dictionary of scikit-learn estimators according to the config.

    Parameters
    ----------
    cfg : BaselineConfig

    Returns
    -------
    dict
        Mapping ``name → unfitted sklearn estimator``.
    """
    if cfg.task not in {"regression", "classification"}:
        raise ValueError("cfg.task must be 'regression' or 'classification'")

    models: Dict[str, BaseEstimator] = {}

    if cfg.task == "regression":
        if cfg._want("dummy"):
            models["dummy"] = DummyRegressor(strategy="mean")
        if cfg._want("linear"):
            # combine standardisation + linear regression
            models["linear"] = make_pipeline(StandardScaler(), LinearRegression())
        if cfg._want("random_forest"):
            models["random_forest"] = RandomForestRegressor(
                n_estimators=200, random_state=cfg.random_state
            )

    else:  # classification
        if cfg._want("dummy"):
            models["dummy"] = DummyClassifier(strategy="most_frequent")
        if cfg._want("logistic"):
            models["logistic"] = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=cfg.random_state)
            )
        if cfg._want("random_forest"):
            models["random_forest"] = RandomForestClassifier(
                n_estimators=200, random_state=cfg.random_state
            )

    if not models:
        raise ValueError("No models were selected/built — check cfg.models list.")

    return models


# ---------------------------------------------------------------------------
# Train & evaluate helper
# ---------------------------------------------------------------------------
def train_and_evaluate(
    cfg: BaselineConfig,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    *,
    save_dir: str | Path = "artifacts/models",
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """
    Fit every baseline model, compute the usual regression metrics and
    SAVE each fitted estimator to <save_dir>/<model_name>.pkl.

    Returns
    -------
    leaderboard : pd.DataFrame
        One row per model with RMSE, MAE, R2.
    trained     : dict[str, estimator]
        The fitted objects so the pipeline can decide which one is “best”.
    """
    # ------------------------------------------------------------
    # 0. Path handling FIRST so we can use it safely everywhere
    # ------------------------------------------------------------
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models     = get_models(cfg)
    rows       = []
    trained    = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # leaderboard row ---------------------------------------
        rows.append(
            {
                "model":  name,
                "family": "baseline",
                "RMSE":   sqrt(mean_squared_error(y_test, y_pred)),
                "MAE":    mean_absolute_error(y_test, y_pred),
                "R2":     r2_score(y_test, y_pred),
            }
        )

        # save artefact -----------------------------------------
        out_path = save_dir / f"{name}.pkl"
        joblib.dump(model, out_path)
        trained[name] = model

    leaderboard = pd.DataFrame(rows).sort_values("RMSE")
    return leaderboard, trained


# ---------------------------------------------------------------------------
# What we export when users do `from src.models.baseline import *`
# ---------------------------------------------------------------------------
__all__: Tuple[str, ...] = ("BaselineConfig", "get_models", "train_and_evaluate")
