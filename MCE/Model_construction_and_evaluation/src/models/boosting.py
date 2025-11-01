"""
src.models.boosting
===================

Gradient-boosting based models for tabular data.

Public API
----------

• BoostingConfig           – configuration object
• get_models(cfg)          – returns dict {name: sklearn-compatible estimator}
• train_and_evaluate(cfg,
                     X_tr, X_te,
                     y_tr, y_te)    – quick benchmarking helper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------------------
# 1.  Which back-end do we have?  Prefer xgboost, else fall back to sklearn.
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    _HAVE_XGB = True
except ModuleNotFoundError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    _HAVE_XGB = False


# ---------------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------------
@dataclass
class BoostingConfig:
    """
    Hyper-parameters shared by the boosting models.
    """

    task: str = "regression"                 # "classification" or "regression"
    random_state: int = 42
    models: List[str] = field(default_factory=list)

    # xgboost-specific knobs (ignored by sklearn fallback)
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6

    def _want(self, name: str) -> bool:
        return not self.models or name in self.models


# ---------------------------------------------------------------------------
# get_models
# ---------------------------------------------------------------------------
def get_models(cfg: BoostingConfig) -> Dict[str, BaseEstimator]:
    """
    Build a dictionary of gradient-boosting estimators.

    • If XGBoost is installed and requested, we build those.
    • Otherwise we fall back to scikit-learn’s GradientBoosting* models.
    """
    if cfg.task not in {"regression", "classification"}:
        raise ValueError("cfg.task must be 'regression' or 'classification'")

    models: Dict[str, BaseEstimator] = {}

    if _HAVE_XGB:
        # ------------------------------------------------------
        # XGBoost back-end
        # ------------------------------------------------------
        if cfg.task == "regression" and cfg._want("xgboost"):
            models["xgboost"] = XGBRegressor(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
        if cfg.task == "classification" and cfg._want("xgboost"):
            models["xgboost"] = XGBClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                random_state=cfg.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss",
            )
    else:
        # ------------------------------------------------------
        # scikit-learn fallback
        # ------------------------------------------------------
        if cfg.task == "regression" and cfg._want("gb"):
            models["gb"] = GradientBoostingRegressor(random_state=cfg.random_state)
        if cfg.task == "classification" and cfg._want("gb"):
            models["gb"] = GradientBoostingClassifier(random_state=cfg.random_state)

    if not models:
        raise ValueError(
            "No boosting models were created. "
            "Check cfg.task / cfg.models or install the xgboost extra."
        )

    # Wrap each model in a standardisation pipeline for good measure
    for name, est in list(models.items()):
        models[name] = make_pipeline(StandardScaler(), est)

    return models


# ---------------------------------------------------------------------------
# train & evaluate helper
# ---------------------------------------------------------------------------
def train_and_evaluate(
    cfg: BoostingConfig,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> Dict[str, float]:
    """
    Fit each boosting model and compute a single metric.

    • Regression → R²  
    • Classification → Accuracy
    """
    models = get_models(cfg)
    scores: Dict[str, float] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if cfg.task == "regression":
            scores[name] = r2_score(y_test, y_pred)
        else:
            scores[name] = accuracy_score(y_test, y_pred)

    return scores


# ---------------------------------------------------------------------------
# public exports
# ---------------------------------------------------------------------------
__all__: Tuple[str, ...] = ("BoostingConfig", "get_models", "train_and_evaluate")
