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

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from math import sqrt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import joblib
from pathlib import Path
from math import sqrt

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
    # ── generic ────────────────────────────────────────────────
    task: str = "regression"           # "regression" | "classification"
    random_state: int | None = 42
    artifacts_dir: Path = Path("artifacts")
    cv_folds: int = 5                  # <- keeps the earlier field

    # ── model selector (empty list ⇒ build everything available) ─
    models: List[str] = field(default_factory=list)

    # ── XGBoost toggles & hyper-parameters ─────────────────────
    xgb_enabled: bool   = True
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int  = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0
    xgb_n_jobs: int = -1

    # helper: True  ⇢  build model “name”; False  ⇢  skip
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
            n_estimators   = cfg.xgb_n_estimators,
            learning_rate  = cfg.xgb_learning_rate,
            max_depth      = cfg.xgb_max_depth,
            subsample      = cfg.xgb_subsample,
            colsample_bytree = cfg.xgb_colsample_bytree,
            reg_lambda     = cfg.xgb_reg_lambda,
            n_jobs         = cfg.xgb_n_jobs,
            random_state   = cfg.random_state,
        )
        if cfg.task == "classification" and cfg._want("xgboost"):
            models["xgboost"] = XGBClassifier(
                n_estimators   = cfg.xgb_n_estimators,
                learning_rate  = cfg.xgb_learning_rate,
                max_depth      = cfg.xgb_max_depth,
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
# Train & evaluate helper for boosting models
# ---------------------------------------------------------------------------
def train_and_evaluate(
    cfg: BoostingConfig,
    X_train: pd.DataFrame | np.ndarray,
    X_test:  pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test:  pd.Series | np.ndarray,
    *,
    save_dir: str | Path = "artifacts/models",
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """
    Fit each boosting estimator, compute metrics and save the artefacts.

    Returns
    -------
    leaderboard : pd.DataFrame
    trained     : dict{name -> fitted model}
    """
    # 0. normalise path BEFORE first use
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models     = get_models(cfg)          # whatever factory you already have
    rows       = []
    trained    = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(
            {
                "model":  name,
                "family": "boosting",
                "RMSE":   sqrt(mean_squared_error(y_test, y_pred)),
                "MAE":    mean_absolute_error(y_test, y_pred),
                "R2":     r2_score(y_test, y_pred),
            }
        )

        out_path = save_dir / f"{name}.pkl"
        joblib.dump(model, out_path)
        trained[name] = model

    leaderboard = pd.DataFrame(rows).sort_values("RMSE")
    return leaderboard, trained

# ---------------------------------------------------------------------------
# public exports
# ---------------------------------------------------------------------------
__all__: Tuple[str, ...] = ("BoostingConfig", "get_models", "train_and_evaluate")
