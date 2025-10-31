"""
src.models.train
================

High-level *single entry point* for model training.

Why this layer?
---------------
• Keeps `scripts/train.py` dead-simple: it only needs to call
  `train_model(**cfg)` without worrying about which concrete estimator is
  used.  
• Centralises common pre-/post-processing, e.g. the log-transform that both
  Baseline and Boosting expect.  
• Makes it trivial to add new algorithms—just register a new key in
  `_REGISTRY`.

Public API
----------
train_model(
    X_train, y_train,
    X_test,  y_test,
    model="baseline",
    *,
    log_target=True,
    cfg=None,
)  → pd.DataFrame
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np
import pandas as pd

from .baseline import BaselineConfig, train_and_evaluate as _run_baseline
from .boosting import BoostingConfig, train_and_evaluate as _run_boosting

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------------------------------------- #
# Registry of back-ends                                                      #
# --------------------------------------------------------------------------- #
# Each entry maps a *model key* to (train_fn, default_cfg_factory)
_REGISTRY: Dict[str, tuple[Callable, Callable[[], object]]] = {
    "baseline": (_run_baseline, BaselineConfig),
    "xgboost": (_run_boosting, BoostingConfig),
    # add new algorithms here, e.g.
    # "lightgbm": (lightgbm.train_and_evaluate, LightGBMConfig),
}


# --------------------------------------------------------------------------- #
# Public helper                                                               #
# --------------------------------------------------------------------------- #
def train_model(  # noqa: D401
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model: str = "baseline",
    log_target: bool = True,
    cfg: object | None = None,
) -> pd.DataFrame:
    """
    Train the requested *model* and return a metrics DataFrame.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Target vectors (un-logged!).  They will be log1p-transformed if
        ``log_target=True`` because all current back-ends expect that.
    model : {"baseline", "xgboost"}, default "baseline"
        Which algorithm to run.  Extendable via `_REGISTRY`.
    log_target : bool, default True
        If True, apply ``np.log1p`` to y before passing it to the back-end and
        later exponentiate predictions for metric calculation.
    cfg : dataclass instance | None
        Custom configuration object (e.g. ``BaselineConfig``).  If ``None``
        a default one is instantiated.

    Returns
    -------
    pd.DataFrame
        Metrics table produced by the selected back-end.
    """
    if model not in _REGISTRY:
        raise ValueError(
            f"Unknown model {model!r}. Available: {', '.join(_REGISTRY)}"
        )

    train_fn, cfg_factory = _REGISTRY[model]
    cfg = cfg or cfg_factory()

    logger.info("Dispatching training to %s back-end …", model)

    if log_target:
        y_train_log = np.log1p(y_train)
    else:
        y_train_log = y_train

    metrics_df = train_fn(
        X_train=X_train,
        y_train_log=y_train_log,
        X_test=X_test,
        y_test=y_test,
        cfg=cfg,
    )

    return metrics_df
