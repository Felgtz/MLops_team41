# src/models/evaluate.py
"""
Common regression-metric helpers.

Design goals
------------
1. Keep the public surface tiny – one function covers the usual trio
   (MAE, RMSE, R²) so all training scripts stay consistent.
2. Assume the model was trained on log-transformed targets (log1p) and
   therefore its raw predictions must be inverse-transformed with
   numpy.expm1 before scoring. If you train directly on the original
   scale, simply pass predictions unchanged.
3. No pandas dependency – pure NumPy + scikit-learn for speed.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def eval_regression(
    *,
    y_true: np.ndarray,
    y_pred_log: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute MAE, RMSE, and R² **on the original target scale**.

    Parameters
    ----------
    y_true : 1-D array-like
        Ground-truth target values on the original scale.
    y_pred_log : 1-D array-like
        Model predictions on the *log1p* scale  
        (i.e. `np.log1p(original_target)`).

    Returns
    -------
    tuple
        (mae, rmse, r2) – floats rounded only by the caller.
    """
    # Inverse log1p → original scale
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    logger.debug(
        "Eval metrics – MAE: %.3f | RMSE: %.3f | R2: %.4f",
        mae,
        rmse,
        r2,
    )
    return mae, rmse, r2
