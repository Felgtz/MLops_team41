"""
src/utils/feature_selection.py
Utility helpers for pruning features.

Right now it provides a single public function:

    drop_high_corr_features(df, thresh=0.95)

which removes columns that are highly correlated with another column
(absolute Pearson correlation > thresh).  The heuristic keeps the first
column of each correlated pair and drops the rest.

You can add more feature-selection helpers here later on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def _high_corr_columns(
    df: pd.DataFrame,
    thresh: float = 0.95,
) -> Iterable[str]:
    """
    Internal helper that returns a list of column names that should be dropped
    because they have absolute Pearson correlation > *thresh* with a *previous*
    column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input features (numeric and/or categorical already encoded).
    thresh : float, default 0.95
        Correlation coefficient above which two columns are considered redundant.

    Returns
    -------
    list[str]
        Columns to drop.
    """
    if not 0.0 < thresh < 1.0:
        raise ValueError("thresh must be between 0 and 1 (exclusive)")

    # Compute absolute correlation for numeric columns only
    corr = df.corr(numeric_only=True).abs()

    # Keep upper triangle of the correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Any column with a correlation higher than thresh is marked for dropping
    to_drop = [col for col in upper.columns if (upper[col] > thresh).any()]
    return to_drop


def drop_high_corr_features(
    df: pd.DataFrame,
    thresh: float = 0.95,
) -> pd.DataFrame:
    """
    Return a copy of *df* with highly correlated features removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature matrix.
    thresh : float, default 0.95
        Absolute Pearson correlation threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame with redundant columns dropped.

    Example
    -------
    >>> cleaned = drop_high_corr_features(X, thresh=0.9)
    """
    to_drop = _high_corr_columns(df, thresh=thresh)
    return df.drop(columns=to_drop, errors="ignore")
