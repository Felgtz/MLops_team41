"""
src.data.load_and_split
=======================

Tiny façade around pandas + scikit-learn’s ``train_test_split`` so every part
of the codebase handles tabular data the same way.

Typical usage
-------------
from src.data.load_and_split import load_csv, train_test_split_df

df = load_csv("data/raw/housing.csv")
X_tr, X_te, y_tr, y_te = train_test_split_df(
    df, target_col="price", test_size=0.2, random_state=42
)

Why not something bigger?
-------------------------
Most projects start with a single CSV.  This module stays minimal on purpose
and can be swapped out later for a feature-store, SQL reader, or Dask loader
without changing the rest of the pipeline API.

Public API
----------
load_csv(path, **read_csv_kwargs)           → pd.DataFrame
train_test_split_df(
    df, target_col, *, test_size=0.2, random_state=0, stratify=None
)                                           → X_train, X_test, y_train, y_test
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# 1. IO helper                                                                #
# --------------------------------------------------------------------------- #
def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the *.csv* file.
    **kwargs
        Extra keyword args forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    logger.info("Loading dataset from %s", path.resolve())
    df = pd.read_csv(path, **kwargs)
    logger.debug("Loaded %d rows × %d cols", *df.shape)
    return df


# --------------------------------------------------------------------------- #
# 2. Train / test split helper                                               #
# --------------------------------------------------------------------------- #
def train_test_split_df(
    df: pd.DataFrame,
    target_col: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = 0,
    stratify: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a tabular DataFrame into X/y, train/test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Whole dataset including the target column.
    target_col : str
        Name of the target column to predict.
    test_size : float, default 0.2
        Fraction of rows to allocate to the test set.
    random_state : int | None, default 0
        RNG seed for reproducibility.  Use ``None`` for non-deterministic.
    stratify : pd.Series | None, default None
        Optional stratification column/Series.  For regression problems you
        normally leave this ``None``; for classification you can pass
        ``df[target_col]`` to preserve class balance.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(
        "Split data: %d→%d train rows, %d test rows", len(df), len(X_tr), len(X_te)
    )
    return X_tr, X_te, y_tr, y_te
