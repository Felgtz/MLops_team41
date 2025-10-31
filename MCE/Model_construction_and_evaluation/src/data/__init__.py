"""
src.data
========

Light-weight helpers for reading raw files and turning them into the
train/test splits consumed by the rest of the pipeline.

Quick start
-----------
>>> from src.data import load_csv, train_test_split_df
>>> df = load_csv("data/raw/housing.csv")
>>> X_tr, X_te, y_tr, y_te = train_test_split_df(
...     df, target_col="price", test_size=0.2, random_state=42
... )

Submodules
----------
load_and_split
    • load_csv(path, **read_csv_kwargs)
    • train_test_split_df(df, target_col, test_size=0.2, ...)
"""

from __future__ import annotations

import logging


from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Re-exports so callers can do `from src.data import load_csv`                #
# --------------------------------------------------------------------------- #
from .load_and_split import load_csv, train_test_split_df  # noqa: F401

__all__: list[str] = ["load_csv", "train_test_split_df"]

# --------------------------------------------------------------------------- #
# Package-level logger (kept silent unless configured by app)                 #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ------------------------------------------------------------------
# 1. The configuration object expected by training_pipeline.py
# ------------------------------------------------------------------
@dataclass
class DataConfig:
    path: str | Path
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = False


# ------------------------------------------------------------------
# 2. Helper that loads a CSV and returns split (X_train, X_test, y_train, y_test)
# ------------------------------------------------------------------
def load_and_split(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame,
                                             pd.Series,   pd.Series]:
    df = pd.read_csv(cfg.path)
    y = df.pop(cfg.target_column)
    stratify = y if cfg.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test


# Re-export so `from src.data import …` works
__all__ = ["DataConfig", "load_and_split"]