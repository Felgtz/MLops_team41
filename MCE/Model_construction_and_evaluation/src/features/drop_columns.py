"""
src.features.drop_columns
=========================

Utility transformer for removing one or more columns inside an
scikit-learn Pipeline.

Example
-------
from sklearn.pipeline import make_pipeline
from src.features.drop_columns import DropColumns

pipe = make_pipeline(
    DropColumns(["id", "free_text"]),      # eliminate leak / high-cardinality
    StandardScaler(),
    LinearRegression(),
).fit(X_train, y_train)

Why a custom transformer?
-------------------------
• Lets you keep preprocessing logic in a single Pipeline object that can be
  cross-validated and exported as a `.pkl`.  
• Avoids editing the DataFrame in-place, so the original data remains intact.  
• Compatible with `sklearn.compose.ColumnTransformer` (it inherits
  `BaseEstimator, TransformerMixin`).

Public API
----------
DropColumns(columns, errors="ignore")
"""

from __future__ import annotations

import logging
from typing import Hashable, Iterable, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from a pandas DataFrame.

    Parameters
    ----------
    columns : Iterable[Hashable]
        Column names (or other indexables) to remove.
    errors : {"ignore", "raise"}, default "ignore"
        • "ignore" – columns not found are silently skipped.  
        • "raise"  – a KeyError is raised if any column is missing.
    copy : bool, default True
        If True, return a new DataFrame; if False, operate on the
        original object (mutates `X`).

    Notes
    -----
    • The transformer *does not* touch the target `y`—just pass it through.
    • Works on any DataFrame-like object that supports `.drop()`.
    """

    def __init__(
        self,
        columns: Iterable[Hashable],
        *,
        errors: str = "ignore",
        copy: bool = True,
    ) -> None:
        self.columns: List[Hashable] = list(columns)
        self.errors = errors
        self.copy = copy

    # --------------------------------------------------------------------- #
    # Fit / transform                                                       #
    # --------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401, N803
        """No-op; kept for API compatibility."""
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        self._validate_columns(X)
        logger.debug("Dropping columns: %s", self.columns)
        return X.drop(columns=self.columns, errors=self.errors, inplace=not self.copy)

    # --------------------------------------------------------------------- #
    # Helper                                                                #
    # --------------------------------------------------------------------- #
    def _validate_columns(self, X: pd.DataFrame) -> None:
        if self.errors == "raise":
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                raise KeyError(f"Columns not found in input DataFrame: {missing}")
