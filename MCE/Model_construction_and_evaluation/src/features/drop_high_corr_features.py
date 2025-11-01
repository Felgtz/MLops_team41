"""
src.features.drop_high_corr_features
====================================

Transformer that identifies columns whose pair-wise correlation exceeds a
user-supplied threshold and drops the second (j-th) column in each offending
pair.  Fits neatly into any scikit-learn Pipeline.

Example
-------
from sklearn.pipeline import make_pipeline
from src.features.drop_high_corr_features import DropHighCorrFeatures

pipe = make_pipeline(
    DropHighCorrFeatures(threshold=0.92, method="spearman"),
    StandardScaler(),
    Ridge()
).fit(X_train, y_train)

Public API
----------
DropHighCorrFeatures(
    threshold=0.9,
    method="pearson",
    absolute=True,
    copy=True,
)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DropHighCorrFeatures(BaseEstimator, TransformerMixin):
    """
    Remove columns whose correlation with *another* column exceeds `threshold`.

    Parameters
    ----------
    threshold : float, default 0.9
        Absolute correlation above which a column will be considered redundant
        w.r.t. an earlier column in the matrix upper triangle.
    method : {"pearson", "spearman", "kendall"}, default "pearson"
        Correlation coefficient to compute (delegated to `DataFrame.corr`).
    absolute : bool, default True
        If True, use ``abs(corr)``; if False, preserve sign (rarely useful).
    copy : bool, default True
        Return a copy of `X`; set False to drop columns in-place (memory saver).

    Attributes
    ----------
    to_drop_ : list[str]
        Column names flagged for removal during `fit`.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        *,
        method: str = "pearson",
        absolute: bool = True,
        copy: bool = True,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("`threshold` must be in (0, 1].")
        self.threshold = threshold
        self.method = method
        self.absolute = absolute
        self.copy = copy
        # populated at fit-time
        self.to_drop_: List[str] = []

    # ------------------------------------------------------------------ #
    # scikit-learn API                                                   #
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401, N803
        """Identify columns to drop; target `y` is ignored."""
        self._check_is_dataframe(X)
        corr = X.corr(method=self.method)
        if self.absolute:
            corr = corr.abs()

        upper = np.triu_indices_from(corr, k=1)  # indices of upper triangle
        to_drop: set[str] = set()
        for i, j in zip(*upper):
            if corr.iat[i, j] > self.threshold:
                col_j = corr.columns[j]
                to_drop.add(col_j)

        self.to_drop_ = sorted(to_drop)
        logger.info(
            "DropHighCorrFeatures: identified %d columns to drop (threshold=%.2f)",
            len(self.to_drop_),
            self.threshold,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        self._check_is_dataframe(X)
        missing = [c for c in self.to_drop_ if c not in X.columns]
        if missing:
            # Happens if downstream pipeline already removed some cols
            logger.warning("Columns %s not present in transform input; skipping.", missing)
        return X.drop(columns=[c for c in self.to_drop_ if c in X.columns],
                      inplace=not self.copy)

    # ------------------------------------------------------------------ #
    # Helper / utils                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_is_dataframe(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "DropHighCorrFeatures expects a pandas DataFrame as input; "
                f"got {type(X).__name__}"
            )

    # Optional: lets users interface with sklearn.feature_selection utilities
    def get_support(self, indices: bool = False):
        """Return a boolean mask or index of the features kept."""
        mask = [col not in self.to_drop_ for col in self._feature_names_in_]
        return np.where(mask)[0] if indices else np.array(mask)

    # scikit-learn 1.2+ compatibility
    def _more_tags(self):
        return {"preserves_dtype": True}
