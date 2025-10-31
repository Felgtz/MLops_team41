"""
src.features
============

Reusable, scikit-learn–compatible feature-engineering transformers.

Quick start
-----------
>>> from src.features import DropColumns, DropHighCorrFeatures
>>> from sklearn.pipeline import make_pipeline
>>> pipe = make_pipeline(
...     DropColumns(["id"]),
...     DropHighCorrFeatures(threshold=0.95),
...     model,
... ).fit(X_train, y_train)

Submodules
----------
drop_columns
    • DropColumns – delete explicit columns.

drop_high_corr_features
    • DropHighCorrFeatures – remove features highly correlated with another.
"""

from __future__ import annotations

import logging

# --------------------------------------------------------------------------- #
# Re-exports                                                                  #
# --------------------------------------------------------------------------- #
from .drop_columns import DropColumns
from .drop_high_corr_features import DropHighCorrFeatures

__all__: list[str] = [
    "DropColumns",
    "DropHighCorrFeatures",
]

# --------------------------------------------------------------------------- #
# Package-level logger (silent by default)                                    #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
