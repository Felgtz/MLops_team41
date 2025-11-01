"""
online_news_popularity
======================

Core utilities for cleaning and calibrating the “Online News Popularity”
dataset in an MLOps-friendly, testable way.

Typical use
-----------
>>> from online_news_popularity import convert_all_numeric, enforce_logical_bounds
>>> df_num, _ = convert_all_numeric(df_raw, except_cols=["url"])
>>> df_clean, report = enforce_logical_bounds(df_num, LDA_COLS, BIN_COLS, SPECIAL_KW_NEG)
"""

from __future__ import annotations

import logging
from importlib import metadata as _metadata

# ----------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------
try:
    __version__: str = _metadata.version("online_news_popularity")
except _metadata.PackageNotFoundError:  # local “editable” install
    __version__ = "0.0.0"

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
# Consumers decide if/when to configure handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ----------------------------------------------------------------------
# Public API re-exports
# ----------------------------------------------------------------------
from .features.groups import (                       # noqa: F401
    LDA_COLS,
    CHANNEL_COLS,
    WEEKDAY_COLS,
    BIN_COLS,
    KW_COLS,
    SPECIAL_KW_NEG,
)
from .cleaning.type_coercion import convert_all_numeric            # noqa: F401
from .cleaning.enforcement import enforce_logical_bounds           # noqa: F401
from .cleaning.imputation import impute                            # noqa: F401
from .cleaning.calibration import calibrate_to_reference           # noqa: F401

__all__ = [
    "__version__",
    # constants
    "LDA_COLS",
    "CHANNEL_COLS",
    "WEEKDAY_COLS",
    "BIN_COLS",
    "KW_COLS",
    "SPECIAL_KW_NEG",
    # main functions
    "convert_all_numeric",
    "enforce_logical_bounds",
    "impute",
    "calibrate_to_reference",
]
