"""
src
===

Public package API for the Team-41 MLOps pipeline.

Anything imported here becomes available as:

    from src import <name>

This file keeps the namespace tidy and avoids “from src.foo import bar”
scattered all over the codebase.
"""

from __future__ import annotations

# Logging --------------------------------------------------------------------
from .logger import logger  # noqa: F401  (re-export)

# Data I/O -------------------------------------------------------------------
from .data import load_dataset, save_dataset  # noqa: F401

# Paths & folders ------------------------------------------------------------
from .config import RAW_DATA_PATH, READY_DATA_PATH, ensure_dirs  # noqa: F401

# Exploratory Data Analysis helpers ------------------------------------------
from .visualization import (  # noqa: F401
    quick_info,
    plot_histograms,
    plot_corr_heatmap,
)

# Feature engineering / preprocessing ----------------------------------------
from .features import scale_features  # noqa: F401

# ---------------------------------------------------------------------------#
# Export control                                                             #
# ---------------------------------------------------------------------------#

__all__ = [
    # logger
    "logger",
    # I/O
    "load_dataset",
    "save_dataset",
    # paths
    "RAW_DATA_PATH",
    "READY_DATA_PATH",
    "ensure_dirs",
    # EDA
    "quick_info",
    "plot_histograms",
    "plot_corr_heatmap",
    # features
    "scale_features",
]
