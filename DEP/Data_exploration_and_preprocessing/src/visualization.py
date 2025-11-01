"""
src.visualization
=================

Lightweight plotting helpers built on seaborn / matplotlib.

Functions
---------
plot_histograms(df, …)
    Draw a separate histogram for each numeric column.

plot_corr_heatmap(df, …)
    Correlation matrix visualised as a heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from . import logger

# --------------------------------------------------------------------------- #
# Try to stay backward-compatible with older code                             #
# --------------------------------------------------------------------------- #
try:
    from .config import FIG_DIR, RANDOM_STATE, ensure_dirs  # legacy names
except ImportError:  # new naming scheme
    from .config import FIGURES_DIR as FIG_DIR, RANDOM_STATE, ensure_dirs

# Final safety net in case RANDOM_STATE wasn’t defined in config.py
RANDOM_STATE = globals().get("RANDOM_STATE", 42)

# Make sure the output folder exists once at import time
ensure_dirs(FIG_DIR)


def quick_info(df: pd.DataFrame, name: str | None = None) -> None:
    """
    Print a compact overview of *df* to the console.

    Parameters
    ----------
    df   : pandas.DataFrame
    name : str, optional  – label shown in the log
    """
    label = f" [{name}]" if name else ""
    logger.info("DataFrame%s  shape = %s", label, df.shape)
    logger.info("First 3 rows%s:\n%s", label, df.head(3).to_string())
    logger.info("dtypes / non-null%s:\n%s", label, df.info(verbose=False, show_counts=True))


# --------------------------------------------------------------------------- #
# Plot helpers                                                                #
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, filename: str | Path) -> None:
    """Utility: save a matplotlib Figure and log where it went."""
    path = Path(filename)
    ensure_dirs(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info("Figure saved → %s", path)


def plot_histograms(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    bins: int = 30,
    figsize: tuple[int, int] = (4, 3),
    save: bool = False,
    prefix: str = "hist",
) -> None:
    """
    Draw an individual histogram for each numeric column (or *columns*).

    Parameters
    ----------
    df :
        Input DataFrame.
    columns :
        Explicit list of columns to plot.  By default every numeric column is
        used.  Non-numeric columns are silently skipped.
    bins :
        Number of histogram bins.
    figsize :
        Size of each subplot in inches.
    save :
        If True, each histogram is written to *FIG_DIR/prefix_<col>.png*.
    prefix :
        Filename prefix when *save* is True.
    """
    cols = (
        list(columns)
        if columns is not None
        else df.select_dtypes("number").columns.tolist()
    )

    if not cols:
        logger.warning("plot_histograms: no numeric columns to plot.")
        return

    logger.info("Plotting histograms for %d column(s)…", len(cols))

    for col in cols:
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(df[col].dropna(), bins=bins, ax=ax, kde=True, color="#1f77b4")
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        if save:
            filename = FIG_DIR / f"{prefix}_{col}.png"
            _save(fig, filename)

        plt.close(fig)


def plot_corr_heatmap(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    figsize: tuple[int, int] = (10, 8),
    save: bool = False,
    filename: str = "corr_heatmap.png",
) -> None:
    """
    Show (and optionally save) a correlation-matrix heatmap.

    Parameters
    ----------
    df :
        Input DataFrame.
    method :
        *pearson* | *spearman* | *kendall* – forwarded to ``DataFrame.corr``.
    figsize :
        Figure size in inches.
    save :
        Whether to save the figure under *FIG_DIR/filename*.
    filename :
        Filename used when *save* is True.
    """
    num_df = df.select_dtypes("number")
    if num_df.shape[1] < 2:
        logger.warning("plot_corr_heatmap: need at least 2 numeric columns.")
        return

    logger.info("Plotting %s correlation heatmap (%d × %d)…", method, *num_df.shape)

    corr = num_df.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"{method.capitalize()} correlation heatmap", fontsize=14)

    if save:
        _save(fig, FIG_DIR / filename)

    plt.close(fig)


# --------------------------------------------------------------------------- #
# Public symbols                                                              #
# --------------------------------------------------------------------------- #
__all__: list[str] = ["plot_histograms", "plot_corr_heatmap"]
