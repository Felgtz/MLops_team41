"""
src.data
========

I/O helpers and quick-inspection utilities.

The functions here are deliberately simple wrappers around pandas so they can
be reused from notebooks, the CLI pipeline, or unit tests without duplicating
boilerplate logging & path handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from . import logger  # package-level singleton logger
from .config import ensure_dirs

# Type alias for readability
PathLike = Union[str, Path]


# ─────────────────────────────────────────────────────────────────────────────
# Core I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(path: PathLike, **pandas_kwargs) -> pd.DataFrame:
    """
    Read a CSV into a pandas DataFrame and log basic info.

    Parameters
    ----------
    path : str | pathlib.Path
        Location of the CSV file.
    pandas_kwargs :
        Any extra keyword arguments are passed straight to
        ``pandas.read_csv`` (e.g. ``usecols=…``).

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)

    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(path)

    logger.info("Loading dataset → %s", path)
    df = pd.read_csv(path, **pandas_kwargs)
    logger.debug("Loaded shape: %s", df.shape)
    return df


def save_dataset(df: pd.DataFrame, path: PathLike, *, index: bool = False, **pandas_kwargs) -> None:
    """
    Save a DataFrame to CSV and make sure the parent folder exists.

    Parameters
    ----------
    df : pandas.DataFrame
        The data to write.
    path : str | pathlib.Path
        Destination CSV file.
    index : bool, default=False
        Whether to include the DataFrame index in the output.
    pandas_kwargs :
        Additional keyword args forwarded to ``DataFrame.to_csv``.
    """
    path = Path(path)
    ensure_dirs(path.parent)

    logger.info("Saving dataset → %s", path)
    df.to_csv(path, index=index, **pandas_kwargs)
    logger.debug("Saved shape: %s", df.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Quick inspection helper
# ─────────────────────────────────────────────────────────────────────────────
def quick_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log basic information about a DataFrame (shape, null counts, dtypes).

    Intended for quick sanity-checks during EDA or debugging; keeps output in
    the unified logging format instead of printing directly to stdout.
    """
    nulls = df.isna().sum().sum()
    logger.info("%s – shape=%s  •  total-nulls=%s", name, df.shape, nulls)

    # More verbose details at DEBUG level
    if logger.isEnabledFor(level=logger.getEffectiveLevel() - 10):
        dtypes = ", ".join(f"{c}:{t}" for c, t in df.dtypes.items())
        logger.debug("%s dtypes: %s", name, dtypes)


# ─────────────────────────────────────────────────────────────────────────────
# Public symbols
# ─────────────────────────────────────────────────────────────────────────────
__all__: list[str] = ["load_dataset", "save_dataset", "quick_info"]
