"""
online_news_popularity.io
=========================

Small façade around pandas + fsspec so the rest of the codebase never has
to deal with:  Pathlib vs strings, creating parent dirs, compression kwargs,
cloud buckets, etc.

Typical use
-----------
>>> import pandas as pd
>>> from online_news_popularity import io
>>> df = io.read_df("s3://mybucket/raw/online_news_modified.csv")
>>> io.save_df(df, "data/interim/clean_stage1.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

try:
    import fsspec  # type: ignore
except ImportError:  # optional, only needed for cloud URLs
    fsspec = None  # pyright: ignore

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# ----------------------------------------------------------------------
# Readers / writers
# ----------------------------------------------------------------------
def _is_parquet(path: str | Path) -> bool:
    return str(path).lower().endswith((".parquet", ".pq"))


def _is_pickle(path: str | Path) -> bool:
    return str(path).lower().endswith((".pkl", ".pickle"))


def _is_compressed_csv(path: str | Path) -> bool:
    return str(path).lower().endswith(
        (
            ".csv.gz",
            ".csv.zip",
            ".csv.bz2",
            ".csv.xz",
            ".csv.zst",
            ".csv.zstd",
        )
    )


def read_df(
    path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
    **pandas_kw: Any,
) -> pd.DataFrame:
    """
    Generic dataframe reader with minimal magic.

    Parameters
    ----------
    path : str or Path
        Local or remote path.  Remote URLs require an `fsspec`-compatible
        protocol (s3://, gs://, az://, etc.) and the corresponding plugin.
    storage_options : dict, optional
        Extra kwargs forwarded to `pandas.read_*` (`storage_options=...`).
    **pandas_kw
        Any keyword pandas would accept (encoding, dtype, etc.).

    Returns
    -------
    pd.DataFrame
    """
    path_str = str(path)

    read_kwargs: dict[str, Any] = dict(storage_options=storage_options or {})
    read_kwargs.update(pandas_kw)

    if _is_parquet(path):
        log.debug("Reading parquet %s", path_str)
        return pd.read_parquet(path_str, **read_kwargs)

    if _is_pickle(path):
        log.debug("Reading pickle %s", path_str)
        return pd.read_pickle(path_str, **read_kwargs)

    # default = CSV (compressed or not)
    log.debug("Reading csv %s", path_str)
    return pd.read_csv(path_str, low_memory=False, **read_kwargs)


def save_df(
    df: pd.DataFrame,
    path: str | Path,
    *,
    fmt: Literal["infer", "csv", "parquet", "pickle"] = "infer",
    index: bool = False,
    storage_options: dict[str, Any] | None = None,
    **extra_kw: Any,
) -> None:
    """
    Persist a dataframe, inferring the format from the filename by default.

    Parameters
    ----------
    df : DataFrame
    path : str or Path
    fmt : "infer" | "csv" | "parquet" | "pickle"
        If "infer", choose based on the suffix: .parquet/.pq → parquet,
        .pkl/.pickle → pickle; else write as csv (compressed if suffix ends
        with .gz, .zip, etc.).
    index : bool, default False
    storage_options : dict, optional
        Passed to pandas for remote URLs.
    **extra_kw
        Forwarded to the underlying pandas writer.
    """
    path_obj = Path(path)
    path_str = str(path_obj)

    # Ensure parent dirs (local only—remote FS handles itself)
    if path_obj.drive or path_obj.anchor:  # local path heuristic
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if fmt == "infer":
        if _is_parquet(path_obj):
            fmt = "parquet"  # type: ignore[assignment]
        elif _is_pickle(path_obj):
            fmt = "pickle"  # type: ignore[assignment]
        else:
            fmt = "csv"  # type: ignore[assignment]

    log.debug("Writing dataframe → %s (%s)", path_str, fmt)

    if fmt == "parquet":
        df.to_parquet(path_str, index=index, storage_options=storage_options, **extra_kw)
    elif fmt == "pickle":
        df.to_pickle(path_str, protocol=extra_kw.pop("protocol", 4))
    elif fmt == "csv":
        df.to_csv(
            path_str,
            index=index,
            storage_options=storage_options,
            float_format=extra_kw.pop("float_format", "%.6g"),
            **extra_kw,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown format: {fmt}")
