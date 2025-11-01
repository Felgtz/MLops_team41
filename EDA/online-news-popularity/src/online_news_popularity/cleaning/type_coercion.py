"""
Type coercion utilities
=======================

Safe, *audit-friendly* helpers that convert every column in a dataframe
(from the Online-News-Popularity project) to a numeric dtype **without
silently swallowing errors**.

The main public helper -- `convert_all_numeric` -- keeps a detailed
change log so you can inspect how many values were forced to NaN, which
columns changed dtype, etc.  Nothing is printed; everything goes through
`logging`.

Example
-------
>>> from online_news_popularity.cleaning import type_coercion
>>> df_num, summary = type_coercion.convert_all_numeric(
...     df_raw, except_cols=["url"], logger=my_logger
... )
>>> summary.head()
        column before_dtype after_dtype  coerced_to_nan  coerced_pct
0  kw_min_min      object    float64            22342       57.763
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Tuple

import pandas as pd

__all__ = ["convert_all_numeric"]


def convert_all_numeric(
    df: pd.DataFrame,
    *,
    except_cols: Iterable[str] | None = None,
    errors: str = "coerce",
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert every dataframe column **except** those listed to `float64`.
    Invalid literals are coerced to NaN (default) or handled per the
    `errors` arg, exactly like :pyfunc:`pandas.to_numeric`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe; will *not* be mutated (a copy is returned).
    except_cols : iterable of str, optional
        Columns to preserve “as is” (e.g. ``["url"]``).
    errors : {"coerce", "raise", "ignore"}, default "coerce"
        Forwarded to :pyfunc:`pandas.to_numeric`.
    logger : logging.Logger, optional
        If provided, receives DEBUG messages per column and an INFO-level
        aggregate.

    Returns
    -------
    df_out : pd.DataFrame
        A *copy* of the input where every non-excluded column is numeric.
    summary : pd.DataFrame
        One row per column with:
        • before / after dtype
        • number / percentage of values coerced to NaN
        • original & final NaN counts

    Notes
    -----
    • We don’t attempt clever dtype inference (int vs float); everything
      goes to float64 to avoid downstream surprises.
    • The function is purposely side-effect-free so you can unit-test it
      in isolation.
    """
    if except_cols is None:
        except_cols = []

    except_set = set(except_cols)
    df_out = df.copy(deep=True)

    records: List[dict[str, Any]] = []
    n_rows = len(df_out)

    for col in df_out.columns:
        before_dtype = df_out[col].dtype
        na_before = int(df_out[col].isna().sum())

        if col in except_set:
            # Leave untouched
            records.append(
                {
                    "column": col,
                    "before_dtype": str(before_dtype),
                    "after_dtype": str(before_dtype),
                    "coerced_to_nan": 0,
                    "coerced_pct": 0.0,
                    "na_before": na_before,
                    "na_after": na_before,
                }
            )
            continue

        numeric_series = pd.to_numeric(df_out[col], errors=errors)
        coerced_mask = numeric_series.isna() & df_out[col].notna()
        n_coerced = int(coerced_mask.sum())
        na_after = int(numeric_series.isna().sum())

        df_out[col] = numeric_series  # assign back

        records.append(
            {
                "column": col,
                "before_dtype": str(before_dtype),
                "after_dtype": str(df_out[col].dtype),
                "coerced_to_nan": n_coerced,
                "coerced_pct": 100.0 * n_coerced / n_rows if n_rows else 0.0,
                "na_before": na_before,
                "na_after": na_after,
            }
        )

        if logger and n_coerced > 0:
            logger.debug(
                "%s: coerced %d values to NaN (%.3f%%)",
                col,
                n_coerced,
                100.0 * n_coerced / n_rows if n_rows else 0.0,
            )

    summary = (
        pd.DataFrame.from_records(records)
        .sort_values("coerced_pct", ascending=False)
        .reset_index(drop=True)
    )

    if logger:
        tot_coerced = int(summary["coerced_to_nan"].sum())
        logger.info(
            "Type-coercion complete: %d columns processed, %d cells coerced to NaN",
            len(df_out.columns) - len(except_set),
            tot_coerced,
        )

    return df_out, summary
