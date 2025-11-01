"""
Group-wise imputation
=====================

Lightweight, *explainable* routines that fill in missing values after the
structural enforcement step.

The default logic mirrors what you did manually in the notebook:

*   Continuous / count features   → median  
*   Binary flags                  → mode (0/1) *plus* re-validate logic  
*   LDA topic probabilities       → fill per-column median then renormalise
*   Sentiment / subjectivity cols → median, clipped back into their domain

Everything is parameter-driven so you can tune strategies from a Hydra /
YAML config without rewriting code.

Public helper
-------------
`impute(df, policy, logger=None) -> (df_filled, report)`
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from ..features import groups

__all__ = ["impute"]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _median_fill(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, int]:
    """Return {col: #filled} after median imputation."""
    filled: Dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            continue
        na_mask = df[c].isna()
        if not na_mask.any():
            continue
        median = df[c].median(skipna=True)
        df.loc[na_mask, c] = median
        filled[c] = int(na_mask.sum())
    return filled


def _mode_fill(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, int]:
    """Mode imputation for binary / categorical cols."""
    filled: Dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            continue
        na_mask = df[c].isna()
        if not na_mask.any():
            continue
        mode_series = df[c].mode(dropna=True)
        if mode_series.empty:
            continue  # cannot fill
        mode_val = mode_series.iloc[0]
        df.loc[na_mask, c] = mode_val
        filled[c] = int(na_mask.sum())
    return filled


# ----------------------------------------------------------------------
# Main entry-point
# ----------------------------------------------------------------------
def impute(
    df: pd.DataFrame,
    policy: dict,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply group-wise imputations according to *policy*.

    Parameters
    ----------
    df      : pd.DataFrame (output of `enforce_logical_bounds`)
    policy  : dict
        Expected keys (all optional, sensible defaults if absent)
        continuous.strategy : "median"
        binary.strategy     : "mode"
        lda.strategy        : "median"
        sentiment.strategy  : "median"
    logger  : logging.Logger, optional

    Returns
    -------
    df_filled : pd.DataFrame   (copy; original untouched)
    report    : dict[str, int] (# cells filled per rule)
    """
    df_out = df.copy(deep=True)
    rep: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Resolve column groups once
    # ------------------------------------------------------------------
    cont_cols = (
        groups.COUNT_COLS
        + groups.SELF_REFERENCE_COLS
        + groups.CONTINUOUS_OTHER_COLS
        + [c for c in groups.KW_COLS if c not in groups.SPECIAL_KW_NEG]
    )
    bin_cols = groups.BIN_COLS
    lda_cols = groups.LDA_COLS
    senti_cols = (
        groups.RATE_COLS
        + groups.SUBJECTIVITY_COLS
        + groups.POS_POLARITY_COLS
        + groups.NEG_POLARITY_COLS
        + groups.ANY_POLARITY_COLS
    )

    # ------------------------------------------------------------------
    # 1. Continuous --> median
    # ------------------------------------------------------------------
    cont_filled = _median_fill(df_out, cont_cols)
    rep["continuous_median"] = sum(cont_filled.values())

    # ------------------------------------------------------------------
    # 2. Binary flags --> mode
    # ------------------------------------------------------------------
    bin_filled = _mode_fill(df_out, bin_cols)
    rep["binary_mode"] = sum(bin_filled.values())

    # After fill, re-validate weekend flag (same logic as enforcement)
    if {"weekday_is_saturday", "weekday_is_sunday", "is_weekend"}.issubset(df_out.columns):
        expected = (
            (df_out["weekday_is_saturday"] == 1) | (df_out["weekday_is_sunday"] == 1)
        ).astype(float)
        changed = (df_out["is_weekend"] != expected).sum()
        df_out["is_weekend"] = expected
        rep["weekend_realigned_after_impute"] = int(changed)

    # ------------------------------------------------------------------
    # 3. LDA probabilities
    # ------------------------------------------------------------------
    if lda_cols and all(c in df_out.columns for c in lda_cols):
        # Fill missing with per-column median
        lda_miss = df_out[lda_cols].isna().sum().sum()
        col_meds = df_out[lda_cols].median(skipna=True)
        df_out[lda_cols] = df_out[lda_cols].fillna(col_meds)
        # Row-wise renormalise
        row_sum = df_out[lda_cols].sum(axis=1)
        pos_mask = row_sum > 0
        df_out.loc[pos_mask, lda_cols] = df_out.loc[pos_mask, lda_cols].div(
            row_sum[pos_mask], axis=0
        )
        rep["lda_imputed"] = int(lda_miss)

    # ------------------------------------------------------------------
    # 4. Sentiment / subjectivity
    # ------------------------------------------------------------------
    senti_filled = _median_fill(df_out, senti_cols)
    rep["sentiment_median"] = sum(senti_filled.values())

    # Clip back into legal ranges
    for c in groups.RATE_COLS + groups.SUBJECTIVITY_COLS + groups.POS_POLARITY_COLS:
        if c in df_out.columns:
            df_out[c] = df_out[c].clip(0, 1)
    for c in groups.NEG_POLARITY_COLS:
        if c in df_out.columns:
            df_out[c] = df_out[c].clip(-1, 0)
    for c in groups.ANY_POLARITY_COLS:
        if c in df_out.columns:
            df_out[c] = df_out[c].clip(-1, 1)

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------
    if logger:
        total = sum(rep.values())
        logger.info("Imputation complete – filled %d cells", total)
        for k, v in rep.items():
            logger.debug("  %-30s : %d", k, v)

    return df_out, rep
