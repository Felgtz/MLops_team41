"""
Logical-bounds enforcement
==========================

Replace *structurally impossible* values with ``NaN`` and repair simple
consistency rules for the Online-News-Popularity dataset.

This module is intentionally **opinionated but small** – it covers the
same rules you used in the notebook, yet keeps them parameterised so they
can be unit-tested or evolved later.

Public helper
-------------
`enforce_logical_bounds(df, …) -> (df_clean, fix_report)`

• `df_clean`  : copy of the input after fixes  
• `fix_report`: dict {str rule → int cells/rows affected}

Nothing is printed; everything goes through a logger.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from ..features import groups  # single source of truth for column families

__all__ = ["enforce_logical_bounds"]


def _count_and_clip(
    df: pd.DataFrame,
    cols: Iterable[str],
    lo: float | None = None,
    hi: float | None = None,
) -> int:
    """Helper → set out-of-range values to NaN, return # changed cells."""
    n = 0
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        mask = s.notna()
        if lo is not None:
            mask &= s < lo
        if hi is not None:
            mask |= s > hi
        bad = mask.sum()
        if bad:
            df.loc[mask, c] = np.nan
            n += int(bad)
    return n


def _snap_binaries(df: pd.DataFrame, cols: Iterable[str]) -> int:
    """
    Force values **very close** to 0/1 into the exact binary set and turn
    everything else into NaN.  Returns # changed cells.
    """
    changed = 0
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        snapped = s.copy()
        # values within ±0.1 of 1 → 1; within ±0.1 of 0 → 0
        snapped[(s >= 0.9) & (s <= 1.1)] = 1.0
        snapped[(s >= -0.1) & (s <= 0.1)] = 0.0
        bad_mask = snapped.notna() & (~snapped.isin([0.0, 1.0]))
        changed += int((snapped != s).sum())
        snapped[bad_mask] = np.nan
        df[c] = snapped
    return changed


def enforce_logical_bounds(
    df: pd.DataFrame,
    *,
    lda_cols: Iterable[str] = groups.LDA_COLS,
    bin_cols: Iterable[str] = groups.BIN_COLS,
    kw_neg_as_na: set[str] = groups.SPECIAL_KW_NEG,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply rule-based fixes.

    Parameters
    ----------
    df            : raw *numeric* dataframe (output of `convert_all_numeric`)
    lda_cols      : five `LDA_*` columns that must lie in [0,1] and sum ≈ 1
    bin_cols      : every binary flag that must be in {0,1}
    kw_neg_as_na  : keyword columns where *negative* values mean “missing”
    logger        : optional logger for DEBUG / INFO messages

    Returns
    -------
    df_clean  : pd.DataFrame          (copy, never mutates original)
    fix_report: dict[str, int]        (# cells / rows touched per rule)
    """
    rep: dict[str, int] = {}
    df_out = df.copy()

    # ------------------------------------------------------------------
    # 1.  Keyword negatives → NaN (special three columns)
    # ------------------------------------------------------------------
    neg_mask_cols = [c for c in kw_neg_as_na if c in df_out.columns]
    rep["kw_sentinel_neg→NaN"] = _count_and_clip(df_out, neg_mask_cols, lo=0)

    # ------------------------------------------------------------------
    # 2.  General counts (negatives invalid)
    # ------------------------------------------------------------------
    count_cols = [c for c in df_out.columns if c.startswith(("n_tokens_", "num_"))] + [
        "num_keywords",
        "average_token_length",
    ]
    rep["counts_neg→NaN"] = _count_and_clip(df_out, count_cols, lo=0)

    # ------------------------------------------------------------------
    # 3.  Rates, subjectivities, positive polarities  ∈ [0,1]
    # ------------------------------------------------------------------
    in01 = (
        groups.RATE_COLS
        + groups.SUBJECTIVITY_COLS
        + groups.POS_POLARITY_COLS
        + list(lda_cols)
    )
    rep["prop_outOf01→NaN"] = _count_and_clip(df_out, in01, lo=0.0, hi=1.0)

    # ------------------------------------------------------------------
    # 4.  Negative polarities ∈ [-1,0]; global/title polarities ∈ [-1,1]
    # ------------------------------------------------------------------
    rep["negPol_outOf[-1,0]→NaN"] = _count_and_clip(
        df_out, groups.NEG_POLARITY_COLS, lo=-1.0, hi=0.0
    )
    rep["anyPol_outOf[-1,1]→NaN"] = _count_and_clip(
        df_out, groups.ANY_POLARITY_COLS, lo=-1.0, hi=1.0
    )

    # ------------------------------------------------------------------
    # 5.  Snap binaries + fix weekday/weekend consistency
    # ------------------------------------------------------------------
    rep["binaries_snapped"] = _snap_binaries(df_out, bin_cols)

    if {"weekday_is_saturday", "weekday_is_sunday", "is_weekend"}.issubset(df_out.columns):
        wknd_expected = (
            (df_out["weekday_is_saturday"] == 1.0)
            | (df_out["weekday_is_sunday"] == 1.0)
        ).astype(float)
        mask = wknd_expected.notna()
        mism = int((df_out.loc[mask, "is_weekend"] != wknd_expected[mask]).sum())
        df_out.loc[mask, "is_weekend"] = wknd_expected[mask]
        rep["weekend_flag_fixed"] = mism

    # ------------------------------------------------------------------
    # 6.  LDA per-row normalisation
    # ------------------------------------------------------------------
    lda_cols = [c for c in lda_cols if c in df_out.columns]
    if lda_cols:
        lda_sum = df_out[lda_cols].sum(axis=1)
        mask_pos = lda_sum > 0
        df_out.loc[mask_pos, lda_cols] = df_out.loc[mask_pos, lda_cols].div(
            lda_sum[mask_pos], axis=0
        )
        rep["lda_rows_renormed"] = int(mask_pos.sum())

    # ------------------------------------------------------------------
    # 7.  Timedelta domain [8, 731]
    # ------------------------------------------------------------------
    if "timedelta" in df_out.columns:
        rep["timedelta_outOf[8,731]"] = _count_and_clip(
            df_out, ["timedelta"], lo=8, hi=731
        )

    # ------------------------------------------------------------------
    # 8.  Logging summary
    # ------------------------------------------------------------------
    if logger:
        total_fixed = sum(rep.values())
        logger.info("Enforcement complete – %d total fixes", total_fixed)
        for k, v in rep.items():
            logger.debug("  %-25s : %d", k, v)

    return df_out, rep
