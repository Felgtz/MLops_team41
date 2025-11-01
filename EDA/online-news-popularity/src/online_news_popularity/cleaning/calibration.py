"""
Statistical calibration
=======================

After structural cleaning and imputation we still want the **modified**
dataset to *look* like the original UCI reference in terms of global
scale and tail behaviour.  This module provides a **light-touch**
calibration helper that:

1.   Clips (μ ± k σ) using the *reference* distribution for a configurable
     list of columns (tokens, counts, `shares`, some `kw_*`).
2.   Optionally falls back to percentile–based clipping if σ≈0 or NaN.
3.   Ensures proportion-like features stay within [0, 1].
4.   Re-normalises LDA rows so they still sum to ≈ 1 after clipping.
5.   Returns both the calibrated dataframe *and* an audit dictionary
     suitable for MLflow / JSON logging.

The idea is **good-enough realism** without over-fitting to the reference
statistics (no full quantile mapping, no target leakage).

Example
-------
>>> df_cal, report = calibrate_to_reference(
...     df_mod, "data/raw/online_news_original.csv",
...     cfg={"clip_k": 3, "clip_k_shares": 4}, logger=my_log
... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from ..features import groups

__all__ = ["calibrate_to_reference"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _zclip(
    cur: pd.Series,
    ref: pd.Series,
    k: float,
) -> Tuple[pd.Series, tuple[float, float, int]]:
    """Clip `cur` to μ ± k σ of `ref`.  Return clipped series + stats."""
    mu, sigma = ref.mean(), ref.std()
    if sigma is None or np.isnan(sigma) or sigma == 0.0:
        low, high = ref.quantile([0.01, 0.99])
    else:
        low, high = mu - k * sigma, mu + k * sigma
    before = int(((cur < low) | (cur > high)).sum())
    clipped = cur.clip(low, high)
    return clipped, (float(low), float(high), before)


def _proportion_clip(df: pd.DataFrame, cols: Iterable[str]) -> int:
    """Clip listed columns into [0,1]; return # values changed."""
    changed = 0
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        clipped = s.clip(0, 1)
        changed += int((clipped != s).sum())
        df[c] = clipped
    return changed


# ----------------------------------------------------------------------
# Public entry-point
# ----------------------------------------------------------------------
def calibrate_to_reference(
    df_mod: pd.DataFrame,
    ref: str | Path | pd.DataFrame,
    cfg: dict | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Light clipping + domain reinforcement using *reference* stats.

    Parameters
    ----------
    df_mod : pd.DataFrame
        Cleaned **modified** dataset (already numeric & imputed).
    ref : str | Path | pd.DataFrame
        Path to the original dataset **or** a dataframe in memory.
    cfg : dict, optional
        Configuration dictionary (all keys optional).

        clip_k            : float, default 3.0
            k for μ ± k σ clipping on *most* columns.
        clip_k_shares     : float, default 4.0
            A laxer k for the target `shares`.
        cols_clip_counts  : list[str], optional
            Overrides default token & count columns.
        cols_clip_kw      : list[str], optional
            Overrides default heavily-tailed kw_* columns.

    logger : logging.Logger, optional

    Returns
    -------
    df_cal : pd.DataFrame
    report : dict[str, int]
        Keys: 'zclip_counts', 'zclip_kw', 'zclip_shares',
              'prop_clipped', 'lda_renorm_rows'
    """
    # -------------------------- config defaults -----------------------
    cfg = cfg or {}
    k = cfg.get("clip_k", 3.0)
    k_shares = cfg.get("clip_k_shares", 4.0)

    # token / count features likely to have fat tails
    default_counts = [
        "n_tokens_title",
        "n_tokens_content",
        "num_hrefs",
        "num_self_hrefs",
        "num_imgs",
        "num_videos",
        "num_keywords",
        "average_token_length",
    ]
    counts_cols = cfg.get("cols_clip_counts", default_counts)

    # a few kw_* that were noisy
    default_kw = ["kw_min_min", "kw_avg_min", "kw_min_avg"]
    kw_cols = cfg.get("cols_clip_kw", default_kw)

    # -------------------------- load reference -----------------------
    if isinstance(ref, (str, Path)):
        df_ref = pd.read_csv(ref, low_memory=False)
    else:
        df_ref = ref.copy()

    df_out = df_mod.copy()
    report: Dict[str, int] = {}

    # -------------------------- Z-score clipping ----------------------
    def _apply_zclip(col_list: Iterable[str], kk: float, tag: str) -> int:
        clipped_cells = 0
        for col in col_list:
            if col not in df_out.columns or col not in df_ref.columns:
                continue
            df_out[col], (lo, hi, n_cut) = _zclip(df_out[col], df_ref[col], kk)
            clipped_cells += n_cut
            if logger and n_cut:
                logger.debug("%s: %s clipped %d cells (%.3g, %.3g)", tag, col, n_cut, lo, hi)
        return clipped_cells

    report["zclip_counts"] = _apply_zclip(counts_cols, k, "counts")
    report["zclip_kw"] = _apply_zclip(kw_cols, k, "kw")
    report["zclip_shares"] = _apply_zclip(["shares"], k_shares, "shares")

    # -------------------------- Proportions [0,1] --------------------
    prop_cols = (
        groups.RATE_COLS
        + groups.SUBJECTIVITY_COLS
        + groups.POS_POLARITY_COLS
        + ["n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens"]
    )
    report["prop_clipped"] = _proportion_clip(df_out, prop_cols)

    # -------------------------- LDA renormalise ----------------------
    lda_cols = [c for c in groups.LDA_COLS if c in df_out.columns]
    if lda_cols:
        row_sum = df_out[lda_cols].sum(axis=1)
        mask = row_sum > 0
        df_out.loc[mask, lda_cols] = df_out.loc[mask, lda_cols].div(
            row_sum[mask], axis=0
        )
        report["lda_renorm_rows"] = int(mask.sum())

    # -------------------------- logging ------------------------------
    if logger:
        logger.info(
            "Calibration complete – total cells clipped: %d",
            sum(report.values()),
        )
        for k_, v_ in report.items():
            logger.debug("  %-20s : %d", k_, v_)

    return df_out, report
