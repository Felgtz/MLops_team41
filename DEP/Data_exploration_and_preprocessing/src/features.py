"""
src.features
============

Feature-engineering helpers:

1. scale_features  – standardise (z-score) numeric columns.
2. run_pca         – principal-component analysis on the scaled data.

Both functions return pandas DataFrames so they fit smoothly into the rest of
the pandas-centric pipeline.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer          # ← NEW

from . import logger

# ─────────────────────────────────────────────────────────────────────────────
# scale_features
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str] | None = None,
    return_scaler: bool = False,
    impute: bool = True,               # default = on
) -> pd.DataFrame | tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale every numeric column except those in *exclude*.
    Numeric NaNs are replaced with the column median; if a column is entirely
    NaN, the value 0.0 is used instead.

    Parameters
    ----------
    df :
        Input DataFrame.
    exclude :
        Column names to leave untouched (e.g. target variable).
    return_scaler :
        If True, also return the fitted StandardScaler instance.
    impute :
        Whether to fill numeric NaNs before scaling.
    """
    exclude = set(exclude or [])

    # 0️⃣  Make numeric-looking object columns into real floats ------------
    df_work = df.copy()
    obj_cols = df_work.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        try:
            df_work[c] = pd.to_numeric(df_work[c])
        except ValueError:
            # non-numeric strings remain as object dtype
            pass

    # 1️⃣  Select numeric feature columns ----------------------------------
    numeric_cols: list[str] = [
        c for c in df_work.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    if not numeric_cols:
        logger.warning("scale_features: no numeric columns found to scale.")
        return (df_work, None) if return_scaler else df_work

    # 2️⃣  Impute -----------------------------------------------------------
    if impute:
        na_cols = [c for c in numeric_cols if df_work[c].isna().any()]
        if na_cols:
            logger.debug(
                "Imputing missing values in %d numeric column(s) with the median.",
                len(na_cols),
            )
            for c in na_cols:
                med = df_work[c].median()
                fill_val = 0.0 if np.isnan(med) else med
                df_work[c].fillna(fill_val, inplace=True)

    # 3️⃣  Scale ------------------------------------------------------------
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_work[numeric_cols])

    df_scaled = df_work.copy()
    df_scaled.loc[:, numeric_cols] = scaled_values

    return (df_scaled, scaler) if return_scaler else df_scaled


# ─────────────────────────────────────────────────────────────────────────────
# run_pca
# ─────────────────────────────────────────────────────────────────────────────
def run_pca(
    df: pd.DataFrame,
    *,
    n_components: float | int = 0.95,
    exclude: Sequence[str] | None = None,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, PCA]:
    """
    Perform PCA on the **scaled** numeric part of *df*.

    Parameters
    ----------
    df :
        Input DataFrame.
    n_components :
        Same meaning as in ``sklearn.decomposition.PCA``.  Typical values:
        • 0 < float < 1  → fraction of variance to keep (e.g. 0.95)
        • int            → number of principal components
    exclude :
        Columns to leave out entirely (e.g. target variable).
    random_state :
        Reproducibility parameter forwarded to PCA.

    Returns
    -------
    (pca_df, pca_model)
        pca_df   – DataFrame whose columns are PC1, PC2, …
                   *plus* any excluded columns appended unchanged.
        pca_model – Fitted ``sklearn.decomposition.PCA`` instance.
    """
    exclude = tuple(exclude or [])
    df_work = df.drop(columns=list(exclude))

    # Scale first
    df_scaled, scaler = scale_features(df_work, return_scaler=True)

    logger.info("Running PCA (n_components=%s)…", n_components)
    pca = PCA(n_components=n_components, random_state=random_state)
    pcs = pca.fit_transform(df_scaled)

    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    df_pca = pd.DataFrame(pcs, columns=pc_cols, index=df.index)

    # Re-attach any excluded (non-numeric / target) columns
    if exclude:
        df_pca = pd.concat([df_pca, df[list(exclude)]], axis=1)

    logger.debug(
        "PCA finished: retained %d components (%.1f%% variance).",
        pca.n_components_,
        pca.explained_variance_ratio_.sum() * 100,
    )

    # We could stash the scaler for later inverse_transform, but simplest is
    # to return just the PCA object; caller can keep the scaler from
    # scale_features if needed.
    return df_pca, pca


# ─────────────────────────────────────────────────────────────────────────────
# Public symbols
# ─────────────────────────────────────────────────────────────────────────────
__all__: list[str] = ["scale_features", "run_pca"]
