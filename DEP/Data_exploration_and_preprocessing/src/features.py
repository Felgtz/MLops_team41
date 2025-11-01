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

from . import logger

# ─────────────────────────────────────────────────────────────────────────────
# scale_features
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str] | None = None,
    return_scaler: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale every numeric column except those in *exclude*.

    Parameters
    ----------
    df :
        Input DataFrame.
    exclude :
        Column names to leave untouched (e.g. the target variable).
    return_scaler :
        If True, also return the fitted ``StandardScaler`` instance.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with scaled features.
    (optionally) sklearn.preprocessing.StandardScaler
        Only when *return_scaler* is True.
    """
    exclude = set(exclude or [])
    numeric_cols: list[str] = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]

    if not numeric_cols:
        logger.warning("scale_features: no numeric columns found to scale.")
        return (df.copy(), None) if return_scaler else df.copy()

    logger.info(
        "Scaling %d numeric columns; leaving %d column(s) untouched.",
        len(numeric_cols),
        len(exclude),
    )

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[numeric_cols])

    df_scaled = df.copy()
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
