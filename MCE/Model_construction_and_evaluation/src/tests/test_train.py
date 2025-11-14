"""
tests/test_train.py
===================

Quick “does-it-run?” tests for the modelling stack.
They purposely use a **tiny synthetic dataset** so CI finishes in seconds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.models.baseline import BaselineConfig, train_and_evaluate as run_baselines
from src.models.boosting import (
    BoostingConfig,
    train_and_evaluate as run_boosting,
)
from src.utils import ensure_dir

import pytest

pytestmark = pytest.mark.skip("Test de plantilla no usado en este proyecto")


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def tiny_regression_data():
    """
    Generate a small, strictly-positive target so log1p is valid.
    Returns X_train, X_test, y_train_log, y_test
    """
    X, y = make_regression(
        n_samples=120,
        n_features=8,
        noise=0.1,
        random_state=0,
    )
    # Shift to strictly positive
    y = y - y.min() + 1.0
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    y_tr_log = np.log1p(y_tr)

    return X_tr, X_te, y_tr_log, y_te


@pytest.fixture()
def tmp_artifacts_dir(tmp_path) -> Path:
    """Return a fresh directory for model artefacts per test."""
    return ensure_dir(tmp_path / "artifacts")


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_baseline_training_smoke(tiny_regression_data, tmp_artifacts_dir):
    X_tr, X_te, y_tr_log, y_te = tiny_regression_data

    cfg = BaselineConfig(
        cv_folds=3,  # keep it fast
        artifacts_dir=tmp_artifacts_dir,
    )

    metrics = run_baselines(X_tr, y_tr_log, X_te, y_te, cfg=cfg)

    # Basic sanity checks
    assert not metrics.empty
    assert {"model", "RMSE_test", "MAE_test", "R2_test"}.issubset(metrics.columns)

    # At least one model file saved
    saved_models = list((tmp_artifacts_dir / "models").glob("*.pkl"))
    assert saved_models, "Expected .pkl files to be written"


def test_boosting_training_smoke(tiny_regression_data, tmp_artifacts_dir):
    # Skip gracefully if XGBoost isn’t installed in the environment
    pytest.importorskip("xgboost", reason="xgboost not available")

    X_tr, X_te, y_tr_log, y_te = tiny_regression_data

    cfg = BoostingConfig(
        cv_folds=3,
        artifacts_dir=tmp_artifacts_dir,
        xgb_n_estimators=50,  # keep it speedy
    )

    metrics = run_boosting(X_tr, y_tr_log, X_te, y_te, cfg=cfg)

    assert not metrics.empty
    assert "xgboost" in metrics["model"].values
