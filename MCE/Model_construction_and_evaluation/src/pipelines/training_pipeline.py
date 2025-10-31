# src/pipelines/training_pipeline.py
"""
End-to-end training pipeline.

Why another file?
-----------------
`src/models/train.py` is a convenient *script* entry-point, but importing a
script from Python code is clunky.  This module provides a reusable,
object-oriented façade (`TrainingPipeline`) that you can call from:

• Jupyter notebooks  
• Unit tests  
• REST / FastAPI services for on-demand retraining  
• Hyper-parameter search frameworks (Optuna, Ray Tune…)

Public API
----------
>>> from src.pipelines.training_pipeline import TrainingPipeline
>>> leaderboard = TrainingPipeline("src/config/experiment.yaml").run()
>>> leaderboard.head()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from src.data import DataConfig, load_and_split
from src.features import drop_columns, drop_high_corr_features
from src.models.baseline import BaselineConfig, train_and_evaluate as run_baselines
from src.models.boosting import BoostingConfig, train_and_evaluate as run_boosting
from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TrainingPipeline:
    """Encapsulates the full experiment defined by a YAML file."""

    def __init__(self, cfg_path: str | Path):
        self.cfg_path = Path(cfg_path)
        self.cfg: Dict = yaml.safe_load(self.cfg_path.read_text())
        logger.info("Loaded config from %s", self.cfg_path)

        # Pre-compute some frequently-used paths
        out_root = Path(self.cfg["output"]["artifacts_dir"])
        self.models_dir = ensure_dir(out_root / self.cfg["output"]["models_subdir"])
        self.metrics_dir = ensure_dir(out_root / self.cfg["output"]["metrics_subdir"])

    # ------------------------------------------------------------------ #
    # 1. DATA                                                            #
    # ------------------------------------------------------------------ #
    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        data_cfg = DataConfig(**self.cfg["data"])
        X_tr, X_te, y_tr, y_te = load_and_split(data_cfg)
        logger.info("Loaded data – train %s | test %s", X_tr.shape, X_te.shape)
        return X_tr, X_te, y_tr, y_te

    # ------------------------------------------------------------------ #
    # 2. FEATURE PRUNING                                                 #
    # ------------------------------------------------------------------ #
    def _prune_features(
        self, X_tr: pd.DataFrame, X_te: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        fs_cfg = self.cfg["feature_selection"]

        # 2.1 explicit drop list
        X_tr = drop_columns(X_tr, fs_cfg["drop"])
        X_te = drop_columns(X_te, fs_cfg["drop"])

        # 2.2 high-correlation pruning
        thresh = fs_cfg.get("corr_threshold", 0.85)
        X_tr = drop_high_corr_features(X_tr, thresh=thresh)
        X_te = X_te[X_tr.columns]

        logger.info("After pruning – %d features left", X_tr.shape[1])

        # Persist the final feature list so predict.py can reuse it
        feature_csv = self.metrics_dir.parent / self.cfg["output"]["feature_list_file"]
        pd.Series(X_tr.columns, name="feature").to_csv(feature_csv, index=False)
        logger.debug("Saved feature list to %s", feature_csv)

        return X_tr, X_te

    # ------------------------------------------------------------------ #
    # 3. TARGET TRANSFORM                                                #
    # ------------------------------------------------------------------ #
    def _transform_target(self, y: pd.Series) -> np.ndarray:
        method = self.cfg["target_transform"].get("method", "none").lower()
        if method == "log1p":
            return np.log1p(y)
        if method in ("none", "", None):
            return y.values  # noqa: NPY002
        raise NotImplementedError(f"Unsupported target transform: {method}")

    # ------------------------------------------------------------------ #
    # 4. MODEL TRAINING                                                  #
    # ------------------------------------------------------------------ #
    def _train_models(
        self,
        X_tr: pd.DataFrame,
        y_tr_trans: np.ndarray,
        X_te: pd.DataFrame,
        y_te: pd.Series,
    ) -> pd.DataFrame:
        seed = self.cfg["seed"]
        cv_folds = self.cfg["modeling"]["cv_folds"]
        artifacts_dir = Path(self.cfg["output"]["artifacts_dir"])

        # 4.1 Baselines
        bl_cfg = BaselineConfig(
            seed=seed,
            cv_folds=cv_folds,
            artifacts_dir=artifacts_dir,
        )
        bl_metrics = run_baselines(X_tr, y_tr_trans, X_te, y_te, cfg=bl_cfg)

        # 4.2 Boosting
        xgb_cfg_yaml = self.cfg["modeling"]["gradient_boosting"]["xgboost"]
        boost_cfg = BoostingConfig(
            seed=seed,
            cv_folds=cv_folds,
            artifacts_dir=artifacts_dir,
            xgb_enabled=xgb_cfg_yaml["enabled"],
            xgb_n_estimators=xgb_cfg_yaml["n_estimators"],
            xgb_learning_rate=xgb_cfg_yaml["learning_rate"],
            xgb_max_depth=xgb_cfg_yaml["max_depth"],
            xgb_subsample=xgb_cfg_yaml["subsample"],
            xgb_colsample_bytree=xgb_cfg_yaml["colsample_bytree"],
            xgb_reg_lambda=xgb_cfg_yaml["reg_lambda"],
            xgb_n_jobs=xgb_cfg_yaml["n_jobs"],
        )
        boost_metrics = run_boosting(X_tr, y_tr_trans, X_te, y_te, cfg=boost_cfg)

        return pd.concat([bl_metrics, boost_metrics], ignore_index=True)

    # ------------------------------------------------------------------ #
    # PUBLIC ENTRY-POINT                                                #
    # ------------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return a leaderboard DataFrame."""
        X_tr, X_te, y_tr, y_te = self._load_data()
        X_tr, X_te = self._prune_features(X_tr, X_te)
        y_tr_trans = self._transform_target(y_tr)

        leaderboard = self._train_models(X_tr, y_tr_trans, X_te, y_te)
        leaderboard = leaderboard.sort_values("RMSE_test").reset_index(drop=True)

        # Persist combined metrics
        out_path = self.metrics_dir / "all_models_metrics.json"
        out_path.write_text(json.dumps(leaderboard.to_dict(orient="records"), indent=2))
        logger.info("Saved leaderboard to %s", out_path)

        return leaderboard


# --------------------------------------------------------------------------- #
#  Optional CLI wrapper – keeps arguments minimal                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    import argparse

    p = argparse.ArgumentParser(description="Run the full training pipeline.")
    p.add_argument(
        "--config",
        "-c",
        required=True,
        type=str,
        help="Path to experiment YAML (e.g. src/config/experiment.yaml)",
    )
    args = p.parse_args()

    lb = TrainingPipeline(args.config).run()
    print("\n=== Leaderboard ===")
    print(lb[["model", "RMSE_test", "MAE_test", "R2_test"]])
