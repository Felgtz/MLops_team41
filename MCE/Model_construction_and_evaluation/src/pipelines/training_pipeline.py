"""
src.pipelines.training_pipeline
===============================

End-to-end training pipeline.

Public usage
------------
>>> from src.pipelines.training_pipeline import TrainingPipeline
>>> lb = TrainingPipeline("src/config/experiment.yaml").run()
>>> lb.head()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
import joblib

# ------------------------------------------------------------------ #
# project helpers
# ------------------------------------------------------------------ #
from src.data import DataConfig, load_and_split
from src.features import drop_columns, drop_high_corr_features
from src.models.baseline import (
    BaselineConfig,
    train_and_evaluate as run_baselines,
)
from src.models.boosting import (
    BoostingConfig,
    train_and_evaluate as run_boosting,
)
from src.utils.io import ensure_dir
from src.utils.dataframe_ops import drop_columns
from src.utils.feature_selection import drop_high_corr_features

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



# ------------------------------------------------------------------ #
# main façade
# ------------------------------------------------------------------ #
class TrainingPipeline:
    """Encapsulates the full experiment defined by a YAML file."""

    def __init__(self, cfg_path: str | Path):
        self.cfg_path = Path(cfg_path)
        self.cfg: Dict = yaml.safe_load(
            self.cfg_path.read_text(encoding="utf-8")
        )
        logger.info("Loaded config from %s", self.cfg_path)

        # Prepare output folders
        out_root = Path(self.cfg["output"]["artifacts_dir"])
        self.models_dir = ensure_dir(out_root / self.cfg["output"]["models_subdir"])
        self.metrics_dir = ensure_dir(out_root / self.cfg["output"]["metrics_subdir"])

    # ------------------------------------------------------------------ #
    # 1. DATA                                                            #
    # ------------------------------------------------------------------ #
    def _load_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        data_cfg = DataConfig(**self.cfg["data"])
        X_tr, X_te, y_tr, y_te = load_and_split(data_cfg)
        logger.info("Loaded data – train %s | test %s", X_tr.shape, X_te.shape)
        return X_tr, X_te, y_tr, y_te

    # ------------------------------------------------------------------ #
    # 2. FEATURE PRUNING                                                 #
    # ------------------------------------------------------------------ #
    def _prune_features(
        self,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
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

        # Persist feature list for predict.py
        feature_csv = (
            self.metrics_dir.parent / self.cfg["output"]["feature_list_file"]
        )
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
        if method in {"none", "", None}:
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

        # 4.1 Baselines --------------------------------------------------
        bl_cfg = BaselineConfig(random_state=seed)
        bl_df, bl_models = run_baselines(bl_cfg, X_tr, X_te, y_tr_trans, y_te)

        # 4.2 Boosting ---------------------------------------------------
        xgb_yaml = self.cfg["modeling"]["gradient_boosting"]["xgboost"]
        boost_cfg = BoostingConfig(
            random_state       = seed,
            xgb_enabled        = xgb_yaml["enabled"],
            xgb_n_estimators   = xgb_yaml["n_estimators"],
            xgb_learning_rate  = xgb_yaml["learning_rate"],
            xgb_max_depth      = xgb_yaml["max_depth"],
            xgb_subsample      = xgb_yaml["subsample"],
            xgb_colsample_bytree = xgb_yaml["colsample_bytree"],
            xgb_reg_lambda     = xgb_yaml["reg_lambda"],
            xgb_n_jobs         = xgb_yaml["n_jobs"],
        )
        boost_df, bst_models = run_boosting(
            boost_cfg, X_tr, X_te, y_tr_trans, y_te
        )

        # 4.3 Combine leaderboards --------------------------------------
        leaderboard = pd.concat([bl_df, boost_df], ignore_index=True)

        # 4.4 Persist models --------------------------------------------
        # directory is prepared in __init__
        all_models = {**bl_models, **bst_models}

        # 4.4.1 save each individual model
        for name, est in all_models.items():
            out_path = self.models_dir / f"{name}.pkl"
            joblib.dump(est, out_path)
            logger.debug("Saved %s", out_path)

        # 4.4.2 save the single best model (lowest RMSE)
        best_name  = leaderboard.sort_values("RMSE").iloc[0]["model"]
        best_model = all_models[best_name]
        joblib.dump(best_model, self.models_dir / "best_model.pkl")
        logger.info(
            "Saved best model (%s) → %s",
            best_name,
            self.models_dir / "best_model.pkl",
        )

        return leaderboard

    # ------------------------------------------------------------------ #
    # PUBLIC ENTRY-POINT                                                #
    # ------------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return a leaderboard DataFrame."""
        X_tr, X_te, y_tr, y_te = self._load_data()
        X_tr, X_te = self._prune_features(X_tr, X_te)
        y_tr_trans = self._transform_target(y_tr)

        leaderboard = self._train_models(X_tr, y_tr_trans, X_te, y_te)

        # Persist combined metrics
        out_path = self.metrics_dir / "leaderboard.json"
        out_path.write_text(
            json.dumps(leaderboard.to_dict(orient="records"), indent=2)
        )
        logger.info("Saved leaderboard to %s", out_path)

        return leaderboard


# --------------------------------------------------------------------------- #
# Optional CLI wrapper                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
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
    print(lb)
