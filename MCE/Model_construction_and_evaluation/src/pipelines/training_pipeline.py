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

# MLflow (ADD)
import mlflow
from mlflow.models import infer_signature
from inspect import signature

from src.data import DataConfig, load_and_split
#from src.features import drop_columns, drop_high_corr_features
#from src.features import drop_columns as mod_drop_columns
#from src.features import drop_high_corr_features as mod_drop_high_corr
#from src.features.drop_columns import drop_columns
#from src.features.drop_high_corr_features import drop_high_corr_features
from src.models.baseline import BaselineConfig, train_and_evaluate as run_baselines
from src.models.boosting import BoostingConfig, train_and_evaluate as run_boosting
from src.utils.io import ensure_dir
# --- Feature helpers (compat con función o clase) ---
try:
    # Si tu repo tuviera funciones
    from src.features.drop_columns import drop_columns as _drop_columns_fn  # type: ignore
except Exception:
    _drop_columns_fn = None

try:
    from src.features.drop_high_corr_features import (
        drop_high_corr_features as _drop_high_corr_fn,  # type: ignore
    )
except Exception:
    _drop_high_corr_fn = None

# Clases (presentes en tu boilerplate)
try:
    from src.features import DropColumns as _DropColumnsCls  # type: ignore
except Exception:
    _DropColumnsCls = None

try:
    from src.features import DropHighCorrFeatures as _DropHighCorrCls  # type: ignore
except Exception:
    _DropHighCorrCls = None


def drop_columns_compat(df, columns):
    """Usa función `drop_columns` si existe; si no, usa la clase `DropColumns`."""
    if _drop_columns_fn is not None:
        return _drop_columns_fn(df, columns)
    if _DropColumnsCls is not None:
        tr = _DropColumnsCls(columns=columns)
        return tr.fit_transform(df)
    raise ImportError("No se encontró ni función ni clase para 'drop_columns'.")


def drop_high_corr_compat(df, thresh=0.85):
    """Usa función `drop_high_corr_features` si existe; si no, usa la clase."""
    if _drop_high_corr_fn is not None:
        return _drop_high_corr_fn(df, thresh=thresh)
    if _DropHighCorrCls is not None:
        tr = _DropHighCorrCls(threshold=thresh)
        return tr.fit_transform(df)
    raise ImportError("No se encontró ni función ni clase para 'drop_high_corr_features'.")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TrainingPipeline:
    """Encapsulates the full experiment defined by a YAML file."""

    def __init__(self, cfg_path: str | Path):
        self.cfg_path = Path(cfg_path)
        #self.cfg: Dict = yaml.safe_load(self.cfg_path.read_text(encoding="utf-8"))
        # Después (robusto en Windows):
        with self.cfg_path.open("r", encoding="utf-8") as f:
            self.cfg: Dict = yaml.safe_load(f)
        logger.info("Loaded config from %s", self.cfg_path)

        # Pre-compute some frequently-used paths
        out_root = Path(self.cfg["output"]["artifacts_dir"])
        self.models_dir = ensure_dir(out_root / self.cfg["output"]["models_subdir"])
        self.metrics_dir = ensure_dir(out_root / self.cfg["output"]["metrics_subdir"])

    # ------------------------------------------------------------------ #
    # MLflow helpers (ADD)                                               #
    # ------------------------------------------------------------------ #
    def _mlflow_enabled(self) -> bool:
        tr = self.cfg.get("tracking", {})
        return bool(tr.get("use_mlflow", False))

    def _mlflow_init(self) -> None:
        tr = self.cfg.get("tracking", {})
        if not tr:
            return
        mlflow.set_tracking_uri(tr["tracking_uri"])
        mlflow.set_experiment(tr["experiment_name"])


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
        
        X_tr = drop_columns_compat(X_tr, fs_cfg["drop"])
        X_te = drop_columns_compat(X_te, fs_cfg["drop"])

                # --- asegurar solo columnas numéricas antes de calcular correlaciones ---
        # (por ejemplo, 'url' es string y rompe la correlación)
        non_numeric_cols = X_tr.columns.difference(
            X_tr.select_dtypes(include=[np.number]).columns
        )
        if len(non_numeric_cols) > 0:
            logger.info("Dropping non-numeric columns before corr: %s",
                        list(non_numeric_cols))
        X_tr = X_tr.select_dtypes(include=[np.number])
        # alinear test con las columnas finales de train
        X_te = X_te[X_tr.columns]


        # 2.2 high-correlation pruning
        thresh = fs_cfg.get("corr_threshold", 0.85)
        X_tr = drop_high_corr_compat(X_tr, thresh=thresh)
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
    from inspect import signature 

    def _train_models(
        self,
        X_tr: pd.DataFrame,
        y_tr_trans: np.ndarray,
        X_te: pd.DataFrame,
        y_te: pd.Series,
    ) -> pd.DataFrame:
        """Entrena baselines y boosting (si aplica) y devuelve un leaderboard."""
        rows: list[pd.DataFrame] = []

        # 4.1 Baselines (robusto a firmas)
        try:
            bl_cfg = BaselineConfig()
        except Exception:
            bl_cfg = None

        # Intentos en orden con argumentos POSICIONALES
        bl_metrics = None
        for args in [
            (X_tr, y_tr_trans, X_te, y_te, bl_cfg, artifacts_dir),
            (X_tr, y_tr_trans, X_te, y_te, bl_cfg),
            (X_tr, y_tr_trans, X_te, y_te),
        ]:
            try:
                bl_metrics = run_baselines(*args)  # <-- sin nombres, solo posicional
                break
            except TypeError:
                continue
        if bl_metrics is not None:
            rows.append(bl_metrics)
        else:
            raise TypeError("run_baselines no aceptó ninguna combinación de argumentos.")

        # 4.2 Boosting (robusto a firmas)
        try:
            xgb_yaml = self.cfg.get("modeling", {}).get("gradient_boosting", {}).get("xgboost", {})
            boost_cfg = BoostingConfig(
                xgb_enabled=xgb_yaml.get("enabled", True),
                xgb_n_estimators=xgb_yaml.get("n_estimators", 500),
                xgb_learning_rate=xgb_yaml.get("learning_rate", 0.05),
                xgb_max_depth=xgb_yaml.get("max_depth", 6),
                xgb_subsample=xgb_yaml.get("subsample", 1.0),
                xgb_colsample_bytree=xgb_yaml.get("colsample_bytree", 1.0),
                xgb_reg_lambda=xgb_yaml.get("reg_lambda", 1.0),
                xgb_n_jobs=xgb_yaml.get("n_jobs", -1),
            )
        except Exception:
            boost_cfg = None

        boost_metrics = None
        for args in [
            (X_tr, y_tr_trans, X_te, y_te, boost_cfg, artifacts_dir),
            (X_tr, y_tr_trans, X_te, y_te, boost_cfg),
            (X_tr, y_tr_trans, X_te, y_te),
        ]:
            try:
                boost_metrics = run_boosting(*args)  # <-- posicional
                break
            except TypeError:
                continue
        if boost_metrics is not None:
            rows.append(boost_metrics)
        else:
            raise TypeError("run_boosting no aceptó ninguna combinación de argumentos.")


        return pd.concat(rows, ignore_index=True)



    def run(self) -> pd.DataFrame:
        """Ejecuta el experimento completo y devuelve el leaderboard."""
        # 1) Carga y split
        X_tr, X_te, y_tr, y_te = self._load_data()

        # 2) Pruning de features
        X_tr, X_te = self._prune_features(X_tr, X_te)

        # 3) Transformación del target
        y_tr_trans = self._transform_target(y_tr)

        # 4) Entrenamiento de modelos
        leaderboard = self._train_models(X_tr, y_tr_trans, X_te, y_te)
        leaderboard = leaderboard.sort_values("RMSE_test").reset_index(drop=True)

        # 5) Persistir métricas combinadas
        out_path = self.metrics_dir / "all_models_metrics.json"
        out_path.write_text(
            json.dumps(leaderboard.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        logger.info("Saved leaderboard to %s", out_path)

        return leaderboard

    # ------------------------------------------------------------------ #
    # PUBLIC ENTRY-POINT                                                #
    # ------------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return a leaderboard DataFrame."""
                # (ADD) MLflow init
        use_mlflow = self._mlflow_enabled()
        if use_mlflow:
            self._mlflow_init()
            # run padre del experimento
            mlflow.start_run(run_name="training-pipeline")
            # tags útiles
            mlflow.set_tags({
                "phase": "training",
                "pipeline": "TrainingPipeline",
                "cfg_path": str(self.cfg_path),
            })
            # params “macro” del experimento (semilla, data, pruning, etc.)
            try:
                mlflow.log_params({
                    "seed": self.cfg.get("seed"),
                    "data.path": self.cfg["data"]["path"],
                    "target": self.cfg["data"]["target_column"],
                    "fs.drop_count": len(self.cfg["feature_selection"]["drop"]),
                    "fs.corr_threshold": self.cfg["feature_selection"].get("corr_threshold", 0.85),
                    "cv_folds": self.cfg["modeling"]["cv_folds"],
                    "xgb.enabled": self.cfg["modeling"]["gradient_boosting"]["xgboost"]["enabled"],
                    "xgb.n_estimators": self.cfg["modeling"]["gradient_boosting"]["xgboost"]["n_estimators"],
                    "xgb.learning_rate": self.cfg["modeling"]["gradient_boosting"]["xgboost"]["learning_rate"],
                    "xgb.max_depth": self.cfg["modeling"]["gradient_boosting"]["xgboost"]["max_depth"],
                })
            except Exception:
                # si cambia el YAML y no existen llaves, evitamos romper el run
                pass

        X_tr, X_te, y_tr, y_te = self._load_data()
        X_tr, X_te = self._prune_features(X_tr, X_te)
        y_tr_trans = self._transform_target(y_tr)

        leaderboard = self._train_models(X_tr, y_tr_trans, X_te, y_te)
        leaderboard = leaderboard.sort_values("RMSE_test").reset_index(drop=True)

                # (ADD) Log de resultados por modelo en runs anidados
        if use_mlflow and not leaderboard.empty:
            for _, row in leaderboard.iterrows():
                model_name = str(row["model"])
                with mlflow.start_run(run_name=model_name, nested=True):
                    # log de métricas con nombres “MLflow-friendly”
                    mlflow.log_metrics({
                        "rmse": float(row["RMSE_test"]),
                        "mae": float(row["MAE_test"]),
                        "r2":  float(row["R2_test"]),
                    })
                    # log de “params” mínimos del modelo (si existen en tu DF)
                    # (opcional) añade más columnas que tengas en el leaderboard
                    mlflow.set_tag("model_name", model_name)


        # Persist combined metrics
        out_path = self.metrics_dir / "all_models_metrics.json"
        out_path.write_text(json.dumps(leaderboard.to_dict(orient="records"), indent=2), encoding="utf-8")
        logger.info("Saved leaderboard to %s", out_path)

                # (ADD) Log de artefactos del experimento (solo una vez, en el run padre)
        if use_mlflow:
            # 1) leaderboard como artifact (CSV + JSON)
            try:
                tmp_csv = self.metrics_dir / "all_models_metrics.csv"
                leaderboard.to_csv(tmp_csv, index=False)
                mlflow.log_artifact(str(tmp_csv))
                mlflow.log_artifact(str(out_path))
            except Exception:
                pass

            # 2) feature list (ya la guardas en _prune_features)
            try:
                feature_csv = self.metrics_dir.parent / self.cfg["output"]["feature_list_file"]
                if feature_csv.exists():
                    mlflow.log_artifact(str(feature_csv))
            except Exception:
                pass

            # 3) PNGs / figuras del artifacts_dir (si tus modelos las guardan ahí)
            try:
                art_dir = Path(self.cfg["output"]["artifacts_dir"])
                if art_dir.exists():
                    for p in art_dir.rglob("*.png"):
                        mlflow.log_artifact(str(p))
            except Exception:
                pass

            # cerrar run padre
            mlflow.end_run()


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
