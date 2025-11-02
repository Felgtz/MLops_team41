#!/usr/bin/env python
"""
scripts/train.py — Runner no-invasivo con MLflow (opcional por YAML).

Uso:
  # usando el config por defecto
  py -m src.scripts.train --config src\config\experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

# Importa tu pipeline sin tocarlo
from src.pipelines.training_pipeline import TrainingPipeline

# MLflow es opcional: sólo se usa si está instalado y el YAML lo pide
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None  # permitirá correr sin MLflow instalado

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena el experimento definido en un YAML (con MLflow opcional).")
    p.add_argument(
        "--config",
        "-c",
        default="src/config/experiment.yaml",
        type=str,
        help="Ruta al YAML del experimento.",
    )
    return p.parse_args()


def maybe_load_yaml(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = maybe_load_yaml(cfg_path)

    # Bandera YAML (seguimos tu convención)
    tracking_cfg = cfg.get("tracking", {}) or {}
    use_mlflow = bool(tracking_cfg.get("use_mlflow", False))

    # Inicializa MLflow sólo si:
    #  - el usuario lo pidió en YAML
    #  - y la librería está instalada/importable
    if use_mlflow and mlflow is not None:
        mlflow.set_tracking_uri(tracking_cfg.get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(tracking_cfg.get("experiment_name", "default"))

        # Corre el pipeline adentro de un run padre
        with mlflow.start_run(run_name="training-pipeline"):
            # Tags y params “macro” (defensivos por si cambian llaves)
            try:
                mlflow.set_tags({
                    "phase": "training",
                    "pipeline": "TrainingPipeline",
                    "cfg_path": str(cfg_path),
                })
                mlflow.log_params({
                    "data.path": cfg.get("data", {}).get("path", ""),
                    "data.target": cfg.get("data", {}).get("target_column", ""),
                    "seed": cfg.get("seed", None),
                })
            except Exception:
                pass

            # Ejecuta el pipeline “tal cual”
            leaderboard = TrainingPipeline(cfg_path).run()

            # Loggea artefactos de salida si existen
            try:
                # rutas declaradas en tu YAML de salida
                out_root = Path(cfg.get("output", {}).get("artifacts_dir", "outputs"))
                models_subdir = cfg.get("output", {}).get("models_subdir", "models")
                metrics_subdir = cfg.get("output", {}).get("metrics_subdir", "metrics")
                feature_list_file = cfg.get("output", {}).get("feature_list_file", "features_used.csv")

                leaderboard_csv = out_root / cfg.get("output", {}).get("leaderboard_filename", "leaderboard.csv")
                all_metrics_json = out_root / metrics_subdir / "all_models_metrics.json"
                features_csv = out_root / feature_list_file

                # si tu pipeline ya los genera, los subimos a MLflow
                for p in [leaderboard_csv, all_metrics_json, features_csv]:
                    if Path(p).exists():
                        mlflow.log_artifact(str(p))
            except Exception:
                logger.exception("No se pudieron loggear artefactos en MLflow (post-ejecución).")

            # imprime algo útil en consola
            if "RMSE_test" in leaderboard.columns:
                print("\n=== Leaderboard (top 5) ===")
                print(leaderboard[["model", "RMSE_test", "MAE_test", "R2_test"]].head().to_string(index=False))
    else:
        # Sin MLflow: ejecuta igual
        logger.info("MLflow desactivado (o no instalado). Ejecutando pipeline sin tracking.")
        leaderboard = TrainingPipeline(cfg_path).run()
        if "RMSE_test" in leaderboard.columns:
            print("\n=== Leaderboard (top 5) ===")
            print(leaderboard[["model", "RMSE_test", "MAE_test", "R2_test"]].head().to_string(index=False))


if __name__ == "__main__":
    main()

