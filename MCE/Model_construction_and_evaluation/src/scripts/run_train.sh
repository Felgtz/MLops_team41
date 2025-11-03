#!/usr/bin/env bash
set -euo pipefail

KERNEL="${1:-python3}"   # kernel a usar (por defecto python3)
NB_IN="EDA_MLops_team41 ML.ipynb"          # <-- ajusta si tu notebook tiene otro nombre
NB_OUT="reports/models/ML_run.ipynb"

# Carpetas de salida
mkdir -p reports/models models

# 1) Ejecuta el notebook con papermill
python -m papermill "$NB_IN" "$NB_OUT" -k "$KERNEL"

# 2) Convierte a HTML como evidencia
python -m jupyter nbconvert --to html --no-input \
  --output-dir "reports/models" "$NB_OUT"

# 3) Validaciones m??nimas de artefactos (ajusta si cambian nombres)
test -f "reports/models/metrics.json"
test -f "reports/models/cv_results_summary.csv"
test -f "models/Linear_Regression.joblib"
test -f "models/KNN.joblib"

echo "[TRAIN] OK: modelos y reportes generados"
