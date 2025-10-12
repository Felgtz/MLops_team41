#!/usr/bin/env bash
set -e

mkdir -p reports/models models

papermill "EDA_MLops_team41 ML.ipynb" "reports/models/ML_run.ipynb" -k python3

jupyter nbconvert --to html --no-input --output-dir "reports/models" "reports/models/ML_run.ipynb"

test -f "reports/models/metrics.json"
test -f "reports/models/cv_results_summary.csv"
test -f "models/Linear_Regression.joblib"
test -f "models/KNN.joblib"

echo "[TRAIN] OK: modelos y reportes generados"
