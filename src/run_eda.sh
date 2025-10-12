#!/usr/bin/env bash
set -e

mkdir -p reports/eda

# Ejecutar notebook de EDA
python -m papermill "EDA_MLops_team41 ML.ipynb" \
                    "reports/eda/EDA_run.ipynb"

# Exportar evidencia HTML
python -m jupyter nbconvert --to html --no-input \
                    --output "reports/eda/EDA_run.html" \
                    "reports/eda/EDA_run.ipynb"

# Validar dependencia
test -f "data/processed/clean.csv" || { echo "[EDA] FALTA data/processed/clean.csv"; exit 1; }
echo "[EDA] OK: evidencia generada y clean.csv presente"
