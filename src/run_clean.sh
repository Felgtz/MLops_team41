#!/usr/bin/env bash
set -e

mkdir -p reports/clean

# Ejecutar notebook de cleaning
python -m papermill "01_EDA_and_Data_Cleaning.ipynb" \
                    "reports/clean/CLEAN_run.ipynb"

# Exportar evidencia HTML
python -m jupyter nbconvert --to html --no-input \
                    --output "reports/clean/CLEAN_run.html" \
                    "reports/clean/CLEAN_run.ipynb"

# Validar salida esperada
test -f "data/processed/clean.csv" || { echo "[CLEAN] FALTA data/processed/clean.csv"; exit 1; }
echo "[CLEAN] OK: data/processed/clean.csv generado"
