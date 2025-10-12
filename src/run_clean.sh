#!/usr/bin/env bash
set -e
# Ejecuta el notebook de CLEANING y valida que exista el CSV limpio

mkdir -p reports/clean

# 1) Corre el notebook de Cleaning (ajusta el nombre si cambiara)
papermill "01_EDA_and_Data_Cleaning.ipynb" \
          "reports/clean/CLEAN_run.ipynb"

# 2) Exporta a HTML (evidencia visual)
jupyter nbconvert --to html --no-input \
          --output "reports/clean/CLEAN_run.html" \
          "reports/clean/CLEAN_run.ipynb"

# 3) Valida que el notebook haya escrito el CSV limpio esperado
test -f "data/processed/clean.csv"
echo "[CLEAN] OK: data/processed/clean.csv generado"
