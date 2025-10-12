#!/usr/bin/env bash
set -e
# Ejecuta el notebook de EDA leyendo clean.csv y genera evidencia

mkdir -p reports/eda

# 1) Corre el notebook de EDA (nota: el nombre tiene espacio)
papermill "EDA_MLops_team41 ML.ipynb" \
          "reports/eda/EDA_run.ipynb"

# 2) Exporta a HTML
jupyter nbconvert --to html --no-input \
          --output "reports/eda/EDA_run.html" \
          "reports/eda/EDA_run.ipynb"

# 3) Valida dependencia
test -f "data/processed/clean.csv"
echo "[EDA] OK: evidencia generada y clean.csv presente"
