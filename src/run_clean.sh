#!/usr/bin/env bash
set -e

mkdir -p reports/clean

echo "Input Notebook:  01_EDA_and_Data_Cleaning.ipynb"
echo "Output Notebook: reports/clean/CLEAN_run.ipynb"

# Ejecuta el notebook
python -m papermill "01_EDA_and_Data_Cleaning.ipynb" "reports/clean/CLEAN_run.ipynb" -k python3

# Exporta HTML en el MISMO directorio del notebook (sin duplicar la ruta)
python -m jupyter nbconvert --to html --no-input \
  --output "CLEAN_run.html" \
  "reports/clean/CLEAN_run.ipynb"
