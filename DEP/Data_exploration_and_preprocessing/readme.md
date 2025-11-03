## How to install on terminal
1. python -m venv mlopstec_env
2. set http_proxy=http://internet.ford.com:83
3. set https_proxy=http://internet.ford.com:83
4. set no_proxy=127.0.0.1,localhost,.ford.com
5. mlopstec_env\Scripts\activate (after running this line you will see mlopstec_env at the beginning of the command line)
6. pip install -e . --extra-index-url https://pypi.org/simple --extra-index-url https://pypi.ford.com/simple

## How to run the program
mlops41 preprocess

## Online News Popularity – MLOps Course Project (Team 41)
This repository contains all code, data pipelines, and documentation for Phase 1 of Team 41’s MLOps capstone at Tecnológico de Monterrey.
The objective is to build a fully reproducible workflow that explores, preprocesses, models, and ultimately deploys predictions for the Online News Popularity dataset.

## Quick links

Project board → https://github.com/<org>/mlops-team41

Dataset source → UCI ML Repository – Online News Popularity

Latest report → reports/ folder (HTML & figures)

## Repository layout
```
project_root/
│
├── notebooks/                     ← Jupyter notebooks (EDA, modeling, …)
│   ├── 02_data_exploration_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/                           ← Re-usable Python package
│   ├── __init__.py
│   ├── config.py                  # central paths & constants
│   ├── data.py                    # load_dataset(), save_dataset(), quick_info()
│   ├── visualization.py           # plot_histograms(), plot_corr_heatmap(), …
│   └── features.py                # scale_features(), run_pca(), …
│
├── data/
│   ├── raw/                       ← original CSV (df_final_validated.csv)
│   ├── interim/                   ← optional, step-wise artefacts
│   └── processed/                 ← df_ready_for_modeling.csv, train/test splits
│
├── reports/
│   └── figures/                   ← automatically saved plots
│
├── tests/                         ← unit tests (pytest)
│
├── pyproject.toml                 ← build / dependency metadata (PEP-621)
├── requirements.txt               ← runtime deps (alt install)
├── requirements-dev.txt           ← lint / test / formatting extras
└── README.md                      ← you are here
```

## Quick start
bash

## 0.  Create & activate a virtual environment (conda, venv, etc.)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

## 1.  Install project in editable mode + dev helpers
pip install -e ".[dev]"            # uses pyproject.toml
**or, if you prefer plain files**
**pip install -r requirements.txt**
**pip install -r requirements-dev.txt**

## 2.  Place the raw dataset in data/raw/
**e.g. data/raw/df_final_validated.csv**

## 3.  Launch JupyterLab
jupyter lab
**open notebooks/02_data_exploration_preprocessing.ipynb and run**

## 4.  (Optional) Run quick CLI helper
mlops41-quickinfo data/raw/df_final_validated.csv

## 5. Project stages
Phase	Notebook / Script	Output artefact	Tools
1. EDA & Data Validation	02_data_exploration_preprocessing.ipynb	reports/figures/*.png	pandas, seaborn
2. Feature Pre-processing	same notebook	data/processed/df_ready_for_modeling.csv	scikit-learn
3. Model Training & Eval	03_model_training.ipynb	models/*.pkl, metrics JSON	scikit-learn, DVC
4. Packaging & Deployment	(upcoming)	REST API / Docker image	FastAPI, Docker, GitHub Actions
5. Key helper functions
python

from src import (
    load_dataset, quick_info,
    plot_histograms, plot_corr_heatmap,
    scale_features, run_pca
)
from src.config import RAW_DATA_PATH, READY_DATA_PATH

df = load_dataset(RAW_DATA_PATH)
quick_info(df)

plot_corr_heatmap(df, save=True)

df_scaled = scale_features(df, exclude=["shares"])
run_pca(df_scaled, hue=None, save=True)
All plots are automatically saved to reports/figures/.

## 6. Testing & linting
bash

pytest -q                 # unit tests
black . && isort .        # formatting
flake8                    # static analysis
pre-commit install        # set up git hooks

## 7. Authors
Angel Iván Ahumada Arguelles — Data Engineer
Steven Sebastian Brutscher Cortez — Data Scientist
Ana Karen Estupiñán Pacheco — Software Engineer
Felipe de Jesús Gutiérrez Dávila — ML Engineer

## 8. License
This project is released under the MIT License – see LICENSE for details.

## 9. Acknowledgements
UCI Machine Learning Repository for providing the Online News Popularity dataset.
The Tecnológico de Monterrey MLOps course team for guidance and feedback.

