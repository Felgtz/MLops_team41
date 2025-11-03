# DEP – Data Exploration and Preprocessing
### Phase 2 – Team 41 | Online News Popularity Project

---

## Overview
This module handles the **data exploration, validation, and preprocessing** stages of the MLOps pipeline.  
It is responsible for transforming the raw datasets (from EDA) into **clean, structured, and model-ready data** that can be directly consumed in the model training phase (MCE).

All code in this module follows **Cookiecutter-style modularization**, allowing reproducibility, maintainability, and automation through **DVC pipelines**.

---

## Pipeline Context
This stage corresponds to the **`dep_preprocess`** step defined in `dvc.yaml`:

```yaml
stages:
  dep_preprocess:
    cmd: python -m src.cli preprocess
    wdir: DEP/Data_exploration_and_preprocessing
    deps:
    - data/raw/df_final_validated.csv
    - src/cli.py
    - src/config.py
    - src/data.py
    - src/features.py
    - src/logger.py
    - src/visualization.py
    outs:
    - data/processed/df_ready_for_modeling.csv
    - reports/figures
 ```

## Main purpose:
Generate the df_ready_for_modeling.csv dataset and complementary visual reports (/reports/figures) used in the MCE module.

## Folder Structure
```
DEP/
└── Data_exploration_and_preprocessing/
    ├── data/               # Processed datasets and intermediate files
    ├── models/             # (Optional) intermediate model checkpoints
    ├── notebooks/          # Data validation & transformation experiments
    ├── reports/            # Visual outputs (correlations, distributions)
    ├── src/                # Core preprocessing scripts
    │   ├── cli.py          # CLI entrypoint (connects modules and DVC)
    │   ├── config.py       # Centralized configuration (paths, params)
    │   ├── data.py         # Data loading, validation, and cleaning logic
    │   ├── features.py     # Feature creation, encoding, and scaling
    │   ├── logger.py       # Logging utilities (console & file logging)
    │   └── visualization.py # Plot generation for reports and diagnostics
    ├── pyproject.toml      # Dependencies and environment metadata
    └── requirements.txt    # Python dependencies for this module
```

## Main Components

1. src/cli.py
Entry point for the preprocessing pipeline.
Executed automatically by DVC using:
python -m src.cli preprocess
It coordinates calls to all other scripts (data.py, features.py, etc.) and ensures consistent logging and reproducibility.

2. src/data.py
Handles:
Loading raw data from data/raw/
Column renaming, type conversion, and null-value handling
Validation against schema and expected columns

3. src/features.py
Performs:
Feature selection and encoding
Scaling and normalization
Outlier treatment and final formatting for ML-ready data

4. src/logger.py
Custom logging setup used across all modules to monitor preprocessing execution.
Outputs logs to both console and file (logs/ folder if configured).

5. src/visualization.py
Generates visual aids for EDA and validation reports, including:
Feature correlations
Missing-value heatmaps
Distribution histograms
All generated visuals are stored under:
reports/figures/

## How to Run This Stage Manually

Ensure all dependencies are installed:
pip install -r DEP/Data_exploration_and_preprocessing/requirements.txt

Run preprocessing (standalone):
cd DEP/Data_exploration_and_preprocessing
python -m src.cli preprocess
Or reproduce via DVC:
dvc repro dep_preprocess

## Outputs

data/processed/df_ready_for_modeling.csv → Clean dataset ready for model training.

reports/figures/ → Graphical diagnostics for feature exploration.

## Notes

The module is self-contained but integrated through DVC to ensure reproducibility.

All outputs are versioned using Google Drive remote storage (configured in .dvc/config).

## Contributors (Phase 2)

| Member        | Role              | Contributions                                                       |
| ------------- | ----------------- | ------------------------------------------------------------------- |
| **Ángel**     | Data Engineer     | DVC integration and data versioning                                 |
| **Steven**    | Data Scientist    | Code documentation, structure validation, and pipeline coordination |
| **Ana Karen** | Software Engineer | Support in logging and configuration improvements                   |
| **Felipe**    | ML Engineer       | Validation of processed datasets for modeling                       |
| **Luis**      | DevOps            | Assistance with repository and environment setup                    |

# Last updated: Phase 2 – November 2025
