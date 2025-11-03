# MCE – Model Construction and Evaluation (Phase 2)

Team 41 — *Model training, evaluation, and experimentation package* for the **Online News Popularity** project.  
This module encapsulates model development, training, evaluation, and reproducibility practices using MLflow and DVC, following the Cookiecutter standard.

> Folder reference: `MCE/Model_construction_and_evaluation/` (this README belongs inside that folder).

---

## What’s here

- **Configuration & metadata**
  - `pyproject.toml` – project metadata and dependency management.
  - `structure.txt` – overview of the internal folder layout.
  - `.DS_Store` – system file (ignored in version control).
  - `readme.md` – documentation specific to this module.

- **Data**
  - `data/raw/df_ready_for_modeling.csv` – final dataset produced from the DEP stage and used for model training.

- **Notebooks**
  - `notebooks/V2_03_Model_Construction_and_Evaluation.ipynb` – interactive exploration, model tuning and validation before code modularization.

- **Source package** (`src/`)
  - `__init__.py` – initializes the Python package and exposes key modules.

  - `config/experiment.yaml` – YAML configuration file defining model parameters, hyperparameters, and experiment metadata.
  - `ci/pre_commit.yaml` – linting and pre-commit hooks configuration for maintaining code quality.

  - **data/**
    - `load_and_split.py` – loads the preprocessed dataset and splits it into training/validation/test sets.
    - `__init__.py` – data package initializer.

  - **features/**
    - `drop_columns.py` – removes irrelevant or redundant features.
    - `drop_high_corr_features.py` – removes highly correlated features based on a configurable correlation threshold.
    - `__init__.py` – feature package initializer.

  - **models/**
    - `baseline.py` – baseline models for comparison (e.g., linear regression, dummy regressors/classifiers).
    - `boosting.py` – tree-based ensemble models (e.g., XGBoost, LightGBM).
    - `train.py` – orchestrates model training using data and config files.
    - `evaluate.py` – computes evaluation metrics (e.g., MAE, RMSE, R²).
    - `predict.py` – generates predictions from trained models.
    - `__init__.py` – models package initializer.

  - **pipelines/**
    - `training_pipeline.py` – builds a complete scikit-learn pipeline, combining preprocessing, training, and evaluation.
    - `__init__.py` – pipelines package initializer.

  - **scripts/**
    - `train.py` – script entry point for training a model from command line or CI/CD pipeline.
    - `predict.py` – generates predictions given trained models and input data.
    - `evaluate.py` – runs evaluation pipeline over trained models and logs metrics to MLflow.
  
  - **tests/**
    - `test_train.py` – minimal test verifying correct model training pipeline execution.

  - **utils/**
    - `io.py` – utility functions for file input/output operations.
    - `__init__.py` – utils package initializer.

> Source: folder structure snapshot provided on 2025-11-02:contentReference[oaicite:0]{index=0}

---

## Installation

> From the repository root (so relative paths and MLflow tracking work):

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv && . .venv/bin/activate      # macOS/Linux
# or for PowerShell:
.\.venv\Scripts\Activate.ps1
```

## Running the training pipeline

The training_pipeline.py file defines the full pipeline for model construction, integrating the steps of feature engineering, model training, and evaluation.

### Option 1 – Using DVC

Run through the DVC stage defined in dvc.yaml (if linked):
```
# Reproduce the pipeline (triggers training and evaluation)
dvc repro
# Install in editable mode
pip install -e MCE/Model_construction_and_evaluation
```

### Option 2 – Using Python CLI

Execute the pipeline directly from the repository root:

`python -m Model_construction_and_evaluation.src.pipelines.training_pipeline`

This will:

Load the dataset from data/raw/df_ready_for_modeling.csv.

Preprocess it (drop columns, handle correlations).

Train one or more models based on the configuration in config/experiment.yaml.

Log metrics and model artifacts to MLflow (if configured).

Store model outputs and predictions in the appropriate directories.


## MLflow integration

This module supports MLflow tracking for experiment management:

Each training run logs:

Parameters (hyperparameters, preprocessing options)

Metrics (MAE, RMSE, R², etc.)

Artifacts (plots, model files)

The tracking URI can be configured in environment variables or via the MLflow UI.

If run locally, MLflow UI can be started with:

`mlflow ui`

and accessed at http://127.0.0.1:5000.

## Configuration

Key configuration file: src/config/experiment.yaml

Example snippet:
```
experiment:
  name: "phase2_mce"
  model: "GradientBoostingRegressor"
  test_size: 0.2
  random_state: 42
  metrics:
    - mae
    - rmse
    - r2
```

## Folder Structure Overview

```
MCE/Model_construction_and_evaluation/
│
├── data/
│   └── raw/
│       └── df_ready_for_modeling.csv
│
├── notebooks/
│   └── V2_03_Model_Construction_and_Evaluation.ipynb
│
├── src/
│   ├── config/experiment.yaml
│   ├── data/load_and_split.py
│   ├── features/drop_columns.py
│   ├── features/drop_high_corr_features.py
│   ├── models/{baseline, boosting, evaluate, predict, train}.py
│   ├── pipelines/training_pipeline.py
│   ├── scripts/{train, predict, evaluate}.py
│   ├── tests/test_train.py
│   ├── utils/io.py
│   └── ci/pre_commit.yaml
│
├── pyproject.toml
└── readme.md
```

## Development guidelines

Maintain modularity between data handling, feature engineering, and model logic.

Keep experiment configurations in YAML files to ensure reproducibility.

Avoid embedding credentials or large files; rely on DVC for large datasets.

Unit tests under src/tests ensure stability of the training scripts.

All experiment outputs should be logged through MLflow for traceability.


## Deliverables this module supports

Versioned model artifacts (via DVC or MLflow registry).

Logged metrics and comparisons for multiple models.

Documented training and evaluation pipeline.

Reproducible results across environments.


##Troubleshooting

Module not found errors: ensure you run commands from the repository root and install with pip install -e ....

MLflow not logging: verify the MLFLOW_TRACKING_URI environment variable or start a local MLflow server.

DVC issues: run dvc pull or check .dvc/config.local for your credentials if remote storage is used.


## Credits

Data Scientist: Steven

Data Engineer: Ángel

Software Engineer: Ana Karen

ML Engineer: Felipe

DevOps: Luis

(Roles per Phase 2 team roster.)


## License

Academic use for Tecnológico de Monterrey — MNA, Phase 2 (MLOps).
