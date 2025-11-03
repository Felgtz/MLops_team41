# Phase 2 - Machine Learning Project Management (MLOps Implementation)
### Team 41 - Online News Popularity Project

---

## Objective
This project demonstrates the ability to manage and extend a **Machine Learning system lifecycle** through **MLOps principles**, including project structuring, code refactoring, experiment tracking, and model versioning using **DVC** and **MLflow**.  
It continues the work developed in *Phase 1*, improving maintainability, traceability, and reproducibility through the **Cookiecutter template** and robust pipeline practices.

---

## Team Members and Roles

| Role | Member | Responsibilities |
|------|---------|------------------|
| **Data Engineer** | Ángel | DVC configuration, data versioning, pipeline integration |
| **Data Scientist** | Steven | Exploratory Data Analysis (EDA), documentation, experiment design |
| **Software Engineer** | Ana Karen | MLflow integration, logging, and visualization of experiments |
| **ML Engineer** | Felipe | Model training, evaluation, and refactoring of model scripts |
| **DevOps Engineer** | Luis | Repository management, workflow automation, and deployment scripts |

---

## Repository Structure (Main Branch)

repo-root/
│
├── .dvc/ # DVC configuration folder (contains remote setup)
│ ├── config # Remote configuration (GDrive link, general info)
│ ├── config.local.example # Example of local credentials template
│
├── .gitignore # Ignored files (caches, pycache, checkpoints)
├── dvc.yaml # Pipeline definition (links EDA, DEP, and MCE stages)
├── dvc.lock # Version lock of pipeline artifacts
│
├── EDA/ # Exploratory Data Analysis Phase
│ └── online-news-popularity/
│ ├── configs/ # YAML configurations (paths, cleaning parameters)
│ ├── data/ # DVC-tracked data (raw.dvc, processed.dvc)
│ ├── notebooks/ # Jupyter notebooks for initial analysis
│ ├── src/ # Core modules (cleaning, features, pipeline)
│ │ ├── cleaning/ # Cleaning functions (imputation, coercion, filters)
│ │ ├── features/ # Feature grouping and transformations
│ │ ├── pipeline/ # Cleaning pipeline orchestrator
│ │ ├── io.py # Input/output data handling
│ │ └── init.py # Module initialization
│ ├── pyproject.toml # Project dependencies and metadata
│ ├── requirements.txt # Package requirements for EDA stage
│ └── readme.md # Local README for this module
│
├── DEP/ # Data Exploration and Preprocessing Phase
│ └── Data_exploration_and_preprocessing/
│ ├── data/ # Intermediate and processed datasets
│ ├── models/ # Placeholder for intermediate models
│ ├── notebooks/ # Jupyter notebooks for transformation and validation
│ ├── reports/ # Visual outputs (figures, correlations)
│ ├── src/ # Preprocessing and feature engineering scripts
│ │ ├── cli.py # CLI entrypoint for preprocessing steps
│ │ ├── config.py # Configuration management for preprocessing
│ │ ├── data.py # Data loading and cleaning functions
│ │ ├── features.py # Feature creation and encoding logic
│ │ ├── logger.py # Logging utilities for pipeline monitoring
│ │ └── visualization.py # Visualization helpers (plots, summaries)
│ ├── pyproject.toml # Project metadata and dependencies
│ └── requirements.txt # Python dependencies for this phase
│
├── MCE/ # Model Construction and Evaluation Phase
│ └── Model_construction_and_evaluation/
│ ├── data/ # Model-ready datasets
│ ├── notebooks/ # Model training & evaluation notebooks
│ ├── src/ # Main ML source code
│ │ ├── models/ # Training, evaluation, prediction scripts
│ │ ├── pipelines/ # Training pipeline orchestration
│ │ ├── scripts/ # Entry scripts for CLI and experiment runs
│ │ ├── config/ # Experiment configuration YAMLs
│ │ └── utils/ # Helper functions (I/O, logging, etc.)
│ ├── pyproject.toml # Metadata and dependencies
│ └── readme.md # Local README for MCE module
│
├── Fase1/ # Legacy Phase 1 (reference data, reports, notebooks)
│ ├── data/ # Original dataset versions
│ ├── notebooks/ # Jupyter notebooks from Phase 1
│ ├── reports/ # PDF and presentation deliverables
│ └── README.md # Phase 1 documentation
│
└── README.md (this file)


---

## ⚙️ Pipeline Overview

The project’s pipeline integrates **DVC** and **MLflow** for reproducible experimentation and structured MLOps workflow.

### Stages (from `dvc.yaml`)
| Stage | Directory | Description |
|--------|------------|-------------|
| `eda_clean` | `EDA/online-news-popularity/` | Cleans and validates raw datasets |
| `dep_preprocess` | `DEP/Data_exploration_and_preprocessing/` | Generates model-ready processed datasets |
| `train_eval` *(planned)* | `MCE/Model_construction_and_evaluation/` | Trains and evaluates models, logs experiments to MLflow |

---

## Tools & Technologies

| Category | Tools / Libraries |
|-----------|------------------|
| **ML & Data** | Python, NumPy, Pandas, Scikit-learn |
| **Tracking & Versioning** | DVC, MLflow, Google Drive Remote |
| **Visualization** | Matplotlib, Seaborn, MLflow UI |
| **Environment & Structure** | Cookiecutter, Git, Pyproject (Poetry-style layout) |
| **Collaboration** | GitHub, Google Drive, Canvas |

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- DVC (`pip install dvc[gdrive]`)
- MLflow (`pip install mlflow`)
- Clone the repository:

git clone https://github.com/Felgtz/MLops_team41.git
cd MLops_team41

### Setup

1. Create and activate a virtual environment:
python -m venv .venv
..venv\Scripts\activate
pip install -r DEP/Data_exploration_and_preprocessing/requirements.txt

2. Retrieve datasets tracked with DVC:
dvc pull

3. Reproduce pipeline:
dvc repro

4. Launch MLflow tracking UI (for local visualization):
mlflow ui --port 5000


---

## Deliverables (Phase 2)

| Deliverable | Description | Status |
|--------------|-------------|---------|
| **Refactored Project Structure** | Implemented via Cookiecutter | ✅ Completed |
| **DVC Integration** | Pipeline reproducibility and dataset versioning | ✅ Completed |
| **MLflow Integration** | Experiment tracking and metrics logging | ✅ In progress |
| **Executive Presentation (PDF)** | Summary of methodology and results | ⏳ Pending |
| **Final Video (≤ 5 min)** | Explanation and walkthrough of project | ⏳ Pending |
| **README Documentation** | Technical and structural documentation | ✅ This file |

---

## Quick Reference

### DVC Commands

Track data remotely
dvc add data/raw
dvc push

Pull latest version
dvc pull

Run pipeline
dvc repro

### MLflow Commands

Launch MLflow tracking UI
mlflow ui --port 5000

Log new run (within code)
mlflow.start_run()
mlflow.log_param("param_name", value)
mlflow.log_metric("metric_name", value)
mlflow.end_run()


---

## Results and Next Steps
Phase 2 consolidates the **MLOps workflow** by ensuring traceable datasets, modularized code, and reproducible experiments.  
The upcoming Phase 3 will extend these results toward **deployment and model monitoring**.

---

## References
- Cookiecutter Data Science Template – [drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)  
- DVC Documentation – [https://dvc.org/doc](https://dvc.org/doc)  
- MLflow Tracking – [https://mlflow.org/docs/latest/tracking.html](https://mlflow.org/docs/latest/tracking.html)  
- Scikit-learn Pipelines – [https://scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html)  
- Tec de Monterrey MLOps Program – Phase 2 Guidelines (Canvas)

---

**Ready to upload to the repository.**
Save as `README_Phase2_Team41.md` in the main branch root.
