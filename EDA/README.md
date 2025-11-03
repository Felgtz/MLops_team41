# EDA – Online News Popularity (Phase 2)

Team 41 — *Exploratory Data Analysis & Cleaning package* for the **Online News Popularity** project.  
This module standardizes data cleaning and EDA code under a Cookiecutter-style layout and exposes a small pipeline you can run as a script or via DVC stages.

> Folder reference: `EDA/online-news-popularity/` (this README belongs inside that folder).

---

## What’s here

- **Configs**
  - `configs/paths.yaml` – canonical locations for raw/processed data files used in this module.
  - `configs/clean.yaml` – cleaning options (imputation, coercions, URL filters, etc).

- **Data (DVC-tracked)**
  - `data/raw.dvc` – pointer to the raw CSV inputs.
  - `data/processed.dvc` – pointer to the cleaned/validated outputs.

- **Notebooks**
  - `notebooks/01_EDA_and_Data_Cleaning.ipynb` – original exploratory pass.
  - `notebooks/V2_01_EDA_and_Data_Cleaning.ipynb` – refactor/second pass.

- **Python package** (`src/online_news_popularity/`)
  - `pipeline/clean_dataset.py` – CLI entry point that reads raw data, applies the cleaning pipeline and writes the validated dataset.
  - `cleaning/*.py` – modular cleaning steps (type coercion, imputations, rule enforcement, URL filters, calibration).
  - `features/groups.py` – helper(s) for feature grouping used during EDA/cleaning.
  - `io.py` – typed I/O utilities.
  - `__init__.py` – package metadata/exports.

- **Build/metadata**
  - `pyproject.toml`, `requirements.txt` – packaging and dependencies.
  - `online_news_popularity.egg-info/` – build artifacts (generated).

---

## Installation

> From the repository root (so relative paths and DVC stages work):

```bash
# 0) (Optional) create & activate a virtual env
python -m venv .venv && . .venv/bin/activate        # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1                  # Windows PowerShell

# 1) Install this module in editable mode with its deps
pip install -e EDA/online-news-popularity
```

If your data is tracked with DVC remotes, pull it first:
```
# at repo root
dvc pull
```

Remote DVC is already configured; if you need credentials locally, follow the repo root README instructions.

## Quick start (CLI)

You can run the cleaning pipeline directly with the module entry script:

```
# from repo root
python -m online_news_popularity.pipeline.clean_dataset \
  --raw EDA/online-news-popularity/data/raw/online_news_modified.csv \
  --reference EDA/online-news-popularity/data/raw/online_news_original.csv \
  --final EDA/online-news-popularity/data/processed/df_final_validated.csv
```

The command:
reads --raw (modified CSV) and --reference (original CSV),
applies the configured cleaning steps (type coercion, imputations, rule enforcement, URL filtering, calibration),
writes --final cleaned dataset and an audit file (if enabled by configs).

Paths above match configs/paths.yaml.
You can change them there and omit CLI flags, or pass the flags explicitly to override.

## Running as a DVC stage

A DVC stage (declared at repo root) already wraps this step so it becomes reproducible and cacheable. Typical flow:

```
# (at repo root)
dvc repro eda_clean
```

This will:
resolve the working directory to EDA/online-news-popularity,
use the declared deps (raw CSVs + pipeline code),
generate outs under data/processed/… and any audit artifacts.
(Stage name may vary if your team renamed it; check dvc.yaml in the repo root.)

## Key module internals

src/online_news_popularity/cleaning/type_coercion.py
Enforces expected dtypes (e.g., numeric/categorical) prior to imputations.

…/cleaning/imputation.py
Imputation policies configurable from configs/clean.yaml.

…/cleaning/enforcement.py
Domain rules (bounds, logical constraints) applied post-imputation.

…/cleaning/url_filters.py
URL parsing and filtering (e.g., malformed or irrelevant sources).

…/cleaning/calibration.py
Optional re-calibration or normalization steps for specific columns.

…/pipeline/clean_dataset.py
Orchestrates I/O, loads configs, sequences the cleaning steps, writes outputs.

…/features/groups.py
Utilities to organize/aggregate feature families during EDA.

## Configuration

Paths – edit configs/paths.yaml to point to your raw and processed locations in the repo (kept relative for portability).

Cleaning – tune configs/clean.yaml to enable/disable steps and adjust parameters (e.g., imputer strategies, allowed ranges).

After config changes, re-run:

`dvc repro eda_clean`

DVC will detect what changed and only re-execute necessary steps.

## Development tips

Keep notebooks for exploration only; ensure production cleaning logic lives in src/…/cleaning/*.py and pipeline/clean_dataset.py.

Any change in src/… or configs/*.yaml should be accompanied by a dvc repro and a commit including the updated dvc.lock.

Large CSVs remain out of Git history; they are referenced by the .dvc files in data/. Use dvc add if you introduce new data files/folders.

Minimal make-do examples (Windows PowerShell):
```
# Install
pip install -e EDA/online-news-popularity

# Pull DVC data (if needed)
dvc pull

# Run cleaning with defaults via stage
dvc repro eda_clean

# Or run explicitly
python -m online_news_popularity.pipeline.clean_dataset `
  --raw EDA/online-news-popularity/data/raw/online_news_modified.csv `
  --reference EDA/online-news-popularity/data/raw/online_news_original.csv `
  --final EDA/online-news-popularity/data/processed/df_final_validated.csv
```

### Deliverables this module supports

Reproducible cleaned dataset (data/processed/df_final_validated.csv) for downstream DEP/MCE steps.

Auditable run (DVC DAG + lock), allowing graders to re-run and verify outputs.

Exploratory notebooks for reviewers to understand the initial analysis.

## Troubleshooting

DVC can’t find remote or asks for auth
Ensure your local DVC credentials are set in .dvc/config.local as described in the repo root README (not committed).

Paths mismatch
Align configs/paths.yaml with your intended locations, or pass CLI paths explicitly.

New raw files
Add with dvc add EDA/online-news-popularity/data/raw/<file>.csv, commit the resulting .dvc.

## Credits

Data Scientist: Steven

Data Engineer: Ángel

Software Engineer: Ana Karen

ML Engineer: Felipe

DevOps: Luis

(Roles per Phase 2 team roster.)

## License
Academic use for Tecnológico de Monterrey — MNA, Phase 2 (MLOps).
